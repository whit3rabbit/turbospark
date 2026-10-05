import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The auto-memory feature: the store, the `memory` tool, the prompt
/// section, and the `#` quick-save path.
///
/// Fixture rule (`swift/CLAUDE.md` Gotcha 43 and the `/private/var` trap):
/// every scratch directory is CREATED FIRST and resolved after, so tests
/// compare real paths; and every store under test is built with an injected
/// base, so nothing here reads or writes the process's own memory root.
final class MemoryFeatureTests: XCTestCase {
    // MARK: - Fixtures

    /// Creates, then resolves: `resolvingSymlinksInPath` on a nonexistent
    /// path is a no-op on macOS, where `/tmp` is a symlink to `/private/tmp`.
    private func makeScratchDirectory(_ label: String) -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("MemoryTests-\(label)-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.resolvingSymlinksInPath()
    }

    private func makeProject(root: URL) -> AppProject {
        AppProject(name: "Memory Demo", rootDirectoryPath: root.path)
    }

    func testLedgerApprovalSupersedesSameProjectTopic() throws {
        let ledger = MemoryLedgerStore(base: makeScratchDirectory("ledger-supersession"))
        let old = MemoryClaim(scope: "project:one", kind: "decision", text: "Use obsolete-format",
                              status: .active, projectTopicName: "format")
        let replacement = MemoryClaim(scope: "project:one", kind: "decision", text: "Use current-format",
                                      projectTopicName: "format")
        try ledger.propose(old)
        try ledger.propose(replacement)
        try ledger.approve(replacement.id)
        let state = ledger.snapshot()
        XCTAssertEqual(state.claims.first { $0.id == old.id }?.status, .superseded)
        XCTAssertEqual(state.claims.first { $0.id == replacement.id }?.supersedesClaimID, old.id)
        XCTAssertEqual(ledger.search("current", scope: "project:one").map(\.id), [replacement.id])
        XCTAssertTrue(ledger.search("obsolete", scope: "project:one").isEmpty)
    }

    func testLegacyImportIsIdempotentAndUnverified() {
        var state = MemoryLedgerState()
        let topics = [
            ("memory:file:projects/demo/memory/release.md", Data("Original topic".utf8)),
            ("memory:file:projects/demo/legacy/IMPORTED_TOPICS.md", Data("Archive only".utf8)),
        ]
        MemoryLedgerStore.importLegacy(profileText: "- Old preference\n", topics: topics, into: &state)
        XCTAssertEqual(state.claims.count, 2)
        XCTAssertEqual(Set(state.claims.map(\.scope)), ["profile", "project:demo"])
        XCTAssertTrue(state.claims.allSatisfy {
            $0.status == .legacy && $0.confidence == "source-unverified" && $0.evidence.isEmpty
        })
        XCTAssertEqual(state.claims.first { $0.scope == "project:demo" }?.legacySourceText,
                       "Original topic")
        MemoryLedgerStore.importLegacy(profileText: "Changed", topics: topics, into: &state)
        XCTAssertEqual(state.claims.count, 2)
    }

    func testLedgerSearchKeepsProjectsIsolatedAndRequiresApproval() throws {
        let ledger = MemoryLedgerStore(base: makeScratchDirectory("ledger-scope"))
        let profile = MemoryClaim(scope: "profile", kind: "preference", text: "Prefer concise answers",
                                  status: .active)
        let first = MemoryClaim(scope: "project:one", kind: "decision", text: "Use the green format",
                                status: .active)
        let second = MemoryClaim(scope: "project:two", kind: "decision", text: "Use the blue format",
                                 status: .active)
        let pending = MemoryClaim(scope: "project:one", kind: "decision", text: "Use the red format")
        for claim in [profile, first, second, pending] { try ledger.propose(claim) }
        XCTAssertEqual(Set(ledger.search("format", scope: "project:one").map(\.id)), [first.id])
        XCTAssertEqual(Set(ledger.search("format", scope: nil).map(\.id)), [])
        XCTAssertEqual(ledger.search("concise", scope: "project:one").map(\.id), [profile.id])
    }

    @MainActor
    func testRecallUsesTheUserTurnAndKeepsOneSnapshotForRetries() throws {
        let model = AppModel()
        model.memoryEnabled = true
        let chat = AppChat(title: "recall")
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.promptText = "orchid"
        model.mutateTurnMessages(for: chat.id) {
            $0.append(AppChatMessage(role: .user, content: "linden"))
        }
        let matching = MemoryClaim(scope: "profile", kind: "preference",
                                   text: "Linden preference", status: .active)
        let unrelated = MemoryClaim(scope: "profile", kind: "preference",
                                    text: "Orchid preference", status: .active)
        let later = MemoryClaim(scope: "profile", kind: "preference",
                                text: "Another linden preference", status: .active)
        let ledger = MemoryLedgerStore.shared
        defer {
            try? ledger.forget(matching.id)
            try? ledger.forget(unrelated.id)
            try? ledger.forget(later.id)
        }
        try ledger.propose(matching)
        try ledger.propose(unrelated)
        XCTAssertEqual(model.memoryRecall(for: chat.id, project: nil).map(\.id), [matching.id])
        try ledger.propose(later)
        XCTAssertEqual(model.memoryRecall(for: chat.id, project: nil).map(\.id), [matching.id])
        model.mutateTurnMessages(for: chat.id) {
            $0.append(AppChatMessage(role: .user, content: "linden"))
        }
        XCTAssertEqual(Set(model.memoryRecall(for: chat.id, project: nil).map(\.id)),
                       [matching.id, later.id])
    }

    func testReviewBatchPersistsUntilItsClaimIsForgotten() throws {
        let ledger = MemoryLedgerStore(base: makeScratchDirectory("review-batch"))
        let proposal = MemoryClaim(scope: "profile", kind: "preference", text: "Prefers tea")
        try ledger.propose(proposal)
        let pending = ledger.snapshot()
        XCTAssertEqual(pending.reviewBatches?.first?.claimIDs, [proposal.id])
        XCTAssertEqual(try JSONDecoder().decode(MemoryLedgerState.self,
                                               from: JSONEncoder().encode(pending)).reviewBatches,
                       pending.reviewBatches)
        try ledger.approve(proposal.id)
        XCTAssertEqual(ledger.explain(proposal.id)?.status, .active)
        try ledger.forget(proposal.id)
        XCTAssertTrue(ledger.snapshot().reviewBatches?.isEmpty == true)
    }

    func testRealEncoderColdAndRepeatedCallMeasurement() async throws {
        guard let model = ProcessInfo.processInfo.environment["TURBOSPARK_ENCODER_TEST_MODEL"] else {
            throw XCTSkip("Set TURBOSPARK_ENCODER_TEST_MODEL to a local encoder directory.")
        }
        func rssKiB() -> Int? {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/ps")
            process.arguments = ["-o", "rss=", "-p", "\(ProcessInfo.processInfo.processIdentifier)"]
            let pipe = Pipe()
            process.standardOutput = pipe
            try? process.run()
            process.waitUntilExit()
            guard let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                                    encoding: .utf8) else { return nil }
            return Int(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let before = rssKiB()
        let coldStart = Date()
        let cold = try await TurboSparkEmbedding.encode(text: "A short local memory", modelPath: model)
        let coldSeconds = Date().timeIntervalSince(coldStart)
        let afterCold = rssKiB()
        let warmStart = Date()
        let warm = try await TurboSparkEmbedding.encode(text: "A short local memory", modelPath: model)
        let warmSeconds = Date().timeIntervalSince(warmStart)
        let afterWarm = rssKiB()
        XCTAssertEqual(cold.count, 384)
        XCTAssertEqual(cold, warm)
        print("encoder cache cold=\(coldSeconds)s warm=\(warmSeconds)s rss_kib=\(String(describing: before))/\(String(describing: afterCold))/\(String(describing: afterWarm))")
    }

    func testLedgerInvalidatesEditedOrDeletedEvidence() throws {
        let ledger = MemoryLedgerStore(base: makeScratchDirectory("ledger-edit"))
        let chatID = UUID()
        let message = AppChatMessage(role: .user, content: "I prefer short replies.")
        let evidence = MemoryEvidence(chatID: chatID, messageID: message.id,
                                      quote: "short replies", source: message.content)
        let claim = MemoryClaim(scope: "profile", kind: "preference", text: "Prefers short replies",
                                status: .active, evidence: [evidence])
        try ledger.propose(claim)
        try ledger.invalidateEvidence(chatID: chatID, messages: [message])
        XCTAssertEqual(ledger.explain(claim.id)?.status, .active)
        var edited = message
        edited.content = "I prefer detailed replies."
        try ledger.invalidateEvidence(chatID: chatID, messages: [edited])
        XCTAssertEqual(ledger.explain(claim.id)?.status, .pending)
        XCTAssertTrue(ledger.explain(claim.id)?.evidence.isEmpty == true)

        let another = MemoryClaim(scope: "profile", kind: "preference", text: "Short replies",
                                  status: .active, evidence: [evidence])
        try ledger.propose(another)
        try ledger.invalidateEvidence(chatID: chatID, messages: [])
        XCTAssertEqual(ledger.explain(another.id)?.status, .pending)
    }

    func testLedgerForgetSuppressesSourceAndClearsLinkedProposals() throws {
        let ledger = MemoryLedgerStore(base: makeScratchDirectory("ledger-forget"))
        let chatID = UUID()
        let messageID = UUID()
        let source = "I prefer short replies."
        let evidence = MemoryEvidence(chatID: chatID, messageID: messageID,
                                      quote: source, source: source)
        let active = MemoryClaim(scope: "profile", kind: "preference", text: "Prefers short replies",
                                 status: .active, evidence: [evidence])
        let pending = MemoryClaim(scope: "profile", kind: "preference", text: "Likes brief answers",
                                  evidence: [evidence])
        try ledger.propose(active)
        try ledger.propose(pending)
        try ledger.forget(active.id)
        let state = ledger.snapshot()
        XCTAssertEqual(state.claims.first { $0.id == active.id }?.status, .retracted)
        XCTAssertEqual(state.claims.first { $0.id == pending.id }?.status, .retracted)
        XCTAssertTrue(state.claims.first { $0.id == active.id }?.text.isEmpty == true)
        XCTAssertTrue(state.suppressedSources.contains(
            MemoryLedgerStore.sourceKey(chatID: chatID, messageID: messageID, source: source)))
        XCTAssertTrue(ledger.search("short", scope: nil).isEmpty)
    }

    func testLedgerUpdateRollsBackOnFailure() throws {
        enum Expected: Error { case failure }
        let ledger = MemoryLedgerStore(base: makeScratchDirectory("ledger-rollback"))
        let before = ledger.snapshot()
        XCTAssertThrowsError(try ledger.update { state in
            state.claims.append(MemoryClaim(scope: "profile", kind: "test", text: "Do not save"))
            throw Expected.failure
        })
        XCTAssertEqual(ledger.snapshot(), before)
    }

    func testLedgerRefusesToOverwriteCorruptSnapshot() throws {
        let directory = makeScratchDirectory("ledger-corrupt")
        let url = directory.appendingPathComponent("ledger.json")
        try Data("not JSON".utf8).write(to: url)
        let ledger = MemoryLedgerStore(base: directory)
        XCTAssertThrowsError(try ledger.saveUserClaim("Keep this", scope: "profile"))
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "not JSON")
    }

    @MainActor
    func testMemorySchedulerDueAndCandidateGuards() {
        let now = Date()
        let user = AppChatMessage(role: .user, content: "I prefer concise replies.")
        let assistant = AppChatMessage(role: .assistant, content: "I prefer long replies.")
        let old = AppChat(messages: [user, assistant], updatedAt: now.addingTimeInterval(-601))
        XCTAssertEqual(AppModel.memoryCandidates(chat: old, now: now,
                                                  state: MemoryLedgerState()).map(\.id), [user.id])
        var ghost = old
        ghost.isGhost = true
        XCTAssertTrue(AppModel.memoryCandidates(chat: ghost, now: now,
                                                state: MemoryLedgerState()).isEmpty)
        var fresh = old
        fresh.updatedAt = now.addingTimeInterval(-599)
        XCTAssertTrue(AppModel.memoryCandidates(chat: fresh, now: now,
                                                state: MemoryLedgerState()).isEmpty)
        XCTAssertTrue(AppModel.memoryNightlyIdle(chats: [old], now: now))
        XCTAssertFalse(AppModel.memoryNightlyIdle(chats: [fresh], now: now))
        XCTAssertTrue(AppModel.memoryNightlyIdle(chats: [ghost], now: now))
        var processed = MemoryLedgerState()
        processed.processedSources.insert(MemoryLedgerStore.sourceKey(
            chatID: old.id, messageID: user.id, source: user.content))
        XCTAssertTrue(AppModel.memoryCandidates(chat: old, now: now, state: processed).isEmpty)
        XCTAssertFalse(AppModel.memoryHourlyDue(now: now, lastRun: nil, enabled: false))
        XCTAssertTrue(AppModel.memoryHourlyDue(now: now, lastRun: nil, enabled: true))
        XCTAssertFalse(AppModel.memoryHourlyDue(
            now: now, lastRun: now.addingTimeInterval(-3599), enabled: true))
        XCTAssertTrue(AppModel.memoryHourlyDue(
            now: now, lastRun: now.addingTimeInterval(-3600), enabled: true))
        let threeAM = Calendar.current.date(bySettingHour: 3, minute: 0, second: 0, of: now)!
        let oneAM = Calendar.current.date(bySettingHour: 1, minute: 0, second: 0, of: now)!
        XCTAssertTrue(AppModel.memoryNightlyDue(now: threeAM, lastDay: nil))
        XCTAssertFalse(AppModel.memoryNightlyDue(now: oneAM, lastDay: nil))
    }

    @MainActor
    func testMemoryCaptureRequiresSuppliedExactUserQuoteAndRejectsSecrets() {
        let message = AppChatMessage(role: .user, content: "I prefer concise replies.")
        let valid = ExtractedMemory(scope: "profile", kind: "preference",
                                    text: "Prefers concise replies", messageID: message.id,
                                    quote: "prefer concise replies")
        let chatID = UUID()
        XCTAssertNotNil(AppModel.validatedMemoryClaim(
            valid, chatID: chatID, source: message, projectScope: nil))
        let wrongQuote = ExtractedMemory(scope: "profile", kind: "preference",
                                         text: valid.text, messageID: message.id,
                                         quote: "prefers concise replies")
        XCTAssertNil(AppModel.validatedMemoryClaim(
            wrongQuote, chatID: chatID, source: message, projectScope: nil))
        let wrongID = ExtractedMemory(scope: "profile", kind: "preference",
                                      text: valid.text, messageID: UUID(), quote: valid.quote)
        XCTAssertNil(AppModel.validatedMemoryClaim(
            wrongID, chatID: chatID, source: message, projectScope: nil))
        let assistant = AppChatMessage(id: message.id, role: .assistant, content: message.content)
        XCTAssertNil(AppModel.validatedMemoryClaim(
            valid, chatID: chatID, source: assistant, projectScope: nil))
        let project = ExtractedMemory(scope: "project", kind: "decision",
                                      text: valid.text, messageID: message.id, quote: valid.quote)
        XCTAssertNil(AppModel.validatedMemoryClaim(
            project, chatID: chatID, source: message, projectScope: nil))
        let secret = ExtractedMemory(scope: "profile", kind: "preference",
                                     text: "api_key: sk-abcdefghijklmnopqrstuvwx",
                                     messageID: message.id, quote: valid.quote)
        XCTAssertNil(AppModel.validatedMemoryClaim(
            secret, chatID: chatID, source: message, projectScope: nil))
    }

    @MainActor
    func testMemorySchedulerSkipsWithoutModelAndForegroundCancelsCapture() async {
        let model = AppModel()
        model.memoryEnabled = true
        model.memoryAutoCaptureEnabled = true
        await model.runMemoryJobsIfDue(now: Date())
        XCTAssertNil(model.memoryCaptureTask)
        let pending = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(5)) } catch { }
        }
        model.memoryCaptureTask = pending
        await model.interruptMemoryCaptureForForeground()
        XCTAssertTrue(pending.isCancelled)
    }

    @MainActor
    private func makeProjectModel(project: AppProject) -> AppModel {
        let model = AppModel()
        model.interactionMode = .projects
        model.projects = [project]
        var chat = AppChat(title: "memory chat")
        chat.projectID = project.id
        model.chats = [chat]
        model.selectedChatID = chat.id
        return model
    }

    // MARK: - Paths

    func testTheProjectKeySanitizesAndAddsACollisionSuffix() {
        let key = MemoryStore.projectKey(forProjectRoot: URL(fileURLWithPath: "/Users/x/a b/c++"))
        XCTAssertTrue(key.hasPrefix("-Users-x-a-b-c--"), "unexpected key: \(key)")
        // 8 hex digits, so `/a-b` and `/a.b` (same sanitized prefix) stay
        // distinct directories.
        let suffix = key.suffix(9)
        XCTAssertTrue(suffix.hasPrefix("-"))
        XCTAssertTrue(suffix.dropFirst().allSatisfy { $0.isHexDigit }, "unexpected suffix: \(suffix)")
    }

    func testSymlinkedRootsShareOneKey() throws {
        let target = makeScratchDirectory("target")
        let link = makeScratchDirectory("link-parent").appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertEqual(
            MemoryStore.projectKey(forProjectRoot: link),
            MemoryStore.projectKey(forProjectRoot: target),
            "a symlinked project and its target must share one memory directory")
    }

    // MARK: - Names and slugs

    func testOnlyKebabCaseNamesAreLegal() {
        XCTAssertTrue(MemoryStore.isValidTopicName("a"))
        XCTAssertTrue(MemoryStore.isValidTopicName("deploy-workflow-2"))
        XCTAssertFalse(MemoryStore.isValidTopicName(""))
        XCTAssertFalse(MemoryStore.isValidTopicName("Deploy"))
        XCTAssertFalse(MemoryStore.isValidTopicName("a/b"))
        XCTAssertFalse(MemoryStore.isValidTopicName(".."))
        XCTAssertFalse(MemoryStore.isValidTopicName("-leading"))
        XCTAssertFalse(MemoryStore.isValidTopicName("double--dash"))
        XCTAssertFalse(MemoryStore.isValidTopicName(String(repeating: "a", count: 81)))
    }

    func testSlugsCoerceArbitraryProseAndOptionallyCarryADate() {
        XCTAssertEqual(MemoryStore.slug(from: "My Memory!", dated: false), "my-memory")
        XCTAssertEqual(MemoryStore.slug(from: "  --  ", dated: false), "memory")
        let dated = MemoryStore.slug(from: "deploy note", dated: true)
        XCTAssertTrue(dated.hasSuffix("-deploy-note"))
        // yyyy-MM-dd-, ten characters of date prefix.
        XCTAssertEqual(dated.count - "-deploy-note".count, 10)
    }

    // MARK: - Index text algebra

    func testIndexParsingAndUpsertAlgebra() {
        let index = MemoryStore.upsertingIndexLine(
            "", fileName: "a.md", title: "a", hook: "first")
        XCTAssertEqual(MemoryStore.parseIndex(index).map(\.fileName), ["a.md"])

        let updated = MemoryStore.upsertingIndexLine(
            index, fileName: "a.md", title: "a", hook: "second")
        XCTAssertEqual(MemoryStore.parseIndex(updated).map(\.hook), ["second"])
        XCTAssertEqual(updated.components(separatedBy: "\n").filter { $0.contains("a.md") }.count, 1)

        let withJunk = "# hand written\n" + index
        let keptJunk = MemoryStore.upsertingIndexLine(
            withJunk, fileName: "b.md", title: "b", hook: "other")
        XCTAssertTrue(keptJunk.contains("# hand written"))
        XCTAssertEqual(Set(MemoryStore.parseIndex(keptJunk).map(\.fileName)), ["a.md", "b.md"])

        XCTAssertNil(MemoryStore.removingIndexLine(index, fileName: "missing.md"))
        // Removing b.md from the junk-bearing index leaves the hand-written
        // line AND a.md, unharmed.
        let removed = MemoryStore.removingIndexLine(keptJunk, fileName: "b.md")
        XCTAssertNotNil(removed)
        XCTAssertTrue(removed?.contains("# hand written") ?? false)
        XCTAssertEqual(MemoryStore.parseIndex(removed ?? "").map(\.fileName), ["a.md"])
    }

    // MARK: - Topic files through the store

    func testSaveReadUpdateAndForgetRoundTrip() throws {
        let store = MemoryStore(base: makeScratchDirectory("store"))
        let root = makeScratchDirectory("project")

        let first = try store.saveTopic(
            projectRoot: root, name: "deploy", type: .feedback,
            description: "how deploys go", body: "Use the staging lane first.")
        XCTAssertTrue(first.created)

        let file = try String(contentsOf: first.url, encoding: .utf8)
        XCTAssertTrue(file.contains("name: deploy"))
        XCTAssertTrue(file.contains("description: how deploys go"))
        XCTAssertTrue(file.contains("type: feedback"))
        XCTAssertTrue(file.contains("Use the staging lane first."))

        let second = try store.saveTopic(
            projectRoot: root, name: "deploy", type: .feedback,
            description: "updated hook", body: "Changed body.")
        XCTAssertFalse(second.created, "a second save to the same name is an update")
        XCTAssertEqual(
            try store.readTopic(projectRoot: root, name: "deploy"),
            try String(contentsOf: second.url, encoding: .utf8))

        let index = store.loadIndex(forProjectRoot: root)
        XCTAssertEqual(MemoryStore.parseIndex(index).map(\.hook), ["updated hook"])

        XCTAssertTrue(try store.forgetTopic(projectRoot: root, name: "deploy"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: second.url.path))
        XCTAssertTrue(store.loadIndex(forProjectRoot: root).trimmingCharacters(in: .whitespaces).isEmpty)
        XCTAssertFalse(try store.forgetTopic(projectRoot: root, name: "deploy"))
    }

    func testADescriptionCarryingNewlinesCannotSmuggleIndexRows() throws {
        let store = MemoryStore(base: makeScratchDirectory("smuggle"))
        let root = makeScratchDirectory("project")
        try store.saveTopic(
            projectRoot: root, name: "trap", type: .user,
            description: "line one\n- [evil](evil.md) -- injected", body: "body")
        let hooks = MemoryStore.parseIndex(store.loadIndex(forProjectRoot: root)).map(\.hook)
        XCTAssertEqual(hooks, ["line one - [evil](evil.md) -- injected"])
    }

    func testAnInvalidNameIsRefusedByTheStore() {
        let store = MemoryStore(base: makeScratchDirectory("invalid"))
        let root = makeScratchDirectory("project")
        XCTAssertThrowsError(try store.saveTopic(
            projectRoot: root, name: "../escape", type: .user, description: "d", body: "b"))
    }

    // MARK: - Truncation

    func testTheIndexIsTruncatedWithASelfDescribingWarning() {
        let under = MemoryPromptBuilder.truncatedIndex("- [a](a.md) -- x")
        XCTAssertFalse(under.truncated)
        XCTAssertFalse(under.content.contains("WARNING"))

        var many = ""
        for i in 0..<(MemoryPromptBuilder.maxIndexLines + 5) {
            many = MemoryStore.upsertingIndexLine(
                many, fileName: "n\(i).md", title: "n\(i)", hook: "hook \(i)")
        }
        let lineCapped = MemoryPromptBuilder.truncatedIndex(many)
        XCTAssertTrue(lineCapped.truncated)
        XCTAssertTrue(lineCapped.content.contains("WARNING"))
        XCTAssertTrue(lineCapped.content.contains("lines"))
        XCTAssertFalse(lineCapped.content.contains("hook \(MemoryPromptBuilder.maxIndexLines + 4)"))

        let wideRow = "- [h](h.md) -- " + String(repeating: "x", count: 300) + "\n"
        let huge = String(repeating: wideRow, count: 120)
        let byteCapped = MemoryPromptBuilder.truncatedIndex(huge)
        XCTAssertTrue(byteCapped.truncated)
        XCTAssertTrue(byteCapped.content.utf8.count <= MemoryPromptBuilder.maxIndexBytes + 400)
        // The cut lands on a row boundary: the warning block starts on its
        // own line after a complete one.
        XCTAssertTrue(byteCapped.content.contains("\n\n> WARNING: MEMORY.md is too large"))
    }

    // MARK: - The prompt section

    func testTheSectionNamesTheDirectoryAndCarriesTheIndex() {
        let store = MemoryStore(base: makeScratchDirectory("prompt"))
        let root = makeScratchDirectory("project")
        _ = try? store.saveTopic(
            projectRoot: root, name: "deploys", type: .project,
            description: "how this deploys", body: "body")
        let section = MemoryPromptBuilder.section(store: store, projectRoot: root)
        XCTAssertTrue(section.hasPrefix("## Memory"))
        XCTAssertTrue(section.contains("deploys.md"), "the index rows must reach the prompt")
        XCTAssertTrue(section.contains("how this deploys"))
        XCTAssertTrue(section.contains("`memory` tool"))
        XCTAssertTrue(section.contains("feedback"), "the type vocabulary must be taught")
    }

    func testAnEmptyIndexStillGetsTheSection() {
        let store = MemoryStore(base: makeScratchDirectory("empty"))
        let root = makeScratchDirectory("project")
        let section = MemoryPromptBuilder.section(store: store, projectRoot: root)
        XCTAssertTrue(section.contains("nothing is remembered yet"))
    }

    // MARK: - Assembler injection

    @MainActor
    func testTheMemorySectionIsProjectScopedAndGated() {
        let model = AppModel()
        let previous = MemoryStore.shared.isModelEnabled
        defer { MemoryStore.shared.isModelEnabled = previous }

        // Chat mode: no project, no memory, whatever the toggle says.
        MemoryStore.shared.isModelEnabled = true
        XCTAssertFalse(
            model.buildSystemPrompt(for: nil, userPrompt: "hi").contains("## Memory"))

        let project = AppProject(name: "Demo", rootDirectoryPath: "/tmp/memory-demo")
        MemoryStore.shared.isModelEnabled = false
        XCTAssertFalse(
            model.buildSystemPrompt(for: project, userPrompt: "hi").contains("## Memory"))
        MemoryStore.shared.isModelEnabled = true
        XCTAssertTrue(
            model.buildSystemPrompt(for: project, userPrompt: "hi").contains("## Memory"))
    }

    func testTheSubagentAssemblerCarriesTheMemorySectionToo() throws {
        let agent = try XCTUnwrap(AgentManager.shared.findAgent(name: "general-purpose"))
        let previous = MemoryStore.shared.isModelEnabled
        defer { MemoryStore.shared.isModelEnabled = previous }
        MemoryStore.shared.isModelEnabled = true

        let project = AppProject(name: "Demo", rootDirectoryPath: "/tmp/memory-demo")
        let prompt = SubagentRunner.buildSystemPrompt(for: agent, project: project)
        XCTAssertTrue(prompt.contains("## Memory"), "a subagent without memory diverges from its parent")
        let without = SubagentRunner.buildSystemPrompt(for: agent, project: nil)
        XCTAssertFalse(without.contains("## Memory"))
    }

    func testMemoryJobsDoNotInheritStoredClaimsAsCaptureEvidence() {
        let previous = MemoryStore.shared.isModelEnabled
        defer { MemoryStore.shared.isModelEnabled = previous }
        MemoryStore.shared.isModelEnabled = true
        let agent = AppAgentDefinition(
            name: "memory-capture", displayName: "Memory Capture",
            agentDescription: "Capture", systemPrompt: "Cite supplied messages only.",
            tools: [], maxTurns: 1, omitsProjectInstructions: true)
        let project = AppProject(name: "Demo", rootDirectoryPath: "/tmp/memory-demo")
        let prompt = SubagentRunner.buildSystemPrompt(for: agent, project: project)
        XCTAssertFalse(prompt.contains("## Profile Memory"))
        XCTAssertFalse(prompt.contains("## Memory\n"))
    }

    // MARK: - The memory tool

    private func toolProject(_ root: URL) -> AppProject {
        makeProject(root: root)
    }

    func testTheToolSavesReadsAndForgets() throws {
        let store = MemoryStore(base: makeScratchDirectory("tool"))
        let ledger = MemoryLedgerStore(base: makeScratchDirectory("tool-ledger"))
        store.isModelEnabled = true
        let root = makeScratchDirectory("project")

        let saved = try MemoryToolExecutor.execute(
            arguments: ["action": "save", "name": "Deploy Flow", "content": "Ship on Thursdays."],
            project: toolProject(root), store: store, ledger: ledger)
        XCTAssertTrue(saved.contains("deploy-flow"), saved)

        let proposed = try XCTUnwrap(ledger.snapshot().claims.first)
        XCTAssertEqual(proposed.status, .pending)

        let index = try MemoryToolExecutor.execute(
            arguments: ["action": "read"], project: toolProject(root), store: store, ledger: ledger)
        XCTAssertTrue(index.contains("empty"), "pending proposals must not reach the model")
        try ledger.approve(proposed.id)
        let approvedIndex = try MemoryToolExecutor.execute(
            arguments: ["action": "read"], project: toolProject(root), store: store, ledger: ledger)
        XCTAssertTrue(approvedIndex.contains("Ship on Thursdays."))

        let named = try MemoryToolExecutor.execute(
            arguments: ["action": "read", "name": "deploy-flow"],
            project: toolProject(root), store: store, ledger: ledger)
        XCTAssertTrue(named.contains("Ship on Thursdays."))

        let forgotten = try MemoryToolExecutor.execute(
            arguments: ["action": "forget", "name": "deploy-flow"],
            project: toolProject(root), store: store, ledger: ledger)
        XCTAssertTrue(forgotten.contains("Proposed forgetting"))
        let forgetProposal = try XCTUnwrap(ledger.snapshot().claims.first { $0.requestedForgetID == proposed.id })
        try ledger.approve(forgetProposal.id)

        let empty = try MemoryToolExecutor.execute(
            arguments: ["action": "read"], project: toolProject(root), store: store, ledger: ledger)
        XCTAssertTrue(empty.contains("empty"))
    }

    func testTheToolRefusesWhatItMust() throws {
        let store = MemoryStore(base: makeScratchDirectory("tool-refusal"))
        XCTAssertThrowsError(try MemoryToolExecutor.execute(
            arguments: ["action": "save", "name": "x", "content": "b"],
            project: nil, store: store))

        let root = makeScratchDirectory("project")
        XCTAssertThrowsError(try MemoryToolExecutor.execute(
            arguments: ["action": "save", "name": "x", "content": ""],
            project: toolProject(root), store: store))
        XCTAssertThrowsError(try MemoryToolExecutor.execute(
            arguments: ["action": "reorganize"],
            project: toolProject(root), store: store))
        XCTAssertThrowsError(try MemoryToolExecutor.execute(
            arguments: ["action": "read", "name": "nope"],
            project: toolProject(root), store: store))
    }

    func testTheToolRefusesExecutionWhileMemoryIsDisabled() throws {
        let store = MemoryStore(base: makeScratchDirectory("tool-disabled"))
        let root = makeScratchDirectory("project")
        store.isModelEnabled = false

        XCTAssertThrowsError(try MemoryToolExecutor.execute(
            arguments: ["action": "save", "name": "blocked", "content": "must not persist"],
            project: toolProject(root), store: store))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: store.directory(forProjectRoot: root)
                    .appendingPathComponent("blocked.md").path))
    }

    func testTheToolIsAdvertisedOnlyWhileMemoryIsEnabled() {
        let previous = MemoryStore.shared.isModelEnabled
        defer { MemoryStore.shared.isModelEnabled = previous }

        MemoryStore.shared.isModelEnabled = true
        XCTAssertTrue(AppToolCatalog.tools(for: .coder).contains { $0.function.name == "memory" })
        XCTAssertTrue(AppToolCatalog.allTools.contains { $0.function.name == "memory" })
        MemoryStore.shared.isModelEnabled = false
        XCTAssertFalse(MemoryToolDefinitions.all.contains { $0.function.name == "memory" })
        XCTAssertFalse(AppToolCatalog.tools(for: .coder).contains { $0.function.name == "memory" })
    }

    func testTheToolIsRootedAndGatedLikeAFileWrite() {
        XCTAssertEqual(
            AppToolCatalog.category(for: "memory"), .fileWrite,
            "memory writes gate like file writes, not like automation")
        for name in ["memory", "remember"] {
            XCTAssertTrue(
                AppToolRegistry.isImplemented(name), "\(name) must have an executor")
        }
    }

    // MARK: - The # quick-save

    @MainActor
    func testAHashDraftSavesAMemoryWithoutGenerating() {
        let root = makeScratchDirectory("quicksave-project")
        let model = makeProjectModel(project: makeProject(root: root))
        model.memoryEnabled = true
        model.promptText = "#  Always deploy with pnpm, never npm"

        model.run()

        // The draft is cleared, one transcript row is appended, nothing generated.
        XCTAssertTrue(model.promptText.isEmpty)
        XCTAssertEqual(model.turnMessages(for: model.selectedChatID).count, 1)
        let row = model.turnMessages(for: model.selectedChatID)[0]
        XCTAssertEqual(row.role, .user)
        let remembered = UserMemoryInputMessage.parse(row.content)
        XCTAssertEqual(remembered, "Always deploy with pnpm, never npm")

        // A quick-save is profile scoped and immediately active.
        let index = MemoryStore.shared.loadIndex(forProjectRoot: root)
        XCTAssertTrue(index.isEmpty)
        XCTAssertTrue(MemoryLedgerStore.shared.snapshot().claims.contains {
            $0.scope == "profile" && $0.status == .active &&
            $0.text == "Always deploy with pnpm, never npm"
        })
    }

    @MainActor
    func testAQuickSaveInAGhostChatStaysOutOfTheRow() {
        let root = makeScratchDirectory("ghost-project")
        let model = makeProjectModel(project: makeProject(root: root))
        model.memoryEnabled = true
        var chat = AppChat(title: "Temporary Chat")
        chat.isGhost = true
        chat.projectID = model.projects[0].id
        model.chats = [chat]
        model.selectedChatID = chat.id

        model.promptText = "# prefers terse answers"
        model.run()

        // The note lands in the sealed payload, never on the row.
        XCTAssertEqual(model.chats[0].messages.count, 0)
        XCTAssertEqual(model.turnMessages(for: chat.id).count, 1)
        XCTAssertNotNil(UserMemoryInputMessage.parse(model.turnMessages(for: chat.id)[0].content))
        // The explicit memory survives, while the ghost transcript remains sealed.
        let index = MemoryStore.shared.loadIndex(forProjectRoot: root)
        XCTAssertTrue(index.isEmpty)
        XCTAssertTrue(MemoryLedgerStore.shared.snapshot().claims.contains {
            $0.scope == "profile" && $0.status == .active && $0.text == "prefers terse answers"
        })
    }

    @MainActor
    func testAHashDraftWithoutAProjectOrModelMemorySavesToProfile() {
        let model = AppModel()
        model.interactionMode = .chat
        model.memoryEnabled = false
        let chat = AppChat(title: "plain")
        model.chats = [chat]
        model.selectedChatID = chat.id

        model.promptText = "# remember this with no project"
        model.run()

        XCTAssertTrue(model.promptText.isEmpty)
        XCTAssertEqual(model.turnMessages(for: chat.id).count, 1)
        XCTAssertTrue(MemoryLedgerStore.shared.snapshot().claims.contains {
            $0.scope == "profile" && $0.status == .active &&
            $0.text == "remember this with no project"
        })
    }

    @MainActor
    func testALoneHashIsProseNotACommand() {
        XCTAssertFalse(UserMemoryInputMessage.isQuickSaveDraft("#"))
        XCTAssertFalse(UserMemoryInputMessage.isQuickSaveDraft("#   "))
        XCTAssertTrue(UserMemoryInputMessage.isQuickSaveDraft("# note"))
    }

    // MARK: - The /memory command

    @MainActor
    func testTheMemoryCommandOpensSettingsForAProject() {
        let root = makeScratchDirectory("slash-project")
        let model = makeProjectModel(project: makeProject(root: root))
        var opened: [AppSettingsView.SettingsTab] = []
        model.handleMemoryCommand { opened.append($0) }
        XCTAssertEqual(opened, [.memory])
    }

    @MainActor
    func testTheMemoryCommandWithoutAProjectOpensSettings() {
        let model = AppModel()
        model.interactionMode = .chat
        let chat = AppChat(title: "plain")
        model.chats = [chat]
        model.selectedChatID = chat.id
        var opened: [AppSettingsView.SettingsTab] = []
        model.handleMemoryCommand { opened.append($0) }
        XCTAssertEqual(opened, [.memory])
    }

    // MARK: - Settings plumbing

    func testMemoryEnabledDefaultsOffAndRoundTripsOn() throws {
        XCTAssertFalse(MacAppSettings().memoryEnabled)
        let off = MacAppSettings(memoryEnabled: true)
        let data = try JSONEncoder().encode(off)
        XCTAssertEqual(try JSONDecoder().decode(MacAppSettings.self, from: data).memoryEnabled, true)
        let legacy = #"{"temperature":0.2}"#
        XCTAssertEqual(
            try JSONDecoder().decode(MacAppSettings.self, from: Data(legacy.utf8)).memoryEnabled,
            false,
            "a settings file written before the field existed decodes to the default")
    }

    func testProfileMemoryAppendsTimestampedMarkdownAndSearchesIt() throws {
        let root = makeScratchDirectory("profile-memory")
        let store = ProfileMemoryStore(base: root)
        try store.write("# Existing preference\n")
        _ = try store.append("Prefer concise release notes", timestamp: Date(timeIntervalSince1970: 0))
        let text = store.load()
        XCTAssertTrue(text.contains("# Existing preference"))
        XCTAssertTrue(text.contains("1970-01-01T00:00:00"))
        XCTAssertEqual(store.lexicalSearch("concise release").count, 1)
    }

    func testProfilePromptContainsDateAndProfileMemoryOnlyWhenEnabled() {
        let previous = MemoryStore.shared.isModelEnabled
        defer { MemoryStore.shared.isModelEnabled = previous }
        MemoryStore.shared.isModelEnabled = true
        let prompt = MemoryPromptBuilder.profileSection(store: ProfileMemoryStore(), userPrompt: "preferences")
        XCTAssertTrue(prompt.contains("## Profile Memory"))
        XCTAssertTrue(prompt.contains("T"))
        MemoryStore.shared.isModelEnabled = false
    }

    @MainActor
    func testMemoryCharacterAssetLoadsFromBundle() {
        let image = MemoryCharacterAssetCache.shared.memoryImage
        XCTAssertNotNil(image, "MemoryCharacter asset should load from bundle resources")
        if let image {
            XCTAssertGreaterThan(image.size.width, 0)
            XCTAssertGreaterThan(image.size.height, 0)
        }
        let view = TSIdlingMemorySparkView(size: 80)
        XCTAssertEqual(view.size, 80)
    }
}
