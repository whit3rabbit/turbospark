import Foundation
import XCTest

@testable import TurboSparkApp

/// Medium data-integrity fixes: rows or records this build cannot decode must
/// survive the next save.
final class DataIntegrityPersistenceTests: XCTestCase {
    private var root: URL!
    private let key = Data(repeating: 0x42, count: 32)

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("DataIntegrity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testUndecodableChatRowSurvivesNextSave() throws {
        let db = try ProfileDatabase(url: root.appendingPathComponent("a.sqlite3"), key: key)
        let keep = AppChat(title: "keep", messages: [])
        let other = AppChat(title: "other", messages: [])
        _ = try db.saveChatArchive(AppChatArchive(selectedChatID: keep.id, chats: [keep, other]))
        try db.execute("UPDATE chats SET payload = X'6E6F74206A736F6E' WHERE id = '\(other.id.uuidString)'")

        let loaded = try XCTUnwrap(db.loadChatArchive())
        XCTAssertEqual(loaded.chats.map(\.id), [keep.id])
        _ = try db.saveChatArchive(loaded)

        XCTAssertTrue(try db.chatIDs().contains(other.id.uuidString))
    }

    func testUndecodableProjectRowSurvivesNextSave() throws {
        let db = try ProfileDatabase(url: root.appendingPathComponent("b.sqlite3"), key: key)
        let keep = AppProject(name: "keep", rootDirectoryPath: root.path)
        let other = AppProject(name: "other", rootDirectoryPath: root.path)
        try db.saveProjectArchive(
            AppProjectArchive(selectedProjectID: keep.id, projects: [keep, other]))
        try db.execute("UPDATE projects SET payload = X'6E6F74206A736F6E' WHERE id = '\(other.id.uuidString)'")

        let loaded = try XCTUnwrap(db.loadProjectArchive())
        XCTAssertEqual(loaded.projects.map(\.id), [keep.id])
        try db.saveProjectArchive(loaded)

        XCTAssertEqual(try db.scalarText("SELECT COUNT(*) FROM projects"), "2")
    }

    /// 14k entries with ~80 byte names is a listing above the old 1 MB cap.
    func testLargeArchiveListingIsNotTruncated() async throws {
        let payload = root.appendingPathComponent("payload", isDirectory: true)
        try FileManager.default.createDirectory(at: payload, withIntermediateDirectories: true)
        let prefix = String(repeating: "n", count: 70)
        for i in 0..<14_000 {
            FileManager.default.createFile(
                atPath: payload.appendingPathComponent("\(prefix)-\(i)").path, contents: Data())
        }
        let archive = root.appendingPathComponent("big.zip")
        let zip = Process()
        zip.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        zip.currentDirectoryURL = payload
        zip.arguments = ["-q", "-r", archive.path, "."]
        try zip.run()
        zip.waitUntilExit()
        XCTAssertEqual(zip.terminationStatus, 0)

        let entries = try await ProfileBackupImport.listEntries(archive: archive)
        XCTAssertGreaterThanOrEqual(entries.count, 14_000)
    }

    // MARK: - undo_edit staleness

    private func runTool(_ name: String, _ args: [String: String], category: AppToolCategory, project: AppProject) async -> AppToolResult {
        await AppToolRegistry.execute(
            call: AppToolCall(name: name, arguments: args, category: category), in: project)
    }

    func testUndoEditRefusedAfterLaterWriteAndNotRepeatable() async throws {
        let project = AppProject(name: "undo", rootDirectoryPath: root.path)
        let file = root.appendingPathComponent("u.txt")
        try "v0 content\n".write(to: file, atomically: true, encoding: .utf8)
        _ = await runTool("read_file", ["path": "u.txt"], category: .fileRead, project: project)
        let edit = await runTool(
            "edit_file", ["path": "u.txt", "old_string": "v0", "new_string": "v1"],
            category: .fileWrite, project: project)
        XCTAssertFalse(edit.isError)

        // A later change (here the user's editor) must not be discarded by undo.
        try "v3 from the user\n".write(to: file, atomically: true, encoding: .utf8)
        let undo = await runTool(
            "editor", ["path": "u.txt", "command": "undo_edit"], category: .fileWrite, project: project)
        XCTAssertTrue(undo.isError)
        XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), "v3 from the user\n")
    }

    func testSecondUndoAfterSuccessfulUndoIsRefused() async throws {
        let project = AppProject(name: "undo2", rootDirectoryPath: root.path)
        let file = root.appendingPathComponent("u2.txt")
        try "v0 content\n".write(to: file, atomically: true, encoding: .utf8)
        _ = await runTool("read_file", ["path": "u2.txt"], category: .fileRead, project: project)
        _ = await runTool(
            "edit_file", ["path": "u2.txt", "old_string": "v0", "new_string": "v1"],
            category: .fileWrite, project: project)
        let first = await runTool(
            "editor", ["path": "u2.txt", "command": "undo_edit"], category: .fileWrite, project: project)
        XCTAssertFalse(first.isError, first.output)
        let second = await runTool(
            "editor", ["path": "u2.txt", "command": "undo_edit"], category: .fileWrite, project: project)
        XCTAssertTrue(second.isError)
        XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), "v0 content\n")
    }

    // MARK: - batch serializes mutating children

    func testMutationGateSerializesCriticalSections() async {
        let gate = MutationGate()
        let counter = Counter()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<50 {
                group.addTask {
                    await gate.acquire()
                    let inside = await counter.enter()
                    // Yield so an unguarded overlap would be observed.
                    await Task.yield()
                    await counter.leave()
                    await gate.release()
                    XCTAssertEqual(inside, 1)
                }
            }
        }
    }

    func testBatchAppliesTwoEditsToTheSameFile() async throws {
        let project = AppProject(name: "batch-edits", rootDirectoryPath: root.path)
        let file = root.appendingPathComponent("b.txt")
        try "alpha\nbeta\n".write(to: file, atomically: true, encoding: .utf8)
        _ = await runTool("read_file", ["path": "b.txt"], category: .fileRead, project: project)
        let args = ["tool_calls": """
        [
          {"tool": "edit_file", "parameters": {"path": "b.txt", "old_string": "alpha", "new_string": "ALPHA"}},
          {"tool": "edit_file", "parameters": {"path": "b.txt", "old_string": "beta", "new_string": "BETA"}}
        ]
        """]
        let result = await AppToolRegistry.execute(
            call: AppToolCall(name: "batch", arguments: args, category: .automation), in: project)
        _ = result
        let text = try String(contentsOf: file, encoding: .utf8)
        // Either both edits landed, or the permission engine refused both
        // (nothing changed). The lost-update outcome is exactly one of them.
        XCTAssertTrue(text == "ALPHA\nBETA\n" || text == "alpha\nbeta\n", text)
    }

    // MARK: - exact backup keeps the previous file on failure

    func testFailedExactExportKeepsThePreviousBackup() throws {
        let vault = root.appendingPathComponent("vault-src", isDirectory: true)
        let store = ProfileVaultStore(
            rootProvider: { vault }, profileIDProvider: { "src" }, migrateLegacyData: false)
        _ = try store.prepareForLaunch()
        let assets = ManagedAssetStore(vault: store)
        let asset = try assets.store(
            data: Data("img".utf8), fileName: "a.png", mimeType: "image/png")
        let chat = AppChat(
            title: "t",
            messages: [AppChatMessage(role: .assistant, content: "x", imagePaths: [asset.storedReference])])
        try ProfileRepository(store: store).saveChatArchive(
            AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        // The row stays but the ciphertext vanishes.
        let ciphertext = store.assetsURL.appendingPathComponent(String(asset.id.prefix(2)))
            .appendingPathComponent("\(asset.id).tsasset")
        try FileManager.default.removeItem(at: ciphertext)

        let destination = root.appendingPathComponent("old.turbospark-profile")
        let sentinel = Data("last week's backup".utf8)
        try sentinel.write(to: destination)

        XCTAssertThrowsError(try EncryptedProfileBackup.export(
            profile: UserProfile(id: "src", name: "n"), destination: destination,
            passphrase: "pw pw pw pw", appVersion: "tests", store: store))
        XCTAssertEqual(try Data(contentsOf: destination), sentinel)
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: root.path)
            .filter { $0.contains(".partial-") }
        XCTAssertTrue(leftovers.isEmpty, "\(leftovers)")
    }

    // MARK: - agent save keeps omit flag and foreign frontmatter

    func testSaveAgentKeepsOmitFlagAndForeignFrontmatterKeys() throws {
        let file = root.appendingPathComponent("reviewer.md")
        try """
        ---
        name: reviewer
        description: Reviews code
        omit_claude_md: true
        color: blue
        permissionMode: acceptEdits
        hooks:
          PreToolUse:
            - run: x
        ---
        Be careful.
        """.write(to: file, atomically: true, encoding: .utf8)
        var agent = try AgentParser.parseFile(at: file)
        XCTAssertTrue(agent.omitsProjectInstructions)
        agent.agentDescription = "Reviews code well"
        try AgentManager.shared.saveAgent(agent)

        let text = try String(contentsOf: file, encoding: .utf8)
        XCTAssertTrue(text.contains("omit_claude_md: true"), text)
        XCTAssertTrue(text.contains("color: blue"), text)
        XCTAssertTrue(text.contains("permissionMode: acceptEdits"), text)
        XCTAssertTrue(text.contains("hooks:\n  PreToolUse:\n    - run: x"), text)
        XCTAssertTrue(text.contains("description: Reviews code well"), text)
        XCTAssertTrue(try AgentParser.parseFile(at: file).omitsProjectInstructions)
    }

    // MARK: - MCP editor arguments

    func testMcpArgumentsRoundTripKeepsSpaces() {
        let args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Library/Application Support/notes"]
        let text = McpServerFormValidation.argsText(for: args)
        XCTAssertEqual(McpServerFormValidation.parseArguments(text), args)
        // The single-line quick entry still splits on spaces.
        XCTAssertEqual(McpServerFormValidation.parseArguments("-y pkg"), ["-y", "pkg"])
    }

    // MARK: - browser settings pane does not write back a stale snapshot

    @MainActor
    func testBrowserSettingsEditKeepsBookmarksImportedAfterTheViewModelWasBuilt() {
        var live = BrowserSettings()
        let viewModel = BrowserSettingsPaneViewModel(
            settings: live, latest: { live }, onChange: { live = $0 })
        // The import sheet updates the owner's copy, not the view model's.
        _ = live.replaceBookmarks([BrowserBookmarkTree(sourceProfileDirectoryLabel: "Default")])
        XCTAssertEqual(live.bookmarks.count, 1)

        viewModel.setDialogPolicy(.autoDismiss)

        XCTAssertEqual(live.dialogPolicy, .autoDismiss)
        XCTAssertEqual(live.bookmarks.count, 1, "bookmarks imported earlier were wiped")
    }

    // MARK: - duplicated chats own their artifacts and observations

    func testDuplicatedChatRetargetsArtifactsAndCopiesObservations() throws {
        var chat = AppChat(title: "src", messages: [])
        chat.artifacts = [AppArtifact(chatID: chat.id, path: "/tmp/x.txt", title: "x", origin: .fileWrite)]
        let copy = chat.duplicated()
        XCTAssertEqual(copy.artifacts.map(\.chatID), [copy.id])

        let store = ToolObservationStore(rootURL: root.appendingPathComponent("obs"))
        let ref = try store.archive(Data("big output".utf8), chatID: chat.id)
        store.copyArchive(from: chat.id, to: copy.id)
        XCTAssertEqual(try store.load(ref, chatID: copy.id), Data("big output".utf8))
        // The source keeps its own copy.
        XCTAssertEqual(try store.load(ref, chatID: chat.id), Data("big output".utf8))
    }

    // MARK: - edit and retry keep every earlier version

    @MainActor
    func testSecondRetryAndSecondEditKeepEarlierVersions() throws {
        let prompt = AppChatMessage(role: .user, content: "p")
        let v1 = AppChatMessage(role: .assistant, content: "v1", stopReason: "endOfTurn")
        var v2 = AppChatMessage(role: .assistant, content: "v2", stopReason: "endOfTurn")
        v2.alternates = [v1]

        guard case .retry(let plan) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: nil, messages: [prompt, v2])
        else { return XCTFail("expected a retry plan") }
        XCTAssertEqual(plan.alternates.map(\.content), ["v1", "v2"])
        XCTAssertTrue(plan.alternates.allSatisfy { $0.alternates.isEmpty })

        // Editing the prompt twice keeps the original text.
        var editedOnce = prompt
        editedOnce.content = "p edited"
        editedOnce.alternates = [prompt]
        let applied = AppModel.editApplied(
            to: [editedOnce, v2], anchorIndex: 0, newText: "p edited twice", now: Date())
        XCTAssertEqual(applied.messages[0].alternates.map(\.content), ["p", "p edited"])
        XCTAssertEqual(applied.responseVariants.map(\.content), ["v1", "v2"])
    }

    // MARK: - project permissions decode tolerantly

    func testUnknownPermissionModeAndCaseDoNotDropTheProject() throws {
        let json = #"{"mode":"someFutureMode","fileWrite":"futureCase","terminal":"allow","mcpDenyRules":["mcp__x", 5]}"#
        let decoded = try JSONDecoder().decode(AppProjectPermissions.self, from: Data(json.utf8))
        XCTAssertEqual(decoded.mode, .ask)
        XCTAssertEqual(decoded.fileWrite, .ask)
        XCTAssertEqual(decoded.terminal, .allow)
        XCTAssertEqual(decoded.fileRead, .allow, "absent keys keep their historical defaults")
        XCTAssertEqual(decoded.mcpDenyRules, ["mcp__x"])
    }

    // MARK: - protected record quarantine

    func testUndecodableProtectedRecordIsQuarantinedBeforeBeingOverwritten() throws {
        let vault = root.appendingPathComponent("vault-q", isDirectory: true)
        let store = ProfileVaultStore(
            rootProvider: { vault }, profileIDProvider: { "q" }, migrateLegacyData: false)
        _ = try store.prepareForLaunch()
        let repo = ProfileRepository(store: store)
        let raw = Data(#"{"servers":[{"transport":"from-the-future"}]}"#.utf8)
        try repo.saveRawRecord(raw, key: "json:global_mcp_servers.json")

        XCTAssertThrowsError(try repo.load([String].self, key: "json:global_mcp_servers.json"))
        // A second failing load does not pile up identical copies.
        XCTAssertThrowsError(try repo.load([String].self, key: "json:global_mcp_servers.json"))
        try repo.save(["fresh"], key: "json:global_mcp_servers.json")

        let kept = try repo.rawRecords(prefix: "quarantine:json:global_mcp_servers.json:")
        XCTAssertEqual(kept.count, 1)
        XCTAssertEqual(kept.first?.1, raw)
    }

    // MARK: - queue drain does not overwrite the composer

    @MainActor
    func testTailDrainLeavesAHalfTypedDraftAndItsAttachmentsAlone() {
        let model = AppModel()
        model.stopCronScheduler()
        let chat = AppChat(title: "Queue")
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.pendingUserMessages[chat.id] = [QueuedUserPrompt(text: "queued A")]
        model.promptText = "typing B"

        XCTAssertNil(model.drainPendingUserMessagesIfIdle(chatID: chat.id))
        XCTAssertEqual(model.promptText, "typing B")
        XCTAssertEqual(model.pendingUserMessages[chat.id]?.map(\.text), ["queued A"])

        // Attachments alone also hold the drain.
        model.promptText = ""
        model.chats[0].draftAttachments = [AppPromptAttachment(
            fileName: "n.txt", formatLabel: "TXT", extractedText: "x",
            wasTruncatedDuringExtraction: false)]
        XCTAssertNil(model.drainPendingUserMessagesIfIdle(chatID: chat.id))
        XCTAssertEqual(model.chats[0].draftAttachments.count, 1)
    }

    @MainActor
    func testBoundarySteerKeepsAttachmentsTheUserAddedToTheComposer() async {
        let model = AppModel()
        model.stopCronScheduler()
        let chat = AppChat(title: "Steer")
        model.chats = [chat]
        model.selectedChatID = chat.id
        let pdf = AppPromptAttachment(
            fileName: "next.pdf", formatLabel: "PDF", extractedText: "for later",
            wasTruncatedDuringExtraction: false)
        model.chats[0].draftAttachments = [pdf]
        model.pendingUserMessages[chat.id] = [QueuedUserPrompt(text: "steer now")]

        let delivered = await model.deliverSteersAtBoundary(
            chatID: chat.id, project: nil, visionAvailable: false)

        XCTAssertTrue(delivered)
        XCTAssertEqual(model.chats[0].draftAttachments, [pdf])
    }

    // MARK: - shutdown preserves a streaming reply

    @MainActor
    func testShutdownMidStreamKeepsThePartialAssistantRow() {
        let model = AppModel()
        model.stopCronScheduler()
        let chat = AppChat(title: "Streaming", messages: [AppChatMessage(role: .user, content: "question")])
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.generating = true
        model.outputText = "half an ans"

        model.shutdown()

        let rows = model.chats.first(where: { $0.id == chat.id })?.messages ?? []
        XCTAssertEqual(rows.map(\.role), [.user, .assistant])
        XCTAssertEqual(rows.last?.content, "half an ans")
    }

    // MARK: - backup import completion uses the current registry

    @MainActor
    func testImportCompletionAddsToTheCurrentRegistryNotTheStartSnapshot() throws {
        let original = UserProfileStore.loadRegistry()
        defer { UserProfileStore.saveRegistry(original) }
        let other = UserProfile(id: UUID().uuidString, name: "Created while restoring")
        var during = original
        try UserProfileStore.adding(other, to: &during)
        XCTAssertTrue(UserProfileStore.saveRegistry(during))

        let imported = UserProfile(id: UUID().uuidString, name: "Restored")
        let fresh = try AppModel.registryAdding(imported)

        XCTAssertTrue(fresh.profiles.contains(where: { $0.id == other.id }))
        XCTAssertTrue(fresh.profiles.contains(where: { $0.id == imported.id }))
    }

    // MARK: - unzip never treats the attached file's name as a wildcard

    private func makeZip(named name: String, entryContent: String) throws -> URL {
        let staging = root.appendingPathComponent("stage-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: staging.appendingPathComponent("word"), withIntermediateDirectories: true)
        try entryContent.write(
            to: staging.appendingPathComponent("word/document.xml"), atomically: true, encoding: .utf8)
        let archive = root.appendingPathComponent(name)
        let zip = Process()
        zip.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        zip.currentDirectoryURL = staging
        zip.arguments = ["-q", "-r", archive.path, "word"]
        try zip.run()
        zip.waitUntilExit()
        XCTAssertEqual(zip.terminationStatus, 0)
        return archive
    }

    func testBracketedArchiveNameDoesNotReadASiblingDocument() throws {
        let sibling = try makeZip(named: "Q3 f.docx", entryContent: "SIBLING SECRET")
        let wanted = try makeZip(named: "Q3 [final].docx", entryContent: "THE ATTACHED ONE")
        _ = sibling
        let archive = try OfficeArchive(url: wanted, limits: .default)
        let data = try archive.data(for: "word/document.xml")
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "THE ATTACHED ONE")
    }

    // MARK: - model store root is machine level, relocation guards

    func testStoreRootIsSharedAcrossProfilesAndMigratesTheProfileValue() throws {
        try? FileManager.default.removeItem(at: ModelStoreRootRecord.url)
        defer { try? FileManager.default.removeItem(at: ModelStoreRootRecord.url) }
        // An existing install: only the profile has a value. It is promoted.
        XCTAssertEqual(ModelStoreRootRecord.resolve(profileValue: " /Volumes/Ext/TS "), "/Volumes/Ext/TS")
        // Another profile with an empty (or stale) setting sees the same root.
        XCTAssertEqual(ModelStoreRootRecord.resolve(profileValue: ""), "/Volumes/Ext/TS")
        XCTAssertEqual(ModelStoreRootRecord.resolve(profileValue: "/old/place"), "/Volumes/Ext/TS")
    }

    @MainActor
    func testRelocationIsRefusedWhileADownloadIsQueued() {
        let model = AppModel()
        model.stopCronScheduler()
        XCTAssertTrue(model.canRelocateModelStore)
        model.modelDownloads = [ModelDownload(
            request: ModelDownload.Request.catalog(alias: "x"), status: .queued)]
        XCTAssertFalse(model.canRelocateModelStore)
    }

    // MARK: - asset reconciliation respects ghost chats and re-imports

    func testReconcileKeepsPinnedGhostAssetsAndReimportRestartsTheGraceWindow() throws {
        let db = try ProfileDatabase(url: root.appendingPathComponent("assets.sqlite3"), key: key)
        let ghostChat = UUID()
        try db.retainAsset(id: "ghostasset", fileName: "g.png", mimeType: "image/png", byteCount: 3)
        try db.retainAsset(id: "orphan", fileName: "o.png", mimeType: "image/png", byteCount: 3)
        try db.retainAsset(id: "reimported", fileName: "r.png", mimeType: "image/png", byteCount: 3)
        try db.execute("UPDATE assets SET created_at = created_at - 3600")
        // A dedupe re-import of an old asset bumps its timestamp.
        try db.retainAsset(id: "reimported", fileName: "r.png", mimeType: "image/png", byteCount: 3)
        ManagedAssetPins.set(["ghostasset"], owner: ghostChat)
        defer { ManagedAssetPins.clear(owner: ghostChat) }

        let keep = AppChat(title: "k", messages: [])
        let unreachable = try db.saveChatArchive(AppChatArchive(selectedChatID: keep.id, chats: [keep]))

        XCTAssertEqual(unreachable, ["orphan"])
        XCTAssertNotNil(try db.assetMetadata(id: "ghostasset"))
        XCTAssertNotNil(try db.assetMetadata(id: "reimported"))
        XCTAssertNil(try db.assetMetadata(id: "orphan"))
    }

    // MARK: - skills

    func testSavingASkillKeepsFrontmatterKeysTheParserDoesNotModel() throws {
        let dir = root.appendingPathComponent("skill", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("SKILL.md")
        let original = """
        ---
        name: deploy
        description: Ship it
        license: MIT
        version: 1.2
        metadata:
          author: me
          tags:
            - a
        ---
        Body text.
        """
        try original.write(to: file, atomically: true, encoding: .utf8)
        var skill = try SkillParser.parseFile(
            at: file, scope: .projectLocal(projectPath: root.path), agentOrigin: .turboSpark, containedIn: dir)
        skill.content = "Edited body."
        let out = SkillParser.serializeSkill(skill, preservingFrontmatterOf: original)
        XCTAssertTrue(out.contains("license: MIT"), out)
        XCTAssertTrue(out.contains("version: 1.2"), out)
        XCTAssertTrue(out.contains("metadata:\n  author: me\n  tags:\n    - a"), out)
        XCTAssertTrue(out.contains("name: deploy"), out)
        XCTAssertTrue(out.contains("Edited body."), out)
    }

    func testCreateSkillRefusesToReplaceAnExistingSkillWithTheSameSanitizedName() throws {
        let manager = SkillManager()
        _ = try manager.createSkill(
            name: "deploy", description: "d", content: "original instructions",
            scope: .projectLocal(projectPath: root.path), projectRootURL: root)
        XCTAssertThrowsError(try manager.createSkill(
            name: "Deploy", description: "d2", content: "new", scope: .projectLocal(projectPath: root.path),
            projectRootURL: root))
        let kept = try String(
            contentsOf: root.appendingPathComponent(".turbospark/skills/deploy/SKILL.md"),
            encoding: .utf8)
        XCTAssertTrue(kept.contains("original instructions"))
    }

    func testSkillDownloadRefusesAnHTTPErrorBody() async throws {
        URLProtocol.registerClass(StubStatusProtocol.self)
        defer { URLProtocol.unregisterClass(StubStatusProtocol.self) }
        let bad = URL(string: "https://stub.invalid/missing/SKILL.md")!
        do {
            _ = try await SkillMarketplaceManager.downloadSkillFile(from: bad)
            XCTFail("a 404 body must not be accepted as SKILL.md")
        } catch {}
        let good = URL(string: "https://stub.invalid/ok/SKILL.md")!
        let data = try await SkillMarketplaceManager.downloadSkillFile(from: good)
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "---\nname: ok\n---\n")
    }
}

/// Answers every request itself: 404 for paths containing "missing", else 200.
private final class StubStatusProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "stub.invalid" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let url = request.url!
        let missing = url.path.contains("missing")
        let response = HTTPURLResponse(
            url: url, statusCode: missing ? 404 : 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(
            self, didLoad: Data((missing ? "404: Not Found" : "---\nname: ok\n---\n").utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}

    func testUserScopeStoresAreRedirectedAwayFromTheRealHomeUnderTest() {
        let dir = UserProfileStore.userScopeSubdirectory("skills").standardizedFileURL.path
        let realHome = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".turbospark").standardizedFileURL.path
        XCTAssertFalse(dir.hasPrefix(realHome), dir)
        XCTAssertTrue(dir.hasPrefix(AppStorageRoot.machineRoot.standardizedFileURL.path), dir)
    }

    // MARK: - legacy UserDefaults migration

    @MainActor
    func testLegacyMigrationNeverRunsAfterAFailedRead() throws {
        let key = "TurboSpark.modelOrganizationMetadata"
        let file = AppStorageRoot.file("model_organization.json")
        let defaults = UserDefaults.standard
        defer {
            defaults.removeObject(forKey: key)
            try? FileManager.default.removeItem(at: file)
        }
        defaults.set(
            try JSONEncoder().encode(["abc": ModelCustomMetadata(nickname: "Legacy")]), forKey: key)

        // A present-but-unreadable record is a failed read, not an absent one.
        try Data("{ this is not json".utf8).write(to: file)
        let failedRead = ModelOrganizationStore()
        XCTAssertTrue(failedRead.metadataByModelKey.isEmpty,
                      "the legacy dictionary must not be imported after a failed read")

        // A genuine first run (no record at all) still migrates for Default.
        for entry in (try? FileManager.default.contentsOfDirectory(atPath: file.deletingLastPathComponent().path)) ?? []
        where entry.hasPrefix("model_organization") {
            try? FileManager.default.removeItem(
                at: file.deletingLastPathComponent().appendingPathComponent(entry))
        }
        let firstRun = ModelOrganizationStore()
        XCTAssertEqual(firstRun.metadataByModelKey["abc"]?.nickname, "Legacy")
    }
}

private actor Counter {
    private var active = 0
    func enter() -> Int { active += 1; return active }
    func leave() { active -= 1 }
}
