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
        try? store.saveTopic(
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

    // MARK: - The memory tool

    private func toolProject(_ root: URL) -> AppProject {
        makeProject(root: root)
    }

    func testTheToolSavesReadsAndForgets() throws {
        let store = MemoryStore(base: makeScratchDirectory("tool"))
        let root = makeScratchDirectory("project")

        let saved = try MemoryToolExecutor.execute(
            arguments: ["action": "save", "name": "Deploy Flow", "content": "Ship on Thursdays."],
            project: toolProject(root), store: store)
        XCTAssertTrue(saved.contains("deploy-flow"), saved)

        let index = try MemoryToolExecutor.execute(
            arguments: ["action": "read"], project: toolProject(root), store: store)
        XCTAssertTrue(index.contains("deploy-flow"))
        XCTAssertTrue(index.contains("Ship on Thursdays."), "the description falls back to the first body line")

        let named = try MemoryToolExecutor.execute(
            arguments: ["action": "read", "name": "deploy-flow"],
            project: toolProject(root), store: store)
        XCTAssertTrue(named.contains("type: project"))

        let forgotten = try MemoryToolExecutor.execute(
            arguments: ["action": "forget", "name": "deploy-flow"],
            project: toolProject(root), store: store)
        XCTAssertTrue(forgotten.contains("Forgot"))

        let empty = try MemoryToolExecutor.execute(
            arguments: ["action": "read"], project: toolProject(root), store: store)
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

        // The memory itself: type user, dated slug, indexed.
        let dir = MemoryStore.shared.directory(forProjectRoot: root)
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        let topic = files.first { $0.hasSuffix(".md") && $0 != "MEMORY.md" }
        XCTAssertNotNil(topic, "expected a topic file, found: \(files)")
        if let topic {
            let text = (try? String(contentsOf: dir.appendingPathComponent(topic), encoding: .utf8)) ?? ""
            XCTAssertTrue(text.contains("type: user"))
            XCTAssertTrue(text.contains("Always deploy with pnpm"))
        }
        let index = MemoryStore.shared.loadIndex(forProjectRoot: root)
        XCTAssertTrue(index.contains("-- Always deploy with pnpm"))
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
        // And the memory is still written: ghost hides the transcript, not the store.
        let dir = MemoryStore.shared.directory(forProjectRoot: root)
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        XCTAssertTrue(files.contains { $0.hasSuffix(".md") })
    }

    @MainActor
    func testAHashDraftWithoutAProjectIsRefusedAndKeepsTheDraft() {
        let model = AppModel()
        model.interactionMode = .chat
        let chat = AppChat(title: "plain")
        model.chats = [chat]
        model.selectedChatID = chat.id

        model.promptText = "# remember this with no project"
        model.run()

        XCTAssertFalse(model.promptText.isEmpty, "a refused save must keep the draft")
        XCTAssertTrue(model.turnMessages(for: chat.id).isEmpty)
    }

    @MainActor
    func testALoneHashIsProseNotACommand() {
        XCTAssertFalse(UserMemoryInputMessage.isQuickSaveDraft("#"))
        XCTAssertFalse(UserMemoryInputMessage.isQuickSaveDraft("#   "))
        XCTAssertTrue(UserMemoryInputMessage.isQuickSaveDraft("# note"))
    }

    // MARK: - The /memory command

    @MainActor
    func testTheMemoryCommandOpensTheProjectFolder() {
        let root = makeScratchDirectory("slash-project")
        let model = makeProjectModel(project: makeProject(root: root))
        var opened: [URL] = []
        model.handleMemoryCommand { opened.append($0) }
        XCTAssertEqual(opened.count, 1)
        XCTAssertEqual(
            opened.first?.lastPathComponent, "memory",
            "/memory opens the project's memory directory, not its base")
    }

    @MainActor
    func testTheMemoryCommandWithoutAProjectOpensProfileMemory() {
        let model = AppModel()
        model.interactionMode = .chat
        let chat = AppChat(title: "plain")
        model.chats = [chat]
        model.selectedChatID = chat.id
        var opened: [URL] = []
        model.handleMemoryCommand { opened.append($0) }
        XCTAssertEqual(opened, [ProfileMemoryStore.shared.directory])
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
