import XCTest

@testable import TurboSparkApp

/// Regression tests for the low-severity State findings fixed in the
/// low-findings round. Each test exercises the behavior its name claims.
final class StateLowFindingsTests: XCTestCase {

    // MARK: - Chat duplication

    func testDuplicatedChatDoesNotCarryTheGoal() {
        var chat = AppChat(title: "src", messages: [])
        chat.goal = ChatGoalState(condition: "ship it")
        XCTAssertNotNil(chat.goal)
        XCTAssertNil(chat.duplicated().goal)
    }

    // MARK: - Skill state merge (RFC 7386)

    func testFirstPatchStripsNullMapEntriesWhenFieldIsAbsent() {
        var state = AppSkillState()
        state.apply(patch: ["files": .object(["a.swift": .null, "b.swift": .string("read")])])
        XCTAssertEqual(state.fields["files"], .object(["b.swift": .string("read")]))
    }

    // MARK: - /copy language filter

    func testCopyLanguageFilterIsCaseInsensitive() {
        let message = "```JSON\n{\"a\":1}\n```"
        XCTAssertEqual(
            CopyCommandParser.resolve(draft: "/copy json", in: message),
            .copied("{\"a\":1}\n"))
    }

    // MARK: - Skill arguments block

    func testTopLevelKeyAfterArgumentsBlockIsNotCapturedAsArgumentDescription() {
        let manifest = SkillParser.parseFrontmatterYAML(
            "arguments:\n  - name: env\ndescription: Deploys the app\n")
        XCTAssertEqual(manifest.description, "Deploys the app")
        XCTAssertEqual(manifest.arguments.map(\.name), ["env"])
        XCTAssertNil(manifest.arguments.first?.description)
    }

    func testUnknownIndentedSubKeyDoesNotDropLaterArguments() {
        let manifest = SkillParser.parseFrontmatterYAML(
            "arguments:\n  - name: a\n    required: true\n  - name: b\n")
        XCTAssertEqual(manifest.arguments.map(\.name), ["a", "b"])
    }

    // MARK: - Plugin cache path

    func testEmptyAndDotVersionsDoNotCollapseTheInstallPath() {
        XCTAssertEqual(PluginMarketplaceManager.sanitizedVersion(""), "unknown")
        XCTAssertEqual(PluginMarketplaceManager.sanitizedVersion("."), "unknown")
        XCTAssertEqual(PluginMarketplaceManager.sanitizedVersion(".."), "unknown")
        XCTAssertEqual(PluginMarketplaceManager.sanitizedVersion("1.0.0"), "1.0.0")
    }

    // MARK: - Install alias

    func testInstallAliasMustBeOnePlainPathComponent() {
        for bad in ["Qwen/Qwen3-8B", "../../Desktop/m", "/tmp/m", ".", "..", "", "a b"] {
            XCTAssertFalse(AppModel.isSafeInstallAlias(bad), bad)
        }
        for good in ["qwen3-8b", "Llama_3.1", "m.v2"] {
            XCTAssertTrue(AppModel.isSafeInstallAlias(good), good)
        }
    }

    // MARK: - Server host

    func testBlankServerHostIsAbsent() {
        XCTAssertNil(AppModel.serverHostOption(from: "   "))
        XCTAssertEqual(AppModel.serverHostOption(from: " 0.0.0.0 "), "0.0.0.0")
    }

    // MARK: - Streamed text cap

    @MainActor
    func testStreamedTextKeepsOnlyABoundedTail() {
        let state = SubagentRunState(id: "x", mode: .foreground, chatID: nil)
        for _ in 0..<50 { state.apply(.content(String(repeating: "a", count: 1_000))) }
        state.apply(.content("END"))
        XCTAssertLessThanOrEqual(state.streamedText.count, SubagentRunState.streamedTextLimit)
        XCTAssertTrue(state.streamedText.hasSuffix("END"))
    }

    // MARK: - Title prompt bound

    func testTitlePromptTruncatesAHugeFirstMessage() async {
        var seen = ""
        _ = await AppTitleGeneration.generateTitle(
            firstUserMessage: String(repeating: "x", count: 50_000)
        ) { messages, _ in
            seen = messages.last?.content ?? ""
            return "Title"
        }
        XCTAssertLessThan(seen.count, AppTitleGeneration.promptCharacterLimit + 10)
    }

    // MARK: - Tool call order

    @MainActor
    func testApprovingAParkedCallKeepsTheModelEmissionOrder() {
        let model = AppModel()
        let chat = AppChat(title: "one")
        model.chats = [chat]
        model.selectedChatID = chat.id
        let a = AppToolCall(name: "read_file", arguments: [:], category: .fileRead)
        let b = AppToolCall(name: "list_directory", arguments: [:], category: .fileRead)
        let c = AppToolCall(name: "grep", arguments: [:], category: .fileRead)
        model.chats[0].messages = [
            AppChatMessage(role: .assistant, content: "", toolCalls: [a, b, c])
        ]
        for call in [a, b, c] {
            model.appendToolExecutionTurn(
                call: call, result: AppToolResult(callID: call.id, output: call.name),
                chatID: chat.id)
        }
        let message = model.chats[0].messages[0]
        XCTAssertEqual(message.toolCalls.map(\.id), [a.id, b.id, c.id])
        XCTAssertEqual(message.toolResults.map(\.callID), [a.id, b.id, c.id])
    }

    // MARK: - Plugin ledger

    func testConcurrentUpsertsDoNotDropRecords() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ledger-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = PluginLedgerStore(root: root)
        let count = 24
        DispatchQueue.concurrentPerform(iterations: count) { i in
            let record = InstalledPluginRecord(
                scope: "user", installPath: "/p/\(i)", version: "1")
            try? store.upsertRecordChecked(record, for: "p\(i)@m")
        }
        XCTAssertEqual(store.load().plugins.count, count)
    }

    // MARK: - Lossy MCP archive

    func testGlobalMcpArchiveKeepsReadableServersWhenOneRowIsUnreadable() throws {
        let good = McpServerConfig(name: "ok", transport: .stdio(command: "x"))
        let goodJSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(good))
        let archive: [String: Any] = ["servers": [["name": 5], goodJSON]]
        let data = try JSONSerialization.data(withJSONObject: archive)
        let decoded = try JSONDecoder().decode(GlobalMcpArchive.self, from: data)
        XCTAssertEqual(decoded.servers.map(\.name), ["ok"])
    }

    // MARK: - Backup header

    func testBackupHeaderWithAbsurdKdfRoundsIsRejectedWithoutDerivingAKey() throws {
        var header = ProfileEncryptedChunkWriter.magic
        header.append(contentsOf: withUnsafeBytes(of: ProfileEncryptedChunkWriter.version.bigEndian, Array.init))
        header.append(contentsOf: withUnsafeBytes(of: UInt32(ProfileEncryptedChunkWriter.chunkSize).bigEndian, Array.init))
        header.append(Data(repeating: 1, count: ProfileVaultCrypto.saltByteCount))
        header.append(contentsOf: withUnsafeBytes(of: UInt32.max.bigEndian, Array.init))
        header.append(Data(repeating: 2, count: 8))
        header.append(contentsOf: withUnsafeBytes(of: UInt32(0).bigEndian, Array.init))
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("hdr-\(UUID().uuidString)")
        try header.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let started = Date()
        XCTAssertThrowsError(
            try ProfileEncryptedChunkWriter.decrypt(
                archive: file, passphrase: "pw",
                destination: file.appendingPathExtension("out")))
        XCTAssertLessThan(Date().timeIntervalSince(started), 5)
    }
}
