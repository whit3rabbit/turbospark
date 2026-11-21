import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Artifact panel history: the version list built from checkpoint manifests
/// and the current file, then the real write tool path in temp directories.
/// Every store here is rooted in a temp directory, never the shared one.
@MainActor
final class ArtifactVersionHistoryTests: XCTestCase {
    private var directories: [URL] = []

    private func temporaryDirectory(_ label: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("artifact-history-\(label)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        directories.append(url)
        return url
    }

    override func tearDown() async throws {
        AppToolRegistry.turnCheckpointCapturer = nil
        for url in directories { try? FileManager.default.removeItem(at: url) }
        directories = []
        try await super.tearDown()
    }

    // MARK: Pure list

    private let chatID = UUID()

    private func manifest(
        _ seconds: TimeInterval, _ entry: TurnCheckpointEntry, turnID: UUID = UUID()
    ) -> TurnCheckpointManifest {
        TurnCheckpointManifest(
            chatID: chatID, turnID: turnID, createdAt: Date(timeIntervalSince1970: seconds),
            rootPath: "/project", entries: [entry])
    }

    private func captured(_ sha: String, path: String = "a.md", bytes: Int = 10) -> TurnCheckpointEntry {
        TurnCheckpointEntry(relativePath: path, state: .captured, blobName: "0.bin", byteCount: bytes, sha256: sha)
    }

    private func list(
        _ manifests: [TurnCheckpointManifest],
        current: ArtifactVersionHistory.CurrentFile = .init(exists: true, byteCount: 10, sha256: "now"),
        turnNumbers: [UUID: Int] = [:],
        missing: Set<UUID> = []
    ) -> [ArtifactVersion] {
        ArtifactVersionHistory.versions(
            manifests: manifests, relativePath: { _ in "a.md" }, current: current,
            turnNumbers: turnNumbers, snapshotExists: { manifest, _ in !missing.contains(manifest.turnID) })
    }

    func testNewestFirstAfterCurrentWithUnavailableReasonsAndNoRowForCreation() {
        let oldest = manifest(100, captured("s1"))
        let created = manifest(50, TurnCheckpointEntry(relativePath: "a.md", state: .absent))
        let skipped = manifest(200, TurnCheckpointEntry(relativePath: "a.md", state: .notCaptured, skipReason: .tooLarge))
        let gone = manifest(300, captured("s3"))
        let newest = manifest(400, captured("s4"))
        let otherFile = manifest(500, captured("x", path: "b.md"))
        let numbers = [created.turnID: 1, oldest.turnID: 2, skipped.turnID: 3, gone.turnID: 4, newest.turnID: 5]

        let rows = list([oldest, created, skipped, gone, newest, otherFile], turnNumbers: numbers, missing: [gone.turnID])

        XCTAssertEqual(rows.map(\.id), ["current", newest.turnID.uuidString, gone.turnID.uuidString,
                                        skipped.turnID.uuidString, oldest.turnID.uuidString])
        XCTAssertEqual(rows.map(\.turnNumber), [nil, 5, 4, 3, 2])
        XCTAssertEqual(rows.map(\.unavailable), [nil, nil, .snapshotMissing, .notCaptured(.tooLarge), nil])
        XCTAssertNil(rows[3].byteCount)
        XCTAssertEqual(ArtifactVersionText.reason(.notCaptured(.tooLarge)), "Not captured: file too large")
        XCTAssertEqual(ArtifactVersionText.title(rows[1]), "Before turn 5")
    }

    func testIdenticalConsecutiveContentsCollapseByHash() {
        let a = manifest(100, captured("same"))
        let b = manifest(200, captured("same"))
        let c = manifest(300, captured("other"))
        let d = manifest(400, captured("now"))
        // d holds what is on disk now (a restore, or a rewrite with the same
        // bytes), b repeats a. c separates nothing it equals.
        let rows = list([a, b, c, d])
        XCTAssertEqual(rows.map(\.id), ["current", c.turnID.uuidString, b.turnID.uuidString])

        // An unavailable row is not content, so it does not hide its neighbour.
        let gap = manifest(150, TurnCheckpointEntry(relativePath: "a.md", state: .notCaptured, skipReason: .sensitive))
        let split = list([a, gap, b])
        XCTAssertEqual(split.map(\.id), ["current", b.turnID.uuidString, gap.turnID.uuidString, a.turnID.uuidString])

        // Non-consecutive equal contents are both kept.
        let e = manifest(500, captured("same"))
        XCTAssertEqual(list([b, c, e]).map(\.id).count, 4)
    }

    func testMissingCurrentFileAndSameSecondTurnsOrderByPrompt() {
        let first = manifest(100, captured("s1"))
        let second = manifest(100, captured("s2"))
        let rows = list([first, second], current: .init(exists: false, byteCount: nil, sha256: nil),
                        turnNumbers: [first.turnID: 1, second.turnID: 2])
        XCTAssertEqual(rows.first?.unavailable, .fileMissing)
        XCTAssertEqual(rows.map(\.turnNumber), [nil, 2, 1])
        XCTAssertEqual(list([]).count, 1)
    }

    func testTurnNumbersCountOnlyPrompts() {
        let first = AppChatMessage(role: .user, content: "one")
        let reply = AppChatMessage(role: .assistant, content: "ok")
        let second = AppChatMessage(role: .user, content: "two")
        let numbers = ArtifactVersionHistory.turnNumbers(in: [first, reply, second])
        XCTAssertEqual(numbers, [first.id: 1, second.id: 2])
    }

    func testRelativePathUsesTheStoresContainmentRule() throws {
        let root = try temporaryDirectory("root")
        let outside = try temporaryDirectory("outside")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("docs"), withIntermediateDirectories: true)
        try Data("x".utf8).write(to: root.appendingPathComponent("docs/a.md"))
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("link"), withDestinationURL: outside)

        XCTAssertEqual(ArtifactVersionHistory.relativePath(
            ofArtifactPath: root.appendingPathComponent("docs/a.md").path, inRoot: root.path), "docs/a.md")
        XCTAssertNil(ArtifactVersionHistory.relativePath(
            ofArtifactPath: outside.appendingPathComponent("a.md").path, inRoot: root.path))
        XCTAssertNil(ArtifactVersionHistory.relativePath(
            ofArtifactPath: root.appendingPathComponent("link/a.md").path, inRoot: root.path))
        XCTAssertNil(ArtifactVersionHistory.relativePath(ofArtifactPath: root.path, inRoot: root.path))
    }

    // MARK: Store and the real tool path

    func testTwoWritesThroughTheToolPathGiveTwoEarlierVersionsWithTheirBytes() async throws {
        let root = try temporaryDirectory("project")
        let store = TurnCheckpointStore(rootURL: try temporaryDirectory("store"))
        let file = root.appendingPathComponent("notes.md")
        try Data("# v0\n".utf8).write(to: file)
        // The first read of `ProfileVaultCoordinator.shared` (inside the
        // capturer) builds the coordinator's own AppModel, whose init installs
        // a capturer on the SHARED store over this test's. Build it first.
        _ = ProfileVaultCoordinator.shared.isProtected
        let model = AppModel()
        let firstPrompt = AppChatMessage(role: .user, content: "rewrite notes")
        var chat = AppChat(title: "history", messages: [firstPrompt])
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.installTurnCheckpointCapture(store: store)
        let project = AppProject(name: "history", rootDirectoryPath: root.path, permissions: .fullAccess)

        // The write tool refuses an existing file this session has not read.
        let read1 = await AppToolRegistry.execute(
            call: AppToolCall(name: "read_file", arguments: ["path": "notes.md"], category: .fileRead),
            in: project, chatID: chat.id)
        XCTAssertFalse(read1.isError, read1.output)
        let first = await model.executeApprovedTool(AppToolCall(
            name: "write_file", arguments: ["path": "notes.md", "content": "# v1\n"], category: .fileWrite),
            project: project, chatID: chat.id)
        XCTAssertFalse(first.result.isError, first.result.output)

        let secondPrompt = AppChatMessage(role: .user, content: "again")
        chat.messages += [AppChatMessage(role: .assistant, content: "done"), secondPrompt]
        model.chats = [chat]
        let read2 = await AppToolRegistry.execute(
            call: AppToolCall(name: "read_file", arguments: ["path": "notes.md"], category: .fileRead),
            in: project, chatID: chat.id)
        XCTAssertFalse(read2.isError, read2.output)
        let second = await model.executeApprovedTool(AppToolCall(
            name: "write_file", arguments: ["path": "notes.md", "content": "# v2\n"], category: .fileWrite),
            project: project, chatID: chat.id)
        XCTAssertFalse(second.result.isError, second.result.output)

        let numbers = ArtifactVersionHistory.turnNumbers(in: chat.messages)
        let rows = ArtifactVersionHistory.load(
            chatID: chat.id, artifactPath: file.standardizedFileURL.path, turnNumbers: numbers, store: store)
        XCTAssertEqual(rows.map(\.turnNumber), [nil, 2, 1])
        guard rows.count == 3 else { return }
        XCTAssertTrue(rows.allSatisfy { $0.unavailable == nil })
        XCTAssertEqual(rows[0].sha256, ToolObservationStore.hash(Data("# v2\n".utf8)))
        XCTAssertEqual(ArtifactVersionHistory.text(of: rows[1], chatID: chat.id, store: store), "# v1\n")
        XCTAssertEqual(ArtifactVersionHistory.text(of: rows[2], chatID: chat.id, store: store), "# v0\n")
        XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), "# v2\n", "history never writes the file")

        // A snapshot that no longer matches its manifest is never shown.
        let manifest = try XCTUnwrap(store.manifest(chatID: chat.id, turnID: firstPrompt.id))
        let blob = store.rootURL.appendingPathComponent(chat.id.uuidString)
            .appendingPathComponent(firstPrompt.id.uuidString)
            .appendingPathComponent("blobs").appendingPathComponent(try XCTUnwrap(manifest.entries.first?.blobName))
        try Data("# tampered\n".utf8).write(to: blob)
        XCTAssertNil(ArtifactVersionHistory.text(of: rows[2], chatID: chat.id, store: store))
        try FileManager.default.removeItem(at: blob)
        let afterRemoval = ArtifactVersionHistory.load(
            chatID: chat.id, artifactPath: file.path, turnNumbers: numbers, store: store)
        XCTAssertEqual(afterRemoval.last?.unavailable, .snapshotMissing)
    }

    func testSnapshotReadsRefuseANameOutsideTheBlobsFolder() throws {
        let store = TurnCheckpointStore(rootURL: try temporaryDirectory("store"))
        let entry = TurnCheckpointEntry(relativePath: "a.md", state: .captured, blobName: "../manifest.json",
                                        byteCount: 1, sha256: "x")
        XCTAssertFalse(store.snapshotExists(chatID: UUID(), turnID: UUID(), entry: entry))
        XCTAssertNil(store.snapshotData(chatID: UUID(), turnID: UUID(), entry: entry))
    }
}
