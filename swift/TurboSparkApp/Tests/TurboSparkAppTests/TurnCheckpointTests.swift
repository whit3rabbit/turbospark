import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Per-turn file checkpoints: the pure rules, then the store and the real
/// write tool path in a temp directory. Every store here is rooted in a temp
/// directory, never the shared one, and trashing goes through a seam.
@MainActor
final class TurnCheckpointTests: XCTestCase {
    private var directories: [URL] = []

    private func temporaryDirectory(_ label: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("turn-checkpoint-\(label)-\(UUID().uuidString)", isDirectory: true)
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

    private func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    // MARK: Pure rules

    func testFileChangingToolsCoverEveryWriteAliasAndNoReader() {
        for name in AppToolRegistry.writeFileAliases.union(AppToolRegistry.editFileAliases) {
            XCTAssertTrue(TurnCheckpointPolicy.isFileChangingTool(name), name)
        }
        for name in ["apply_patch", "MultiEdit", "document_write", "notebook_edit"] {
            XCTAssertTrue(TurnCheckpointPolicy.isFileChangingTool(name), name)
        }
        for name in ["read_file", "run_command", "bash", "grep", "list_directory"] {
            XCTAssertFalse(TurnCheckpointPolicy.isFileChangingTool(name), name)
        }
    }

    func testTargetPathsReadEachToolsOwnArguments() {
        XCTAssertEqual(TurnCheckpointPolicy.targetPaths(for: AppToolCall(
            name: "write_file", arguments: ["filePath": "a.txt", "content": "x"], category: .fileWrite)), ["a.txt"])
        let patch = "*** Begin Patch\n*** Add File: new.txt\n+hi\n*** Update File: old.txt\n@@\n-a\n+b\n*** End Patch"
        XCTAssertEqual(TurnCheckpointPolicy.targetPaths(for: AppToolCall(
            name: "apply_patch", arguments: ["patch_text": patch], category: .fileWrite)), ["new.txt", "old.txt"])
        let edits = #"[{"file_path":"m.txt","old_string":"a","new_string":"b"},{"file_path":"m.txt","old_string":"c","new_string":"d"}]"#
        XCTAssertEqual(TurnCheckpointPolicy.targetPaths(for: AppToolCall(
            name: "multiedit", arguments: ["edits": edits], category: .fileWrite)), ["m.txt"])
    }

    func testContainedTargetRefusesEverythingOutsideTheRoot() throws {
        let root = try temporaryDirectory("root")
        let outside = try temporaryDirectory("outside")
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("link"), withDestinationURL: outside)
        for path in ["../escape.txt", "/etc/hosts", "~/notes.txt", "", ".", "link/new.txt", "link"] {
            XCTAssertNil(TurnCheckpointPolicy.containedTarget(path, root: root), path)
        }
        let inside = try XCTUnwrap(TurnCheckpointPolicy.containedTarget("src/./a.swift", root: root))
        XCTAssertEqual(inside.relativePath, "src/a.swift")
        XCTAssertEqual(inside.url.path, PathContainment.canonical(root).appendingPathComponent("src/a.swift").path)
    }

    func testDecideAppliesFirstCaptureRuleThenSensitivityThenCaps() {
        func decide(_ kind: TurnCheckpointPolicy.FileKind, recorded: Set<String> = [], sensitive: Bool = false,
                    captured: Int = 0) -> TurnCheckpointPolicy.CaptureDecision {
            TurnCheckpointPolicy.decide(
                relativePath: "a.txt", recorded: recorded, isSensitive: sensitive, kind: kind,
                capturedBytes: captured, perFileLimit: 100, perTurnLimit: 250)
        }
        // A path seen once keeps its first state, even a skipped one.
        XCTAssertEqual(decide(.regular(byteCount: 1), recorded: ["a.txt"]), .alreadyRecorded)
        XCTAssertEqual(decide(.regular(byteCount: 1), sensitive: true), .skip(.sensitive))
        XCTAssertEqual(decide(.absent), .recordAbsent)
        XCTAssertEqual(decide(.other), .skip(.notRegularFile))
        XCTAssertEqual(decide(.regular(byteCount: 101)), .skip(.tooLarge))
        XCTAssertEqual(decide(.regular(byteCount: 100)), .capture)
        XCTAssertEqual(decide(.regular(byteCount: 100), captured: 151), .skip(.turnBudget))
        XCTAssertEqual(decide(.regular(byteCount: 100), captured: 150), .capture)
    }

    func testRestorePlanFromManifestAndDisk() {
        var manifest = TurnCheckpointManifest(chatID: UUID(), turnID: UUID(), createdAt: Date(), rootPath: "/r")
        manifest.entries = [
            .init(relativePath: "kept.txt", state: .captured, blobName: "0.bin", byteCount: 1),
            .init(relativePath: "gone.txt", state: .captured, blobName: "1.bin", byteCount: 1),
            .init(relativePath: "made.txt", state: .absent),
            .init(relativePath: "made-then-removed.txt", state: .absent),
            .init(relativePath: "big.bin", state: .notCaptured, skipReason: .tooLarge),
            .init(relativePath: "swapped.txt", state: .captured, blobName: "2.bin", byteCount: 1),
            .init(relativePath: "now-a-dir", state: .captured, blobName: "3.bin", byteCount: 1),
        ]
        let disk: [String: TurnCheckpointPolicy.FileKind?] = [
            "kept.txt": .regular(byteCount: 3), "gone.txt": .absent, "made.txt": .regular(byteCount: 2),
            "made-then-removed.txt": .absent, "big.bin": .regular(byteCount: 9), "swapped.txt": nil,
            "now-a-dir": .other,
        ]
        let plan = TurnCheckpointRestorePlan.make(
            manifest: manifest, current: { disk[$0] ?? nil }, laterTurnPaths: ["kept.txt"])
        XCTAssertEqual(plan.items, [
            .init(relativePath: "kept.txt", action: .overwrite, changedInLaterTurn: true),
            .init(relativePath: "gone.txt", action: .recreate, changedInLaterTurn: false),
            .init(relativePath: "made.txt", action: .trash, changedInLaterTurn: false),
            .init(relativePath: "big.bin", action: .cannotRestore(.tooLarge), changedInLaterTurn: false),
            .init(relativePath: "swapped.txt", action: .cannotRestore(.outsideProject), changedInLaterTurn: false),
            .init(relativePath: "now-a-dir", action: .cannotRestore(.notRegularFile), changedInLaterTurn: false),
        ])
        XCTAssertEqual(plan.actionable.map(\.relativePath), ["kept.txt", "gone.txt", "made.txt"])
    }

    func testRetentionKeepsTheNewestTurns() {
        let base = Date(timeIntervalSince1970: 1_000)
        let turns = (0..<25).map { (turnID: UUID(), createdAt: base.addingTimeInterval(Double($0))) }
        let pruned = TurnCheckpointPolicy.turnsToPrune(turns.shuffled(), keep: 20)
        XCTAssertEqual(pruned, turns.prefix(5).map(\.turnID))
        XCTAssertEqual(TurnCheckpointPolicy.turnsToPrune(Array(turns.prefix(20)), keep: 20), [])
    }

    func testConfirmationTextNamesLaterTurnsAndCapsTheList() {
        let items = (0..<15).map {
            TurnCheckpointRestorePlan.Item(relativePath: "f\($0).txt", action: .overwrite, changedInLaterTurn: $0 == 0)
        }
        let message = TurnCheckpointRestoreText.message(TurnCheckpointRestorePlan(items: items))
        XCTAssertTrue(message.contains("f0.txt (changed again in a later turn)"), message)
        XCTAssertTrue(message.contains("More files not shown: 3"), message)
        XCTAssertFalse(message.contains("f12.txt"), message)
        XCTAssertTrue(message.contains("shell commands"), message)
    }

    // MARK: Store

    func testStoreCapturesOnceAndNeverSensitiveOrOutsideFiles() throws {
        let root = try temporaryDirectory("root")
        let outside = try temporaryDirectory("outside")
        let store = TurnCheckpointStore(rootURL: try temporaryDirectory("store"))
        try write("secret", to: root.appendingPathComponent(".env"))
        try write("outside", to: outside.appendingPathComponent("o.txt"))
        try write("first", to: root.appendingPathComponent("a.txt"))
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("link"), withDestinationURL: outside)
        let chatID = UUID(), turnID = UUID()

        let manifest = try XCTUnwrap(store.capture(
            paths: ["a.txt", ".env", "../\(outside.lastPathComponent)/o.txt", "link/o.txt", "/etc/hosts"],
            rootURL: root, chatID: chatID, turnID: turnID))
        XCTAssertEqual(manifest.entries.map(\.relativePath), ["a.txt", ".env"])
        XCTAssertEqual(manifest.entries[1].state, .notCaptured)
        XCTAssertEqual(manifest.entries[1].skipReason, .sensitive)
        XCTAssertNil(manifest.entries[1].blobName)
        XCTAssertTrue(manifest.isIncomplete)

        // The turn's own later bytes never replace the first capture.
        try write("second", to: root.appendingPathComponent("a.txt"))
        XCTAssertNil(store.capture(paths: ["a.txt"], rootURL: root, chatID: chatID, turnID: turnID))

        let blobs = store.rootURL.appendingPathComponent("\(chatID.uuidString)/\(turnID.uuidString)/blobs")
        let names = try FileManager.default.contentsOfDirectory(atPath: blobs.path)
        XCTAssertEqual(names, ["0.bin"])
        XCTAssertEqual(try String(contentsOf: blobs.appendingPathComponent("0.bin"), encoding: .utf8), "first")
    }

    func testPerFileCapRecordsNotCaptured() throws {
        let root = try temporaryDirectory("root")
        let store = TurnCheckpointStore(rootURL: try temporaryDirectory("store"))
        let big = root.appendingPathComponent("big.bin")
        FileManager.default.createFile(atPath: big.path, contents: nil)
        let handle = try FileHandle(forWritingTo: big)
        try handle.truncate(atOffset: UInt64(TurnCheckpointPolicy.perFileByteLimit + 1))
        try handle.close()
        let manifest = try XCTUnwrap(store.capture(paths: ["big.bin"], rootURL: root, chatID: UUID(), turnID: UUID()))
        XCTAssertEqual(manifest.entries.first?.skipReason, .tooLarge)
    }

    func testCorruptManifestIsQuarantinedAndReadsAsNoCheckpoint() throws {
        let root = try temporaryDirectory("root")
        let storeRoot = try temporaryDirectory("store")
        try write("x", to: root.appendingPathComponent("a.txt"))
        let chatID = UUID(), turnID = UUID()
        XCTAssertNotNil(TurnCheckpointStore(rootURL: storeRoot)
            .capture(paths: ["a.txt"], rootURL: root, chatID: chatID, turnID: turnID))
        let turnDirectory = storeRoot.appendingPathComponent("\(chatID.uuidString)/\(turnID.uuidString)")
        let manifestURL = turnDirectory.appendingPathComponent("manifest.json")
        try Data("{ not json".utf8).write(to: manifestURL)

        let reopened = TurnCheckpointStore(rootURL: storeRoot)
        XCTAssertNil(reopened.manifest(chatID: chatID, turnID: turnID))
        XCTAssertNil(reopened.restorePlan(chatID: chatID, turnID: turnID))
        XCTAssertFalse(FileManager.default.fileExists(atPath: manifestURL.path))
        let kept = try FileManager.default.contentsOfDirectory(atPath: turnDirectory.path)
            .filter { $0.hasPrefix("manifest.corrupt-") }
        XCTAssertEqual(kept.count, 1)
        XCTAssertEqual(try String(contentsOf: turnDirectory.appendingPathComponent(kept[0]), encoding: .utf8), "{ not json")
        // Not started over: a fresh manifest would call the turn's bytes "before".
        try write("y", to: root.appendingPathComponent("b.txt"))
        XCTAssertNil(reopened.capture(paths: ["b.txt"], rootURL: root, chatID: chatID, turnID: turnID))
        XCTAssertFalse(FileManager.default.fileExists(atPath: manifestURL.path))
    }

    func testRetentionPrunesOldTurnsAndDeleteRemovesTheChat() throws {
        let root = try temporaryDirectory("root")
        let store = TurnCheckpointStore(rootURL: try temporaryDirectory("store"))
        try write("x", to: root.appendingPathComponent("a.txt"))
        let chatID = UUID()
        let base = Date(timeIntervalSince1970: 1_000)
        let turns = (0..<(TurnCheckpointPolicy.retainedTurnsPerChat + 2)).map { _ in UUID() }
        for (offset, turnID) in turns.enumerated() {
            store.capture(paths: ["a.txt"], rootURL: root, chatID: chatID, turnID: turnID,
                          now: base.addingTimeInterval(Double(offset)))
        }
        let kept = store.manifests(chatID: chatID).map(\.turnID)
        XCTAssertEqual(kept, Array(turns.suffix(TurnCheckpointPolicy.retainedTurnsPerChat)))
        store.delete(chatID: chatID)
        XCTAssertTrue(store.manifests(chatID: chatID).isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.rootURL.appendingPathComponent(chatID.uuidString).path))
    }

    // MARK: Real write tool path

    func testWriteToolPathCapturesAndRestoreBringsBytesBack() async throws {
        let root = try temporaryDirectory("project")
        let store = TurnCheckpointStore(rootURL: try temporaryDirectory("store"))
        let fakeTrash = try temporaryDirectory("trash")
        try write("original\n", to: root.appendingPathComponent("a.txt"))
        let model = AppModel()
        let prompt = AppChatMessage(role: .user, content: "change a.txt and add b.txt")
        let chat = AppChat(title: "checkpoint", messages: [prompt])
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.installTurnCheckpointCapture(store: store)
        let project = AppProject(name: "checkpoint", rootDirectoryPath: root.path, permissions: .fullAccess)

        let read = await AppToolRegistry.execute(
            call: AppToolCall(name: "read_file", arguments: ["path": "a.txt"], category: .fileRead),
            in: project, chatID: chat.id)
        XCTAssertFalse(read.isError, read.output)
        let edited = await model.executeApprovedTool(AppToolCall(
            name: "write_file", arguments: ["path": "a.txt", "content": "changed\n"], category: .fileWrite),
            project: project, chatID: chat.id)
        XCTAssertFalse(edited.result.isError, edited.result.output)
        let created = await model.executeApprovedTool(AppToolCall(
            name: "write_file", arguments: ["path": "b.txt", "content": "new\n"], category: .fileWrite),
            project: project, chatID: chat.id)
        XCTAssertFalse(created.result.isError, created.result.output)
        // Capture leaves the tool's own result alone.
        XCTAssertTrue(created.result.output.hasPrefix("Successfully wrote"), created.result.output)

        let manifest = try XCTUnwrap(store.manifest(chatID: chat.id, turnID: prompt.id))
        XCTAssertEqual(manifest.entries.map(\.relativePath), ["a.txt", "b.txt"])
        XCTAssertEqual(manifest.entries.map(\.state), [.captured, .absent])

        let plan = try XCTUnwrap(store.restorePlan(chatID: chat.id, turnID: prompt.id))
        XCTAssertEqual(plan.items.map(\.action), [.overwrite, .trash])
        var trashed: [URL] = []
        let outcome = try XCTUnwrap(store.restore(chatID: chat.id, turnID: prompt.id) { url in
            trashed.append(url)
            try FileManager.default.moveItem(at: url, to: fakeTrash.appendingPathComponent(url.lastPathComponent))
        })
        XCTAssertEqual(outcome.restored, ["a.txt"])
        XCTAssertEqual(outcome.trashed, ["b.txt"])
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("a.txt")), Data("original\n".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("b.txt").path))
        XCTAssertEqual(try String(contentsOf: fakeTrash.appendingPathComponent("b.txt"), encoding: .utf8), "new\n")
        XCTAssertEqual(trashed.map(\.lastPathComponent), ["b.txt"])
        XCTAssertNotNil(store.manifest(chatID: chat.id, turnID: prompt.id)?.restoredAt)
    }

    func testGhostChatAndRunningTurnGetNoCheckpointWork() throws {
        let model = AppModel()
        let ghost = AppChat(title: "ghost", messages: [AppChatMessage(role: .user, content: "hi")], isGhost: true)
        let normal = AppChat(title: "normal", messages: [AppChatMessage(role: .user, content: "hi")])
        model.chats = [ghost, normal]
        XCTAssertNil(model.checkpointTurnID(forChat: ghost.id))
        XCTAssertFalse(model.checkpointsAvailable(inChat: ghost.id))
        XCTAssertEqual(model.checkpointTurnID(forChat: normal.id), normal.messages[0].id)
        model.generating = true
        defer { model.generating = false }
        XCTAssertEqual(
            model.checkpointRestoreRefusal(chatID: normal.id, turnID: normal.messages[0].id),
            "Wait for the current turn to finish before restoring.")
    }
}
