import PencilKit
import XCTest
@testable import TurboSparkApp

/// ZP-10: board persistence, naming, corruption handling and the debounce.
@MainActor
final class WhiteboardStoreTests: XCTestCase {
    private var directory: URL!

    override func setUp() {
        super.setUp()
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiteboard-tests-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
        super.tearDown()
    }

    private func makeStore(debounce: Duration = .milliseconds(20)) -> WhiteboardStore {
        WhiteboardStore(directory: directory, debounce: debounce)
    }

    /// A drawing with `count` horizontal strokes, as PKDrawing data.
    static func drawingData(strokes count: Int) -> Data {
        var drawing = PKDrawing()
        for i in 0..<count {
            let y = CGFloat(20 + i * 10)
            drawing.strokes.append(
                WhiteboardStrokes.stroke(
                    through: [CGPoint(x: 10, y: y), CGPoint(x: 90, y: y)],
                    color: .black, width: 4)!)
        }
        return drawing.dataRepresentation()
    }

    private func strokeCount(_ data: Data) throws -> Int {
        try PKDrawing(data: data).strokes.count
    }

    // MARK: - Naming

    func testAutoTitlesNeverCollideAndReuseTheLowestFreeNumber() {
        let store = makeStore()
        let a = store.createBoard(workspaceKey: "w")
        let b = store.createBoard(workspaceKey: "w")
        let c = store.createBoard(workspaceKey: "w")
        let titles = [a, b, c].map(\.title)
        XCTAssertEqual(Set(titles).count, 3, "three boards, three distinct titles")
        store.delete(id: b.id)
        let d = store.createBoard(workspaceKey: "w")
        XCTAssertEqual(d.title, b.title, "the freed number is reused")
        XCTAssertEqual(Set(store.boards(in: "w").map(\.title)).count, 3)
    }

    func testPureNamingHelpers() {
        let make = { (n: Int) in "Board \(n)" }
        XCTAssertEqual(WhiteboardStore.nextAutoTitle(existing: [], make: make), "Board 1")
        XCTAssertEqual(WhiteboardStore.nextAutoTitle(existing: ["Board 1", "board 2"], make: make), "Board 3")
        XCTAssertEqual(WhiteboardStore.nextAutoTitle(existing: ["Board 2"], make: make), "Board 1")
        XCTAssertEqual(WhiteboardStore.uniqueTitle("Plan", existing: ["plan", "Plan 2"]), "Plan 3")
        XCTAssertEqual(WhiteboardStore.uniqueTitle("Plan", existing: []), "Plan")
    }

    func testTitlesAreScopedPerWorkspace() {
        let store = makeStore()
        let a = store.createBoard(workspaceKey: "one")
        let b = store.createBoard(workspaceKey: "two")
        XCTAssertEqual(a.title, b.title, "each workspace numbers from 1")
        XCTAssertEqual(store.boards(in: "one").count, 1)
    }

    func testRenameDedupesTrimsAndIgnoresEmpty() {
        let store = makeStore()
        let a = store.createBoard(workspaceKey: "w")
        let b = store.createBoard(workspaceKey: "w")
        store.rename(id: b.id, to: "  \(a.title.uppercased())  ")
        XCTAssertNotEqual(store.board(id: b.id)?.title.lowercased(), a.title.lowercased())
        store.rename(id: b.id, to: "   ")
        XCTAssertNotEqual(store.board(id: b.id)?.title, "")
        store.rename(id: b.id, to: String(repeating: "x", count: 200))
        XCTAssertEqual(store.board(id: b.id)?.title.count, WhiteboardStore.maxTitleLength)
    }

    // MARK: - Persistence

    func testBoardsAndStrokesSurviveARestart() throws {
        let first = makeStore()
        let board = first.createBoard(workspaceKey: "w")
        first.scheduleSave(id: board.id, data: Self.drawingData(strokes: 3))
        first.flush()

        let second = makeStore()
        XCTAssertEqual(second.boards(in: "w").map(\.id), [board.id])
        XCTAssertEqual(second.board(id: board.id)?.title, board.title)
        guard case .drawing(let data) = second.loadDrawing(id: board.id) else {
            return XCTFail("expected the saved drawing")
        }
        XCTAssertEqual(try strokeCount(data), 3)
    }

    func testDebouncedWriteLandsWithoutAnExplicitFlush() async throws {
        let store = makeStore(debounce: .milliseconds(20))
        let board = store.createBoard(workspaceKey: "w")
        store.scheduleSave(id: board.id, data: Self.drawingData(strokes: 1))
        XCTAssertTrue(store.hasPendingWrites)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.blobURL(for: board).path),
            "nothing is written before the debounce elapses")
        for _ in 0..<100 where store.hasPendingWrites { try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertFalse(store.hasPendingWrites)
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.blobURL(for: board).path))
    }

    func testBurstOfChangesWritesTheLastOne() throws {
        let store = makeStore(debounce: .seconds(30))
        let board = store.createBoard(workspaceKey: "w")
        for count in 1...4 { store.scheduleSave(id: board.id, data: Self.drawingData(strokes: count)) }
        // Still queued, and a read sees the queued value, not an older disk copy.
        guard case .drawing(let queued) = store.loadDrawing(id: board.id) else { return XCTFail() }
        XCTAssertEqual(try strokeCount(queued), 4)
        store.flush()
        guard case .drawing(let disk) = makeStore().loadDrawing(id: board.id) else { return XCTFail() }
        XCTAssertEqual(try strokeCount(disk), 4)
    }

    func testDeleteRemovesTheBlobAndCancelsItsQueuedWrite() {
        let store = makeStore(debounce: .seconds(30))
        let board = store.createBoard(workspaceKey: "w")
        store.scheduleSave(id: board.id, data: Self.drawingData(strokes: 1))
        store.flush()
        store.scheduleSave(id: board.id, data: Self.drawingData(strokes: 2))
        let url = store.blobURL(for: board)
        store.delete(id: board.id)
        store.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        XCTAssertTrue(makeStore().boards.isEmpty)
    }

    // MARK: - Corruption

    func testCorruptBlobIsSetAsideNotOverwrittenAndTheBoardOpensEmpty() throws {
        let store = makeStore()
        let board = store.createBoard(workspaceKey: "w")
        store.scheduleSave(id: board.id, data: Self.drawingData(strokes: 1))
        store.flush()
        let url = store.blobURL(for: board)
        try Data("not a drawing".utf8).write(to: url)

        let reopened = makeStore()
        XCTAssertEqual(reopened.loadDrawing(id: board.id), .recoveredFromCorruption)
        XCTAssertNotNil(reopened.lastIssue)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path), "the bad file was moved")
        let kept = try FileManager.default.contentsOfDirectory(atPath: url.deletingLastPathComponent().path)
        XCTAssertTrue(kept.contains { $0.hasPrefix("corrupt-") }, "and kept: \(kept)")
        XCTAssertNotNil(reopened.board(id: board.id), "the board row stays")
        // A later save writes a fresh blob and the board works again.
        reopened.scheduleSave(id: board.id, data: Self.drawingData(strokes: 2))
        reopened.flush()
        guard case .drawing(let data) = makeStore().loadDrawing(id: board.id) else { return XCTFail() }
        XCTAssertEqual(try strokeCount(data), 2)
    }

    func testCorruptMetadataIsQuarantinedAndBlobsAreAdopted() throws {
        let store = makeStore()
        let board = store.createBoard(workspaceKey: "w")
        store.scheduleSave(id: board.id, data: Self.drawingData(strokes: 2))
        store.flush()
        try Data("{ not json".utf8).write(to: directory.appendingPathComponent("boards.json"))

        let recovered = makeStore()
        XCTAssertEqual(recovered.boards(in: "w").map(\.id), [board.id], "the sketch is not lost")
        guard case .drawing(let data) = recovered.loadDrawing(id: board.id) else { return XCTFail() }
        XCTAssertEqual(try strokeCount(data), 2)
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertTrue(names.contains { $0.hasPrefix("boards.corrupt-") },
            "the unreadable metadata was kept, not overwritten: \(names)")
    }

    func testMetadataRowWithoutABlobIsAnEmptyBoard() {
        let store = makeStore()
        let board = store.createBoard(workspaceKey: "w")
        XCTAssertEqual(makeStore().loadDrawing(id: board.id), .empty)
    }
}
