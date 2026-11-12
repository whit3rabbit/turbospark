import XCTest
@testable import TurboSparkApp

final class TaskIDCollisionTests: XCTestCase {
    func testCreateDoesNotOverwriteOnIDCollision() {
        let manager = TaskManager.shared
        let original = manager.idGenerator
        defer { manager.idGenerator = original }
        let counter = Counter()
        // First two calls collide with each other; the third is unique.
        manager.idGenerator = { counter.next() < 2 ? "deadbeef" : "cafef00d" }

        let a = manager.createTask(subject: "first", description: "")
        let b = manager.createTask(subject: "second", description: "")
        defer {
            _ = manager.updateTask(id: a.id, status: "deleted")
            _ = manager.updateTask(id: b.id, status: "deleted")
        }
        XCTAssertNotEqual(a.id, b.id)
        XCTAssertEqual(manager.getTask(id: a.id)?.subject, "first")
        XCTAssertEqual(manager.getTask(id: b.id)?.subject, "second")
    }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func next() -> Int { lock.lock(); defer { lock.unlock() }; let v = n; n += 1; return v }
}
