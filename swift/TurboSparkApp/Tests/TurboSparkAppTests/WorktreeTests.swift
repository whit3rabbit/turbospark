import XCTest
@testable import TurboSparkApp

final class WorktreeTests: XCTestCase {
    @MainActor
    func testWorktreeFileChangeInitialization() {
        let change = WorktreeFileChange(
            relativePath: "crates/tools/src/bash.rs",
            status: .modified,
            additions: 12,
            deletions: 4
        )

        XCTAssertEqual(change.relativePath, "crates/tools/src/bash.rs")
        XCTAssertEqual(change.fileName, "bash.rs")
        XCTAssertEqual(change.directoryPath, "crates/tools/src")
        XCTAssertEqual(change.status, .modified)
        XCTAssertEqual(change.additions, 12)
        XCTAssertEqual(change.deletions, 4)
        XCTAssertEqual(change.status.label, "Modified")
    }

    @MainActor
    func testAppInteractionModeProperties() {
        let chat = AppModel.AppInteractionMode.chat
        let cowork = AppModel.AppInteractionMode.cowork

        XCTAssertEqual(chat.id, "chat")
        XCTAssertEqual(chat.title, "Chat")
        XCTAssertEqual(cowork.id, "cowork")
        XCTAssertEqual(cowork.title, "Cowork")
    }

    @MainActor
    func testWorktreeModelRefreshOnEmptyPath() {
        let worktree = WorktreeModel(rootDirectoryPath: "")
        XCTAssertFalse(worktree.isGitRepository)
        XCTAssertTrue(worktree.files.isEmpty)
    }
}
