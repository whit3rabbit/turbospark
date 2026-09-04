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
        let projects = AppModel.AppInteractionMode.projects

        XCTAssertEqual(chat.id, "chat")
        XCTAssertEqual(chat.title, "Chat")
        XCTAssertEqual(projects.id, "projects")
        XCTAssertEqual(projects.title, "Projects")
    }

    /// state#89: `--numstat` and `--porcelain` disagree about how a rename is
    /// spelled, and the join between them only normalized the porcelain side.
    /// Every rename and every non-ASCII path therefore looked up a key that
    /// could not exist and read 0/0.
    func testNumstatPathsAreNormalizedTheSameWayPorcelainPathsAre() {
        // The plain rename form.
        XCTAssertEqual(
            WorktreeModel.numstatPath("old/name.swift => new/name.swift"), "new/name.swift")
        // The FACTORED form, which porcelain never emits at all.
        XCTAssertEqual(
            WorktreeModel.numstatPath("crates/{old => new}/src/lib.rs"),
            "crates/new/src/lib.rs")
        // A factored half can be empty, which is how git spells adding or
        // removing a directory level.
        XCTAssertEqual(WorktreeModel.numstatPath("d/{ => sub}/f.txt"), "d/sub/f.txt")
        XCTAssertEqual(WorktreeModel.numstatPath("d/{sub => }/f.txt"), "d/f.txt")
        // C-quoting, which numstat applies for exactly the reasons porcelain
        // does. The two sides must agree byte for byte or the join misses.
        XCTAssertEqual(
            WorktreeModel.numstatPath("\"caf\\303\\251.txt\""),
            WorktreeModel.porcelainPath("\"caf\\303\\251.txt\""))
        // And an ordinary path is untouched, which is the common case.
        XCTAssertEqual(WorktreeModel.numstatPath("src/main.swift"), "src/main.swift")
    }

    @MainActor
    func testWorktreeModelRefreshOnEmptyPath() {
        let worktree = WorktreeModel(rootDirectoryPath: "")
        XCTAssertFalse(worktree.isGitRepository)
        XCTAssertTrue(worktree.files.isEmpty)
    }
}
