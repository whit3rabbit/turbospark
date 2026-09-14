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

    @MainActor
    func testWorktreeTreeNodeHierarchyBuilding() {
        let changes = [
            WorktreeFileChange(relativePath: "crates/bench/src/main.rs", status: .modified, additions: 10, deletions: 2),
            WorktreeFileChange(relativePath: "crates/bench/src/real.rs", status: .added, additions: 5, deletions: 0),
            WorktreeFileChange(relativePath: "crates/catalog/src/lib.rs", status: .modified, additions: 20, deletions: 1),
            WorktreeFileChange(relativePath: "AGENTS.md", status: .modified, additions: 50, deletions: 9)
        ]

        let tree = WorktreeModel.buildTree(from: changes)
        // Top-level contains crates (directory) and AGENTS.md (leaf file)
        XCTAssertEqual(tree.count, 2)

        let cratesNode = tree.first { $0.isDirectory }
        XCTAssertNotNil(cratesNode)
        XCTAssertEqual(cratesNode?.name, "crates")
        XCTAssertEqual(cratesNode?.relativePath, "crates")
        XCTAssertEqual(cratesNode?.additions, 35) // 10 + 5 + 20
        XCTAssertEqual(cratesNode?.deletions, 3)  // 2 + 0 + 1
        XCTAssertEqual(cratesNode?.children.count, 2) // bench and catalog

        let fileNode = tree.first { !$0.isDirectory }
        XCTAssertNotNil(fileNode)
        XCTAssertEqual(fileNode?.name, "AGENTS.md")
        XCTAssertEqual(fileNode?.additions, 50)
        XCTAssertEqual(fileNode?.deletions, 9)
    }

    @MainActor
    func testWorktreeComparisonModes() {
        let uncommitted = WorktreeComparisonMode.uncommitted
        XCTAssertEqual(uncommitted.title, "Uncommitted changes")

        let againstBranch = WorktreeComparisonMode.againstBranch("main")
        XCTAssertEqual(againstBranch.title, "All changes vs main")

        let commit = WorktreeComparisonMode.commit(hash: "dc63fbf123456", summary: "Fix bug in parser")
        XCTAssertEqual(commit.title, "dc63fbf: Fix bug in parser")
    }

    @MainActor
    func testWorktreeViewModes() {
        XCTAssertEqual(WorktreeViewMode.tree.title, "Tree View")
        XCTAssertEqual(WorktreeViewMode.flat.title, "Flat List")
        XCTAssertEqual(WorktreeViewMode.tree.systemImage, "list.bullet.indent")
        XCTAssertEqual(WorktreeViewMode.flat.systemImage, "list.bullet")
    }

    @MainActor
    func testWorktreeCommitProperties() {
        let commit = WorktreeCommit(
            hash: "dc63fbf1234567890",
            shortHash: "dc63fbf",
            author: "whit3rabbit",
            relativeDate: "6m ago",
            summary: "docs(notes): point the bench page"
        )
        XCTAssertEqual(commit.id, "dc63fbf1234567890")
        XCTAssertEqual(commit.shortHash, "dc63fbf")
        XCTAssertEqual(commit.author, "whit3rabbit")
        XCTAssertEqual(commit.relativeDate, "6m ago")
        XCTAssertEqual(commit.summary, "docs(notes): point the bench page")
    }

    @MainActor
    func testWorktreeFilteredFilesSearch() {
        let worktree = WorktreeModel(rootDirectoryPath: "")
        worktree.files = [
            WorktreeFileChange(relativePath: "crates/bench/src/main.rs", status: .modified),
            WorktreeFileChange(relativePath: "crates/catalog/src/lib.rs", status: .modified),
            WorktreeFileChange(relativePath: "AGENTS.md", status: .modified)
        ]

        XCTAssertEqual(worktree.filteredFiles.count, 3)

        worktree.searchQuery = "bench"
        XCTAssertEqual(worktree.filteredFiles.count, 1)
        XCTAssertEqual(worktree.filteredFiles.first?.relativePath, "crates/bench/src/main.rs")

        worktree.searchQuery = "agents"
        XCTAssertEqual(worktree.filteredFiles.count, 1)
        XCTAssertEqual(worktree.filteredFiles.first?.relativePath, "AGENTS.md")

        worktree.searchQuery = ""
        XCTAssertEqual(worktree.filteredFiles.count, 3)
    }

    @MainActor
    func testWorktreeTabModes() {
        XCTAssertEqual(WorktreeTabMode.changes.title, "Changes")
        XCTAssertEqual(WorktreeTabMode.timeline.title, "Timeline")
        XCTAssertEqual(WorktreeTabMode.worktrees.title, "Worktrees")
    }

    @MainActor
    func testGitWorktreeInfoProperties() {
        let wt = GitWorktreeInfo(
            path: "/path/to/worktree",
            head: "dc63fbf",
            branch: "feature-branch",
            isCurrent: true
        )
        XCTAssertEqual(wt.id, "/path/to/worktree")
        XCTAssertEqual(wt.path, "/path/to/worktree")
        XCTAssertEqual(wt.head, "dc63fbf")
        XCTAssertEqual(wt.branch, "feature-branch")
        XCTAssertTrue(wt.isCurrent)
    }

    func testParseWorktreeListOutput() {
        let raw = """
        worktree /Users/dev/repo
        HEAD a1b2c3d4e5f6
        branch refs/heads/main

        worktree /Users/dev/repo-feature
        HEAD f6e5d4c3b2a1
        branch refs/heads/feature-x

        """
        let worktrees = WorktreeModel.parseWorktreeListOutput(raw, currentRoot: "/Users/dev/repo")
        XCTAssertEqual(worktrees.count, 2)

        XCTAssertEqual(worktrees[0].path, "/Users/dev/repo")
        XCTAssertEqual(worktrees[0].head, "a1b2c3d")
        XCTAssertEqual(worktrees[0].branch, "main")
        XCTAssertTrue(worktrees[0].isCurrent)

        XCTAssertEqual(worktrees[1].path, "/Users/dev/repo-feature")
        XCTAssertEqual(worktrees[1].head, "f6e5d4c")
        XCTAssertEqual(worktrees[1].branch, "feature-x")
        XCTAssertFalse(worktrees[1].isCurrent)
    }

    func testGitCheckoutTreatsDashPrefixedBranchAsAnOperand() {
        let arguments = WorktreeModel.gitCheckoutArguments(branch: "--force")
        XCTAssertEqual(arguments.switchArgs, ["switch", "--", "--force"])
        XCTAssertEqual(arguments.checkoutArgs, ["checkout", "--", "--force"])
    }

    func testParseNameStatusAndNumstat() {
        let numstat = "12\t4\tsrc/main.swift\n20\t0\tsrc/new.swift\n0\t15\tsrc/old.swift\n"
        let nameStatus = "M\tsrc/main.swift\nA\tsrc/new.swift\nD\tsrc/old.swift\n"

        let changes = WorktreeModel.parseNameStatusAndNumstat(
            numstatOutput: numstat,
            nameStatusOutput: nameStatus
        )
        XCTAssertEqual(changes.count, 3)

        XCTAssertEqual(changes[0].relativePath, "src/main.swift")
        XCTAssertEqual(changes[0].status, .modified)
        XCTAssertEqual(changes[0].additions, 12)
        XCTAssertEqual(changes[0].deletions, 4)

        XCTAssertEqual(changes[1].relativePath, "src/new.swift")
        XCTAssertEqual(changes[1].status, .added)
        XCTAssertEqual(changes[1].additions, 20)
        XCTAssertEqual(changes[1].deletions, 0)

        XCTAssertEqual(changes[2].relativePath, "src/old.swift")
        XCTAssertEqual(changes[2].status, .deleted)
        XCTAssertEqual(changes[2].additions, 0)
        XCTAssertEqual(changes[2].deletions, 15)
    }

    @MainActor
    func testScopedFilesByComparisonMode() {
        let worktree = WorktreeModel(rootDirectoryPath: "")
        let uncommittedFile = WorktreeFileChange(relativePath: "local.swift", status: .modified)
        let branchFile = WorktreeFileChange(relativePath: "branch.swift", status: .modified)
        let commitFile = WorktreeFileChange(relativePath: "commit.swift", status: .modified)

        worktree.files = [uncommittedFile]
        worktree.branchComparisonFiles = [branchFile]
        worktree.selectedCommitFiles = [commitFile]

        worktree.comparisonMode = .uncommitted
        XCTAssertEqual(worktree.scopedFiles.map(\.relativePath), ["local.swift"])

        worktree.comparisonMode = .againstBranch("main")
        XCTAssertEqual(worktree.scopedFiles.map(\.relativePath), ["branch.swift"])

        worktree.comparisonMode = .commit(hash: "123", summary: "test")
        XCTAssertEqual(worktree.scopedFiles.map(\.relativePath), ["commit.swift"])
    }
}
