import XCTest

@testable import TurboSparkApp

/// C1-C4: the project and worktree lifecycle. `deleteProject` did not undo
/// what `selectProject` does, `loadProjects` restored an id it never checked,
/// and `WorktreeModel`'s two async paths both raced themselves.
@MainActor
final class ProjectLifecycleTests: XCTestCase {
    private func makeProject(named name: String) throws -> (AppProject, () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return (
            AppProject(name: name, rootDirectoryPath: dir.path),
            { try? FileManager.default.removeItem(at: dir) }
        )
    }

    // MARK: - C2: deleting undoes what selecting did

    func testDeletingTheSelectedProjectClearsItsWorktree() throws {
        let appModel = AppModel()
        let (project, cleanup) = try makeProject(named: "doomed")
        defer { cleanup() }

        appModel.projects = [project]
        appModel.selectProject(id: project.id)
        XCTAssertNotNil(appModel.worktree, "Precondition: selecting binds the git pane.")

        appModel.deleteProject(id: project.id)

        XCTAssertNil(
            appModel.worktree,
            "The git pane must not keep rendering a repository the user just removed.")
        XCTAssertNil(appModel.selectedProjectID)
        XCTAssertTrue(appModel.projects.isEmpty)
    }

    func testDeletingAnUnselectedProjectLeavesTheSelectionAlone() throws {
        let appModel = AppModel()
        let (keep, cleanupKeep) = try makeProject(named: "keep")
        let (drop, cleanupDrop) = try makeProject(named: "drop")
        defer {
            cleanupKeep()
            cleanupDrop()
        }

        appModel.projects = [keep, drop]
        appModel.selectProject(id: keep.id)
        appModel.deleteProject(id: drop.id)

        XCTAssertEqual(appModel.selectedProjectID, keep.id)
        XCTAssertNotNil(
            appModel.worktree, "Deleting a different project must not tear down the active one.")
    }

    // MARK: - C4: a restored selection is validated and rebuilt

    func testAnUnknownRestoredProjectIDIsDroppedRatherThanFilteringChatsToNothing() throws {
        let file = AppStorageRoot.file("projects_archive.json")
        let saved = try? Data(contentsOf: file)
        defer {
            try? FileManager.default.removeItem(at: file)
            if let saved { try? saved.write(to: file) }
        }

        // An id with no matching project: what a half-finished delete, a
        // hand-edited file, or a schema change leaves behind.
        AppProjectFileStore.save(
            AppProjectArchive(selectedProjectID: UUID(), projects: []))

        let appModel = AppModel()
        XCTAssertNil(
            appModel.selectedProjectID,
            "An id naming no project filters the chat list to nothing with no way to see why.")
    }

    func testAValidRestoredProjectRebuildsItsWorktree() throws {
        let file = AppStorageRoot.file("projects_archive.json")
        let saved = try? Data(contentsOf: file)
        defer {
            try? FileManager.default.removeItem(at: file)
            if let saved { try? saved.write(to: file) }
        }

        let (project, cleanup) = try makeProject(named: "restored")
        defer { cleanup() }
        AppProjectFileStore.save(
            AppProjectArchive(selectedProjectID: project.id, projects: [project]))

        let appModel = AppModel()
        XCTAssertEqual(appModel.selectedProjectID, project.id)
        XCTAssertNotNil(
            appModel.worktree,
            "The git pane came up empty on every relaunch until the project was re-picked.")
        XCTAssertEqual(appModel.worktree?.rootDirectoryPath, project.rootDirectoryPath)
    }

    // MARK: - C1: two overlapping diff loads do not race
    //
    // **THIS CASE IS A GUARD, NOT A PROOF, AND THE DIFFERENCE IS WORTH
    // STATING.** The defect is an ORDERING one: an unstored `Task` meant
    // `guard !Task.isCancelled` could never fire, so a superseded `git diff`
    // that happened to finish LAST overwrote the newer selection's diff.
    // Reproducing that needs the first process to outlive the second, and two
    // `git diff` calls on a two-line file are uniformly fast -- deleting
    // either half of the fix leaves this green. Making it deterministic would
    // mean a delay seam in `queryGitDiff`, i.e. shipping a test hook in the
    // production path for a race that is already fixed by construction.
    // What this DOES pin is that the normal ordering shows the selected
    // file's own diff, and that the spinner comes down.

    /// Runs `git` in `dir`, failing the test if it does not succeed.
    private func git(_ args: [String], in dir: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.currentDirectoryURL = dir
        process.arguments = args
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, "git \(args.joined(separator: " ")) failed")
    }

    func testTheLastSelectedFilesDiffIsTheOneThatSurvives() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // A REAL repository with two files whose diffs DIFFER. Against an
        // empty directory both `git diff` calls return "", so the two
        // outcomes are indistinguishable and the test proves nothing -- the
        // fixture has to be able to tell them apart.
        try git(["init", "-q"], in: dir)
        try git(["config", "user.email", "t@example.com"], in: dir)
        try git(["config", "user.name", "T"], in: dir)
        try "base\n".write(
            to: dir.appendingPathComponent("alpha.txt"), atomically: true, encoding: .utf8)
        try "base\n".write(
            to: dir.appendingPathComponent("beta.txt"), atomically: true, encoding: .utf8)
        try git(["add", "."], in: dir)
        try git(["commit", "-qm", "base"], in: dir)
        try "base\nALPHA_ONLY_MARKER\n".write(
            to: dir.appendingPathComponent("alpha.txt"), atomically: true, encoding: .utf8)
        try "base\nBETA_ONLY_MARKER\n".write(
            to: dir.appendingPathComponent("beta.txt"), atomically: true, encoding: .utf8)

        let worktree = WorktreeModel(rootDirectoryPath: dir.path)
        // `init` kicks off a refresh, and a refresh that no longer lists the
        // selected file clears `selectedFilePath`. Let it settle first, or
        // this measures that interaction rather than the diff race.
        // Waiting on `isRefreshing` would return immediately -- it is set
        // INSIDE the task, so it is still false when this line runs.
        // `lastRefreshTime` is only written once the refresh has landed.
        var settle = Date().addingTimeInterval(5)
        while worktree.lastRefreshTime == nil && Date() < settle {
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        // The `Task` inside `loadDiff` used to be referenced by nothing, so
        // its own `guard !Task.isCancelled` could never fire: clicking A then
        // B left whichever `git diff` finished last on screen, under B's name.
        worktree.loadDiff(for: "alpha.txt")
        worktree.loadDiff(for: "beta.txt")

        XCTAssertEqual(worktree.selectedFilePath, "beta.txt")
        settle = Date().addingTimeInterval(5)
        while worktree.isLoadingDiff && Date() < settle {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(
            worktree.isLoadingDiff, "The spinner must come down rather than being stranded.")
        XCTAssertEqual(worktree.selectedFilePath, "beta.txt")

        let shown = worktree.selectedFileDiff ?? ""
        XCTAssertTrue(
            shown.contains("BETA_ONLY_MARKER"),
            "The selected file's own diff must be the one on screen. Got: \(shown)")
        XCTAssertFalse(
            shown.contains("ALPHA_ONLY_MARKER"),
            "The superseded load must not land its diff under the new file's name.")
    }

    // MARK: - C3: an empty root clears every field, not just the file list

    func testClearingTheRootResetsBranchAndTotalsToo() async throws {
        let worktree = WorktreeModel(rootDirectoryPath: "")
        worktree.currentBranch = "left-over-branch"
        worktree.totalAdditions = 42
        worktree.totalDeletions = 7

        worktree.refresh()
        let deadline = Date().addingTimeInterval(5)
        while !worktree.currentBranch.isEmpty && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        XCTAssertEqual(
            worktree.currentBranch, "",
            "A stale branch name beside an empty file list reads as a clean repository.")
        XCTAssertEqual(worktree.totalAdditions, 0)
        XCTAssertEqual(worktree.totalDeletions, 0)
        XCTAssertFalse(worktree.isRefreshing)
    }
}
