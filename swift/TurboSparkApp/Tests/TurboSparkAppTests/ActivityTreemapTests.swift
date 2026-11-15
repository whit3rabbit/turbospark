import CoreGraphics
import XCTest
@testable import TurboSparkApp

final class ActivityTreemapTests: XCTestCase {
    private func change(_ path: String, _ status: WorktreeFileChange.Status, _ add: Int, _ del: Int) -> WorktreeFileChange {
        WorktreeFileChange(relativePath: path, status: status, additions: add, deletions: del)
    }

    private var sample: [WorktreeFileChange] {
        [
            change("crates/a/lib.rs", .modified, 10, 2),
            change("crates/a/new.rs", .untracked, 30, 0),
            change("crates/b/gone.rs", .deleted, 0, 40),
            change("docs/README.md", .renamed, 0, 0),
            change("top.txt", .added, 5, 5),
            change("odd.bin", .unknown, 0, 0),
        ]
    }

    func testSummaryCountsEveryStatusAndTotals() {
        let s = ActivitySummary(files: sample)
        XCTAssertEqual(s.files, 6)
        XCTAssertEqual(s.additions, 45)
        XCTAssertEqual(s.deletions, 47)
        XCTAssertEqual(s.added, 2)       // untracked plus added
        XCTAssertEqual(s.modified, 1)
        XCTAssertEqual(s.deleted, 1)
        XCTAssertEqual(s.renamed, 1)
        XCTAssertEqual(s.withoutLineChanges, 2)
    }

    func testSummaryMatchesTheTreeRollup() {
        let s = ActivitySummary(files: sample)
        let roots = WorktreeModel.buildTree(from: sample)
        XCTAssertEqual(roots.reduce(0) { $0 + $1.additions }, s.additions)
        XCTAssertEqual(roots.reduce(0) { $0 + $1.deletions }, s.deletions)
    }

    func testEmptySummary() {
        XCTAssertEqual(ActivitySummary(files: []), ActivitySummary(files: []))
        XCTAssertEqual(ActivitySummary(files: []).files, 0)
    }

    func testChildrenAtPathDrillsThroughDirectories() {
        let roots = WorktreeModel.buildTree(from: sample)
        XCTAssertEqual(ActivityTreemap.children(at: "", in: roots)?.count, roots.count)
        let crates = ActivityTreemap.children(at: "crates", in: roots)
        XCTAssertEqual(crates?.map(\.name).sorted(), ["a", "b"])
        let a = ActivityTreemap.children(at: "crates/a", in: roots)
        XCTAssertEqual(a?.map(\.name).sorted(), ["lib.rs", "new.rs"])
        XCTAssertNil(ActivityTreemap.children(at: "crates/zzz", in: roots))
        // A file path is not a directory.
        XCTAssertNil(ActivityTreemap.children(at: "top.txt", in: roots))
    }

    func testSurvivingPathFallsBackToNearestAncestor() {
        var roots = WorktreeModel.buildTree(from: sample)
        XCTAssertEqual(ActivityTreemap.survivingPath("crates/a", in: roots), "crates/a")
        // crates/a disappears on refresh: stay in crates.
        roots = WorktreeModel.buildTree(from: sample.filter { !$0.relativePath.hasPrefix("crates/a/") })
        XCTAssertEqual(ActivityTreemap.survivingPath("crates/a", in: roots), "crates")
        // The whole subtree disappears: back to the top.
        roots = WorktreeModel.buildTree(from: sample.filter { !$0.relativePath.hasPrefix("crates/") })
        XCTAssertEqual(ActivityTreemap.survivingPath("crates/a", in: roots), "")
        XCTAssertEqual(ActivityTreemap.survivingPath("", in: []), "")
    }

    func testBreadcrumbSegments() {
        XCTAssertTrue(ActivityTreemap.breadcrumb(for: "").isEmpty)
        let crumbs = ActivityTreemap.breadcrumb(for: "crates/a/src")
        XCTAssertEqual(crumbs.map(\.name), ["crates", "a", "src"])
        XCTAssertEqual(crumbs.map(\.path), ["crates", "crates/a", "crates/a/src"])
    }

    func testPlanSizesByChurnAndSkipsZeroWeight() {
        let roots = WorktreeModel.buildTree(from: sample)
        let plan = ActivityTreemap.plan(nodes: roots, in: CGRect(x: 0, y: 0, width: 300, height: 200))
        let names = Set(plan.cells.map(\.node.name))
        // docs holds only a zero-line rename, so it takes no area.
        XCTAssertEqual(names, ["crates", "top.txt"])
        XCTAssertEqual(plan.hiddenCount, 0)
        let crates = plan.cells.first { $0.node.name == "crates" }!
        let top = plan.cells.first { $0.node.name == "top.txt" }!
        let ratio = (crates.rect.width * crates.rect.height) / (top.rect.width * top.rect.height)
        XCTAssertEqual(Double(ratio), 82.0 / 10.0, accuracy: 0.01)
    }

    func testSingleFileRendersOneRectangle() {
        let roots = WorktreeModel.buildTree(from: [change("only.rs", .modified, 3, 1)])
        let plan = ActivityTreemap.plan(nodes: roots, in: CGRect(x: 0, y: 0, width: 120, height: 80))
        XCTAssertEqual(plan.cells.count, 1)
        let r = plan.cells[0].rect
        XCTAssertEqual(r.width, 120, accuracy: 0.001)
        XCTAssertEqual(r.height, 80, accuracy: 0.001)
    }

    func testPlanCapsCellsAndReportsTheRest() {
        let many = (0..<250).map { change("dir/f\($0).rs", .modified, $0 + 1, 0) }
        let roots = WorktreeModel.buildTree(from: many)
        let files = ActivityTreemap.children(at: "dir", in: roots)!
        let plan = ActivityTreemap.plan(nodes: files, in: CGRect(x: 0, y: 0, width: 800, height: 600))
        XCTAssertEqual(plan.cells.count, ActivityTreemap.maxCells)
        XCTAssertEqual(plan.hiddenCount, 50)
        XCTAssertEqual(ActivityTreemap.hiddenCount(nodes: files), 50)
        // The heaviest survive, the lightest are the ones dropped.
        XCTAssertTrue(plan.cells.contains { $0.node.name == "f249.rs" })
        XCTAssertFalse(plan.cells.contains { $0.node.name == "f0.rs" })
    }

    func testDeepNestingResolvesAndPlans() {
        let path = (0..<40).map { "d\($0)" }.joined(separator: "/")
        let roots = WorktreeModel.buildTree(from: [change(path + "/leaf.rs", .added, 9, 1)])
        XCTAssertEqual(ActivityTreemap.children(at: path, in: roots)?.first?.name, "leaf.rs")
        XCTAssertEqual(ActivityTreemap.survivingPath(path, in: roots), path)
    }

    func testActivityTabIsRegistered() {
        XCTAssertTrue(WorktreeTabMode.allCases.contains(.activity))
        XCTAssertEqual(WorktreeTabMode.activity.title, "Activity")
    }
}
