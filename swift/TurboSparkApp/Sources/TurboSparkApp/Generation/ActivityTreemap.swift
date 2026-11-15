import CoreGraphics
import Foundation

/// Totals for the treemap's summary row. Computed from the same file list the
/// tree is built from, so the row can never disagree with the rectangles.
struct ActivitySummary: Equatable {
    var additions = 0
    var deletions = 0
    var files = 0
    /// `.added` and `.untracked`: both are files that are new to the index.
    var added = 0
    var modified = 0
    var deleted = 0
    var renamed = 0
    /// Files with zero added plus removed lines (binary files, pure renames,
    /// empty files). They count above but take no area in the treemap.
    var withoutLineChanges = 0

    init(files changes: [WorktreeFileChange]) {
        for change in changes {
            files += 1
            additions += change.additions
            deletions += change.deletions
            if change.additions + change.deletions == 0 { withoutLineChanges += 1 }
            switch change.status {
            case .added, .untracked: added += 1
            case .modified: modified += 1
            case .deleted: deleted += 1
            case .renamed: renamed += 1
            case .ignored, .unknown: break
            }
        }
    }
}

/// Pure helpers behind the activity treemap: weights, drill-down paths and the
/// per-level layout plan. The directory tree itself comes from
/// `WorktreeModel.buildTree(from:)`, the same one the Changes tab shows.
///
/// v1 scope: the data is the worktree's uncommitted (or compared) change list.
/// It has no per-chat attribution, viewed-only files or pending state, so the
/// treemap does not claim any.
enum ActivityTreemap {
    /// Rectangles drawn per level. A level with more entries keeps the
    /// heaviest and reports the rest, so a vendored directory with thousands
    /// of files cannot build thousands of controls.
    static let maxCells = 200

    static func weight(_ node: WorktreeTreeNode) -> Int {
        max(0, node.additions) + max(0, node.deletions)
    }

    /// The children shown when drilled into the directory at `path` ("" is
    /// the top level), or nil when no such directory exists in `roots`.
    static func children(at path: String, in roots: [WorktreeTreeNode]) -> [WorktreeTreeNode]? {
        if path.isEmpty { return roots }
        var level = roots
        for part in path.split(separator: "/").map(String.init) {
            guard let next = level.first(where: { $0.isDirectory && $0.name == part }) else {
                return nil
            }
            level = next.children
        }
        return level
    }

    /// `path` itself when it still exists, else its nearest surviving
    /// ancestor ("" at worst). A refresh can remove the drilled directory.
    static func survivingPath(_ path: String, in roots: [WorktreeTreeNode]) -> String {
        var parts = path.split(separator: "/").map(String.init)
        while !parts.isEmpty {
            let candidate = parts.joined(separator: "/")
            if children(at: candidate, in: roots) != nil { return candidate }
            parts.removeLast()
        }
        return ""
    }

    /// Breadcrumb segments below the root, each with the path that selects it.
    static func breadcrumb(for path: String) -> [(name: String, path: String)] {
        var result: [(name: String, path: String)] = []
        var acc = ""
        for part in path.split(separator: "/").map(String.init) {
            acc = acc.isEmpty ? part : acc + "/" + part
            result.append((part, acc))
        }
        return result
    }

    /// One level laid out: the rectangles with their nodes, plus what was
    /// left out of them.
    struct Plan {
        var cells: [(node: WorktreeTreeNode, rect: CGRect)]
        /// Positive-weight entries beyond `maxCells`.
        var hiddenCount: Int
    }

    /// Positive-weight entries a level cannot draw because of `limit`.
    static func hiddenCount(nodes: [WorktreeTreeNode], limit: Int = ActivityTreemap.maxCells) -> Int {
        max(0, nodes.filter { weight($0) > 0 }.count - max(0, limit))
    }

    static func plan(
        nodes: [WorktreeTreeNode], in bounds: CGRect, limit: Int = ActivityTreemap.maxCells
    ) -> Plan {
        let weighted = nodes
            .filter { weight($0) > 0 }
            .sorted { lhs, rhs in
                let l = weight(lhs), r = weight(rhs)
                return l != r ? l > r : lhs.id < rhs.id
            }
        let kept = Array(weighted.prefix(max(0, limit)))
        let byID = Dictionary(uniqueKeysWithValues: kept.map { ($0.id, $0) })
        let rects = TreemapLayout.layout(
            items: kept.map { TreemapItem(id: $0.id, weight: Double(weight($0))) },
            in: bounds)
        return Plan(
            cells: rects.compactMap { placed in byID[placed.id].map { ($0, placed.rect) } },
            hiddenCount: weighted.count - kept.count)
    }
}
