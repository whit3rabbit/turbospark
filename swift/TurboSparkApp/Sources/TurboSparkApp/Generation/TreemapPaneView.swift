import SwiftUI

/// Activity treemap tab of the Git inspector: the changed files as rectangles
/// sized by added plus removed lines, rolled up through directories.
///
/// It lives beside the other Worktree views because it reads `WorktreeModel`
/// and is routed from `WorktreeView`'s tab bar. Every rectangle is a native
/// `Button`, so keyboard focus and VoiceOver come with it, instead of one
/// drawn canvas with no per-cell accessibility.
@MainActor
struct TreemapPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var worktree: WorktreeModel

    /// Built once per change list, not per body: a resize or a drill re-runs
    /// the body, and the tree is the expensive part for a large change set.
    @State private var tree: [WorktreeTreeNode]
    /// Path of the directory drilled into ("" is the top level). Keyed by
    /// path, not node identity, so it survives a refresh that rebuilds the tree.
    @State private var drillPath: String = ""

    init(worktree: WorktreeModel) {
        self.worktree = worktree
        _tree = State(initialValue: WorktreeModel.buildTree(from: worktree.scopedFiles))
    }

    private var summary: ActivitySummary { ActivitySummary(files: worktree.scopedFiles) }

    var body: some View {
        Group {
            if worktree.scopedFiles.isEmpty {
                AppEmptyState(
                    systemImage: "square.grid.2x2",
                    title: Text("No File Activity", bundle: .module),
                    message: Text("Changed files appear here as a treemap sized by added and removed lines.", bundle: .module)
                )
            } else {
                content
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: worktree.scopedFiles) {
            tree = WorktreeModel.buildTree(from: worktree.scopedFiles)
            // A refresh can remove the drilled directory: fall back to the
            // nearest ancestor that still exists.
            drillPath = ActivityTreemap.survivingPath(drillPath, in: tree)
        }
    }

    private var content: some View {
        let stats = summary
        let level = ActivityTreemap.children(at: drillPath, in: tree) ?? tree
        return VStack(alignment: .leading, spacing: 8) {
            summaryRow(stats)
            breadcrumb
            GeometryReader { proxy in
                treemap(level: level, size: proxy.size)
            }
            .accessibilityElement(children: .contain)
            .accessibilityLabel(Text("Treemap of changed files", bundle: .module))
            footnotes(stats: stats, level: level)
        }
        .padding(10)
    }

    // MARK: Summary

    private func summaryRow(_ stats: ActivitySummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(verbatim: "+\(stats.additions)")
                    .themedFont(.small, weight: .semibold).monospacedDigit()
                    .foregroundStyle(.appSuccess)
                Text(verbatim: "-\(stats.deletions)")
                    .themedFont(.small, weight: .semibold).monospacedDigit()
                    .foregroundStyle(.appDanger)
            }
            .accessibilityElement(children: .combine)
            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 84 * AppMetrics.textScale(theme)), alignment: .leading)],
                alignment: .leading, spacing: 4
            ) {
                countChip(Text("Files", bundle: .module), stats.files, systemImage: "doc.on.doc")
                if stats.added > 0 {
                    countChip(Text("Added", bundle: .module), stats.added, systemImage: "plus.circle")
                }
                if stats.modified > 0 {
                    countChip(Text("Modified", bundle: .module), stats.modified, systemImage: "pencil.circle")
                }
                if stats.deleted > 0 {
                    countChip(Text("Deleted", bundle: .module), stats.deleted, systemImage: "minus.circle")
                }
                if stats.renamed > 0 {
                    countChip(Text("Renamed", bundle: .module), stats.renamed, systemImage: "arrow.triangle.swap")
                }
            }
        }
    }

    private func countChip(_ label: Text, _ count: Int, systemImage: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: systemImage)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            label.themedFont(.tiny).foregroundStyle(.appSecondary)
            Text(verbatim: "\(count)")
                .themedFont(.tiny, weight: .semibold).monospacedDigit()
                .foregroundStyle(.appText)
        }
        .accessibilityElement(children: .combine)
    }

    // MARK: Breadcrumb

    private var breadcrumb: some View {
        let crumbs = ActivityTreemap.breadcrumb(for: drillPath)
        return ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                crumbButton(Text("All files", bundle: .module), path: "", isCurrent: crumbs.isEmpty)
                ForEach(Array(crumbs.enumerated()), id: \.offset) { index, crumb in
                    Image(systemName: "chevron.right")
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .accessibilityHidden(true)
                    crumbButton(Text(verbatim: crumb.name), path: crumb.path, isCurrent: index == crumbs.count - 1)
                }
            }
        }
    }

    private func crumbButton(_ label: Text, path: String, isCurrent: Bool) -> some View {
        Button {
            TSMotion.animate(TSMotion.select) { drillPath = path }
        } label: {
            label
                .themedFont(.small, weight: isCurrent ? .semibold : .regular)
                .foregroundStyle(isCurrent ? AnyShapeStyle(.appText) : AnyShapeStyle(.appAccentText))
                .lineLimit(1)
                .padding(.horizontal, 4)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isCurrent)
        .accessibilityAddTraits(isCurrent ? [.isSelected] : [])
    }

    // MARK: Treemap

    @ViewBuilder
    private func treemap(level: [WorktreeTreeNode], size: CGSize) -> some View {
        let plan = ActivityTreemap.plan(nodes: level, in: CGRect(origin: .zero, size: size))
        ZStack(alignment: .topLeading) {
            ForEach(plan.cells, id: \.node.id) { cell in
                cellView(cell.node, rect: cell.rect)
            }
        }
        .frame(width: size.width, height: size.height, alignment: .topLeading)
    }

    private func cellView(_ node: WorktreeTreeNode, rect: CGRect) -> some View {
        let inset: CGFloat = rect.width > 6 && rect.height > 6 ? 1 : 0
        let frame = rect.insetBy(dx: inset, dy: inset)
        let scale = AppMetrics.textScale(theme)
        let showsLabel = frame.width >= 56 * scale && frame.height >= 34 * scale
        let tint = node.fileChange.map { $0.status.color(in: theme) }
        let shape = RoundedRectangle(cornerRadius: 3, style: .continuous)

        return Button {
            activate(node)
        } label: {
            ZStack(alignment: .topLeading) {
                shape.fill(tint.map { AnyShapeStyle($0.opacity(0.18)) } ?? AnyShapeStyle(.appElevated))
                shape.strokeBorder(tint.map { AnyShapeStyle($0) } ?? AnyShapeStyle(.appBorder), lineWidth: 1)
                if showsLabel {
                    VStack(alignment: .leading, spacing: 1) {
                        HStack(spacing: 3) {
                            // Glyph carries the status so color is never the
                            // only signal.
                            Image(systemName: node.fileChange?.status.systemImage ?? "folder")
                                .themedFont(.tiny)
                                .accessibilityHidden(true)
                            Text(verbatim: node.name).lineLimit(1)
                        }
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(.appText)
                        Text(verbatim: "+\(node.additions) -\(node.deletions)")
                            .themedFont(.micro).monospacedDigit()
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                    }
                    .padding(4)
                }
            }
            .frame(width: frame.width, height: frame.height, alignment: .topLeading)
            .clipShape(shape)
            .contentShape(shape)
        }
        .buttonStyle(.plain)
        .offset(x: frame.minX, y: frame.minY)
        .help(Text(verbatim: node.relativePath))
        .accessibilityLabel(accessibilityLabel(for: node))
        .accessibilityHint(node.isDirectory
            ? Text("Opens this folder in the treemap", bundle: .module)
            : Text("Shows this file's diff in the Changes tab", bundle: .module))
    }

    private func accessibilityLabel(for node: WorktreeTreeNode) -> Text {
        if let file = node.fileChange {
            let status = AppStrings.localized(String.LocalizationValue(stringLiteral: file.status.label))
            return Text("\(node.name), \(status), \(node.additions) added, \(node.deletions) removed", bundle: .module)
        }
        return Text("\(node.name), folder, \(node.additions) added, \(node.deletions) removed", bundle: .module)
    }

    private func activate(_ node: WorktreeTreeNode) {
        if node.isDirectory {
            TSMotion.animate(TSMotion.select) { drillPath = node.relativePath }
        } else {
            worktree.loadDiff(for: node.relativePath)
            TSMotion.animate(TSMotion.select) { worktree.activeTab = .changes }
        }
    }

    // MARK: Footnotes

    @ViewBuilder
    private func footnotes(stats: ActivitySummary, level: [WorktreeTreeNode]) -> some View {
        let hidden = ActivityTreemap.hiddenCount(nodes: level)
        if stats.withoutLineChanges > 0 {
            Text("Files without line changes: \(stats.withoutLineChanges)", bundle: .module)
                .themedFont(.tiny).foregroundStyle(.appSecondary)
        }
        if hidden > 0 {
            Text("Smaller items not shown: \(hidden)", bundle: .module)
                .themedFont(.tiny).foregroundStyle(.appSecondary)
        }
    }
}
