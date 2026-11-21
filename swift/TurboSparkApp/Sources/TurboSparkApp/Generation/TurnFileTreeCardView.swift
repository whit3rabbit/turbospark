import SwiftUI

/// The collapsed card at the end of a turn listing the files it touched
/// (component audit, file-tree). The model is `TurnFileTree`; this view only
/// draws it. A file row reveals the tool card of the call behind it, the
/// same path the sources popover uses, so no new way to open files is added.
struct TurnFileTreeCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearance = AppearanceManager.shared
    @ObservedObject var model: AppModel
    let tree: TurnFileTree
    @State private var isExpanded = false

    private var indentStep: CGFloat { (14 * AppMetrics.textScale(theme)).rounded() }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
            if isExpanded {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(tree.rows.enumerated()), id: \.offset) { _, row in
                        rowView(row)
                    }
                }
            }
        }
        .appCardSurface(padding: 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
    }

    private var header: some View {
        Button {
            // TSMotion.animate skips the animation under Reduce Motion.
            TSMotion.animate(TSMotion.select) { isExpanded.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .themedFont(.micro, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
                Image(systemName: "folder")
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appAccentText)
                    .accessibilityHidden(true)
                Text("Files in this turn", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appText)
                Text("\(tree.changedCount) changed", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                if tree.readCount > 0 {
                    Text("\(tree.readCount) viewed", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(.isHeader)
        .accessibilityValue(isExpanded
            ? Text("Expanded", bundle: .module) : Text("Collapsed", bundle: .module))
    }

    @ViewBuilder
    private func rowView(_ row: TurnFileTree.Row) -> some View {
        switch row {
        case .directory(let name, let depth):
            HStack(spacing: 6) {
                Image(systemName: "folder")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
                Text(verbatim: name)
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .padding(.leading, CGFloat(depth) * indentStep)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(Text(verbatim: name))
            .accessibilityValue(Text("Folder", bundle: .module))
        case .file(let entry, let name, let depth):
            fileRow(entry, name: name)
                .padding(.leading, CGFloat(depth) * indentStep)
        }
    }

    private func fileRow(_ entry: TurnFileTree.Entry, name: String) -> some View {
        Button {
            model.revealToolCard(for: entry)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: symbol(entry.change))
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(tint(entry.change))
                    .accessibilityHidden(true)
                Text(verbatim: name)
                    .themedCode(.small)
                    .foregroundStyle(.appText)
                    .strikethrough(entry.change == .deleted)
                    .lineLimit(1)
                    .truncationMode(.middle)
                changeWord(entry.change)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                if let additions = entry.additions, let deletions = entry.deletions {
                    Text(verbatim: "+\(additions)")
                        .themedCode(.tiny, weight: .bold)
                        .foregroundStyle(appearance.diffMarkers == .color ? theme.success : theme.foreground)
                    Text(verbatim: "-\(deletions)")
                        .themedCode(.tiny, weight: .bold)
                        .foregroundStyle(appearance.diffMarkers == .color ? theme.danger : theme.secondaryText)
                }
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(Text("Show the tool call for \(entry.path)", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(rowLabel(entry, name: name))
        .accessibilityHint(Text("Show the tool call for \(entry.path)", bundle: .module))
        .accessibilityAddTraits(.isButton)
    }

    private func rowLabel(_ entry: TurnFileTree.Entry, name: String) -> Text {
        let word = localizedWord(entry.change)
        if let additions = entry.additions, let deletions = entry.deletions {
            return Text("\(name), \(word), \(additions) added, \(deletions) removed", bundle: .module)
        }
        return Text(verbatim: "\(name), \(word)")
    }

    /// The word beside the glyph, so the state never rests on color.
    private func changeWord(_ change: TurnFileTree.Change) -> Text {
        switch change {
        case .added: return Text("Added", bundle: .module)
        case .modified: return Text("Modified", bundle: .module)
        case .written: return Text("Written", bundle: .module)
        case .deleted: return Text("Deleted", bundle: .module)
        case .read: return Text("Viewed", bundle: .module)
        }
    }

    /// `changeWord` as a String, for the composed VoiceOver label.
    private func localizedWord(_ change: TurnFileTree.Change) -> String {
        switch change {
        case .added: return AppStrings.localized("Added")
        case .modified: return AppStrings.localized("Modified")
        case .written: return AppStrings.localized("Written")
        case .deleted: return AppStrings.localized("Deleted")
        case .read: return AppStrings.localized("Viewed")
        }
    }

    private func symbol(_ change: TurnFileTree.Change) -> String {
        switch change {
        case .added: return "plus.circle"
        case .modified: return "pencil.circle"
        case .written: return "square.and.pencil"
        case .deleted: return "minus.circle"
        case .read: return "eye"
        }
    }

    private func tint(_ change: TurnFileTree.Change) -> AnyShapeStyle {
        switch change {
        case .added: return AnyShapeStyle(.appSuccess)
        case .deleted: return AnyShapeStyle(.appDanger)
        case .modified: return AnyShapeStyle(.appAccentText)
        case .written, .read: return AnyShapeStyle(.appSecondary)
        }
    }
}
