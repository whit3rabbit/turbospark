import SwiftUI
import TurboSpark

/// Settings pane presenting the fixed keyboard shortcuts. The rows come from
/// `KeyboardShortcutCatalog`, which is what keeps them honest against the
/// commands in `TurboSparkApp.swift`.
@MainActor
struct KeyboardShortcutsSettingsPaneView: View {
    @Environment(\.appTheme) private var theme

    var body: some View {
        Form {
            Text("Shortcuts are fixed. There is no rebinding in this version.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)

            ForEach(KeyboardShortcutCatalog.sections) { section in
                Section {
                    ForEach(section.rows) { row in
                        shortcutRow(row)
                    }
                } header: {
                    // Section titles and row labels are catalog keys, the
                    // same keys the menu items use, so the pane and the menu
                    // bar read alike in every language.
                    Text(LocalizedStringKey(section.title), bundle: .module)
                }
            }
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
    }

    /// One row per command. Side by side when it fits; the keys drop under
    /// the label at a large text size or with a long translation instead of
    /// truncating the label. VoiceOver reads the row as one element,
    /// "label, keys", rather than three unrelated fragments.
    private func shortcutRow(_ row: KeyboardShortcutRow) -> some View {
        let label = Text(LocalizedStringKey(row.label), bundle: .module)
        return ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) {
                label.font(theme.ui(.base))
                Spacer(minLength: 12)
                keys(row)
            }
            VStack(alignment: .leading, spacing: 6) {
                label
                    .font(theme.ui(.base))
                    .fixedSize(horizontal: false, vertical: true)
                keys(row)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(spokenLabel(row, label: label))
    }

    private func keys(_ row: KeyboardShortcutRow) -> some View {
        HStack(spacing: 8) {
            if let alt = row.altKeys {
                Text("also \(alt)", bundle: .module)
                    .font(theme.code(.small))
                    .foregroundStyle(.appSecondary)
            }
            Text(verbatim: row.keys)
                .font(theme.code(.small))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.appSurface)
                .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                        .stroke(.appBorder, lineWidth: 1)
                )
        }
        .fixedSize()
    }

    private func spokenLabel(_ row: KeyboardShortcutRow, label: Text) -> Text {
        let keys = label + Text(verbatim: ", ") + Text(verbatim: row.keys)
        guard let alt = row.altKeys else { return keys }
        return keys + Text(verbatim: ", ") + Text("also \(alt)", bundle: .module)
    }
}
