import SwiftUI
import TurboSpark

/// Settings pane presenting the fixed keyboard shortcuts. The rows come from
/// `KeyboardShortcutCatalog`, which is what keeps them honest against the
/// commands in `TurboSparkApp.swift`.
struct KeyboardShortcutsSettingsPaneView: View {
    @Environment(\.appTheme) private var theme

    var body: some View {
        Form {
            ForEach(KeyboardShortcutCatalog.sections) { section in
                Section(section.title) {
                    ForEach(section.rows) { row in
                        shortcutRow(label: row.label, shortcut: row.keys, alt: row.altKeys)
                    }
                }
            }

            Section {
                Text("Shortcuts are fixed. There is no rebinding in this version.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private func shortcutRow(label: String, shortcut: String, alt: String?) -> some View {
        HStack {
            Text(label)
                .font(theme.ui(.base))
            Spacer()
            if let alt {
                Text("also \(alt)", bundle: .module)
                    .font(theme.code(.small))
                    .foregroundStyle(.appSecondary)
            }
            Text(shortcut)
                .font(theme.code(.small))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.appSurface)
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(.appBorder.opacity(0.3), lineWidth: 1)
                )
        }
    }
}
