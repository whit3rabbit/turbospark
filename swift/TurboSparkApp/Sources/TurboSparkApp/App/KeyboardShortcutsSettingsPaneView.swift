import SwiftUI
import TurboSpark

/// Settings pane presenting reference keyboard shortcuts for navigation, zoom, and chat.
struct KeyboardShortcutsSettingsPaneView: View {
    @Environment(\.appTheme) private var theme

    var body: some View {
        Form {
            Section("Navigation & View") {
                shortcutRow(label: "Chat", shortcut: "\u{2318} 1")
                shortcutRow(label: "Files", shortcut: "\u{2318} 2")
                shortcutRow(label: "Installed Models", shortcut: "\u{2318} 3")
                shortcutRow(label: "Discover Models", shortcut: "\u{2318} 4")
                shortcutRow(label: "Toggle Chat Sidebar", shortcut: "\u{2303} \u{2318} S")
                shortcutRow(label: "Toggle Inspector", shortcut: "\u{21E7} \u{2318} I")
            }

            Section("Text Size & Zoom") {
                shortcutRow(label: "Make Text Bigger", shortcut: "\u{2318} +")
                shortcutRow(label: "Make Text Smaller", shortcut: "\u{2318} -")
                shortcutRow(label: "Default Text Size", shortcut: "\u{2318} 0")
            }

            Section("Chat & Generation") {
                shortcutRow(label: "New Chat", shortcut: "\u{2318} N")
                shortcutRow(label: "Previous Chat", shortcut: "\u{2318} [")
                shortcutRow(label: "Next Chat", shortcut: "\u{2318} ]")
                shortcutRow(label: "Focus Prompt", shortcut: "\u{2318} L")
                shortcutRow(label: "Clear History", shortcut: "\u{21E7} \u{2318} K")
                shortcutRow(label: "Generate Response", shortcut: "\u{2318} \u{21A9}")
                shortcutRow(label: "Cancel Generation", shortcut: "\u{2318} .")
            }
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private func shortcutRow(label: String, shortcut: String) -> some View {
        HStack {
            Text(label)
                .font(theme.ui(.base))
            Spacer()
            Text(shortcut)
                .font(theme.code(.small))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1)
                )
        }
    }
}
