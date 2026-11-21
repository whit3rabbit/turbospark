import SwiftUI

/// The inline editor a pending shell card swaps in for its command cell
/// (`ToolCallCommandEdit`). It only hands the text back: the model decides
/// what an edit means (an unchanged or empty edit is a cancel), and nothing
/// here runs or approves a command.
///
/// No default-action shortcut: Return is a newline in a command and belongs
/// to the composer elsewhere. Esc cancels and restores the model's text.
struct PendingCommandEditorView: View {
    @Environment(\.appTheme) private var theme
    @Binding var text: String
    let onSubmit: () -> Void
    let onCancel: () -> Void

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextEditor(text: $text)
                .font(theme.code(.small))
                .frame(maxWidth: .infinity, maxHeight: 160, alignment: .leading)
                .appEditorSurface(minHeight: 56)
                .appFocusGlow(isFocused)
                .focused($isFocused)
                .accessibilityLabel(Text("Edited command", bundle: .module))
                .onExitCommand { onCancel() }

            Text(
                "The edited command is checked again and needs its own approval before it runs.",
                bundle: .module
            )
            .themedFont(.tiny)
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                Button(action: onSubmit) {
                    Label { Text("Use edited command", bundle: .module) } icon: { Image(systemName: "checkmark") }
                }
                .appAction(.primary, size: .small)
                .help(Text("Replace the proposed command with this one. Nothing runs until you approve it.", bundle: .module))

                Button(role: .cancel, action: onCancel) {
                    Text("Cancel", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .help(Text("Keep the proposed command (Esc)", bundle: .module))
            }
        }
        .task { isFocused = true }
    }
}
