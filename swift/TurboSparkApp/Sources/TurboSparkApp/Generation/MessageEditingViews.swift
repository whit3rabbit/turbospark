import SwiftUI

/// The three affordances behind message edit / regenerate / branch: the
/// variant switcher that rides in a message's hover bar, the in-place edit
/// composer that replaces a prompt row, and the sheet that forks a branch.
/// All of them are thin: the state machine lives in
/// `AppModel+MessageEditing.swift`, and each view calls one model method.
extension AppModel {
    /// Identifiable wrapper so a row can present the branch editor in a
    /// `.sheet(item:)` keyed by the message being branched from.
    struct BranchTarget: Identifiable {
        let id: UUID
        let originalText: String
    }
}

/// `< n / m >` navigation between a message's versions. Rendered only when
/// `count > 1`; the left chevron is older, the right newer.
struct MessageVariantSwitcherView: View {
    let position: Int
    let count: Int
    let step: (Int) -> Void

    var body: some View {
        HStack(spacing: 2) {
            Button {
                step(-1)
            } label: {
                Image(systemName: "chevron.left")
                    .themedFont(.tiny, weight: .semibold)
                    .frame(width: 18, height: 18)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(position <= 1)
            .help("Show earlier version")

            Text("\(position) / \(count)")
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .monospacedDigit()
                .padding(.horizontal, 2)

            Button {
                step(1)
            } label: {
                Image(systemName: "chevron.right")
                    .themedFont(.tiny, weight: .semibold)
                    .frame(width: 18, height: 18)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(position >= count)
            .help("Show later version")
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Message version \(position) of \(count)")
        .accessibilityHint("Steps between the earlier and later versions of this message")
    }
}

/// The in-place editor that replaces a prompt row while it is open. Save
/// re-runs the turn from the edited prompt; Escape or Cancel restores the
/// row untouched.
struct MessageEditComposerView: View {
    let initialText: String
    var saveLabel: String = "Save & rerun"
    let onSave: (String) -> Void
    let onCancel: () -> Void

    @Environment(\.appTheme) private var theme
    @State private var text: String = ""
    @FocusState private var isFocused: Bool

    private var trimmedText: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        VStack(alignment: .trailing, spacing: 8) {
            TextEditor(text: $text)
                .font(theme.uiFont)
                .scrollContentBackground(.hidden)
                .frame(minWidth: 320, maxWidth: 640, minHeight: 56, maxHeight: 240, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(.appSurface.opacity(0.85))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(.appAccent.opacity(0.45), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .focused($isFocused)

            HStack(spacing: 8) {
                Button { onCancel() } label: { Text("Cancel", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
                Button(saveLabel) { onSave(text) }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(trimmedText.isEmpty)
            }
        }
        .onAppear {
            text = initialText
        }
        .task {
            isFocused = true
        }
        .onExitCommand { onCancel() }
    }
}

/// The branch editor for an EARLIER prompt: editing one of those forks the
/// conversation into a new chat (the prefix up to this prompt, with the edit
/// applied) rather than rewriting history the later turns were built on.
struct BranchEditSheet: View {
    let originalText: String
    let chatTitle: String
    let onCreate: (String) -> Void
    let onCancel: () -> Void

    @Environment(\.appTheme) private var theme
    @State private var text: String = ""

    private var trimmedText: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Edit & branch", bundle: .module)
                    .themedFont(.large, weight: .semibold)
                Text(
                    "Creates a new chat containing the conversation up to this message, with your edit in place, and continues there. \"\(chatTitle)\" is left unchanged.",
                    bundle: .module
                )
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }

            TextEditor(text: $text)
                .font(theme.uiFont)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 120, maxHeight: 280, alignment: .leading)
                .padding(8)
                .background(.appSurface.opacity(0.85))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Color.primary.opacity(0.12), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

            HStack {
                Spacer()
                Button { onCancel() } label: { Text("Cancel", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
                Button { onCreate(text) } label: { Text("Create branch & rerun", bundle: .module) }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(trimmedText.isEmpty)
            }
        }
        .padding(20)
        .frame(width: 520)
        .onAppear {
            text = originalText
        }
        .onExitCommand { onCancel() }
    }
}
