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
                    .frame(minWidth: 18, minHeight: 18)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(position <= 1)
            .help(Text("Show earlier version", bundle: .module))

            // A numeric readout: nothing in it to translate.
            Text(verbatim: "\(position) / \(count)")
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .monospacedDigit()
                .padding(.horizontal, 2)

            Button {
                step(1)
            } label: {
                Image(systemName: "chevron.right")
                    .themedFont(.tiny, weight: .semibold)
                    .frame(minWidth: 18, minHeight: 18)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(position >= count)
            .help(Text("Show later version", bundle: .module))
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Message version \(position) of \(count)", bundle: .module))
        .accessibilityHint(Text("Steps between the earlier and later versions of this message", bundle: .module))
        // The children are ignored above, so the two chevrons are only
        // reachable through this: VoiceOver's increment and decrement step
        // the version the way the hint promises.
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment:
                if position < count { step(1) }
            case .decrement:
                if position > 1 { step(-1) }
            @unknown default:
                break
            }
        }
    }
}

/// The in-place editor that replaces a prompt row while it is open. Save
/// re-runs the turn from the edited prompt; Escape or Cancel restores the
/// row untouched.
struct MessageEditComposerView: View {
    let initialText: String
    var saveLabel: LocalizedStringKey = "Save & rerun"
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
                .frame(minWidth: 320, maxWidth: 640, maxHeight: 240, alignment: .leading)
                .appEditorSurface(minHeight: 56)
                .focused($isFocused)
                .accessibilityLabel(Text("Edit message text", bundle: .module))

            HStack(spacing: 8) {
                Button { onCancel() } label: { Text("Cancel", bundle: .module) }
                    .appAction(.secondary)
                    .keyboardShortcut(.cancelAction)
                Button { onSave(text) } label: { Text(saveLabel, bundle: .module) }
                    .appAction(.primary)
                    .keyboardShortcut(.defaultAction)
                    .disabled(trimmedText.isEmpty)
                    .help(trimmedText.isEmpty
                        ? Text("Enter text to save", bundle: .module)
                        : Text("Save and rerun prompt", bundle: .module))
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
                .frame(maxHeight: 280, alignment: .leading)
                .appEditorSurface(minHeight: 120)
                .accessibilityLabel(Text("Branch message text", bundle: .module))

            HStack {
                Spacer()
                Button { onCancel() } label: { Text("Cancel", bundle: .module) }
                    .appAction(.secondary)
                    .keyboardShortcut(.cancelAction)
                Button { onCreate(text) } label: { Text("Create branch & rerun", bundle: .module) }
                    .appAction(.primary)
                    .keyboardShortcut(.defaultAction)
                    .disabled(trimmedText.isEmpty)
                    .help(trimmedText.isEmpty
                        ? Text("Enter text to create branch", bundle: .module)
                        : Text("Create branch and rerun prompt", bundle: .module))
            }
        }
        .padding(20)
        .frame(minWidth: 460, idealWidth: 520, maxWidth: 640)
        .onAppear {
            text = originalText
        }
        .onExitCommand { onCancel() }
    }
}
