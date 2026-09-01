import SwiftUI

/// The prompt text field itself, kept separate from the composer's footer
/// controls so the two can be reasoned about (and restyled) independently.
///
/// The editor grows with its text between a one-line floor and a scroll
/// ceiling.
///
/// `TextEditor` is greedy vertically and has no intrinsic height, so a
/// plain `maxHeight` frame makes an empty composer as tall as a full one.
/// So the height is MEASURED: a hidden `Text` with the same font and
/// insets, sized with `fixedSize(vertical:)` at the editor's own width,
/// reports its ideal height through a preference and the editor gets that
/// height clamped into range.
struct PromptComposerEditor: View {
    @ObservedObject var model: AppModel
    var promptFocused: FocusState<Bool>.Binding
    @Environment(\.appTheme) private var theme

    @ScaledMetric private var editorMinHeight: CGFloat = 34
    @ScaledMetric private var editorMaxHeight: CGFloat = 200
    @State private var measuredTextHeight: CGFloat = 0

    var body: some View {
        Color.clear
            .frame(height: min(max(measuredTextHeight, editorMinHeight), editorMaxHeight))
            .background(alignment: .topLeading) {
                Text(model.promptText.isEmpty ? " " : model.promptText)
                    .font(theme.uiFont)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .background {
                        GeometryReader { geometry in
                            Color.clear.preference(
                                key: PromptTextHeightKey.self,
                                value: geometry.size.height)
                        }
                    }
                    .hidden()
                    .accessibilityHidden(true)
            }
            .onPreferenceChange(PromptTextHeightKey.self) { height in
                measuredTextHeight = height
            }
            .overlay(alignment: .topLeading) {
                TextEditor(text: $model.promptText)
                    .accessibilityLabel("Prompt")
                    .accessibilityHint("Type your message here. Press Return to send, Shift-Return for a new line.")
                    .font(theme.uiFont)
                    .scrollContentBackground(.hidden)
                    .focused(promptFocused)
                    .overlay(alignment: .topLeading) {
                        if model.promptText.isEmpty {
                            Text("Ask a question, request code, or explore ideas...")
                                .font(theme.uiFont)
                                .foregroundStyle(.tertiary)
                                .padding(.leading, 5)
                                .padding(.vertical, 8)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                    }
                    .onKeyPress(phases: .down) { press in
                        if press.key == .escape {
                            if model.isRunning && model.canCancel {
                                model.cancel()
                                return .handled
                            }
                            promptFocused.wrappedValue = false
                            return .handled
                        }
                        if press.key == .return {
                            if press.modifiers.contains(.shift) {
                                return .ignored
                            }
                            if model.canRun {
                                model.run()
                                return .handled
                            }
                            return .handled
                        }
                        return .ignored
                    }
            }
    }
}

/// Ideal height of the composer's text, measured off a hidden sizer.
private struct PromptTextHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
