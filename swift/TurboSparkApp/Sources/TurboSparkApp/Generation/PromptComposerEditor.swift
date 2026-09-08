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
    /// The shared `/` and `@` autocomplete state. While its popup is
    /// visible, the arrow keys, Tab, Return and Escape are the popup's
    /// before they are the editor's.
    var autocomplete: ComposerAutocompleteController? = nil
    @Environment(\.appTheme) private var theme

    @ScaledMetric private var editorMinHeight: CGFloat = 34
    @ScaledMetric private var editorMaxHeight: CGFloat = 200
    @ScaledMetric private var expandedMaxHeight: CGFloat = 420
    @State private var measuredTextHeight: CGFloat = 0
    @State private var isExpanded: Bool = false
    /// Up/Down recall through `model.promptHistory` (qwen-code input
    /// history). Nil while the editor is not browsing. The text each step
    /// writes is remembered so an unrelated edit (typing, pasting) ends the
    /// walk instead of leaving it armed under a draft that moved.
    @State private var historyBrowser: ComposerHistoryBrowser? = nil
    @State private var lastHistoryAppliedText: String? = nil

    private var currentMaxHeight: CGFloat {
        isExpanded ? expandedMaxHeight : editorMaxHeight
    }

    var body: some View {
        Color.clear
            .frame(height: min(max(measuredTextHeight, editorMinHeight), currentMaxHeight))
            .animation(.smooth(duration: 0.15), value: measuredTextHeight)
            .animation(.smooth(duration: 0.2), value: isExpanded)
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
            .onChange(of: model.promptText) { _, newValue in
                // A sent (or cleared) prompt starts the next message fresh --
                // an expand toggled for one long message must not carry its
                // taller ceiling and "collapse" affordance into an unrelated
                // new one.
                if newValue.isEmpty {
                    isExpanded = false
                }
                // A text change that was not a history step ends the walk.
                if historyBrowser != nil, newValue != lastHistoryAppliedText {
                    historyBrowser?.noteExternalEdit()
                    if historyBrowser?.isActive != true { historyBrowser = nil }
                }
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
                            Text("Ask anything", bundle: .module)
                                .font(theme.uiFont)
                                .foregroundStyle(.tertiary)
                                .padding(.leading, 5)
                                .padding(.vertical, 8)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                    }
                    .onKeyPress(phases: .down) { press in
                        if let autocomplete, autocomplete.isVisible {
                            switch press.key {
                            case .upArrow:
                                autocomplete.moveSelection(-1)
                                return .handled
                            case .downArrow:
                                autocomplete.moveSelection(1)
                                return .handled
                            case .tab:
                                acceptAutocomplete()
                                return .handled
                            case .escape:
                                // The popup consumes the first Escape;
                                // canceling a running turn stays on the
                                // second.
                                autocomplete.dismiss()
                                return .handled
                            case .return where !press.modifiers.contains(.shift):
                                acceptAutocomplete()
                                return .handled
                            default:
                                break
                            }
                        }
                        if press.key == .escape {
                            if model.isRunning && model.canCancel
                                && model.handleEscCancelIntent() {
                                // Two-stage cancel (qwen-code parity): the
                                // first press ARMS ("Press Esc again to
                                // stop" in the footer), the second stops.
                                return .handled
                            }
                            promptFocused.wrappedValue = false
                            return .handled
                        }
                        // History recall (qwen-code parity): Up from an
                        // empty draft starts a walk back through submitted
                        // prompts; Down reverses it and past the newest
                        // restores the parked draft. Only from an EMPTY
                        // draft or an active walk, so it never steals the
                        // caret move inside a half-written multi-line
                        // message; the autocomplete popup outranks it.
                        if press.key == .upArrow, press.modifiers.isEmpty,
                            historyBrowser?.canStepBack(draftText: model.promptText) == true {
                            applyHistoryStep(backward: true)
                            return .handled
                        }
                        if press.key == .downArrow, press.modifiers.isEmpty,
                            historyBrowser?.isActive == true {
                            applyHistoryStep(backward: false)
                            return .handled
                        }
                        if press.key == .return {
                            if press.modifiers.contains(.shift) {
                                return .ignored
                            }
                            // CanRunOrQueue, not canRun: a Return during a
                            // running turn QUEUES the draft (Claude Code's
                            // message queue) instead of dying here.
                            if model.canRunOrQueue {
                                model.run()
                                return .handled
                            }
                            return .handled
                        }
                        return .ignored
                    }
            }
            .overlay(alignment: .topTrailing) {
                if measuredTextHeight > editorMinHeight + 6 || isExpanded {
                    Button {
                        withAnimation(.smooth(duration: 0.2)) {
                            isExpanded.toggle()
                        }
                    } label: {
                        Image(systemName: isExpanded ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
                            .themedFont(points: 10, weight: .medium)
                            .foregroundStyle(.tertiary)
                            .padding(4)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 2)
                    .padding(.trailing, 2)
                    .help(isExpanded ? "Collapse editor" : "Expand editor")
                    .accessibilityLabel(isExpanded ? "Collapse editor" : "Expand editor")
                    .transition(.opacity)
                }
            }
    }

    private func acceptAutocomplete() {
        guard let autocomplete,
            let newText = autocomplete.accept(in: model.promptText)
        else { return }
        model.promptText = newText
    }

    /// One step of the history walk, shared by both arrows. The applied
    /// text is recorded so the `onChange` above can tell a history step
    /// from a keystroke.
    private func applyHistoryStep(backward: Bool) {
        if historyBrowser == nil { historyBrowser = ComposerHistoryBrowser() }
        var browser = historyBrowser!
        let applied: String?
        if backward {
            applied = browser.stepBack(
                store: model.promptHistory, currentDraft: model.promptText)
        } else {
            applied = browser.stepForward(store: model.promptHistory)
        }
        guard let applied else { return }
        lastHistoryAppliedText = applied
        model.promptText = applied
        if browser.isActive {
            historyBrowser = browser
        } else {
            historyBrowser = nil
            lastHistoryAppliedText = nil
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
