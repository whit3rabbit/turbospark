import SwiftUI

/// OC-08: the Talk control in the call panel (push-to-talk for auto-send).
///
/// Pointer: hold to talk, or tap once to keep talking and tap again to stop
/// (`VoiceCallTalkSwitch` decides by press length). A pointer hold cannot be
/// performed with VoiceOver, so the element is exposed as a toggle: activating
/// it flips Talk. The Toggle Talk chord does the same from the keyboard.
/// State is a glyph plus a word, never color alone.
@MainActor
struct VoiceCallTalkButton: View {
    @ObservedObject var talk: VoiceCallTalkSwitch
    @Environment(\.appTheme) private var theme
    @State private var pressing = false

    var body: some View {
        Label {
            talk.isTalking ? Text("Talking", bundle: .module) : Text("Talk", bundle: .module)
        } icon: {
            Image(systemName: talk.isTalking ? "mic.fill" : "mic")
        }
        .themedFont(.small, weight: .medium)
        .foregroundStyle(talk.isTalking ? .appAccentText : .appText)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Capsule().fill(.appElevated))
        .overlay(Capsule().strokeBorder(talk.isTalking ? AnyShapeStyle(.appAccentText) : AnyShapeStyle(.appBorder), lineWidth: 1))
        .contentShape(Capsule())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !pressing else { return }
                    pressing = true
                    talk.pointerDown()
                }
                .onEnded { _ in
                    pressing = false
                    talk.pointerUp()
                })
        .help(Text("Hold to talk, or press once to keep talking and again to stop.", bundle: .module))
        .accessibilityRepresentation {
            Toggle(isOn: Binding(get: { talk.isTalking }, set: { _ in talk.toggle() })) {
                Text("Talk", bundle: .module)
            }
        }
    }
}
