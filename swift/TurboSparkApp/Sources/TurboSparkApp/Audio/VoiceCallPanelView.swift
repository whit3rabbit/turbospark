import SwiftUI

/// OC-08: the floating call panel. State as glyph plus text, the last few
/// utterances, Mute, Stop speaking and End. It is state-only: the transcript
/// itself is the chat, and the panel closes with the call.
@MainActor
struct VoiceCallPanelView: View {
    @ObservedObject var call: VoiceCallController
    let onDownloadModel: () -> Void
    /// Auto-send is on, so Talk (push to talk) means something.
    var showsTalk = false
    @Environment(\.appTheme) private var theme
    /// Where the user dragged the panel to. Not persisted: it resets with
    /// the call.
    @State private var resting = CGSize.zero
    @GestureState private var dragging = CGSize.zero

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            if call.state.isActive {
                if let notice = call.notice { noticeText(notice) }
                if !call.captions.isEmpty { captions }
                if call.draftReady {
                    Label { Text("Draft ready. Press Send to ask.", bundle: .module) } icon: { Image(systemName: "square.and.pencil") }
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Text("Use headphones so the speaker does not feed back into the microphone.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                controls
            } else if let refusal = call.refusal {
                refusalBody(refusal)
            }
        }
        .frame(minWidth: 260 * scale, idealWidth: 320 * scale, maxWidth: 380 * scale, alignment: .leading)
        .padding(14)
        .appGlassSurface(.rect(cornerRadius: AppMetrics.cardRadius))
        .offset(x: resting.width + dragging.width, y: resting.height + dragging.height)
        .accessibilityElement(children: .contain)
        .onChange(of: call.state) { _, new in announce(Text(LocalizedStringKey(new.titleKey), bundle: .module)) }
        .onChange(of: call.captions.last?.text) { _, text in
            if let text { announceVerbatim(text) }
        }
        .onChange(of: call.talk.isTalking) { _, talking in
            announceVerbatim(AppStrings.localized(talking ? "Talking" : "Not talking"))
        }
        // Focus left the app: a held or latched Talk must not outlive it.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in
            call.talk.clear()
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: call.state.isActive ? call.state.symbol : "phone")
                .foregroundStyle(call.state == .listening ? .appAccentText : .appSecondary)
                .tsPulse(active: call.state == .listening || call.state == .preparing)
                .accessibilityHidden(true)
            Text("Voice call", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 8)
            if call.state.isActive {
                // Glyph above, words here: state is never color alone.
                Text(LocalizedStringKey(call.state.titleKey), bundle: .module)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appSecondary)
                    .accessibilityLabel(Text("Voice call", bundle: .module))
                    .accessibilityValue(Text(LocalizedStringKey(call.state.titleKey), bundle: .module))
            }
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture()
                .updating($dragging) { value, state, _ in state = value.translation }
                .onEnded { value in
                    resting.width += value.translation.width
                    resting.height += value.translation.height
                })
    }

    private func noticeText(_ notice: String) -> some View {
        Label { Text(verbatim: notice) } icon: {
            Image(systemName: call.state.symbol).accessibilityHidden(true)
        }
        .themedFont(.small)
        .foregroundStyle(.appSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: Captions

    private var captions: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(call.captions) { caption in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Image(systemName: symbol(for: caption.speaker)).accessibilityHidden(true)
                    label(for: caption.speaker).themedFont(.small, weight: .semibold)
                    Text(verbatim: caption.text).themedFont(.small)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .foregroundStyle(caption.speaker == .note ? .appSecondary : .appText)
                .accessibilityElement(children: .combine)
            }
        }
    }

    private func symbol(for speaker: VoiceCallCaption.Speaker) -> String {
        switch speaker {
        case .you: return "person.fill"
        case .reply: return "speaker.wave.2"
        case .note: return "info.circle"
        }
    }

    @ViewBuilder private func label(for speaker: VoiceCallCaption.Speaker) -> some View {
        switch speaker {
        case .you: Text("You", bundle: .module)
        case .reply: Text("Reply", bundle: .module)
        case .note: EmptyView()
        }
    }

    // MARK: Controls

    private var controls: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            switch call.state {
            case .failed:
                Button { call.end() } label: { Text("Close", bundle: .module) }
                    .appAction(.primary, size: .small)
                    .keyboardShortcut(.defaultAction)
            default:
                if showsTalk { VoiceCallTalkButton(talk: call.talk) }
                Button { Task { await call.toggleMute() } } label: {
                    Label {
                        call.state == .paused ? Text("Unmute", bundle: .module) : Text("Mute", bundle: .module)
                    } icon: {
                        Image(systemName: call.state == .paused ? "mic.fill" : "mic.slash")
                    }
                }
                .appAction(.secondary, size: .small)
                // Mute is a listening state: while a reply runs there is
                // nothing to mute but the speaker, which has its own button.
                .disabled(call.state != .listening && call.state != .paused && call.state != .preparing)
                if call.state == .speaking {
                    Button { call.stopSpeaking() } label: {
                        Label { Text("Stop speaking", bundle: .module) } icon: { Image(systemName: "stop.fill") }
                    }
                    .appAction(.secondary, size: .small)
                    // Escape always stops playback and returns to listening.
                    .keyboardShortcut(.cancelAction)
                }
                Spacer(minLength: 4)
                Button { call.end() } label: {
                    Label { Text("End call", bundle: .module) } icon: { Image(systemName: "phone.down.fill") }
                }
                .appAction(.primary, size: .small)
            }
        }
    }

    // MARK: Refusal

    private func refusalBody(_ refusal: VoiceCallRefusal) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label { refusalText(refusal) } icon: { Image(systemName: "exclamationmark.triangle.fill").accessibilityHidden(true) }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: AppMetrics.actionSpacing) {
                if case .noSpeechModel(_, true) = refusal {
                    Button { onDownloadModel(); call.dismissRefusal() } label: { Text("Download", bundle: .module) }
                        .appAction(.primary, size: .small)
                }
                Button { call.dismissRefusal() } label: { Text("Dismiss", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .keyboardShortcut(.cancelAction)
            }
        }
    }

    private func refusalText(_ refusal: VoiceCallRefusal) -> Text {
        switch refusal {
        case .microphone: return Text("Microphone access is not allowed for voice calls.", bundle: .module)
        case .turnRunning: return Text("Finish or stop the current reply before starting a call.", bundle: .module)
        case .recordingActive: return Text("A recording is already active. Stop it before starting another.", bundle: .module)
        case .noSpeechModel(let reason, _): return Text(verbatim: reason)
        }
    }

    // MARK: VoiceOver

    private func announce(_ text: Text) {
        // Mirrors `showToast`: a state change is spoken, not only drawn.
        _ = AccessibilityNotification.Announcement.post(.init(AppStrings.localized(String.LocalizationValue(call.state.titleKey))))
    }

    private func announceVerbatim(_ text: String) {
        _ = AccessibilityNotification.Announcement.post(.init(text))
    }
}
