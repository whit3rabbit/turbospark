import SwiftUI

/// What speaks on the Text-to-Speech page.
enum AudioSpeechEngine: Equatable {
    /// No model is chosen: the macOS voices read the script through
    /// `AppSpeechSynthesizer`, so the page works before any download.
    case systemVoice
    /// The chosen audio model, by alias.
    case model(String)
}

extension AudioWorkspaceController {
    /// The synthesizer is shared with chat read-aloud and keys its state by a
    /// message id; this one marks the workspace's reading.
    static let systemVoiceUtteranceID = UUID(uuidString: "6E1D6D0A-5F2B-4C59-9B57-3A0C5B2F7E11")!

    /// Text-to-Speech only. Voice Cloning and Voice Design have no system
    /// fallback (they need a reference or a description a model reads), and
    /// keep their gate and consent rules, so they return nil.
    /// `restoreModelSelection` leaves the choice empty only when no runnable
    /// model is listed, so "no model" here means none is installed.
    var speechEngine: AudioSpeechEngine? {
        guard page == .voiceover else { return nil }
        guard let id = recipe.modelID else { return .systemVoice }
        return .model(id)
    }

    var usesSystemVoice: Bool { speechEngine == .systemVoice }

    /// `canRun` for the system voice. A model download in flight does not
    /// block it, since nothing here needs the model.
    var canSpeakWithSystemVoice: Bool {
        usesSystemVoice && !isBusy && !isShutDown && !hasRecordingActivity && importTask == nil
            && !recipe.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var isSpeakingWithSystemVoice: Bool {
        let synthesizer = AppSpeechSynthesizer.shared
        return synthesizer.isSpeaking && synthesizer.speakingMessageID == Self.systemVoiceUtteranceID
    }

    /// Reads `text` aloud, or stops when the workspace is already reading
    /// and `toggle` is set (the Read aloud button). Nothing is saved: the
    /// synthesizer plays live and writes no take.
    func speakWithSystemVoice(_ text: String, toggle: Bool) {
        let synthesizer = AppSpeechSynthesizer.shared
        if toggle {
            synthesizer.toggleSpeech(text: text, messageID: Self.systemVoiceUtteranceID)
        } else {
            synthesizer.speak(text: text, messageID: Self.systemVoiceUtteranceID)
        }
    }

    /// Stops the workspace's reading, leaving a chat message's alone.
    func stopSystemVoice() {
        if isSpeakingWithSystemVoice { AppSpeechSynthesizer.shared.stop() }
    }
}

/// Says the system voice is speaking and offers the model download without
/// blocking the page. Models are downloaded in Discover.
@MainActor
struct AudioSystemVoiceNotice: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { message; Spacer(minLength: 8); upgrade }
            VStack(alignment: .leading, spacing: 8) { message; upgrade }
        }
        .themedFont(.small)
    }

    private var message: some View {
        Label {
            Text("Reads the script aloud with the macOS system voice. Nothing is saved.", bundle: .module)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: "speaker.wave.2").foregroundStyle(.appAccentText)
        }
        .foregroundStyle(.appSecondary)
    }

    private var upgrade: some View {
        Button { model.openModelHub() } label: {
            Text("Download a model for better voices", bundle: .module)
        }
        .appAction(.secondary, size: .small)
    }
}
