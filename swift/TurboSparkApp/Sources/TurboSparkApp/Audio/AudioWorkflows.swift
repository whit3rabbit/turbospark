import AppKit
import Foundation
import SwiftUI

/// The composer's recording state machine (docs/AUDIO_UI.md, capability 1).
///
/// One instance, because there is one microphone: two composers (chat and a
/// future pop-out) recording at once would fight over the input node.
@MainActor
final class ComposerAudioRecorder: ObservableObject {
    static let shared = ComposerAudioRecorder()

    enum Mode: Equatable, Sendable {
        /// Transcribe and insert the text into the prompt.
        case dictate
        /// Attach the recording itself as an audio attachment.
        case voiceNote
    }

    enum Phase: Equatable {
        case idle
        case requestingPermission
        case recording(Mode)
        case transcribing
        case denied(AudioPermissionKind)
    }

    @Published private(set) var phase: Phase = .idle
    let capture = AudioCaptureService()

    var isActive: Bool { phase != .idle }

    private init() {
        capture.onLimitReached = { [weak self] in
            guard let self, case .recording = self.phase, let model = self.model else { return }
            self.finish(model: model)
        }
    }

    /// The model the current recording belongs to, so the clip-length limit
    /// can finish it without a view in hand.
    private weak var model: AppModel?

    /// Starts recording in `mode`, prompting for permission first when the
    /// system has never asked. With the audio flag off, or with macOS
    /// Dictation chosen, forwards to system dictation exactly as before.
    func start(mode: Mode, model: AppModel, promptFocused: FocusState<Bool>.Binding) {
        let capabilities = AudioCapabilities.shared
        capabilities.refresh()
        if !AudioPreferences.experimentalEnabled
            || (mode == .dictate && AudioPreferences.speechEngine == .systemDictation)
        {
            Self.startSystemDictation(promptFocused: promptFocused)
            return
        }
        guard phase == .idle else { return }
        self.model = model
        phase = .requestingPermission
        Task {
            guard await capabilities.requestMicrophoneAccess() else {
                phase = .denied(.microphone)
                return
            }
            if mode == .dictate,
                case .needsPermission(.speechRecognition) = capabilities.snapshot.transcription,
                !(await capabilities.requestSpeechAccess())
            {
                phase = .denied(.speechRecognition)
                return
            }
            do {
                try capture.start()
                phase = .recording(mode)
            } catch {
                phase = .idle
                model.showToast(error.localizedDescription, style: .error)
            }
        }
    }

    /// Stops recording and delivers the result for the current mode.
    func finish(model: AppModel) {
        guard case .recording(let mode) = phase, let url = capture.stop() else { return }
        switch mode {
        case .voiceNote:
            phase = .idle
            attach(url, model: model)
        case .dictate:
            phase = .transcribing
            Task {
                do {
                    let transcript = try await SpeechToTextRouter.transcribe(fileURL: url)
                    try? FileManager.default.removeItem(at: url)
                    insert(transcript.text, into: model)
                    phase = .idle
                    _ = AccessibilityNotification.Announcement.post(.init(
                        String(localized: "Transcript inserted", bundle: .module)))
                    if AudioPreferences.autoSendAfterDictation && !model.isRunning {
                        model.run()
                    }
                } catch {
                    // Nothing the user said is thrown away: a recording that
                    // could not be transcribed becomes a voice note instead.
                    phase = .idle
                    attach(url, model: model)
                    model.showToast(
                        String(
                            localized: "Could not transcribe, attached the recording instead: \(error.localizedDescription)",
                            bundle: .module),
                        style: .warning, duration: 6)
                }
            }
        }
    }

    /// Stops and discards.
    func cancel() {
        capture.cancel()
        phase = .idle
    }

    /// Clears the denied notice.
    func dismissNotice() {
        if case .denied = phase { phase = .idle }
    }

    static func startSystemDictation(promptFocused: FocusState<Bool>.Binding) {
        promptFocused.wrappedValue = true
        NSApp.sendAction(Selector(("startDictation:")), to: nil, from: nil)
    }

    /// Appends at the end of the draft. `TextEditor` exposes no caret, and
    /// appending is the behaviour the plus menu's inserts already use.
    private func insert(_ text: String, into model: AppModel) {
        let current = model.promptText
        let separator = current.isEmpty || current.hasSuffix(" ") || current.hasSuffix("\n") ? "" : " "
        model.writePromptTextDirectly(current + separator + text)
    }

    private func attach(_ url: URL, model: AppModel) {
        let chatID = model.selectedChatID
        Task {
            let outcome = await AttachmentImporter.importDocuments([url], into: model, chatID: chatID)
            try? FileManager.default.removeItem(at: url)
            if let error = outcome.errorText {
                model.showToast(error, style: .error)
            }
        }
    }
}

/// Background transcription of audio ATTACHMENTS, with per-attachment
/// status the chip reads (docs/AUDIO_UI.md, capability 2).
@MainActor
final class AudioAttachmentTranscriber: ObservableObject {
    static let shared = AudioAttachmentTranscriber()

    @Published private(set) var inFlight: Set<UUID> = []
    @Published private(set) var failures: [UUID: String] = [:]
    /// Who produced each finished transcript, for the preview pane label.
    @Published private(set) var providers: [UUID: SpeechProvider] = [:]

    func isTranscribing(_ id: UUID) -> Bool { inFlight.contains(id) }

    /// Starts transcription unless one is running or the capability says no.
    func transcribe(_ attachment: AppPromptAttachment, chatID: UUID?, model: AppModel) {
        guard attachment.isAudio, !inFlight.contains(attachment.id) else { return }
        let availability = AudioCapabilities.shared.snapshot.transcription
        guard availability.isUsable else {
            failures[attachment.id] = availability.reason
            return
        }
        guard let url = attachment.sourceURL else {
            failures[attachment.id] = String(
                localized: "The source file is no longer at its original path.", bundle: .module)
            return
        }
        inFlight.insert(attachment.id)
        failures[attachment.id] = nil
        let id = attachment.id
        Task {
            if case .needsPermission(.speechRecognition) = availability {
                _ = await AudioCapabilities.shared.requestSpeechAccess()
            }
            do {
                let transcript = try await SpeechToTextRouter.transcribe(fileURL: url)
                model.updatePromptAttachment(id: id, inChatID: chatID) {
                    $0.extractedText = transcript.text
                }
                providers[id] = transcript.provider
            } catch {
                failures[id] = error.localizedDescription
            }
            inFlight.remove(id)
        }
    }
}
