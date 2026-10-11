import AVFoundation
import Foundation
import TurboSpark

extension AudioWorkspaceController {
    var modelSelectionKey: String {
        switch page {
        case .voiceClone: return "text_to_speech:clone"
        case .voiceDesign: return "text_to_speech:design"
        case .voiceover: return AudioTask.textToSpeech.rawValue
        default:
            if task == .textToSpeech, let mode = recipe.speech?.mode, mode != .customVoice {
                return "text_to_speech:" + (mode == .clone ? "clone" : "design")
            }
            return task.rawValue
        }
    }
    nonisolated static func speechMode(for profile: AudioProfile) -> AudioSpeechSettings.Mode {
        // Picker routing uses catalog metadata. Execution validates config.json independently.
        if profile.identity.repository.contains("VoiceDesign") { return .design }
        if profile.family == "qwen3_tts" && profile.identity.repository.contains("-Base-") { return .clone }
        if clonesNatively(profile) { return .clone }
        return .customVoice
    }
    /// A Rust session clones from reference audio passed as its PCM (MegaTTS3).
    /// Qwen3-TTS clones in the Swift MLX provider instead.
    nonisolated static func clonesNatively(_ profile: AudioProfile?) -> Bool {
        guard let profile else { return false }
        return profile.family != "qwen3_tts" && profile.capabilities.supportsVoiceClone
    }
    var selectedProfileClonesNatively: Bool { Self.clonesNatively(selectedProfile) }
    func speechProfileMatchesPage(_ profile: AudioProfile) -> Bool {
        switch page {
        case .voiceClone:
            return (profile.family == "qwen3_tts" || Self.clonesNatively(profile)) && Self.speechMode(for: profile) == .clone
        case .voiceDesign: return profile.family == "qwen3_tts" && Self.speechMode(for: profile) == .design
        case .voiceover: return Self.speechMode(for: profile) == .customVoice
        default: return true
        }
    }
    var speechSettings: AudioSpeechSettings {
        get { recipe.speech ?? AudioSpeechSettings() }
        set { recipe.speech = newValue }
    }
    var speechSupportsStyle: Bool {
        speechSettings.mode == .design || selectedProfile?.identity.repository.contains("1.7B-CustomVoice") == true
    }
    var cloneOptions: AudioCloneOptions {
        get { speechSettings.cloneOptions ?? AudioCloneOptions() }
        set { var settings = speechSettings; settings.cloneOptions = newValue; speechSettings = settings }
    }
    /// Consent covers one exact reference region.
    var cloneConsentGiven: Bool {
        guard let reference = speechSettings.reference else { return false }
        return cloneOptions.consentedReference == reference
    }
    func setCloneConsent(_ granted: Bool) {
        cloneOptions.consentedReference = granted ? speechSettings.reference : nil
        persistPreferences()
    }
    var speechValidationMessage: String? {
        if selectedProfileClonesNatively { return nativeCloneValidationMessage }
        guard selectedProfile?.family == "qwen3_tts" else { return nil }
        if recipe.text.count > 6000 { return String(localized: "Split speech text longer than 6,000 characters into sections.", bundle: .module) }
        if speechSettings.mode == .clone {
            guard let reference = speechSettings.reference, reference.isValid else {
                return String(localized: "Choose 1 to 30 seconds of reference speech.", bundle: .module)
            }
            if !speechSettings.embeddingOnly && speechSettings.referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return String(localized: "Enter the exact transcript of the selected reference region.", bundle: .module)
            }
        }
        if speechSettings.mode == .design && speechSettings.instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return String(localized: "Describe the voice before generating speech.", bundle: .module)
        }
        return nil
    }
    private var nativeCloneValidationMessage: String? {
        if recipe.text.count > 6000 { return String(localized: "Split speech text longer than 6,000 characters into sections.", bundle: .module) }
        guard let reference = speechSettings.reference, reference.isValid else {
            return String(localized: "Choose 1 to 30 seconds of reference speech.", bundle: .module)
        }
        if cloneOptions.consentedReference != reference {
            return String(localized: "Confirm you have the right to use this voice.", bundle: .module)
        }
        return nil
    }
    func setVoiceReference(_ item: AudioLibraryItem) {
        guard item.hasPlayableAudio, !isBusy, !hasRecordingActivity else { return }
        stopReferencePreview()
        var settings = speechSettings
        settings.reference = AudioVoiceReference(sourceID: item.id, title: item.title, clips: item.clips, duration: min(30, item.duration))
        settings.referenceText = item.preferredTranscript?.segments.map(\.text).joined(separator: " ")
            ?? (item.kind == .voiceover ? item.recipe.text : "")
        if item.duration > 30 { settings.referenceText = "" }
        speechSettings = settings
        flushPreferences()
    }
    func reuseVoice(_ item: AudioLibraryItem) {
        guard !isBusy, !hasRecordingActivity else { return }
        selectPage(.voiceClone)
        setVoiceReference(item)
    }
    func clearVoiceReference() { stopReferencePreview(); speechSettings.reference = nil; speechSettings.referenceText = ""; flushPreferences() }
    func stopReferencePreview() {
        referencePreviewTask?.cancel(); referencePreviewTask = nil
        referencePlayer?.stop(); referencePlayer = nil; referencePlaying = false
    }
    func previewVoiceReference() {
        if referencePlaying || referencePreviewTask != nil { stopReferencePreview(); return }
        guard let reference = speechSettings.reference, reference.isValid, !hasRecordingActivity, !isShutDown else { return }
        stopPlayback()
        let assets = self.assets; let lifetime = self.lifetime; let token = epoch
        referencePreviewTask = Task { [weak self] in
            do {
                let bytes = try await Task.detached {
                    let samples = try AudioMediaIO.readMono24k(reference.clips, store: assets, start: reference.start, duration: reference.duration, cancel: lifetime.check)
                    return try AudioMediaIO.wav(samples: samples, sampleRate: 24_000, channels: 1)
                }.value
                try Task.checkCancellation()
                guard let self, self.valid(token), !self.hasRecordingActivity, self.speechSettings.reference == reference else { return }
                let player = try AVAudioPlayer(data: bytes)
                self.referencePlayer = player; self.referencePlaying = player.play(); self.referencePreviewTask = nil
            } catch {
                guard let self, self.valid(token), !Task.isCancelled else { return }
                self.referencePreviewTask = nil; self.error = error.localizedDescription
            }
        }
    }
}
