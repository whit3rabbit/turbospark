import AVFoundation
import Foundation
import TurboSpark

extension AudioWorkspaceController {
    var modelSelectionKey: String {
        switch page {
        case .voiceClone: return "text_to_speech:clone"
        case .voiceDesign: return "text_to_speech:design"
        case .voiceover: return AudioTask.textToSpeech.rawValue
        // Each Speech-to-Text mode keeps its own final model.
        case .transcribe: return SpeechToTextModels.key(speechToTextMode, .final)
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
        if profile.family == "omnivoice" {
            let backend = profile.capabilities.backend.lowercased()
            let hasMetalOrMLX = backend.contains("metal") || backend.contains("mlx")
            return hasMetalOrMLX && profile.capabilities.supportsVoiceClone
        }
        return profile.family != "qwen3_tts" && profile.capabilities.supportsVoiceClone
    }
    /// The native cloners whose runtime also speaks with no reference, in a
    /// voice the model picks (`fish_drama3_request`, `omnivoice_request`).
    /// MegaTTS3 cannot, so it stays off Voiceover.
    nonisolated static let referenceFreeSpeechFamilies: Set<String> = ["omnivoice", "fish_drama3"]
    /// Whether a run clones from a reference. A native cloner that can speak
    /// without one clones only in clone mode, which `normalizeModelOptions`
    /// sets on every page but Voiceover. Any other native cloner always does.
    nonisolated static func runClonesNatively(_ profile: AudioProfile, speech: AudioSpeechSettings?) -> Bool {
        guard clonesNatively(profile) else { return false }
        return speech?.mode == .clone || !referenceFreeSpeechFamilies.contains(profile.family)
    }
    /// The selected run clones from a reference (Voice Cloning, Advanced).
    var selectedProfileClonesNatively: Bool {
        selectedProfile.map { Self.runClonesNatively($0, speech: recipe.speech) } ?? false
    }
    /// The selected native cloner speaks without a reference (Voiceover).
    var selectedProfileSpeaksWithoutReference: Bool {
        Self.clonesNatively(selectedProfile) && !selectedProfileClonesNatively
    }
    /// The native families whose request shape `nativeCloneRequest` builds.
    nonisolated static let nativeCloneRequestFamilies: Set<String> = ["megatts3", "omnivoice", "fish_drama3"]
    /// Drama 3 clones only from the reference's exact words, and has no
    /// speed, steps or guidance weights.
    var selectedProfileNeedsReferenceTranscript: Bool { selectedProfile?.family == "fish_drama3" }
    func speechProfileMatchesPage(_ profile: AudioProfile) -> Bool {
        // A streaming-only model (Parakeet Unified) serves live text, not a
        // batch transcription job, which would refuse to open it.
        if profile.identity.task == .speechToText && !profile.capabilities.operations.contains("transcribe") { return false }
        // Any other native cloner would get MegaTTS3's request, which the
        // runtime refuses for another family. A new cloner stays unlisted
        // until `nativeCloneRequest` builds its shape.
        if Self.clonesNatively(profile) && !Self.nativeCloneRequestFamilies.contains(profile.family) { return false }
        switch page {
        case .voiceClone:
            if profile.family == "omnivoice" {
                let backend = profile.capabilities.backend.lowercased()
                let hasMetalOrMLX = backend.contains("metal") || backend.contains("mlx")
                if !hasMetalOrMLX { return false }
            }
            return (profile.family == "qwen3_tts" || Self.clonesNatively(profile)) && Self.speechMode(for: profile) == .clone
        case .voiceDesign: return profile.family == "qwen3_tts" && Self.speechMode(for: profile) == .design
        case .voiceover:
            if Self.clonesNatively(profile) { return Self.referenceFreeSpeechFamilies.contains(profile.family) }
            return Self.speechMode(for: profile) == .customVoice
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
    /// The longest script either engine accepts in one run.
    nonisolated static let speechTextCharacterLimit = 6_000
    var speechValidationMessage: String? {
        if selectedProfileClonesNatively { return nativeCloneValidationMessage }
        if selectedProfileSpeaksWithoutReference {
            return recipe.text.count > Self.speechTextCharacterLimit
                ? AppStrings.localized("Split speech text longer than 6,000 characters into sections.") : nil
        }
        guard selectedProfile?.family == "qwen3_tts" else { return nil }
        if recipe.text.count > Self.speechTextCharacterLimit { return AppStrings.localized("Split speech text longer than 6,000 characters into sections.") }
        if speechSettings.mode == .clone {
            guard let reference = speechSettings.reference, reference.isValid else {
                return AppStrings.localized("Choose 1 to 30 seconds of reference speech.")
            }
            if !speechSettings.embeddingOnly && speechSettings.referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return AppStrings.localized("Enter the exact transcript of the selected reference region.")
            }
        }
        if speechSettings.mode == .design && speechSettings.instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return AppStrings.localized("Describe the voice before generating speech.")
        }
        return nil
    }
    private var nativeCloneValidationMessage: String? {
        if recipe.text.count > Self.speechTextCharacterLimit { return AppStrings.localized("Split speech text longer than 6,000 characters into sections.") }
        guard let reference = speechSettings.reference, reference.isValid else {
            return AppStrings.localized("Choose 1 to 30 seconds of reference speech.")
        }
        if cloneOptions.consentedReference != reference {
            return AppStrings.localized("Confirm you have the right to use this voice.")
        }
        if selectedProfileNeedsReferenceTranscript && speechSettings.referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return AppStrings.localized("Enter the exact transcript of the selected reference region.")
        }
        return nil
    }
    func setVoiceReference(_ item: AudioLibraryItem) {
        guard item.hasPlayableAudio, !isBusy, !hasRecordingActivity else { return }
        assignVoiceReference(item)
    }
    /// The body of `setVoiceReference` without its idle guard, for a
    /// recording that just stopped while an unrelated job holds `isBusy`.
    func assignVoiceReference(_ item: AudioLibraryItem) {
        guard item.hasPlayableAudio else { return }
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
    /// Moves one end of the reference region. A transcript of a different
    /// region is invalid conditioning: OmniVoice speaks a mismatched
    /// transcript's words in place of the script
    /// (`crates/audio/src/tts/omnivoice/README.md`, reference transcript A/B).
    func setVoiceReferenceTrim(_ path: WritableKeyPath<AudioVoiceReference, Double>, to value: Double) {
        stopReferencePreview()
        speechSettings.reference?[keyPath: path] = value
        speechSettings.referenceText = ""
    }
    func stopReferencePreview() {
        referencePreviewTask?.cancel(); referencePreviewTask = nil
        referencePlayer?.stop(); referencePlayer = nil; referencePlaying = false
        speakerSampleKey = nil
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
