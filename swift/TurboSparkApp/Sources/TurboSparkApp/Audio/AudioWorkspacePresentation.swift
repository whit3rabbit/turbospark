import Foundation

extension AudioWorkspaceController {
    var runTitle: String {
        switch task {
        case .speechToText: return "Transcribe"
        case .textToSpeech: return "Generate speech"
        case .music: return "Generate music"
        default: return "Run"
        }
    }

    var workspaceResult: AudioLibraryItem? {
        guard let item = selectedItem else { return nil }
        switch page {
        case .music: return item.kind == .music ? item : nil
        case .voiceover, .voiceClone, .voiceDesign: return item.kind == .voiceover && (item.recipe.speech?.mode ?? .customVoice) == (recipe.speech?.mode ?? .customVoice) ? item : nil
        default: return item
        }
    }

    var workspaceHistory: [AudioLibraryItem] {
        guard AudioWorkspacePage.destinations.contains(page) else { return [] }
        return items.filter { $0.kind == kind && (task != .textToSpeech || ($0.recipe.speech?.mode ?? .customVoice) == (recipe.speech?.mode ?? .customVoice)) && $0.id != selectedID && $0.status == .completed && $0.hasPlayableAudio }
            .sorted { $0.createdAt > $1.createdAt }
    }
}

/// One step of the Voice Cloning steps strip.
enum VoiceCloneStep: Hashable, CaseIterable, Sendable { case voice, consent, transcript, script, generate }

/// What a step shows. Each state renders with its own glyph and text, never
/// color alone.
enum VoiceCloneStepState: Equatable, Sendable { case todo, attention, done, ready, active }

struct VoiceCloneStepStatus: Identifiable, Equatable, Sendable {
    let step: VoiceCloneStep
    let state: VoiceCloneStepState
    var id: VoiceCloneStep { step }
}

/// Why Generate speech is disabled, shown beside the button and in its
/// tooltip. Nil exactly when an idle workspace can run.
enum VoiceCloneRunBlocker: Equatable, Sendable {
    case recording, importing, installing, noModel, notInstalled, runtimeUnavailable, emptyScript
    case invalid(String)
}

extension AudioWorkspaceController {
    /// Every saved take of the current speech mode, newest first. Unlike
    /// `workspaceHistory` it keeps the selected take, so the row a user just
    /// played does not vanish from under VoiceOver focus.
    var speechTakes: [AudioLibraryItem] {
        guard task == .textToSpeech else { return [] }
        let mode = recipe.speech?.mode ?? .customVoice
        return items.filter {
            $0.kind == .voiceover && ($0.recipe.speech?.mode ?? .customVoice) == mode
                && $0.status == .completed && $0.hasPlayableAudio
        }.sorted { $0.createdAt > $1.createdAt }
    }

    var voiceCloneSteps: [VoiceCloneStepStatus] {
        var steps = [VoiceCloneStepStatus(step: .voice, state: voiceStepState)]
        if selectedProfileClonesNatively {
            steps.append(VoiceCloneStepStatus(step: .consent, state: cloneConsentGiven ? .done : .todo))
        } else if selectedProfile?.family == "qwen3_tts" {
            let transcribed = speechSettings.embeddingOnly
                || !speechSettings.referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            steps.append(VoiceCloneStepStatus(step: .transcript, state: transcribed ? .done : .todo))
        }
        steps.append(VoiceCloneStepStatus(step: .script, state: scriptStepState))
        let generate: VoiceCloneStepState = isBusy && speechRun != nil ? .active : (canRun ? .ready : .todo)
        steps.append(VoiceCloneStepStatus(step: .generate, state: generate))
        return steps
    }

    private var voiceStepState: VoiceCloneStepState {
        guard let reference = speechSettings.reference else { return .todo }
        return reference.isValid ? .done : .attention
    }

    private var scriptStepState: VoiceCloneStepState {
        if recipe.text.count > Self.speechTextCharacterLimit { return .attention }
        return recipe.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .todo : .done
    }

    var voiceCloneRunBlocker: VoiceCloneRunBlocker? {
        guard !isBusy, !isShutDown, !canRun else { return nil }
        if hasRecordingActivity { return .recording }
        if importTask != nil { return .importing }
        if isInstalling { return .installing }
        if selectedProfile == nil { return .noModel }
        if !modelInstalled { return .notInstalled }
        if !selectedProfileIsRunnable { return .runtimeUnavailable }
        if recipe.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return .emptyScript }
        if let message = speechValidationMessage { return .invalid(message) }
        // canRun said no for a reason not listed above. Say so rather than
        // showing an enabled-looking explanation-free disabled button.
        return .runtimeUnavailable
    }
}
