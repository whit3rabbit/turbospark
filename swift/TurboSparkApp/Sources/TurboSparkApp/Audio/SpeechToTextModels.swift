import Foundation
import TurboSpark

/// The jobs a Speech-to-Text mode runs a model for. Each mode picks its own
/// model per job, saved in `AudioWorkspacePreferences.selectedModels` under
/// `speech_to_text:<mode>:<job>`.
enum SpeechToTextJob: String, CaseIterable, Identifiable, Sendable {
    case live, final, speakers
    var id: String { rawValue }

    /// The catalog task this job's models come from.
    var task: AudioTask { self == .speakers ? .diarization : .speechToText }

    /// Whether a profile can do this job. Live text streams, the final pass
    /// transcribes a whole take, and speakers diarize. A model this build
    /// cannot run at all is never offered.
    func accepts(_ profile: AudioProfile) -> Bool {
        guard profile.identity.task == task, AudioRuntimeProvider(profile) != .unsupported else { return false }
        let operations = profile.capabilities.operations
        switch self {
        case .live: return operations.contains("stream_transcribe")
        case .final: return operations.contains("transcribe")
        case .speakers: return operations.contains("diarize")
        }
    }

    /// Live text opens its own stream for each take and closes it when the
    /// take stops, so it is never held loaded between takes.
    var staysLoaded: Bool { self != .live }

    /// One line on when the job runs, for its slot in the models popover.
    var hintKey: String {
        switch self {
        case .live: return "Words appear as you speak, about a second behind (English)."
        case .final: return "Runs when a take stops and replaces the live draft."
        case .speakers: return "Finds who spoke when, after the take stops."
        }
    }
}

extension SpeechToTextMode {
    /// Every mode can label speakers. Only a meeting does by default
    /// (`labelsSpeakersByDefault`); the others opt in with a switch.
    var jobs: [SpeechToTextJob] { [.live, .final, .speakers] }

    var labelsSpeakersByDefault: Bool { self == .meeting }

    /// Any Audio's live text is captions of what another app plays.
    func titleKey(for job: SpeechToTextJob) -> String {
        switch job {
        case .live: return self == .anyAudio ? "Live Captions" : "Live Text"
        case .final: return "Final Transcript"
        case .speakers: return "Speakers"
        }
    }

    /// The mode whose preset recorded a take. The Record page shares the
    /// voice note and meeting presets, so its takes map here too.
    init?(preset: AudioRecordingPreset) {
        guard let mode = Self.allCases.first(where: { $0.recordingPreset == preset }) else { return nil }
        self = mode
    }
}

/// Which model each mode uses for each job. Pure, so tests can drive it
/// without a controller. Main-actor bound like the preference order it reuses.
@MainActor
enum SpeechToTextModels {
    nonisolated static func key(_ mode: SpeechToTextMode, _ job: SpeechToTextJob) -> String {
        "speech_to_text:\(mode.rawValue):\(job.rawValue)"
    }

    /// Where the one shared Speech-to-Text model was saved before modes had
    /// their own. The Record page still uses it.
    nonisolated static let sharedKey = AudioTask.speechToText.rawValue

    /// Catalog profiles that can do a job, best first. Final passes keep the
    /// workspace's preference order.
    static func candidates(for job: SpeechToTextJob, in profiles: [AudioProfile]) -> [AudioProfile] {
        let fit = profiles.filter { job.accepts($0) }
        return job == .final ? AudioWorkspaceController.preferredSpeechProfiles(fit) : fit
    }

    /// The model a mode uses for a job. The mode's own choice wins while the
    /// catalog lists it. Otherwise the job keeps the app's earlier behavior:
    /// the shared Speech-to-Text choice for the final pass, Parakeet Unified
    /// for live text, and the first installed diarization model for speakers.
    /// Failing those, the best installed model this Mac can run, then the
    /// best candidate, which the popover offers to download.
    static func resolve(_ mode: SpeechToTextMode, _ job: SpeechToTextJob, profiles: [AudioProfile],
                        selected: [String: String], isInstalled: (AudioProfile) -> Bool) -> AudioProfile? {
        let options = candidates(for: job, in: profiles)
        func listed(_ alias: String?) -> AudioProfile? {
            alias.flatMap { alias in options.first { $0.identity.alias == alias } }
        }
        if let chosen = listed(selected[key(mode, job)]) { return chosen }
        switch job {
        case .final: if let shared = listed(selected[sharedKey]) { return shared }
        case .live: if let live = listed(AudioWorkspaceController.liveTextAlias) { return live }
        case .speakers: break
        }
        let runnable = options.filter { AudioRuntimeProvider($0).canRun($0) }
        return runnable.first(where: isInstalled) ?? runnable.first ?? options.first
    }

    /// The selections after `alias` was deleted. Every mode that used it for
    /// a job moves to the best installed model left for that job. With none
    /// left, its own choice is dropped and the default decides, which may
    /// name the deleted model again as one to download. `isInstalled`
    /// describes the store after the delete.
    static func selections(afterDeleting alias: String, from selected: [String: String], profiles: [AudioProfile],
                           isInstalled: (AudioProfile) -> Bool) -> [String: String] {
        // A checkpoint folder chosen for the same model still serves it.
        if let profile = profiles.first(where: { $0.identity.alias == alias }), isInstalled(profile) { return selected }
        var next = selected
        let before: (AudioProfile) -> Bool = { $0.identity.alias == alias || isInstalled($0) }
        for mode in SpeechToTextMode.allCases {
            for job in mode.jobs {
                let used = resolve(mode, job, profiles: profiles, selected: selected, isInstalled: before)
                guard used?.identity.alias == alias else { continue }
                let replacement = candidates(for: job, in: profiles).first {
                    $0.identity.alias != alias && AudioRuntimeProvider($0).canRun($0) && isInstalled($0)
                }
                next[key(mode, job)] = replacement?.identity.alias
            }
        }
        return next
    }

    /// Every mode and job that uses `alias`, for the Installed Used By card.
    static func uses(of alias: String, profiles: [AudioProfile], selected: [String: String],
                     isInstalled: (AudioProfile) -> Bool) -> [(mode: SpeechToTextMode, job: SpeechToTextJob)] {
        SpeechToTextMode.allCases.flatMap { mode in
            mode.jobs.compactMap { job in
                resolve(mode, job, profiles: profiles, selected: selected, isInstalled: isInstalled)?.identity.alias == alias
                    ? (mode, job) : nil
            }
        }
    }
}

extension AudioWorkspaceController {
    /// A managed install or a checkpoint folder the user chose.
    func isInstalled(_ profile: AudioProfile) -> Bool {
        installedPaths[profile.identity.alias] != nil || preferences.localModels[profile.identity.alias] != nil
    }

    /// The model a mode uses for a job. The current mode's final pass is the
    /// recipe's model, which a run and a new take read, so the two never
    /// disagree.
    func speechToTextProfile(_ mode: SpeechToTextMode, _ job: SpeechToTextJob) -> AudioProfile? {
        if job == .final, page == .transcribe, mode == speechToTextMode, let selectedProfile { return selectedProfile }
        let selected = speechToTextSelections
        return SpeechToTextModels.resolve(mode, job, profiles: profiles, selected: selected, isInstalled: isInstalled)
    }

    func speechToTextCandidates(_ job: SpeechToTextJob) -> [AudioProfile] {
        SpeechToTextModels.candidates(for: job, in: profiles)
    }

    /// The saved choices with the current mode's final model as the recipe
    /// holds it, which reaches the preferences only at the next flush.
    var speechToTextSelections: [String: String] {
        var selected = preferences.selectedModels
        if page == .transcribe, let model = recipe.modelID {
            selected[SpeechToTextModels.key(speechToTextMode, .final)] = model
        }
        return selected
    }

    /// Changing a model mid-run would split a take between two models.
    var canChangeSpeechToTextModels: Bool { !isBusy && !hasRecordingActivity && !isShutDown }

    /// Uses `alias` for a mode's job from the next take on.
    func setSpeechToTextModel(_ alias: String, mode: SpeechToTextMode, job: SpeechToTextJob) {
        guard canChangeSpeechToTextModels, speechToTextCandidates(job).contains(where: { $0.identity.alias == alias }) else { return }
        preferences.selectedModels[SpeechToTextModels.key(mode, job)] = alias
        if job == .final, page == .transcribe, mode == speechToTextMode { selectModel(alias) }
        flushPreferences()
    }

    /// The Speech-to-Text mode an item belongs to: a take by the preset it
    /// recorded with, an imported file to Any Audio, which lists imports.
    func speechToTextMode(for item: AudioLibraryItem) -> SpeechToTextMode? {
        if let mode = item.recordingPreset.flatMap(AudioRecordingPreset.init(rawValue:)).flatMap(SpeechToTextMode.init(preset:)) {
            return mode
        }
        return item.kind == .imported ? .anyAudio : nil
    }

    /// Whether a mode's final pass labels speakers.
    func labelsSpeakers(in mode: SpeechToTextMode) -> Bool {
        preferences.speakerLabels?[mode.rawValue] ?? mode.labelsSpeakersByDefault
    }

    /// Turns speaker labels on or off for a mode, from the next take on.
    func setLabelsSpeakers(_ on: Bool, in mode: SpeechToTextMode) {
        guard canChangeSpeechToTextModels else { return }
        var labels = preferences.speakerLabels ?? [:]
        labels[mode.rawValue] = on
        preferences.speakerLabels = labels
        flushPreferences()
    }

    /// After a delete, the modes that used `alias` move on (see
    /// `SpeechToTextModels.selections(afterDeleting:)`), and the current
    /// mode's recipe follows its new choice.
    func reassignSpeechToTextModels(afterDeleting alias: String) {
        let before = speechToTextSelections
        preferences.selectedModels = SpeechToTextModels.selections(
            afterDeleting: alias, from: before, profiles: profiles, isInstalled: isInstalled)
        guard page == .transcribe, recipe.modelID == alias else { return }
        recipe.modelID = preferences.selectedModels[modelSelectionKey]
        restoreModelSelection()
        normalizeModelOptions()
    }
}
