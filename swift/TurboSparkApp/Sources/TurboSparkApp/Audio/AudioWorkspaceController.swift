import AppKit
import AVFoundation
import Combine
import Foundation
import Metal
import MLX
import Qwen3TTS
import TurboSpark
import UniformTypeIdentifiers

enum AudioTranscriptExportFormat: String, CaseIterable { case text, srt, vtt, json }
struct AudioCaptureApplication: Identifiable { var id: Int32; var name: String }
struct AudioCaptureMicrophone: Identifiable { var id: String; var name: String }

@MainActor
final class AudioWorkspaceController: ObservableObject {
    @Published var page: AudioWorkspacePage = .library { didSet { persistPreferences() } }
    @Published var recipe = AudioRecipe() { didSet { persistPreferences() } }
    @Published var items: [AudioLibraryItem] = []
    // Several paths select a take by assigning here (a new generation, an
    // import, a recording), not through `select`. A player left over from the
    // previous selection would otherwise be resumed by Play and replay the old take.
    @Published var selectedID: UUID? { didSet { if selectedID != oldValue { stopPlayback() } } }
    @Published var profiles: [AudioProfile] = []
    @Published var isRefreshingModels = false
    @Published var modelCatalogError: String?
    /// Installs the receipt store does not recognise yet but whose identity
    /// matches a pinned profile (for example a music model pulled by the CLI,
    /// which writes no receipt). Adopting one verifies it and makes it usable.
    @Published var needsAdoption: [AudioLegacyInstall] = []
    /// Installs that exist on disk but cannot be used as they are, with the
    /// engine's reason. Shown so a broken install is visible, not just absent.
    @Published var incompatibleInstalls: [AudioIncompatibleInstall] = []
    /// Installs in a library whose drive is not connected (task 6.10). A state to
    /// show, not an error: they are not `incompatibleInstalls`, they are not in
    /// `installedPaths`, and none of them can be run.
    @Published var notConnectedInstalls: [AudioNotConnectedInstall] = []
    /// The verified installs with their sizes, and managed audio installs the
    /// catalog no longer lists, for Installed > Audio. The rows carry the library
    /// and the size a move needs (task 6.11); `installedPaths` is derived from
    /// the same rows.
    @Published var installedRecords: [AudioInstalledRecord] = []
    @Published var otherAudioInstalls: [AudioInstalledRecord] = []
    @Published var isManagingInstall = false
    @Published var familyCapabilities: [AudioFamilyCapability] = []
    @Published var search = ""
    @Published var isBusy = false
    @Published var isInstalling = false
    @Published var isRecording = false
    @Published var recordingSetupStatus: String?
    /// The page that started the current capture (SpeechToTextMode.swift).
    @Published var captureOrigin: AudioCaptureOrigin?
    var hasRecordingActivity: Bool { isRecording || recordingSetupStatus != nil }
    @Published var isPaused = false
    @Published var recordingSeconds: Double = 0
    @Published var inputLevel: Float = 0
    @Published var status = AppStrings.localized("Saved in this profile")
    @Published var error: String?
    @Published var progress: Double?
    @Published var generationProgress: AudioGenerationProgress?
    @Published var activeGeneration: AudioLibraryItem?
    /// Stages of the active speech run, for the Voice Cloning progress card.
    @Published var speechRun: AudioSpeechRunProgress?
    var activeRunID: UUID?
    @Published var transcriptBacklog = 0
    @Published var isPlaying = false
    @Published var isExporting = false
    @Published var playbackTime: Double = 0
    @Published var captureMicrophone = true
    @Published var captureSystemAudio = false
    @Published var selectedApplicationID: Int32?
    @Published var audioApplications: [AudioCaptureApplication] = []
    @Published var microphoneID: String?
    @Published var microphones: [AudioCaptureMicrophone] = []
    var allowPortableExperiment: Bool {
        get { recipe.experimentalBackend == "portable" }
        set { recipe.experimentalBackend = newValue ? "portable" : nil }
    }
    var allowExperimentalMetal: Bool {
        get { recipe.experimentalBackend == "experimental_metal" }
        set { recipe.experimentalBackend = newValue ? "experimental_metal" : nil }
    }
    @Published var liveTranscription = true
    /// Streaming live text for the take being recorded (AudioWorkspaceLive.swift).
    let live = AudioLiveText()
    /// Words, meaning and sound search over the library (AudioLibrarySearch.swift).
    let librarySearch = AudioLibrarySearch()
    @Published var pendingSaves: [UUID: AudioLibraryItem] = [:]
    @Published var pendingTranscriptEdits: [UUID: [AudioLibrarySegment]] = [:]
    @Published var preferences = AudioWorkspacePreferences()
    @Published var installedPaths: [String: String] = [:]
    private var cachedMusicSamplingPath: String?
    private var cachedMusicSampling: AudioMusicSamplingRecommendation?

    let pendingAssetOwner = UUID()
    let lifetime = AudioOperationLifetime()
    var summaryHandler: ((AudioLibraryItem, InstalledModel?) async throws -> AudioMeetingSummary)?
    /// Whether the model at a path is being moved to another library. The move removes
    /// the original folder when it ends, so a session is never opened from a path for
    /// which this says yes (`residentSession`, the Qwen speech job). Set by the owner.
    var modelMoving: ((String) -> Bool)?
    var summaryAvailable: (() -> Bool)?
    var availableChatModels: (() -> [InstalledModel])?
    var selectedChatModel: (() -> InstalledModel?)?
    var selectChatModel: ((InstalledModel) -> Void)?
    var canSummarizeMeeting: Bool {
        selectedItem?.preferredTranscript != nil &&
            (summaryAvailable?() == true || selectedChatModel?() != nil || !(availableChatModels?().isEmpty ?? true))
    }
    /// Queues a managed download. The second argument lists the ids of the
    /// profile's acknowledgments the user accepted; the third is the library it
    /// goes into (nil is This Mac).
    var installHandler: ((AudioProfile, [String], String?) -> Bool)?
    /// The default library for downloads, nil for This Mac (requirement 2.2), for
    /// installs that offer no destination menu. Set by the owner.
    var defaultInstallLibrary: (() -> String?)?
    /// How the catalog is read: the live engine, or a test's answer. It runs off
    /// the main actor.
    let catalogReader: @Sendable () throws -> AudioCatalogSnapshot
    /// The one door to the engine for an install that no queue owns.
    let installService: ModelLibraryService
    let library: AudioLibraryStore
    let assets: ManagedAssetStore
    var profileID: String?
    var epoch = UUID()
    var isShutDown = false
    var loading = true
    var modelsRefreshPending = false
    var jobTask: Task<Void, Never>?
    var preferencesTask: Task<Void, Never>?
    var transcriptSaveTasks: [UUID: Task<Void, Never>] = [:]
    var idleTask: Task<Void, Never>?
    var deletionCleanupTask: Task<Void, Never>?
    @Published var opening: AudioModelOpening?
    var qwenSpeechJob: Qwen3TTSSpeechJob?
    var qwenLoadDependencies = AudioQwenLoadDependencies()
    var referencePreviewTask: Task<Void, Never>?
    var referencePlayer: AVAudioPlayer?
    @Published var referencePlaying = false
    /// Told whenever the resident session is set or cleared, from any of the
    /// sites that do so, so the eject coordinator's snapshot of loaded models
    /// stays current without polling.
    var onResidentModelChanged: (() -> Void)?
    /// The Meeting take whose Name the Speakers sheet is due, and the voice
    /// whose sample plays there, by speaker key (AudioSpeakerLabels.swift).
    @Published var speakerNamingID: UUID?
    @Published var speakerSampleKey: String?
    @Published var session: AudioSession? { didSet { onResidentModelChanged?() } }
    @Published var qwenModel: Qwen3TTSModel?
    /// A resident FluidAudio recognizer (AudioWorkspaceFluid.swift). Released
    /// wherever the Rust session and Qwen model are.
    var fluidTranscriber: (any FluidAudioTranscriber)?
    @Published var loadedModelAlias: String?
    var sessionKey: String?
    /// The folder `session` was opened from. Only meaningful while `session`
    /// is set; read it through `residentModelPath`.
    var sessionPath: String?
    var residentModelPath: String? { session == nil ? nil : sessionPath }
    var modelAccessURL: URL?
    var capture: AudioCapture?
    var recordingSetupTask: Task<Void, Never>?
    var recordingSetupAttempt: UUID?
    var recordingSetupDependencies = AudioRecordingSetupDependencies()
    var isDraftJob = false
    var recordingID: UUID?
    /// The recording that becomes the cloning reference when it stops.
    @Published var voiceReferenceRecordingID: UUID?
    var pendingRefinementIDs: [UUID] = []
    var nextDraftAttemptAt = Date.distantPast
    var draftThrough: Double = 0
    var draftRequestedThrough: Double = 0
    var timer: Timer?
    var player: AVAudioPlayer?
    var playbackURL: URL?
    var playbackTask: Task<Void, Never>?
    var playbackLifetime: AudioOperationLifetime?
    var exportTask: Task<Void, Never>?
    var importTask: Task<Void, Never>?
    var scratchURLs: Set<URL> = []

    init(
        library: AudioLibraryStore = AudioLibraryStore(), assets: ManagedAssetStore = .shared,
        catalogReader: @escaping @Sendable () throws -> AudioCatalogSnapshot = AudioCatalogSnapshot.live,
        installService: ModelLibraryService = LiveModelLibraryService()
    ) {
        self.library = library; self.assets = assets
        self.catalogReader = catalogReader; self.installService = installService
        profileID = ProfileVaultStore.shared.session?.profileID
        do {
            preferences = try library.preferences()
            recipe = preferences.recipe; page = preferences.page
            restoreCaptureSources()
            // A saved page or task from a build that offered controls the
            // runtime cannot open would restore into a dead end.
            if !AudioWorkspacePage.tools.contains(page), page == .cleanup { page = .library }
            if AudioTask(rawValue: recipe.task)?.isRunnable == false {
                recipe.task = AudioTask.speechToText.rawValue
            }
            let loaded = try library.loadReporting()
            items = loaded.items
            if loaded.unreadable > 0 {
                status = AppStrings.localized("Recordings set aside because they could not be read: \(loaded.unreadable)")
            }
            try library.collectUnusedAssets()
            for index in items.indices where items[index].recoverInterrupted() { try library.save(items[index]) }
        } catch { self.error = error.localizedDescription }
        loading = false
        refreshModels()
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }
    var selectedItem: AudioLibraryItem? {
        if let activeGeneration, activeGeneration.id == selectedID { return activeGeneration }
        return items.first { $0.id == selectedID }
    }
    var task: AudioTask {
        switch page {
        case .record, .transcribe: return .speechToText
        case .voiceover, .voiceClone, .voiceDesign: return .textToSpeech
        case .music: return .music
        case .cleanup: return recipe.task == AudioTask.separation.rawValue ? .separation : .enhancement
        case .library, .advanced: return AudioTask(rawValue: recipe.task) ?? .speechToText
        }
    }
    var availableProfiles: [AudioProfile] {
        var choices = profiles.filter {
            $0.identity.task == task && speechProfileMatchesPage($0) && AudioRuntimeProvider($0) != .unsupported
        }
        if page == .advanced && task == .speechToText { choices.append(Self.localQwenProfile) }
        if task == .music { choices = Self.preferredMusicProfiles(choices) }
        return choices
    }
    static let defaultMusicProfileAlias = "minimax-music3-turbo-4bit-8step"
    static let localQwenProfile = AudioProfile(
        identity: AudioProfileIdentity(task: .speechToText, alias: "local-qwen3-asr", repository: "", revision: "", assetFingerprint: ""),
        family: "qwen3_asr", displayName: "Qwen3-ASR (local checkpoint)",
        capabilities: AudioCapabilities(operations: ["transcribe"], timing: "clip", backend: "experimental_metal_or_portable", cancellation: "decode_step", canRun: false,
            unavailableReason: "Choose a local checkpoint and an experimental backend in Advanced."),
        pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "implemented_unqualified")
    var selectedProfile: AudioProfile? { availableProfiles.first { $0.identity.alias == recipe.modelID } }
    var modelInstalled: Bool {
        guard let id = recipe.modelID else { return false }
        return installedPaths[id] != nil || preferences.localModels[id] != nil
    }
    /// Whether `profile` can be used, is installed on a drive that is away, or is
    /// not installed. The picker and the install button read this, so an offline
    /// install is shown as such and never offered a download.
    func installStatus(for profile: AudioProfile) -> AudioInstallStatus {
        AudioInstallStatus.resolve(
            alias: profile.identity.alias, installedPaths: installedPaths,
            localModels: Set(preferences.localModels.keys), notConnected: notConnectedInstalls)
    }
    var selectedLocalModelURL: URL? {
        guard let id = recipe.modelID, let bookmark = preferences.localModels[id] else { return nil }
        var stale = false
        return try? URL(resolvingBookmarkData: bookmark, options: [.withSecurityScope, .withoutUI, .withoutMounting], relativeTo: nil, bookmarkDataIsStale: &stale)
    }
    var selectedProfileIsRunnable: Bool {
        selectedProfile.map { AudioRuntimeProvider($0).canRun($0) } == true
            || (page == .advanced && allowPortableExperiment && ["kokoro", "omnivoice", "qwen3_asr"].contains(selectedProfile?.family ?? ""))
            || (page == .advanced && allowExperimentalMetal && selectedProfile?.family == "qwen3_asr")
    }
    var canRun: Bool {
        guard !isBusy, !isInstalling, !isShutDown, !hasRecordingActivity, importTask == nil, modelInstalled else { return false }
        guard selectedProfileIsRunnable else { return false }
        switch task {
        case .speechToText, .enhancement, .separation, .alignment, .diarization, .speechDetection, .codec, .languageIdentification:
            return selectedItem?.clips.isEmpty == false
        case .textToSpeech: return !recipe.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && speechValidationMessage == nil
        case .music:
            return !recipe.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && musicValidationMessage == nil
                && (recipe.effectiveMusicStepsOverride.map { (1...30).contains($0) } ?? true)
        }
    }
    var musicMemoryAdvice: MusicMemoryAdvisor.Profile {
        MusicMemoryAdvisor.evaluate(for: selectedProfile)
    }
    var musicSamplingRecommendation: AudioMusicSamplingRecommendation {
        let installed = selectedProfile.flatMap { installedPaths[$0.identity.alias] }
        let modelURL = selectedLocalModelURL ?? installed.map { URL(fileURLWithPath: $0) }
        guard let modelURL else { return .fast }
        if cachedMusicSamplingPath == modelURL.path, let cachedMusicSampling {
            return cachedMusicSampling
        }
        let scoped = modelURL.startAccessingSecurityScopedResource()
        defer { if scoped { modelURL.stopAccessingSecurityScopedResource() } }
        let recommendation = AudioMusicSamplingRecommendation.read(configAt: modelURL)
        cachedMusicSamplingPath = modelURL.path
        cachedMusicSampling = recommendation
        return recommendation
    }
    var musicValidationMessage: String? {
        guard task == .music else { return nil }
        return MusicMemoryAdvisor.validate(recipe: recipe, profile: selectedProfile)
    }
    func setMusicContextToMaxIfSupported() {
        let advice = musicMemoryAdvice
        if recipe.music == nil {
            recipe.music = AudioMusicSettings(
                contextSize: advice.maxModelContextTokens,
                maxFrames: advice.maxSupportedFrames,
                frameRate: advice.frameRate
            )
        } else {
            recipe.music?.contextSize = advice.maxModelContextTokens
            recipe.music?.maxFrames = advice.maxSupportedFrames
            recipe.music?.frameRate = advice.frameRate
        }
        recipe.durationSeconds = advice.maxSupportedSeconds
    }
    var presets: [AudioPreset] { preferences.presets }
    var playbackDuration: Double { player?.duration ?? selectedItem?.duration ?? 0 }
    var filteredItems: [AudioLibraryItem] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        return items.filter {
            !$0.isEmptyGeneratedAttempt && (query.isEmpty || $0.title.localizedCaseInsensitiveContains(query)
                || $0.collection.localizedCaseInsensitiveContains(query)
                || ($0.preferredTranscript?.segments.contains { $0.text.localizedCaseInsensitiveContains(query) } ?? false))
        }.sorted { $0.favorite == $1.favorite ? $0.createdAt > $1.createdAt : $0.favorite }
    }
    func valid(_ token: UUID) -> Bool {
        guard !isShutDown, epoch == token else { return false }
        if profileID == nil, let current = ProfileVaultStore.shared.session?.profileID {
            profileID = current
        }
        if let profileID {
            return ProfileVaultStore.shared.session?.profileID == profileID
        }
        return true
    }
    func tick() {
        if let capture, isRecording { recordingSeconds = capture.currentElapsed() }
        enforceVoiceReferenceRecordingLimit()
        // Assigning an unchanged value still publishes, which redrew every
        // view observing the controller four times a second.
        let playing = referencePlayer?.isPlaying == true
        if referencePlaying != playing { referencePlaying = playing }
        if let player { playbackTime = player.currentTime; isPlaying = player.isPlaying }
        if !isBusy { startNextDraft() }
    }
    func persistPreferences() {
        guard !loading, !isShutDown else { return }
        preferencesTask?.cancel()
        preferencesTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled, let self else { return }
            self.flushPreferences()
        }
    }
    func flushPreferences() {
        preferences.page = page; preferences.recipe = recipe
        if task == .textToSpeech, let speech = recipe.speech {
            if preferences.speechModes == nil { preferences.speechModes = [:] }
            preferences.speechModes?[speech.mode.rawValue] = speech
        }
        if let model = recipe.modelID { preferences.selectedModels[modelSelectionKey] = model }
        do {
            try library.savePreferences(preferences)
            if !isBusy && error == nil { status = AppStrings.localized("Saved in this profile") }
        }
        catch { fail(error) }
    }
    func selectPage(_ page: AudioWorkspacePage) {
        guard self.page != page else { return }
        flushPreferences()
        stopReferencePreview()
        let leaving = self.page
        self.page = page
        // Speech-to-Text and Record set sources differently; each gets its own back.
        if page == .transcribe || (page == .record && leaving == .transcribe) { restoreCaptureSources() }
        recipe.task = task.rawValue
        recipe.modelID = preferences.selectedModels[modelSelectionKey]
        if page != .advanced { recipe.experimentalBackend = nil }
        restoreModelSelection()
        normalizeModelOptions()
        if task == .textToSpeech, let mode = recipe.speech?.mode,
           let saved = preferences.speechModes?[mode.rawValue] { recipe.speech = saved }
    }
    func selectTask(_ value: String) {
        flushPreferences()
        recipe.task = value
        recipe.modelID = preferences.selectedModels[value]
        restoreModelSelection()
        normalizeModelOptions()
    }
    func restoreModelSelection() {
        // Catalog revisions can remove a saved choice. Keep the picker and
        // run request on the same task, with a usable fallback after refresh.
        if let id = recipe.modelID, availableProfiles.contains(where: { $0.identity.alias == id }) { return }
        if let remembered = preferences.selectedModels[modelSelectionKey],
           availableProfiles.contains(where: { $0.identity.alias == remembered }) {
            recipe.modelID = remembered
        } else if page == .transcribe {
            // A mode with no choice of its own starts from the shared one.
            recipe.modelID = SpeechToTextModels.resolve(
                speechToTextMode, .final, profiles: profiles, selected: preferences.selectedModels, isInstalled: isInstalled)?.identity.alias
        } else if task == .speechToText {
            // Catalog order lists Whisper first; the preference order puts the
            // most accurate runnable transcriber first. An installed runnable
            // model still wins over a better one that needs a download.
            let preferred = Self.preferredSpeechProfiles(availableProfiles)
            recipe.modelID = preferred.first(where: { AudioRuntimeProvider($0).canRun($0) && installedPaths[$0.identity.alias] != nil })?.identity.alias
                ?? preferred.first?.identity.alias
        } else if task == .music {
            // New music workspaces should download the pinned Turbo checkpoint
            // by default, even when another music profile is already installed.
            recipe.modelID = Self.preferredMusicProfiles(availableProfiles).first?.identity.alias
        } else {
            // A new speech workspace should start with an executable provider.
            let preferred = task == .textToSpeech
                ? availableProfiles.filter { AudioRuntimeProvider($0).canRun($0) }
                : availableProfiles
            recipe.modelID = preferred.first(where: { installedPaths[$0.identity.alias] != nil })?.identity.alias
                ?? preferred.first?.identity.alias ?? availableProfiles.first?.identity.alias
        }
    }
    static let speechPreference = ["parakeet-ultra-coreml", "parakeet-redux-coreml", "whisper-base"]
    static func preferredMusicProfiles(_ profiles: [AudioProfile]) -> [AudioProfile] {
        profiles.enumerated().sorted { left, right in
            let leftIsDefault = left.element.identity.alias == defaultMusicProfileAlias
            let rightIsDefault = right.element.identity.alias == defaultMusicProfileAlias
            return leftIsDefault != rightIsDefault ? leftIsDefault : left.offset < right.offset
        }.map(\.element)
    }
    /// Runnable before not runnable, then the single-language final-pass model
    /// for the UI's language (`languageCode`, nil for none), then
    /// `speechPreference` order, then catalog order. Runnable is the provider's
    /// verdict: a FluidAudio row's Rust `canRun` is false by design.
    static func preferredSpeechProfiles(_ profiles: [AudioProfile],
                                        languageCode: String? = AudioModelLanguage.uiLanguageCode) -> [AudioProfile] {
        let rank = { (profile: AudioProfile) in speechPreference.firstIndex(of: profile.identity.alias) ?? speechPreference.count }
        let runs = { (profile: AudioProfile) in AudioRuntimeProvider(profile).canRun(profile) }
        let dedicated = { (profile: AudioProfile) in AudioModelLanguage.isDedicated(profile, to: languageCode) }
        return profiles.enumerated().sorted { a, b in
            if runs(a.element) != runs(b.element) { return runs(a.element) }
            if dedicated(a.element) != dedicated(b.element) { return dedicated(a.element) }
            let (left, right) = (rank(a.element), rank(b.element))
            return left != right ? left < right : a.offset < b.offset
        }.map(\.element)
    }
    func selectModel(_ alias: String?) {
        recipe.modelID = alias
        normalizeModelOptions()
    }
    func normalizeModelOptions() {
        guard let profile = selectedProfile else { return }
        let capabilities = profile.capabilities
        if profile.family == "qwen3_tts" {
            var speech = recipe.speech ?? AudioSpeechSettings()
            speech.mode = Self.speechMode(for: profile)
            recipe.speech = speech
            if !speechSupportsStyle { recipe.speech?.instruction = "" }
        } else if Self.clonesNatively(profile) && page != .voiceover {
            // Keep the reference across model switches; it is the voice.
            // Voiceover runs the same model without one (`runClonesNatively`).
            var speech = recipe.speech ?? AudioSpeechSettings()
            speech.mode = .clone
            speech.instruction = ""
            if speech.cloneOptions == nil { speech.cloneOptions = AudioCloneOptions() }
            recipe.speech = speech
        } else {
            recipe.speech = nil
        }
        if task == .music || profile.identity.task == .music {
            let advice = musicMemoryAdvice
            if recipe.music == nil {
                recipe.music = AudioMusicSettings(
                    contextSize: capabilities.maxContextTokens ?? advice.maxModelContextTokens,
                    maxFrames: capabilities.maxAudioFrames ?? advice.maxSupportedFrames,
                    frameRate: Double(capabilities.frameRateHz ?? Int(advice.frameRate))
                )
                if advice.canSupportMaxContext && recipe.durationSeconds == 10 {
                    recipe.durationSeconds = advice.maxSupportedSeconds
                }
            } else {
                if let ctx = capabilities.maxContextTokens { recipe.music?.contextSize = ctx }
                if let frames = capabilities.maxAudioFrames { recipe.music?.maxFrames = frames }
                if let rate = capabilities.frameRateHz { recipe.music?.frameRate = Double(rate) }
            }
        }
        if !capabilities.languages.isEmpty, !capabilities.languages.contains(recipe.language) {
            recipe.language = task == .speechToText ? "auto" : capabilities.languages[0]
        }
        if !capabilities.voices.isEmpty, !capabilities.voices.contains(recipe.voice) {
            recipe.voice = capabilities.voices[0]
        }
    }
    func useDownloadedModel() {
        guard !isBusy, !hasRecordingActivity, let id = recipe.modelID,
              installedPaths[id] != nil else { return }
        preferences.localModels.removeValue(forKey: id)
        flushPreferences()
    }
    func select(_ item: AudioLibraryItem) {
        guard item.id != selectedID else { return }
        stopPlayback(); selectedID = item.id
        if page == .library { recipe = item.recipe }
    }
    func publish(_ item: AudioLibraryItem) {
        if let index = items.firstIndex(where: { $0.id == item.id }) { items[index] = item }
        else { items.insert(item, at: 0) }
        if !isBusy { status = AppStrings.localized("Saved in this profile") }
    }
    func refreshPendingPins() {
        ManagedAssetPins.set(Set(pendingSaves.values.flatMap(\.assetReferences).compactMap(ManagedAssetStore.assetID(from:))), owner: pendingAssetOwner)
    }
    func save(_ item: AudioLibraryItem) throws {
        do {
            try library.save(item); pendingSaves.removeValue(forKey: item.id); refreshPendingPins(); publish(item)
            if activeGeneration?.id == item.id { activeGeneration = nil }
        }
        catch { pendingSaves[item.id] = item; refreshPendingPins(); throw error }
    }
    @discardableResult func persistMutation(_ id: UUID, _ mutation: (inout AudioLibraryItem) -> Void) throws -> AudioLibraryItem {
        if var pending = pendingSaves[id] {
            mutation(&pending); try save(pending); return pending
        }
        if var active = activeGeneration, active.id == id {
            mutation(&active); try save(active); return active
        }
        var candidate: AudioLibraryItem?
        do {
            let saved = try library.update(id) { mutation(&$0); candidate = $0 }
            pendingSaves.removeValue(forKey: id); refreshPendingPins(); publish(saved)
            return saved
        } catch {
            if let candidate { pendingSaves[id] = candidate; refreshPendingPins() }
            throw error
        }
    }
    func retryAutosave() {
        for item in Array(pendingSaves.values) {
            do { try save(item) } catch { fail(error); return }
        }
        for (id, segments) in pendingTranscriptEdits { _ = saveTranscript(segments, for: id) }
        flushPreferences()
        if pendingSaves.isEmpty && pendingTranscriptEdits.isEmpty { error = nil }
    }
    func fail(_ failure: Error) {
        error = failure.localizedDescription
        status = AppStrings.localized("Not saved. Retry after checking available storage.")
    }
    func update(_ id: UUID, _ mutation: (inout AudioLibraryItem) -> Void) {
        do { try persistMutation(id, mutation) } catch { fail(error) }
    }
    func toggleFavorite(_ item: AudioLibraryItem) { update(item.id) { $0.favorite.toggle() } }
    func rename(_ item: AudioLibraryItem, title: String) {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { return }
        update(item.id) { $0.title = title; $0.updatedAt = Date() }
    }
    func organize(_ item: AudioLibraryItem, collection: String) { update(item.id) { $0.collection = collection } }
    func canDelete(_ item: AudioLibraryItem) -> Bool {
        !isShutDown && !hasRecordingActivity && !isExporting && item.id != activeRunID
            && item.id != recordingID && !(isBusy && item.status == .processing)
    }
    func delete(_ item: AudioLibraryItem) {
        // Recheck at confirmation time: the item may have started processing.
        guard canDelete(item) else { return }
        do {
            try library.delete(item.id); items.removeAll { $0.id == item.id }
            pendingSaves.removeValue(forKey: item.id); pendingTranscriptEdits.removeValue(forKey: item.id); refreshPendingPins()
            transcriptSaveTasks.removeValue(forKey: item.id)?.cancel()
            if selectedID == item.id { stopPlayback(); selectedID = nil }
            deletionCleanupTask?.cancel()
            deletionCleanupTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(61))
                guard !Task.isCancelled, let self, !self.isShutDown else { return }
                do { try self.library.collectUnusedAssets() } catch { self.fail(error) }
            }
        } catch { fail(error) }
    }
    func duplicateSelected() {
        guard var copy = selectedItem else { return }
        copy.sourceID = copy.id; copy.id = UUID(); copy.createdAt = Date(); copy.updatedAt = Date()
        copy.title += AppStrings.localized(" copy"); copy.recipe = recipe
        copy.status = .draft; copy.failure = nil
        do { try save(copy); select(copy) } catch { fail(error) }
    }
    func saveDraft() {
        var draft = AudioLibraryItem(title: AppStrings.localized("Untitled audio project"), kind: kind)
        draft.status = .draft; draft.recipe = recipe; draft.sourceID = runSource?.id
        do { try save(draft); select(draft) } catch { fail(error) }
    }
    var kind: AudioLibraryItem.Kind {
        switch task {
        case .speechToText: return .transcription
        case .textToSpeech: return .voiceover
        case .music: return .music
        case .enhancement: return .enhancement
        case .separation: return .separation
        default: return .experiment
        }
    }
    func transcriptDraft(for id: UUID) -> [AudioLibrarySegment]? { pendingTranscriptEdits[id] }
    func stageTranscriptEdits(_ segments: [AudioLibrarySegment], for id: UUID) {
        guard !isShutDown else { return }
        pendingTranscriptEdits[id] = segments
        transcriptSaveTasks[id]?.cancel()
        let token = epoch
        transcriptSaveTasks[id] = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled, let self, self.valid(token), let latest = self.pendingTranscriptEdits[id] else { return }
            _ = self.saveTranscript(latest, for: id)
        }
    }
    @discardableResult func saveTranscript(_ segments: [AudioLibrarySegment]) -> Bool {
        guard let id = selectedID else { return false }; return saveTranscript(segments, for: id)
    }
    @discardableResult func saveTranscript(_ segments: [AudioLibrarySegment], for itemID: UUID) -> Bool {
        guard !isShutDown else { return false }
        do {
            publish(try library.update(itemID) { item in
                guard item.preferredTranscript?.segments != segments else { return }
                item.addTranscript(segments, source: .user, language: item.preferredTranscript?.language)
            })
            pendingTranscriptEdits.removeValue(forKey: itemID)
            transcriptSaveTasks.removeValue(forKey: itemID)?.cancel()
            return true
        } catch { pendingTranscriptEdits[itemID] = segments; fail(error); return false }
    }
    @discardableResult func saveSummaryText(_ text: String, for itemID: UUID) -> Bool {
        guard !isShutDown else { return false }
        do {
            try persistMutation(itemID) { current in
                if current.summary != nil {
                    current.summary?.text = text
                } else if let rev = current.preferredTranscript?.id {
                    current.summary = AudioMeetingSummary(text: text, sourceTimes: [], transcriptRevisionID: rev)
                }
                current.updatedAt = Date()
            }
            return true
        } catch {
            fail(error)
            return false
        }
    }
    func savePreset(name: String) {
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        preferences.presets.append(AudioPreset(name: name, recipe: recipe)); flushPreferences()
    }
    func applyPreset(_ preset: AudioPreset) {
        restoreRecipe(preset.recipe)
    }
    func restoreRecipe(_ value: AudioRecipe) {
        recipe = value
        if recipe.experimentalBackend != nil { page = .advanced; return }
        switch AudioTask(rawValue: recipe.task) {
        case .textToSpeech: page = value.speech?.mode == .clone ? .voiceClone : value.speech?.mode == .design ? .voiceDesign : .voiceover
        case .music: page = .music
        case .speechToText: page = .transcribe
        case .enhancement, .separation: page = .cleanup
        default: page = .advanced
        }
    }
    func runAgain(_ item: AudioLibraryItem) {
        guard !isBusy, !hasRecordingActivity else { return }
        select(item)
        restoreRecipe(item.recipe)
        do {
            if let profile = selectedProfile, item.modelFamily != nil {
                preferences.localModels[profile.identity.alias] = try modelBookmark(for: profile, source: item)
            }
        } catch { fail(error); return }
        run()
    }
    func modelBookmark(for profile: AudioProfile, source: AudioLibraryItem? = nil) throws -> Data? {
        guard let source else { return preferences.localModels[profile.identity.alias] }
        if let family = source.modelFamily, family != profile.family {
            throw AudioWorkspaceError.message(AppStrings.localized("The saved model is unavailable. Choose a model and create a new take."))
        }
        if let identity = source.modelIdentity,
           try JSONDecoder().decode(AudioProfileIdentity.self, from: identity) != profile.identity {
            throw AudioWorkspaceError.message(AppStrings.localized("The saved model is unavailable. Choose a model and create a new take."))
        }
        return source.localModelBookmark
    }
    func refreshModels() {
        guard !isShutDown else { return }
        if isRefreshingModels {
            // A completed install may request a refresh while an older scan
            // is still reading receipts. Do not lose that newer snapshot.
            modelsRefreshPending = true
            return
        }
        isRefreshingModels = true
        modelCatalogError = nil
        let token = epoch
        let reader = catalogReader
        Task { [weak self] in
            let result = await Task.detached { Result { try reader() } }.value
            guard let self else { return }
            defer {
                self.isRefreshingModels = false
                if self.modelsRefreshPending {
                    self.modelsRefreshPending = false
                    self.refreshModels()
                }
            }
            guard !self.isShutDown, self.epoch == token else { return }
            switch result {
            case .success(let snapshot):
                let installed = snapshot.installed
                self.familyCapabilities = snapshot.capabilities
                self.profiles = snapshot.profiles
                self.installedPaths = Dictionary(installed.installed.map { ($0.alias, $0.path) }, uniquingKeysWith: { first, _ in first })
                self.installedRecords = installed.installed
                self.needsAdoption = installed.needsAdoption
                self.incompatibleInstalls = installed.incompatible
                // Installs on a drive that is away: kept as their own state.
                self.notConnectedInstalls = installed.notConnected
                self.otherAudioInstalls = installed.otherAudio
                self.restoreModelSelection()
                self.normalizeModelOptions()
            case .failure(let error): self.modelCatalogError = error.localizedDescription
            }
        }
    }
    func chooseModelFolder() {
        if let profile = selectedProfile { chooseModelFolder(for: profile) }
    }
    /// Uses a checkpoint folder for `profile` instead of its managed install.
    func chooseModelFolder(for profile: AudioProfile) {
        guard AudioRuntimeProvider(profile).acceptsLocalFolder, !isBusy, !hasRecordingActivity else { return }
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = AppStrings.localized("Choose a checkpoint folder for the selected audio model.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            preferences.localModels[profile.identity.alias] = try url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
            flushPreferences()
        } catch { fail(error) }
    }
    /// `acknowledged` lists the accepted ids of the profile's
    /// `requiredAcknowledgments`; the native installer refuses a missing one.
    /// `library` is where the model goes (nil is This Mac).
    func installSelectedModel(acknowledged: [String] = [], library: String? = nil) {
        if let profile = selectedProfile { install(profile, acknowledged: acknowledged, library: library) }
    }
    func install(_ profile: AudioProfile, acknowledged: [String] = [], library: String? = nil) {
        guard !profile.identity.repository.isEmpty, !isBusy else { return }
        // An install on a drive that is away is not a download to start: the
        // engine would refuse it as `alias_elsewhere`, and the picker offers the
        // statement instead of the button.
        if case .notConnected = installStatus(for: profile) { return }
        if let installHandler { _ = installHandler(profile, acknowledged, library); return }
        guard !isInstalling else { return }
        isInstalling = true; error = nil
        status = AppStrings.localized("Downloading audio model")
        let token = epoch
        let service = installService
        Task { [weak self] in
            let result = await Task.detached {
                Result {
                    try service.installAudio(
                        profile.identity, acknowledged: acknowledged,
                        library: AppModel.destinationLibrary(library), progress: { _ in })
                }
            }.value
            guard let self else { return }
            defer { self.isInstalling = false }
            guard self.valid(token) else { return }
            switch result {
            case .success(let record):
                self.installedPaths[record.alias] = record.path
                self.installedRecords.removeAll { $0.alias == record.alias }
                self.installedRecords.append(record)
                self.status = AppStrings.localized("Audio model installed")
            case .failure(let error): self.error = error.localizedDescription
            }
        }
    }
    func residentSession(for profile: AudioProfile, portable: Bool = false, experimentalMetal: Bool = false, source: AudioLibraryItem? = nil) async throws -> AudioSession {
        idleTask?.cancel()
        var url: URL?
        let path: String
        if let bookmark = try modelBookmark(for: profile, source: source) {
            var stale = false
            let resolved = try URL(resolvingBookmarkData: bookmark, options: .withSecurityScope, relativeTo: nil, bookmarkDataIsStale: &stale)
            guard !stale else { throw AudioWorkspaceError.message(AppStrings.localized("The model folder moved. Choose it again.")) }
            _ = resolved.startAccessingSecurityScopedResource(); url = resolved; path = resolved.path
        } else if let installed = installedPaths[profile.identity.alias] { path = installed }
        else { throw AudioWorkspaceError.message(AppStrings.localized("Download this model or choose its checkpoint folder.")) }
        // **NOT A MODEL THAT IS BEING MOVED**: the engine removes the folder it is
        // opened from when the move ends. Checked on the resolved path, so a local
        // checkpoint bookmark that points into a moving install is covered too.
        if modelMoving?(path) == true {
            url?.stopAccessingSecurityScopedResource()
            throw AudioWorkspaceError.message(ModelMoveText.modelBeingMoved())
        }
        let key = "\(profile.family):\(profile.identity.task.rawValue):\(path):\(portable):\(experimentalMetal)"
        if sessionKey == key, let session { url?.stopAccessingSecurityScopedResource(); return session }
        if let previous = session { await Task.detached { previous.close() }.value }
        modelAccessURL?.stopAccessingSecurityScopedResource(); modelAccessURL = nil
        session = nil; sessionKey = nil
        if qwenModel != nil {
            qwenModel = nil
            MLX.Memory.clearCache()
        }
        fluidTranscriber = nil
        loadedModelAlias = nil
        let token = epoch
        do {
            let opening = AudioModelOpening()
            opening.onProgress = { [weak self, weak opening] fraction in
                Task { @MainActor in
                    guard let self, let opening, self.valid(token), self.opening === opening else { return }
                    self.status = AppStrings.localized("Verifying")
                    self.progress = fraction
                }
            }
            self.opening = opening
            let opened = try await Task.detached { try opening.open(path: path, task: profile.identity.task, portable: portable, experimentalMetal: experimentalMetal, expectedFamily: profile.family) }.value
            self.opening = nil
            self.progress = nil
            guard valid(token), !Task.isCancelled else {
                await Task.detached { opened.close() }.value
                throw CancellationError()
            }
            sessionPath = path
            session = opened; sessionKey = key; loadedModelAlias = profile.identity.alias; modelAccessURL = url
            return opened
        } catch { opening = nil; progress = nil; url?.stopAccessingSecurityScopedResource(); throw error }
    }

    func residentQwenSession(for profile: AudioProfile, source: AudioLibraryItem? = nil) async throws -> Qwen3TTSModel {
        idleTask?.cancel()
        var url: URL?
        let directory: URL
        let path: String
        if let bookmark = try modelBookmark(for: profile, source: source) {
            var stale = false
            let resolved = try URL(resolvingBookmarkData: bookmark, options: .withSecurityScope, relativeTo: nil, bookmarkDataIsStale: &stale)
            guard !stale else { throw AudioWorkspaceError.message(AppStrings.localized("The model folder moved. Choose it again.")) }
            _ = resolved.startAccessingSecurityScopedResource(); url = resolved; directory = resolved; path = resolved.path
        } else if let installed = installedPaths[profile.identity.alias] {
            path = installed
            directory = URL(fileURLWithPath: installed, isDirectory: true)
        } else {
            throw AudioWorkspaceError.message(AppStrings.localized("Download this model or choose its checkpoint folder."))
        }
        let key = "qwen3_tts:\(path)"
        if sessionKey == key, let qwenModel {
            url?.stopAccessingSecurityScopedResource()
            return qwenModel
        }
        if let previous = session { await Task.detached { previous.close() }.value }
        session = nil
        modelAccessURL?.stopAccessingSecurityScopedResource(); modelAccessURL = nil
        qwenModel = nil; fluidTranscriber = nil; sessionKey = nil; loadedModelAlias = nil
        MLX.Memory.clearCache()

        let token = epoch
        do {
            let weightBytes = try [directory, directory.appendingPathComponent("speech_tokenizer")].reduce(UInt64(0)) { sum, folder in
                guard FileManager.default.fileExists(atPath: folder.path) else { return sum }
                return try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.fileSizeKey])
                    .filter { $0.pathExtension == "safetensors" }.reduce(sum) { try $0 + UInt64($1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) }
            }
            let budget = min(ProcessInfo.processInfo.physicalMemory * 3 / 4, MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
            guard weightBytes > 0, weightBytes <= budget / 2 else {
                throw AudioWorkspaceError.message(AppStrings.localized("This checkpoint exceeds the available audio model memory budget."))
            }

            let opening = AudioModelOpening(token: qwenLoadDependencies.openToken())
            opening.onProgress = { [weak self, weak opening] fraction in
                Task { @MainActor in
                    guard let self, let opening, self.valid(token), self.opening === opening else { return }
                    self.status = AppStrings.localized("Verifying")
                    self.progress = fraction
                }
            }
            self.opening = opening

            // The verification runs in the opening's place, and so does its
            // absence for a chosen folder: either way the barrier ends, so a
            // shutdown drain never waits for an open that never comes.
            let verifies = url == nil
            let verify = qwenLoadDependencies.verify
            let verified = try await Task.detached {
                try opening.perform { () throws -> String in
                    guard verifies else { return path }
                    // A managed install loads only after its bytes verify, and
                    // a failed token creation leaves nothing to verify with.
                    // Refusing here, inside perform, still ends the barrier.
                    guard let openToken = opening.token else { throw CancellationError() }
                    return try verify(profile.identity, openToken) { progress in
                        guard let total = progress.total, total > 0 else { return }
                        opening.onProgress?(min(1, Double(progress.completed) / Double(total)))
                    }
                }
            }.value
            guard verified == path else { throw CancellationError() }

            self.status = AppStrings.localized("Loading model")
            let load = qwenLoadDependencies.load
            let loaded = try await Task.detached {
                try Task.checkCancellation()
                return try await load(directory)
            }.value

            self.opening = nil
            self.progress = nil
            guard valid(token), !Task.isCancelled else {
                throw CancellationError()
            }
            qwenModel = loaded
            sessionKey = key
            loadedModelAlias = profile.identity.alias
            modelAccessURL = url
            return loaded
        } catch {
            opening = nil
            progress = nil
            url?.stopAccessingSecurityScopedResource()
            throw error
        }
    }

    func isProfileLoaded(_ profile: AudioProfile) -> Bool {
        if let alias = loadedModelAlias {
            return alias == profile.identity.alias
        }
        return session != nil && selectedProfile?.identity.alias == profile.identity.alias
    }

    var isModelLoaded: Bool {
        guard let profile = selectedProfile else { return false }
        return isProfileLoaded(profile)
    }
    var isModelLoading: Bool { opening != nil }

    var canLoadModel: Bool {
        modelInstalled && selectedProfileIsRunnable && !isModelLoaded && opening == nil && !isBusy && !isInstalling
    }

    /// Whether `profile` can be loaded now, outside the page's own picker.
    /// A streaming-only model opens per take and is never held loaded.
    func canLoad(_ profile: AudioProfile) -> Bool {
        let streamingOnly = profile.identity.task == .speechToText && !profile.capabilities.operations.contains("transcribe")
        return isInstalled(profile) && AudioRuntimeProvider(profile).canRun(profile) && profile.identity.task.isRunnable
            && !streamingOnly && !isProfileLoaded(profile) && opening == nil && !isBusy && !isInstalling
    }

    var canUnloadModel: Bool {
        (session != nil || qwenModel != nil || loadedModelAlias != nil) && !isBusy && !isRecording
    }

    func loadModel() {
        guard let profile = selectedProfile, canLoadModel else { return }
        load(profile, portable: allowPortableExperiment, experimentalMetal: allowExperimentalMetal)
    }

    /// Makes `profile` the one resident audio model, replacing any other.
    func load(_ profile: AudioProfile) {
        guard canLoad(profile) else { return }
        load(profile, portable: false, experimentalMetal: false)
    }

    private func load(_ profile: AudioProfile, portable: Bool, experimentalMetal: Bool) {
        let token = epoch
        Task { [weak self] in
            guard let self, self.valid(token) else { return }
            do {
                switch AudioRuntimeProvider(profile) {
                case .qwenMLX:
                    _ = try await self.residentQwenSession(for: profile)
                case .fluidAudio(let family):
                    _ = try await self.residentFluidTranscriber(for: profile, family: family)
                case .rust, .unsupported:
                    _ = try await self.residentSession(for: profile, portable: portable, experimentalMetal: experimentalMetal)
                }
                self.status = AppStrings.localized("Loaded \(profile.displayName)")
            } catch {
                if !Task.isCancelled {
                    self.error = error.localizedDescription
                }
            }
        }
    }

    func unloadModel() {
        guard canUnloadModel else { return }
        idleTask?.cancel()
        let old = session
        let oldAccess = modelAccessURL
        session = nil
        sessionKey = nil
        modelAccessURL = nil
        let oldQwen = qwenModel
        qwenModel = nil
        fluidTranscriber = nil
        loadedModelAlias = nil
        Task {
            if let old { await Task.detached { old.close() }.value }
            oldAccess?.stopAccessingSecurityScopedResource()
            if oldQwen != nil {
                MLX.Memory.clearCache()
            }
        }
        status = AppStrings.localized("Model unloaded")
    }

    func scheduleUnload() {
        idleTask?.cancel()
        idleTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(300))
            guard !Task.isCancelled, let self, !self.isBusy, !self.isRecording else { return }
            let old = self.session; let oldAccess = self.modelAccessURL
            let oldQwen = self.qwenModel
            self.session = nil; self.sessionKey = nil; self.modelAccessURL = nil
            self.qwenModel = nil
            self.fluidTranscriber = nil
            self.loadedModelAlias = nil
            if let old { await Task.detached { old.close() }.value }
            oldAccess?.stopAccessingSecurityScopedResource()
            if oldQwen != nil {
                MLX.Memory.clearCache()
            }
        }
    }
    func cancel() { jobTask?.cancel(); qwenSpeechJob?.cancel(); session?.cancel(); opening?.cancelOpen() }
    /// Takes the resident session out of this controller NOW, whatever is
    /// running on it: the eject of its drive is under way (requirement 6.1).
    /// The caller has already called `cancel()`. The same release
    /// `deleteSelectedModel` and the idle timer do, minus their guards. The
    /// returned closer closes it off the main actor (`close()` cancels and
    /// drains) and then ends the security-scoped access.
    ///
    /// **THE CLOSE IS SEPARATE BECAUSE IT CAN BLOCK WITHOUT LIMIT.**
    /// `AudioSession.close()` waits on the session's operation lock until the
    /// native job returns, so the caller bounds its wait and treats a close
    /// that is still running as the model still being held. Reads nothing from
    /// the drive.
    func detachResidentSession() -> DetachedSession? {
        idleTask?.cancel()
        let old = session; let oldAccess = modelAccessURL
        session = nil; sessionKey = nil; modelAccessURL = nil
        guard let old else {
            oldAccess?.stopAccessingSecurityScopedResource()
            return nil
        }
        return DetachedSession {
            await Task.detached { old.close() }.value
            oldAccess?.stopAccessingSecurityScopedResource()
        }
    }
    func shutdown() {
        guard !isShutDown else { return }
        preferencesTask?.cancel()
        transcriptSaveTasks.values.forEach { $0.cancel() }; transcriptSaveTasks.removeAll()
        retryAutosave()
        recordingSetupTask?.cancel(); recordingSetupTask = nil
        recordingSetupAttempt = nil; recordingSetupStatus = nil
        // Drain durable capture before the vault gate invalidates its keys.
        if let capture { _ = capture.stopAcceptingAndDrain(); Task { _ = await capture.stop() } }
        live.transcriber?.stopAndDrain(); live.clear(); librarySearch.stopIndexing()
        if let id = recordingID { update(id) { $0.status = .interrupted; $0.updatedAt = Date() } }
        lifetime.end()
        isShutDown = true; epoch = UUID()
        timer?.invalidate(); timer = nil; idleTask?.cancel(); deletionCleanupTask?.cancel(); cancel(); importTask?.cancel(); exportTask?.cancel()
        capture = nil; recordingID = nil; pendingRefinementIDs.removeAll(); isRecording = false
        stopPlayback(); stopReferencePreview()
        qwenSpeechJob?.stopAndDrain(); qwenSpeechJob = nil
        if let opening { DispatchQueue.global(qos: .userInitiated).sync { opening.stopAndDrain() } }
        opening = nil
        if let old = session { DispatchQueue.global(qos: .userInitiated).sync { old.close() } }
        session = nil; sessionKey = nil; modelAccessURL?.stopAccessingSecurityScopedResource(); modelAccessURL = nil
        qwenModel = nil; fluidTranscriber = nil; loadedModelAlias = nil
        MLX.Memory.clearCache()
        for url in scratchURLs { try? FileManager.default.removeItem(at: url) }
        scratchURLs.removeAll(); items.removeAll(); activeGeneration = nil; activeRunID = nil; generationProgress = nil
        speechRun = nil; voiceReferenceRecordingID = nil
        pendingTranscriptEdits.removeAll(); pendingSaves.removeAll(); ManagedAssetPins.clear(owner: pendingAssetOwner); recipe = AudioRecipe(); preferences = AudioWorkspacePreferences()
    }
}

/// The two slow steps of a resident Qwen3-TTS load and the token that stops
/// its verification. Tests replace them to hold a load in either step, or to
/// load without a token.
struct AudioQwenLoadDependencies {
    var openToken: @Sendable () -> AudioOpenToken? = { try? AudioOpenToken() }
    var verify: @Sendable (AudioProfileIdentity, AudioOpenToken, @escaping @Sendable (AudioProgress) -> Void) throws -> String = {
        try AudioCatalog.verify($0, token: $1, progress: $2)
    }
    var load: @Sendable (URL) async throws -> Qwen3TTSModel = { try await Qwen3TTSModel.fromModelDirectory($0) }
}

enum AudioWorkspaceError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let text) = self { return text }; return nil }
}

/// Model construction cannot call back to MainActor: profile teardown waits on
/// this barrier before releasing old profile state, even while a load is pending.
final class AudioModelOpening: @unchecked Sendable {
    private let lock = NSLock()
    private let finished = DispatchGroup()
    private var stopped = false
    private var opened: AudioSession?
    /// Stops the byte verification an open of a managed model starts with. A
    /// multi-gigabyte model used to make Stop and shutdown wait it out.
    let token: AudioOpenToken?
    /// Fraction (0...1) of the verification done, called on the opening thread.
    var onProgress: (@Sendable (Double) -> Void)?
    init(token: AudioOpenToken?) { self.token = token; finished.enter() }
    convenience init() { self.init(token: try? AudioOpenToken()) }
    /// Stops an open in flight. Safe from any thread; does not wait.
    func cancelOpen() { token?.cancel() }
    func open(path: String, task: AudioTask, portable: Bool, experimentalMetal: Bool, expectedFamily: String) throws -> AudioSession {
        defer { finished.leave() }
        lock.lock(); let cancelled = stopped; lock.unlock()
        guard !cancelled else { throw CancellationError() }
        let report = onProgress
        let session: AudioSession
        do {
            session = try AudioSession(
                modelPath: path, task: task, allowPortable: portable,
                allowExperimentalMetal: experimentalMetal, expectedFamily: expectedFamily,
                openToken: token,
                onProgress: { update in
                    guard let total = update.total, total > 0 else { return }
                    report?(min(1, Double(update.completed) / Double(total)))
                })
        } catch let error as TurboSparkError where error.code == .cancelled {
            // The caller's own stop, which every caller of this already treats
            // as a cancellation rather than a failure to show.
            throw CancellationError()
        }
        lock.lock(); let close = stopped
        if !close { opened = session }
        lock.unlock()
        if close { session.close(); throw CancellationError() }
        return session
    }
    /// Runs work that is not a Rust session open, such as a Swift-hosted
    /// model's verification, in this opening's place: Stop cancels it through
    /// `token`, and a drain waits for it. Without this a drain would wait for
    /// an open that never comes. Call it once, as `open` is.
    func perform<T>(_ work: () throws -> T) throws -> T {
        defer { finished.leave() }
        lock.lock(); let cancelled = stopped; lock.unlock()
        guard !cancelled else { throw CancellationError() }
        return try work()
    }
    func stopAndDrain() {
        lock.lock(); stopped = true; lock.unlock()
        // Before waiting: an open inside its verification now ends within a
        // chunk of hashing instead of after the whole model.
        token?.cancel()
        finished.wait()
        lock.lock(); let session = opened; opened = nil; lock.unlock()
        session?.close()
    }
}
