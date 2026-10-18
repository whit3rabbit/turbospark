import AppKit
import AVFoundation
import Combine
import Foundation
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
    @Published var isManagingInstall = false
    @Published var familyCapabilities: [AudioFamilyCapability] = []
    @Published var search = ""
    @Published var isBusy = false
    @Published var isInstalling = false
    @Published var isRecording = false
    @Published var recordingSetupStatus: String?
    var hasRecordingActivity: Bool { isRecording || recordingSetupStatus != nil }
    @Published var isPaused = false
    @Published var recordingSeconds: Double = 0
    @Published var inputLevel: Float = 0
    @Published var status = AppStrings.localized("Saved in this profile")
    @Published var error: String?
    @Published var progress: Double?
    @Published var generationProgress: AudioGenerationProgress?
    @Published var activeGeneration: AudioLibraryItem?
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
    @Published var pendingSaves: [UUID: AudioLibraryItem] = [:]
    @Published var pendingTranscriptEdits: [UUID: [AudioLibrarySegment]] = [:]
    @Published var preferences = AudioWorkspacePreferences()
    @Published var installedPaths: [String: String] = [:]
    /// The receipt-backed installs the report listed, by alias: the library and the
    /// size a move needs (task 6.11). `installedPaths` is derived from the same rows.
    @Published var installedRecords: [String: AudioInstalledRecord] = [:]

    let pendingAssetOwner = UUID()
    let lifetime = AudioOperationLifetime()
    var summaryHandler: ((AudioLibraryItem) async throws -> AudioMeetingSummary)?
    /// Whether the model at a path is being moved to another library. The move removes
    /// the original folder when it ends, so a session is never opened from a path for
    /// which this says yes (`residentSession`, the Qwen speech job). Set by the owner.
    var modelMoving: ((String) -> Bool)?
    var summaryAvailable: (() -> Bool)?
    var canSummarizeMeeting: Bool { selectedItem?.preferredTranscript != nil && summaryAvailable?() == true }
    /// Queues a managed download. The second argument lists the ids of the
    /// profile's acknowledgments the user accepted; the third is the library it
    /// goes into (nil is This Mac).
    var installHandler: ((AudioProfile, [String], String?) -> Bool)?
    /// How the catalog is read: the live engine, or a test's answer. It runs off
    /// the main actor.
    let catalogReader: @Sendable () throws -> AudioCatalogSnapshot
    /// The one door to the engine for an install that no queue owns.
    let installService: ModelLibraryService
    let library: AudioLibraryStore
    let assets: ManagedAssetStore
    let profileID: String?
    var epoch = UUID()
    var isShutDown = false
    var loading = true
    var modelsRefreshPending = false
    var jobTask: Task<Void, Never>?
    var preferencesTask: Task<Void, Never>?
    var transcriptSaveTasks: [UUID: Task<Void, Never>] = [:]
    var idleTask: Task<Void, Never>?
    var deletionCleanupTask: Task<Void, Never>?
    var opening: AudioModelOpening?
    var qwenSpeechJob: Qwen3TTSSpeechJob?
    var referencePreviewTask: Task<Void, Never>?
    var referencePlayer: AVAudioPlayer?
    @Published var referencePlaying = false
    /// Told whenever the resident session is set or cleared, from any of the
    /// sites that do so, so the eject coordinator's snapshot of loaded models
    /// stays current without polling.
    var onResidentModelChanged: (() -> Void)?
    var session: AudioSession? { didSet { onResidentModelChanged?() } }
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
            // A saved page or task from a build that offered controls the
            // runtime cannot open would restore into a dead end.
            if !AudioWorkspacePage.tools.contains(page), page == .cleanup { page = .library }
            if AudioTask(rawValue: recipe.task)?.isRunnable == false {
                recipe.task = AudioTask.speechToText.rawValue
            }
            items = try library.load()
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
        var choices = profiles.filter { $0.identity.task == task && speechProfileMatchesPage($0) }
        if page == .advanced && task == .speechToText { choices.append(Self.localQwenProfile) }
        return choices
    }
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
        selectedProfile?.family == "qwen3_tts" || selectedProfile?.capabilities.canRun == true
            || (page == .advanced && allowPortableExperiment && ["kokoro", "qwen3_asr"].contains(selectedProfile?.family ?? ""))
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
                && recipe.durationSeconds.isFinite && recipe.durationSeconds > 0
                && recipe.durationSeconds <= Double(selectedProfile?.capabilities.maxInputSeconds ?? 360)
                && (1...30).contains(recipe.steps)
        }
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
        !isShutDown && epoch == token && ProfileVaultStore.shared.session?.profileID == profileID
    }
    func tick() {
        if let capture, isRecording { recordingSeconds = capture.currentElapsed() }
        referencePlaying = referencePlayer?.isPlaying == true
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
        self.page = page
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
        } else {
            // A new speech workspace should start with an executable provider.
            let preferred = task == .textToSpeech
                ? availableProfiles.filter { $0.capabilities.canRun || $0.family == "qwen3_tts" }
                : availableProfiles
            recipe.modelID = preferred.first(where: { installedPaths[$0.identity.alias] != nil })?.identity.alias
                ?? preferred.first?.identity.alias ?? availableProfiles.first?.identity.alias
        }
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
        } else if Self.clonesNatively(profile) {
            // Keep the reference across model switches; it is the voice.
            var speech = recipe.speech ?? AudioSpeechSettings()
            speech.mode = .clone
            speech.instruction = ""
            if speech.cloneOptions == nil { speech.cloneOptions = AudioCloneOptions() }
            recipe.speech = speech
        } else {
            recipe.speech = nil
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
            guard let self, self.valid(token) else { return }
            self.isRefreshingModels = false
            defer {
                if self.modelsRefreshPending {
                    self.modelsRefreshPending = false
                    self.refreshModels()
                }
            }
            switch result {
            case .success(let snapshot):
                let installed = snapshot.installed
                self.familyCapabilities = snapshot.capabilities
                self.profiles = snapshot.profiles
                self.installedPaths = Dictionary(installed.installed.map { ($0.alias, $0.path) }, uniquingKeysWith: { first, _ in first })
                self.installedRecords = Dictionary(installed.installed.map { ($0.alias, $0) }, uniquingKeysWith: { first, _ in first })
                self.needsAdoption = installed.needsAdoption
                self.incompatibleInstalls = installed.incompatible
                // Installs on a drive that is away: kept as their own state.
                self.notConnectedInstalls = installed.notConnected
                self.restoreModelSelection()
                self.normalizeModelOptions()
            case .failure(let error): self.modelCatalogError = error.localizedDescription
            }
        }
    }
    func chooseModelFolder() {
        guard let profile = selectedProfile, !isBusy, !hasRecordingActivity else { return }
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
        guard let profile = selectedProfile, !profile.identity.repository.isEmpty, !isBusy else { return }
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
            guard let self, self.valid(token) else { return }
            self.isInstalling = false
            switch result {
            case .success(let record):
                self.installedPaths[record.alias] = record.path
                self.installedRecords[record.alias] = record
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
            session = opened; sessionKey = key; modelAccessURL = url
            return opened
        } catch { opening = nil; progress = nil; url?.stopAccessingSecurityScopedResource(); throw error }
    }
    func scheduleUnload() {
        idleTask?.cancel()
        idleTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(300))
            guard !Task.isCancelled, let self, !self.isBusy, !self.isRecording else { return }
            let old = self.session; let oldAccess = self.modelAccessURL
            self.session = nil; self.sessionKey = nil; self.modelAccessURL = nil
            if let old { await Task.detached { old.close() }.value }
            oldAccess?.stopAccessingSecurityScopedResource()
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
        for url in scratchURLs { try? FileManager.default.removeItem(at: url) }
        scratchURLs.removeAll(); items.removeAll(); activeGeneration = nil; activeRunID = nil; generationProgress = nil
        pendingTranscriptEdits.removeAll(); pendingSaves.removeAll(); ManagedAssetPins.clear(owner: pendingAssetOwner); recipe = AudioRecipe(); preferences = AudioWorkspacePreferences()
    }
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
    private let token = try? AudioOpenToken()
    /// Fraction (0...1) of the verification done, called on the opening thread.
    var onProgress: (@Sendable (Double) -> Void)?
    init() { finished.enter() }
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
