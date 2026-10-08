import AppKit
import AVFoundation
import Combine
import Foundation
import TurboSpark
import UniformTypeIdentifiers

enum AudioExportPreset: String, CaseIterable { case nativeWAV, videoWAV, m4a }
enum AudioTranscriptExportFormat: String, CaseIterable { case text, srt, vtt, json }
struct AudioCaptureApplication: Identifiable { var id: Int32; var name: String }
struct AudioCaptureMicrophone: Identifiable { var id: String; var name: String }

@MainActor
final class AudioWorkspaceController: ObservableObject {
    @Published var page: AudioWorkspacePage = .library { didSet { persistPreferences() } }
    @Published var recipe = AudioRecipe() { didSet { persistPreferences() } }
    @Published var items: [AudioLibraryItem] = []
    @Published var selectedID: UUID?
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
    @Published var status = String(localized: "Saved in this profile", bundle: .module)
    @Published var error: String?
    @Published var progress: Double?
    @Published var transcriptBacklog = 0
    @Published var isPlaying = false
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

    let pendingAssetOwner = UUID()
    let lifetime = AudioOperationLifetime()
    var summaryHandler: ((AudioLibraryItem) async throws -> AudioMeetingSummary)?
    var summaryAvailable: (() -> Bool)?
    var canSummarizeMeeting: Bool { selectedItem?.preferredTranscript != nil && summaryAvailable?() == true }
    var installHandler: ((AudioProfile) -> Bool)?
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
    var session: AudioSession?
    var sessionKey: String?
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

    init(library: AudioLibraryStore = AudioLibraryStore(), assets: ManagedAssetStore = .shared) {
        self.library = library; self.assets = assets
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
    var selectedItem: AudioLibraryItem? { items.first { $0.id == selectedID } }
    var task: AudioTask {
        switch page {
        case .record, .transcribe: return .speechToText
        case .voiceover: return .textToSpeech
        case .music: return .music
        case .cleanup: return recipe.task == AudioTask.separation.rawValue ? .separation : .enhancement
        case .library, .advanced: return AudioTask(rawValue: recipe.task) ?? .speechToText
        }
    }
    var availableProfiles: [AudioProfile] {
        var choices = profiles.filter { $0.identity.task == task }
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
    var selectedLocalModelURL: URL? {
        guard let id = recipe.modelID, let bookmark = preferences.localModels[id] else { return nil }
        var stale = false
        return try? URL(resolvingBookmarkData: bookmark, options: [.withSecurityScope, .withoutUI, .withoutMounting], relativeTo: nil, bookmarkDataIsStale: &stale)
    }
    var selectedProfileIsRunnable: Bool {
        selectedProfile?.capabilities.canRun == true
            || (page == .advanced && allowPortableExperiment && ["kokoro", "qwen3_asr"].contains(selectedProfile?.family ?? ""))
            || (page == .advanced && allowExperimentalMetal && selectedProfile?.family == "qwen3_asr")
    }
    var canRun: Bool {
        guard !isBusy, !isInstalling, !isShutDown, !hasRecordingActivity, modelInstalled else { return false }
        guard selectedProfileIsRunnable else { return false }
        switch task {
        case .speechToText, .enhancement, .separation, .alignment, .diarization, .speechDetection, .codec, .languageIdentification:
            return selectedItem?.clips.isEmpty == false
        case .textToSpeech: return !recipe.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .music: return !recipe.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }
    var presets: [AudioPreset] { preferences.presets }
    var playbackDuration: Double { player?.duration ?? selectedItem?.duration ?? 0 }
    var filteredItems: [AudioLibraryItem] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        return items.filter {
            query.isEmpty || $0.title.localizedCaseInsensitiveContains(query)
                || $0.collection.localizedCaseInsensitiveContains(query)
                || ($0.preferredTranscript?.segments.contains { $0.text.localizedCaseInsensitiveContains(query) } ?? false)
        }.sorted { $0.favorite == $1.favorite ? $0.createdAt > $1.createdAt : $0.favorite }
    }
    func valid(_ token: UUID) -> Bool {
        !isShutDown && epoch == token && ProfileVaultStore.shared.session?.profileID == profileID
    }
    func tick() {
        if let capture, isRecording { recordingSeconds = capture.currentElapsed() }
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
        if let model = recipe.modelID { preferences.selectedModels[task.rawValue] = model }
        do { try library.savePreferences(preferences); status = String(localized: "Saved in this profile", bundle: .module) }
        catch { fail(error) }
    }
    func selectPage(_ page: AudioWorkspacePage) {
        guard self.page != page else { return }
        flushPreferences()
        self.page = page
        recipe.task = task.rawValue
        recipe.modelID = preferences.selectedModels[task.rawValue]
        if page != .advanced { recipe.experimentalBackend = nil }
        restoreModelSelection()
        normalizeModelOptions()
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
        if let remembered = preferences.selectedModels[task.rawValue],
           availableProfiles.contains(where: { $0.identity.alias == remembered }) {
            recipe.modelID = remembered
        } else {
            recipe.modelID = availableProfiles.first(where: { installedPaths[$0.identity.alias] != nil })?.identity.alias
                ?? availableProfiles.first?.identity.alias
        }
    }
    func selectModel(_ alias: String?) {
        recipe.modelID = alias
        normalizeModelOptions()
    }
    func normalizeModelOptions() {
        guard let capabilities = selectedProfile?.capabilities else { return }
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
        status = String(localized: "Saved in this profile", bundle: .module)
    }
    func refreshPendingPins() {
        ManagedAssetPins.set(Set(pendingSaves.values.flatMap(\.assetReferences).compactMap(ManagedAssetStore.assetID(from:))), owner: pendingAssetOwner)
    }
    func save(_ item: AudioLibraryItem) throws {
        do { try library.save(item); pendingSaves.removeValue(forKey: item.id); refreshPendingPins(); publish(item) }
        catch { pendingSaves[item.id] = item; refreshPendingPins(); throw error }
    }
    @discardableResult func persistMutation(_ id: UUID, _ mutation: (inout AudioLibraryItem) -> Void) throws -> AudioLibraryItem {
        if var pending = pendingSaves[id] {
            mutation(&pending); try save(pending); return pending
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
        status = String(localized: "Not saved. Retry after checking available storage.", bundle: .module)
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
    func delete(_ item: AudioLibraryItem) {
        guard item.id != recordingID, !(isBusy && item.status == .processing) else { return }
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
        copy.title += String(localized: " copy", bundle: .module); copy.recipe = recipe
        copy.status = .draft; copy.failure = nil
        do { try save(copy); select(copy) } catch { fail(error) }
    }
    func saveDraft() {
        var draft = AudioLibraryItem(title: String(localized: "Untitled audio project", bundle: .module), kind: kind)
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
        case .textToSpeech: page = .voiceover
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
            throw AudioWorkspaceError.message(String(localized: "The saved model is unavailable. Choose a model and create a new take.", bundle: .module))
        }
        if let identity = source.modelIdentity,
           try JSONDecoder().decode(AudioProfileIdentity.self, from: identity) != profile.identity {
            throw AudioWorkspaceError.message(String(localized: "The saved model is unavailable. Choose a model and create a new take.", bundle: .module))
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
        Task { [weak self] in
            let result = await Task.detached { Result { (try AudioCatalog.profiles(), try AudioCatalog.installed(), try AudioCatalog.capabilities()) } }.value
            guard let self, self.valid(token) else { return }
            self.isRefreshingModels = false
            defer {
                if self.modelsRefreshPending {
                    self.modelsRefreshPending = false
                    self.refreshModels()
                }
            }
            switch result {
            case .success(let (profiles, installed, capabilities)):
                self.familyCapabilities = capabilities
                self.profiles = profiles
                self.installedPaths = Dictionary(installed.installed.map { ($0.alias, $0.path) }, uniquingKeysWith: { first, _ in first })
                self.needsAdoption = installed.needsAdoption
                self.incompatibleInstalls = installed.incompatible
                self.restoreModelSelection()
                self.normalizeModelOptions()
            case .failure(let error): self.modelCatalogError = error.localizedDescription
            }
        }
    }
    func chooseModelFolder() {
        guard let profile = selectedProfile, !isBusy, !hasRecordingActivity else { return }
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = String(localized: "Choose a checkpoint folder for the selected audio model.", bundle: .module)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            preferences.localModels[profile.identity.alias] = try url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
            flushPreferences()
        } catch { fail(error) }
    }
    func installSelectedModel() {
        guard let profile = selectedProfile, !profile.identity.repository.isEmpty, !isBusy else { return }
        if let installHandler { _ = installHandler(profile); return }
        guard !isInstalling else { return }
        isInstalling = true; error = nil
        status = String(localized: "Downloading audio model", bundle: .module)
        let token = epoch
        Task { [weak self] in
            let result = await Task.detached { Result { try AudioCatalog.install(profile.identity) } }.value
            guard let self, self.valid(token) else { return }
            self.isInstalling = false
            switch result {
            case .success(let record): self.installedPaths[record.alias] = record.path; self.status = String(localized: "Audio model installed", bundle: .module)
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
            guard !stale else { throw AudioWorkspaceError.message(String(localized: "The model folder moved. Choose it again.", bundle: .module)) }
            _ = resolved.startAccessingSecurityScopedResource(); url = resolved; path = resolved.path
        } else if let installed = installedPaths[profile.identity.alias] { path = installed }
        else { throw AudioWorkspaceError.message(String(localized: "Download this model or choose its checkpoint folder.", bundle: .module)) }
        let key = "\(profile.family):\(profile.identity.task.rawValue):\(path):\(portable):\(experimentalMetal)"
        if sessionKey == key, let session { url?.stopAccessingSecurityScopedResource(); return session }
        if let previous = session { await Task.detached { previous.close() }.value }
        modelAccessURL?.stopAccessingSecurityScopedResource(); modelAccessURL = nil
        session = nil; sessionKey = nil
        let token = epoch
        do {
            let opening = AudioModelOpening()
            opening.onProgress = { [weak self] fraction in
                Task { @MainActor in
                    guard let self, self.opening != nil else { return }
                    self.status = String(localized: "Verifying", bundle: .module)
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
    func cancel() { jobTask?.cancel(); session?.cancel(); opening?.cancelOpen() }
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
        stopPlayback()
        if let opening { DispatchQueue.global(qos: .userInitiated).sync { opening.stopAndDrain() } }
        opening = nil
        if let old = session { DispatchQueue.global(qos: .userInitiated).sync { old.close() } }
        session = nil; sessionKey = nil; modelAccessURL?.stopAccessingSecurityScopedResource(); modelAccessURL = nil
        for url in scratchURLs { try? FileManager.default.removeItem(at: url) }
        scratchURLs.removeAll(); items.removeAll(); pendingTranscriptEdits.removeAll(); pendingSaves.removeAll(); ManagedAssetPins.clear(owner: pendingAssetOwner); recipe = AudioRecipe(); preferences = AudioWorkspacePreferences()
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
