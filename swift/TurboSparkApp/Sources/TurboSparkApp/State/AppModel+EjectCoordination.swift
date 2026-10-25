import Combine
import Foundation
import TurboSpark

/// Pure: which models the sessions hold, as the coordinator wants them.
///
/// One entry per model, in a fixed order (chat, then server by id, then
/// audio), so a toast naming them reads the same every time. The chat session
/// that is also served appears once, as chat: unloading it detaches it from
/// the server too (`detachChatSessionFromServer`).
enum LoadedModelInventory {
    struct ServerSession: Equatable {
        let id: String
        let path: String
        /// The chat pane's own session, served as well.
        let isChatSession: Bool
    }

    static func make(
        chat: String?, retiredChats: [String] = [], serverSessions: [ServerSession],
        serverAudio: [(id: String, path: String)], audio: String?, retiredAudio: [String] = [],
        name: (String) -> String
    ) -> [LoadedModel] {
        var models: [LoadedModel] = []
        if let chat { models.append(LoadedModel(category: .chat, name: name(chat), path: chat)) }
        // Chat sessions the app already dropped but something still holds
        // are loaded as far as the drive is concerned. One entry per path.
        var seen = Set(models.map { AppModel.normalizedModelPath($0.path) })
        for path in retiredChats where seen.insert(AppModel.normalizedModelPath(path)).inserted {
            models.append(LoadedModel(category: .chat, name: name(path), path: path))
        }
        for session in serverSessions.sorted(by: { $0.id < $1.id }) where !session.isChatSession {
            models.append(LoadedModel(category: .server, name: name(session.path), path: session.path))
        }
        for entry in serverAudio.sorted(by: { $0.id < $1.id }) {
            models.append(LoadedModel(category: .server, name: name(entry.path), path: entry.path))
        }
        if let audio { models.append(LoadedModel(category: .audio, name: name(audio), path: audio)) }
        // An audio session whose close has not returned is loaded as far as
        // the drive is concerned, the same as a retired chat session.
        var seenAudio = Set(models.filter { $0.category == .audio }.map { AppModel.normalizedModelPath($0.path) })
        for path in retiredAudio where seenAudio.insert(AppModel.normalizedModelPath(path)).inserted {
            models.append(LoadedModel(category: .audio, name: name(path), path: path))
        }
        return models
    }

    /// The file or folder name without the `.gturbo` extension, for a model
    /// the installed list does not know.
    static func fallbackName(forPath path: String) -> String {
        let last = (path as NSString).lastPathComponent
        return last.hasSuffix(".gturbo") ? String(last.dropLast(".gturbo".count)) : last
    }
}

/// The eject coordinator, its registration as the monitor's approval delegate,
/// and the subscriptions that keep its snapshot current.
///
/// The monitor holds its delegate weakly, so `AppModel` holds this, and this
/// holds the coordinator.
@MainActor
final class LibraryEjectCoordination {
    let coordinator: LibraryEjectCoordinator
    private let host: UnmountApprovalHosting
    private var subscriptions: Set<AnyCancellable> = []

    init(coordinator: LibraryEjectCoordinator, host: UnmountApprovalHosting) {
        self.coordinator = coordinator
        self.host = host
    }

    func register() {
        host.delegate = coordinator
    }

    /// Re-reads the loaded set whenever a session, a server attachment or the
    /// library list changes. A publisher fires before the new value is stored,
    /// so `scheduleRefresh` defers the read to a main-actor task.
    func follow(_ model: AppModel) {
        let refresh: () -> Void = { [weak coordinator] in coordinator?.scheduleRefresh() }
        model.$session.dropFirst().sink { _ in refresh() }.store(in: &subscriptions)
        model.$serverAttachedSessions.dropFirst().sink { _ in refresh() }.store(in: &subscriptions)
        model.$serverAudioModelPaths.dropFirst().sink { _ in refresh() }.store(in: &subscriptions)
        model.libraryState.$libraries.dropFirst().sink { _ in refresh() }.store(in: &subscriptions)
    }

    /// Makes the coordinator approve everything and ends its work. The
    /// registration is removed only if it is still this one: a replacement
    /// model (a vault unlock) may have registered its own already. The monitor
    /// itself is never stopped.
    func stop() {
        coordinator.stop()
        if host.delegate === coordinator { host.delegate = nil }
        subscriptions.removeAll()
    }
}

/// The live `LoadedModelSource`: chat, server and audio sessions, read from
/// the app model, stopped and released with the app's own paths. It holds the
/// model weakly.
///
/// What it does NOT cover: the image session, the Humanize session and the
/// server's image and speech attachments. They are not chat, server-model or
/// audio-workspace sessions, so an eject with only those loaded proceeds as it
/// did before this feature.
@MainActor
final class AppModelLoadedModels: LoadedModelSource {
    /// How long a stopped turn may take to wind down before the sessions are
    /// released anyway.
    static let stopPatience: Duration = .seconds(5)

    /// Stand-ins for the parts of the app a unit test cannot build: a model
    /// session needs a multi-gigabyte checkpoint, an audio session a native
    /// engine, and an empty server never refuses a detach. The release code
    /// that runs against them is the live code. All nil in the app.
    struct Seams {
        var chatSession: ChatSessionSlot?
        var audioSession: AudioSessionSlot?
        var serverEngine: ServerDetachEngine?

        init(
            chatSession: ChatSessionSlot? = nil, audioSession: AudioSessionSlot? = nil,
            serverEngine: ServerDetachEngine? = nil
        ) {
            self.chatSession = chatSession
            self.audioSession = audioSession
            self.serverEngine = serverEngine
        }
    }

    private weak var model: AppModel?
    private let patience: Duration
    private let seams: Seams
    /// Sessions an earlier release dropped that something still holds.
    let retired = RetiredSessionLedger()

    init(model: AppModel, patience: Duration = AppModelLoadedModels.stopPatience, seams: Seams = Seams()) {
        self.model = model
        self.patience = patience
        self.seams = seams
    }

    private func chatSlot(of model: AppModel) -> ChatSessionSlot {
        seams.chatSession ?? model.liveChatSessionSlot()
    }

    private func audioSlot(of model: AppModel) -> AudioSessionSlot {
        seams.audioSession ?? model.liveAudioSessionSlot()
    }

    func loadedModels() -> [LoadedModel] {
        guard let model else { return [] }
        return LoadedModelInventory.make(
            chat: chatSlot(of: model).loadedPath(),
            retiredChats: retired.alive(.chat).map(\.path),
            serverSessions: model.serverAttachedSessions.map {
                LoadedModelInventory.ServerSession(
                    id: $0.key, path: $0.value.info.modelPath, isChatSession: $0.value === model.session)
            },
            serverAudio: model.serverAudioModelPaths.map { (id: $0.key, path: $0.value) },
            audio: audioSlot(of: model).residentPath(),
            retiredAudio: retired.alive(.audio).map(\.path),
            name: { model.displayName(forModelAt: $0) })
    }

    /// Requirement 6.2: generation stops BEFORE the model goes.
    ///
    /// **THE TURN IS NOT THE ONLY HOLDER OF THE CHAT SESSION.** `cancel()`
    /// stops the turn and deliberately nothing else (see its note), but title
    /// generation, memory capture and background agents each capture the
    /// session strongly and outlive it. Dropping the app's reference with one
    /// of them running frees nothing and leaves the drive busy, so they are
    /// cancelled and awaited here too, within the same patience. Whatever
    /// refuses to end is caught later by the weak-reference check, not here.
    func stopGeneration(in category: LoadedModelCategory, models: [LoadedModel]) async {
        guard let model else { return }
        switch category {
        case .chat:
            // **LOOPS TO QUIESCENCE, CANCELLING ON EVERY POLL.** Anything
            // that ends here can be replaced in the same hop: a cancelled
            // turn's tail starts the next queued prompt, a cancelled title
            // task starts its successor, a killed agent's watcher injects a
            // notification. The release hold (`beginRelease`) stops the
            // prompts and notifications from starting a turn at all; this
            // loop catches the rest by cancelling again until the turn, the
            // title and memory tasks and the background agents are all quiet
            // on two polls in a row (the second is the extra yield), or the
            // patience runs out. The turn winds down on the main actor after
            // the cancel, so the session is released when it has, or when the
            // wait runs out (`shutdown()` releases a running turn's session
            // the same way).
            //
            // At least one cancel pass always runs, and a cancelled caller
            // ends the loop after it instead of spinning: `Task.sleep` throws
            // at once in a cancelled task.
            let deadline = ContinuousClock.now + patience
            var quietPolls = 0
            repeat {
                quietPolls = model.cancelTurnAndSessionBoundWork() ? 0 : quietPolls + 1
                if quietPolls >= 2 || ContinuousClock.now >= deadline { break }
                try? await Task.sleep(for: .milliseconds(25))
            } while !Task.isCancelled
        case .server:
            // The server has no per-request cancel, and the engine REFUSES to
            // detach a model that has a request in flight (it does not end
            // the request). `unload` therefore retries the detach so that
            // in-flight requests can drain, and the caller verifies the
            // result; nothing here stops them.
            return
        case .audio:
            if model.audioWorkspaceCreated { model.audioWorkspace.cancel() }
        }
    }

    /// Holds the chat against starting any turn for the whole release, when a
    /// chat model is part of it. A release that only takes a server or audio
    /// model off a drive leaves an unrelated chat alone.
    func beginRelease(of models: [LoadedModel]) {
        guard models.contains(where: { $0.category == .chat }) else { return }
        model?.beginEjectHold()
    }

    func endRelease(of models: [LoadedModel]) {
        guard models.contains(where: { $0.category == .chat }) else { return }
        model?.endEjectHold()
    }

    /// Releases the sessions that hold `models`. Not a promise: the caller
    /// asks `loadedModels()` afterwards, because a session can outlive this.
    func unload(_ category: LoadedModelCategory, models: [LoadedModel]) async {
        guard let model else { return }
        let paths = Set(models.map { AppModel.normalizedModelPath($0.path) })
        switch category {
        case .chat:
            // Sessions an earlier release dropped that something still held:
            // wait for them again, cancelling what holds them.
            for entry in retired.alive(.chat) where paths.contains(AppModel.normalizedModelPath(entry.path)) {
                _ = await SessionRelease.waitUntilGone(entry.ref, patience: patience) {
                    model.cancelTurnAndSessionBoundWork()
                }
            }
            let slot = chatSlot(of: model)
            guard let loaded = slot.loadedPath(), paths.contains(AppModel.normalizedModelPath(loaded)) else { return }
            if let holder = await model.releaseChatSessionForEject(patience: patience, slot: slot) {
                retired.add(holder, path: loaded)
            }
        case .server:
            await model.detachServerModels(atPaths: paths, patience: patience, engine: seams.serverEngine)
        case .audio:
            for entry in retired.alive(.audio) where paths.contains(AppModel.normalizedModelPath(entry.path)) {
                _ = await SessionRelease.waitUntilGone(entry.ref, patience: patience)
            }
            let slot = audioSlot(of: model)
            guard let loaded = slot.residentPath(), paths.contains(AppModel.normalizedModelPath(loaded)),
                let detached = slot.detach()
            else { return }
            // **THE CLOSE IS BOUNDED.** `AudioSession.close()` waits on the
            // session's operation lock until the native job returns, which
            // can be never. The wait gives up with the patience and the
            // session is remembered as still held: the close goes on in the
            // background, holds the object until it returns, and until then
            // the model is listed as loaded and the drive is not ejected.
            let ref = WeakObject(detached)
            let closed = await BoundedAwait.finishes(within: patience) { await detached.close() }
            if !closed { retired.add(ref, path: loaded, category: .audio) }
        }
    }
}

extension AppModel {
    nonisolated static func normalizedModelPath(_ path: String) -> String {
        URL(fileURLWithPath: path).standardizedFileURL.path
    }

    /// The name a toast gives a model: its installed alias, else its folder.
    func displayName(forModelAt path: String) -> String {
        let target = Self.normalizedModelPath(path)
        let known =
            installed.first { Self.normalizedModelPath($0.path) == target }
            ?? libraryState.allInstalled.first { Self.normalizedModelPath($0.path) == target }
        return known?.alias ?? LoadedModelInventory.fallbackName(forPath: path)
    }

    /// The chat session, released the way `shutdown()` releases it (detached
    /// from the server, then dropped) and then VERIFIED: it waits until the
    /// session object is really gone. Not `unloadModel()`, which refuses while
    /// a turn is running or a load is in flight and posts its own toast; the
    /// eject coordinator has already stopped the turn, and tells the user
    /// itself.
    ///
    /// Returns nil when nothing holds the session any more, or when it was
    /// KEPT because the server still serves it (it is then still loaded, and
    /// `loadedModelPath` says so). Returns the weak reference to the holder
    /// when the app let go but a task still holds the session: the model is
    /// still mapped, and the caller must keep reporting it as loaded.
    func releaseChatSessionForEject(patience: Duration, slot: ChatSessionSlot? = nil) async -> WeakObject? {
        // The server's half first, and checked: a model the engine still
        // serves (a request in flight) cannot be freed by dropping the Swift
        // side, and dropping it would hide the model from the pane.
        if let session, server != nil {
            let served = serverAttachedSessions.filter { $0.value === session }.map(\.key).sorted()
            let stillServed = await BoundedRetry.drain(
                served, patience: patience, interval: Self.serverDetachRetryInterval
            ) { self.attemptServerDetach(id: $0).releasedTheModel }
            if !stillServed.isEmpty { return nil }
        }
        let slot = slot ?? liveChatSessionSlot()
        // Settled before every look: a turn started since the stop step ended
        // (a queued prompt the user sent meanwhile) holds the session as much
        // as a background task does, so the turn is cancelled too, not only
        // the helpers.
        return await SessionRelease.dropAndWait(
            get: slot.object, clear: slot.clear, patience: patience
        ) { self.cancelTurnAndSessionBoundWork() }
    }

    /// The chat session as the eject path reads and clears it.
    func liveChatSessionSlot() -> ChatSessionSlot {
        ChatSessionSlot(
            loadedPath: { [weak self] in self?.loadedModelPath },
            object: { [weak self] in self?.session },
            clear: { [weak self] in self?.session = nil })
    }

    /// The audio workspace's resident session as the eject path reads it.
    /// Reading `audioWorkspace` builds it, so it is asked only once built.
    func liveAudioSessionSlot() -> AudioSessionSlot {
        AudioSessionSlot(
            residentPath: { [weak self] in
                guard let self, self.audioWorkspaceCreated else { return nil }
                return self.audioWorkspace.residentModelPath
            },
            detach: { [weak self] in
                guard let self, self.audioWorkspaceCreated else { return nil }
                return self.audioWorkspace.detachResidentSession()
            })
    }

    /// How often a refused detach is tried again while a request drains.
    static let serverDetachRetryInterval: Duration = .milliseconds(100)

    /// Detaches every served model whose files are at one of `paths`, with the
    /// same call the Server pane uses, so the server's reference and the
    /// app's both go.
    ///
    /// **THE ENGINE REFUSES A BUSY MODEL, SO THIS RETRIES.** `crates/ffi`
    /// will not detach a model with a request in flight or a loaded queue; it
    /// does not end the request. A refused detach is tried again for the
    /// patience so the request can drain, and a model the engine still holds
    /// afterwards keeps its record (see `attemptServerDetach`), so it is still
    /// reported as loaded.
    func detachServerModels(
        atPaths paths: Set<String>, patience: Duration, engine: ServerDetachEngine? = nil
    ) async {
        let textIds = serverAttachedSessions
            .filter { $0.value !== session && paths.contains(Self.normalizedModelPath($0.value.info.modelPath)) }
            .map(\.key)
        let audioIds = serverAudioModelPaths
            .filter { paths.contains(Self.normalizedModelPath($0.value)) }
            .map(\.key)
        let ids = (textIds + audioIds).sorted()
        guard !ids.isEmpty else { return }
        _ = await BoundedRetry.drain(ids, patience: patience, interval: Self.serverDetachRetryInterval) {
            self.attemptServerDetach(id: $0, engine: engine).releasedTheModel
        }
    }

    /// Shows what the eject coordinator reports. Records are not touched: a
    /// message is all a surprise removal produces (requirement 6.4).
    func presentEjectNotice(_ notice: EjectNotice) {
        showToast(
            EjectNoticeText.message(for: notice), style: EjectNoticeText.style(for: notice), duration: 10)
    }

    /// Creates the coordinator, registers it as the monitor's approval delegate
    /// and starts it on its own event stream. Calling it again does nothing.
    func startEjectCoordination(
        events: AsyncStream<VolumeEvent>, host: UnmountApprovalHosting, ejector: DiskEjecting,
        locator: ModelVolumeLocating, source: (any LoadedModelSource)? = nil
    ) {
        guard ejectCoordination == nil else { return }
        let coordinator = LibraryEjectCoordinator(
            source: source ?? AppModelLoadedModels(model: self), ejector: ejector, locator: locator,
            present: { [weak self] notice in self?.presentEjectNotice(notice) })
        let coordination = LibraryEjectCoordination(coordinator: coordinator, host: host)
        ejectCoordination = coordination
        coordination.follow(self)
        coordinator.start(events: events)
        coordination.register()
        coordinator.scheduleRefresh()
    }
}
