import Foundation
import TurboSpark

/// The three calls one detach attempt makes on the running server, so a test
/// can stand in for the engine's refusals, which no empty server produces.
struct ServerDetachEngine {
    var detach: (String) throws -> Void
    var detachAudio: (String) throws -> Void
    var info: () throws -> ServerInfo
}

/// Which models the running server is holding.
///
/// **TWO REFERENCES HOLD AN ATTACHED MODEL AND BOTH HAVE TO GO.** The server
/// holds one on the Rust side and `serverAttachedSessions` holds the Swift
/// one; dropping either alone keeps the weights, the KV cache and the
/// compiled pipelines resident with nothing in the UI still showing them
/// (`swift/CLAUDE.md` Gotcha 26). The CHAT session is the deliberate
/// exception and needs no branch: `AppModel.session` is a third reference
/// the Chat pane still holds, so ejecting it here stops it being SERVED and
/// leaves it loaded.
///
/// That pairing is what this file is about, and it is where the review found
/// its defects: state#53 (a throwing detach kept the Swift reference),
/// state#72 (a stop during an attach orphaned a session no remover can
/// reach) and state#104 (a `try?` that reported nothing).
extension AppModel {
    // MARK: - Attaching and detaching

    nonisolated static func detachedServerModelID(from event: ServerEvent) -> String? {
        guard case let .modelDetached(_, model) = event else { return nil }
        return model
    }

    nonisolated static func removeServerAttachment<T>(
        id: String,
        from attachments: inout [String: T]
    ) {
        attachments.removeValue(forKey: id)
    }

    /// Whether the engine still lists `id` as attached, per a fresh reading.
    ///
    /// **NIL (NO READING) COUNTS AS HELD.** A read that failed is not proof
    /// the engine let go: a refused detach followed by a throwing read on a
    /// server that is still running used to drop the app's record of a model
    /// the engine still held. The cases where nothing is attached are decided
    /// before this is asked: no server at all (stopping it released every
    /// model), and a server that answers without listing the id.
    nonisolated static func engineStillHolds(_ id: String, info: ServerInfo?) -> Bool {
        guard let info else { return true }
        return info.models.contains(id) || info.audioModels.contains(id)
    }

    /// Drops the app-owned session reference after the server reports a
    /// detach, then refreshes every surface derived from server state.
    func releaseServerAttachment(id: String, refresh: Bool = true) {
        Self.removeServerAttachment(id: id, from: &serverAttachedSessions)
        Self.removeServerAttachment(id: id, from: &serverAudioModelPaths)
        if refresh {
            refreshServerInfo()
        }
    }

    /// Opens `model` in its own session and attaches it to the running
    /// server, without disturbing the Chat pane's session.
    ///
    /// **A SECOND SESSION, NOT A SHARED ONE.** Opening the same install twice
    /// would map its weights twice, so the pane offers models the app does
    /// not already have open; attaching the CHAT session is
    /// `attachChatSession()` and is the cheap path.
    ///
    /// **THROUGH THE CHOKE POINT** (task 6.7): opening a second session reads the
    /// whole model from its drive, so a warned model asks first, and a model whose
    /// drive is away is refused.
    public func attachModelToServer(_ model: InstalledModel) {
        requestLoad(model, purpose: .serverAttach)
    }

    /// The attach itself, once the gate has said yes.
    func beginServerAttach(_ model: InstalledModel) {
        guard let server, !serverBusy else { return }
        // The chat model is already resident: serve THAT session instead of
        // mapping the same install a second time (doubled weights and KV, or
        // a load-guard refusal).
        if Self.isChatInstall(
            modelPath: model.path, selectedPath: selected?.path, hasSession: session != nil)
        {
            attachChatSession()
            return
        }
        serverBusy = true
        Task { await performServerAttach(model, to: server) }
    }

    func attachServerModelAndWait(_ model: InstalledModel) async {
        guard let server, !serverBusy else { return }
        if Self.isChatInstall(
            modelPath: model.path, selectedPath: selected?.path, hasSession: session != nil)
        {
            attachChatSession()
            return
        }
        serverBusy = true
        await performServerAttach(model, to: server)
    }

    /// Whether `modelPath` is the install the Chat pane's live session was
    /// opened from (`selected` is set only by a successful open). Pure so the
    /// routing decision can be asserted without a multi-gigabyte install.
    static func isChatInstall(modelPath: String, selectedPath: String?, hasSession: Bool) -> Bool {
        hasSession && selectedPath == modelPath
    }

    private func performServerAttach(_ model: InstalledModel, to server: TurboSparkServer) async {
        defer { serverBusy = false }
        guard self.server === server else { return }
        // The net under the attach paths, the favorite's auto-start among them: a
        // model whose drive is away or whose folder is gone is never opened.
        // `loadServerFavorite` already skips such a model with its own toast;
        // this one covers a row that went stale between that check and here.
        guard canOpenNow(model) else { return }
        loadObserver?(model)
        do {
            let session = try await TurboSparkSession(
                modelPath: model.path, options: buildOpenOptions(modelPath: model.path))
            // **THE SERVER CAN BE STOPPED WHILE THIS OPEN IS IN FLIGHT**
            // (state#72). `stopServer()` has no `serverBusy` guard on
            // purpose (state#28 needs it reachable during a bind), and it
            // clears `server` and `serverAttachedSessions` -- while this
            // task holds `server` as a local captured BEFORE the await
            // and then writes into the map it just emptied. Every remover
            // starts with `guard let server`, so that resurrected entry
            // is unreachable: the weights, the KV cache and the compiled
            // pipelines stay resident for the life of the process, and
            // `deleteModel` refuses that model forever. Returning here
            // drops the only reference and releases the mapping.
            guard let live = self.server,
                Self.attachMayComplete(captured: server, current: live)
            else {
                showToast(
                    AppStrings.localized("Server stopped while \(model.alias) was loading; it was not attached."),
                    style: .warning)
                return
            }
            let id: String
            do {
                id = try live.attach(session)
            } catch {
                // The session opened and the attach did not, so nothing
                // holds the engine but this local -- letting it go is
                // what releases the mapping. Named because the ordering
                // is the whole reason a failed attach is not a leak.
                throw error
            }
            serverAttachedSessions[id] = session
            refreshServerInfo()
            showToast(AppStrings.localized("Serving \(id)"), style: .success)
        } catch {
            let msg = AppStrings.localized("Could not serve \(model.alias): \(error.localizedDescription)")
            self.error = msg
            showToast(msg, style: .error)
        }
    }

    /// Whether an attach begun against `captured` may still write its session
    /// into `serverAttachedSessions` (state#72).
    ///
    /// **IDENTITY, NOT "IS THERE A SERVER".** Stop-then-Start during a long
    /// open leaves a DIFFERENT server running, and attaching to that one
    /// would serve a model the user never asked it to. A pure static because
    /// the call site it guards sits after `TurboSparkSession(modelPath:)`,
    /// which no test can reach without a real multi-gigabyte install; this is
    /// the decision that guard makes, and it is what can be asserted.
    static func attachMayComplete(captured: TurboSparkServer, current: TurboSparkServer?) -> Bool {
        guard let current else { return false }
        return current === captured
    }

    /// Attaches the Chat pane's own session, so one resident model answers
    /// both the app and the network.
    public func attachChatSession() {
        guard let server, let session, !serverBusy else { return }
        do {
            let id = try server.attach(session)
            serverAttachedSessions[id] = session
            refreshServerInfo()
            showToast(AppStrings.localized("Serving \(id)"), style: .success)
        } catch {
            let msg = AppStrings.localized("Could not serve the loaded model: \(error.localizedDescription)")
            self.error = msg
            showToast(msg, style: .error)
        }
    }

    /// Removes a model from the server, releasing the server's reference.
    ///
    /// **THE APP'S OWN REFERENCE GOES TOO, WHICH IS WHAT ACTUALLY FREES IT.**
    /// The server holds one and `serverAttachedSessions` holds another, so
    /// dropping only one keeps the weights, the KV cache and the compiled
    /// pipelines resident with nothing in the UI still showing them.
    ///
    /// The CHAT session survives this and is meant to: `AppModel.session`
    /// is a third reference the Chat pane still needs, so detaching it here
    /// stops it being SERVED and leaves it loaded. That falls out of the
    /// references rather than needing a branch, which is why there is not
    /// one.
    public func detachModelFromServer(id: String) {
        guard server != nil else { return }
        switch attemptServerDetach(id: id) {
        case .detached:
            showToast(AppStrings.localized("Stopped serving \(id)"), style: .info)
        case .alreadyGone(let reason), .stillServed(let reason):
            let msg = AppStrings.localized("Could not stop serving \(id): \(reason)")
            self.error = msg
            showToast(msg, style: .error)
        }
    }

    /// How one attempt to take a model off the running server ended.
    enum ServerDetachOutcome: Equatable {
        case detached
        /// The call threw, but the engine no longer lists the model (nothing
        /// was attached under that id, or the server stopped). The app's
        /// reference is released, as state#53 requires.
        case alreadyGone(String)
        /// The engine REFUSED and still holds the model (a request is using
        /// it). The app's reference is KEPT: it is the only record that the
        /// model is still resident, and dropping it made the model vanish
        /// from the pane and from the eject coordinator while its weights, KV
        /// cache and pipelines stayed mapped. The row stays so it can be
        /// retried.
        case stillServed(String)

        /// True when the engine no longer holds the model.
        var releasedTheModel: Bool {
            switch self {
            case .detached, .alreadyGone: return true
            case .stillServed: return false
            }
        }
    }

    /// One attempt, without a toast: the Server pane toasts the outcome, and
    /// the eject coordinator retries it until the model drains.
    ///
    /// **THE SWIFT REFERENCE GOES WHEN THE ENGINE LETS GO** (state#53). A
    /// throwing detach used to keep the session in `serverAttachedSessions`
    /// while the row vanished from the pane anyway, so the user saw nothing
    /// serving it with the memory still committed; the fix dropped the
    /// reference whatever the engine said. That was wrong in the other
    /// direction for a REFUSED detach (`crates/ffi` refuses while a request is
    /// in flight or the queue is loaded): the engine still held the model and
    /// the app forgot it. So after a failure the engine is asked, and the
    /// reference is dropped only when the engine no longer lists the model.
    func attemptServerDetach(id: String, engine: ServerDetachEngine? = nil) -> ServerDetachOutcome {
        guard let engine = engine ?? liveServerDetachEngine() else {
            // No server: stopping it released every model already.
            releaseServerAttachment(id: id, refresh: false)
            return .detached
        }
        do {
            if serverInfo?.audioModels.contains(id) == true {
                try engine.detachAudio(id)
            } else {
                try engine.detach(id)
            }
            releaseServerAttachment(id: id)
            return .detached
        } catch {
            let reading = try? engine.info()
            if let reading { serverInfo = reading }
            if Self.engineStillHolds(id, info: reading) {
                return .stillServed(error.localizedDescription)
            }
            releaseServerAttachment(id: id)
            return .alreadyGone(error.localizedDescription)
        }
    }

    /// The calls to the running server, or nil when none is running.
    func liveServerDetachEngine() -> ServerDetachEngine? {
        guard let server else { return nil }
        return ServerDetachEngine(
            detach: { try server.detach(modelId: $0) },
            detachAudio: { try server.detachAudioModel(id: $0) },
            info: { try server.info() })
    }

    /// Attaches an audio model (speech-to-text, text-to-speech, or music) to the running server.
    public func attachAudioModelToServer(path: String, task: String = "") {
        guard let server, !serverBusy else { return }
        serverBusy = true
        Task {
            defer { serverBusy = false }
            do {
                let id = try server.attachAudioModel(path, task: task)
                // The server reports ids only; the path is what tells the
                // eject coordinator which drive holds this model.
                serverAudioModelPaths[id] = path
                refreshServerInfo()
                showToast(AppStrings.localized("Serving \(id)"), style: .success)
            } catch {
                let msg = AppStrings.localized("Could not serve audio model \(path): \(error.localizedDescription)")
                self.error = msg
                showToast(msg, style: .error)
            }
        }
    }

    /// Detaches whatever entry `session` was attached under, if any.
    ///
    /// **CALLED BY EVERY SITE THAT CLEARS `session`**, and that obligation is
    /// the multi-model form of `swift/CLAUDE.md` Gotcha 26: with one model,
    /// `unloadModel()` and friends called `stopServer()`; with several,
    /// stopping the whole server to unload one is wrong and doing nothing
    /// leaks the one being unloaded. A site that clears `session` without
    /// coming through here keeps that model resident, served, and invisible
    /// in the Chat pane.
    public func detachChatSessionFromServer() {
        guard let server, let session else { return }
        let ids = serverAttachedSessions.filter { $0.value === session }.map(\.key)
        var failures: [String] = []
        for id in ids {
            do {
                try server.detach(modelId: id)
            } catch {
                // **`try?` HERE MEANT THE ENGINE STILL HELD IT** (state#104).
                // The Swift reference is dropped either way, which is the
                // recoverable direction (state#53's own reasoning), but a
                // failed detach leaves the model resident and SERVED with
                // nothing in either pane still showing it -- exactly the
                // invisible-resident state this function exists to prevent,
                // reported nowhere.
                failures.append("\(id): \(error.localizedDescription)")
            }
            releaseServerAttachment(id: id, refresh: false)
        }
        if !failures.isEmpty {
            let serving = failures.joined(separator: ", ")
            let msg = failures.count == 1
                ? AppStrings.localized("The server could not release a model; it may still be serving \(serving).")
                : AppStrings.localized("The server could not release some models; it may still be serving \(serving).")
            self.error = msg
            showToast(msg, style: .warning)
        }
        if !ids.isEmpty {
            refreshServerInfo()
        }
    }
}
