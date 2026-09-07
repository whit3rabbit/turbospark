import Foundation
import TurboSpark

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

    /// Opens `model` in its own session and attaches it to the running
    /// server, without disturbing the Chat pane's session.
    ///
    /// **A SECOND SESSION, NOT A SHARED ONE.** Opening the same install twice
    /// would map its weights twice, so the pane offers models the app does
    /// not already have open; attaching the CHAT session is
    /// `attachChatSession()` and is the cheap path.
    public func attachModelToServer(_ model: InstalledModel) {
        guard let server, !serverBusy else { return }
        serverBusy = true
        Task {
            defer { serverBusy = false }
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
                        "Server stopped while \(model.alias) was loading; it was not attached.",
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
                showToast("Serving \(id)", style: .success)
            } catch {
                let msg = "Could not serve \(model.alias): \(error.localizedDescription)"
                self.error = msg
                showToast(msg, style: .error)
            }
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
            showToast("Serving \(id)", style: .success)
        } catch {
            let msg = "Could not serve the loaded model: \(error.localizedDescription)"
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
        guard let server else { return }
        // **THE SWIFT REFERENCE GOES WHATEVER THE ENGINE SAYS** (state#53).
        // This sat INSIDE the `do`, so a throwing detach kept the session in
        // `serverAttachedSessions` -- which is the reference that holds the
        // weights, the KV cache and the compiled pipelines alive -- while
        // `refreshServerInfo` never ran and the row vanished from the pane
        // anyway. The user sees nothing serving it and the memory is still
        // committed. Dropping it here is the recoverable direction: the
        // engine's own half either succeeded or is reported below.
        defer {
            serverAttachedSessions.removeValue(forKey: id)
            refreshServerInfo()
        }
        do {
            try server.detach(modelId: id)
            showToast("Stopped serving \(id)", style: .info)
        } catch {
            let msg = "Could not stop serving \(id): \(error.localizedDescription)"
            self.error = msg
            showToast(msg, style: .error)
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
            serverAttachedSessions[id] = nil
        }
        if !failures.isEmpty {
            let msg =
                "The server could not release \(failures.count == 1 ? "a model" : "some models"); "
                + "it may still be serving \(failures.joined(separator: ", "))."
            self.error = msg
            showToast(msg, style: .warning)
        }
        if !ids.isEmpty {
            refreshServerInfo()
        }
    }
}
