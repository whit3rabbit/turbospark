import Foundation
import TurboSpark

/// The two rows the settings panel draws for a running server, derived from
/// what that server REPORTED rather than from what was asked for.
///
/// **A VALUE RATHER THAN A VIEW, so the rows can be tested with no model, no
/// session and no bound socket.** Every fact here used to be spelled inline
/// in `AppSettingsView`, where the only way to check it was to launch the app
/// and look -- which is how a `127.0.0.1` literal sat beside a read-back port
/// for the life of the feature, and how `authEnabled` came to be decoded and
/// displayed nowhere.
public struct ServerStatusRows: Equatable {
    /// What to put in front of a user to copy. Falls back to `host:port`
    /// only if the URL will not build, which no address this engine binds
    /// can cause -- a visible pair of numbers beats an empty row.
    public let address: String
    public let authLabel: String
    /// True when the server is answering with no key. Drawn in a warning
    /// colour because "none" is the state a user needs to notice, and it is
    /// reachable by typing nothing but spaces into the key field.
    public let authIsWarning: Bool
    /// The tool-call guardrails this server STARTED with.
    ///
    /// **A SECOND ENFORCEMENT POINT FROM THE CHAT PANE'S, and worth stating
    /// because the two are easy to conflate.** `ForgeGuardrailsEngine` runs
    /// in this app's own agent loop, over a reply it read itself; every HTTP
    /// client of this server bypasses that entirely. The server resolves its
    /// own at start, from the same setting, and cannot be changed without a
    /// restart -- which is exactly why it is reported rather than assumed.
    public let guardrailsLabel: String

    public init(info: ServerInfo, guardrails: ServerOptions.Guardrails? = nil) {
        self.address = info.baseURL?.absoluteString ?? "\(info.host):\(info.port)"
        self.authLabel = info.authEnabled ? "API key required" : "none"
        self.authIsWarning = !info.authEnabled
        switch guardrails {
        case .some(.on): self.guardrailsLabel = "on"
        case .some(.off): self.guardrailsLabel = "off"
        // Not "on": the server was started before this app tracked the value,
        // so what it is running is genuinely unknown here and saying "on"
        // would be a guess presented as a reading (Gotcha 23).
        case .none: self.guardrailsLabel = "unknown (restart the server to report it)"
        }
    }
}

/// One model the server is serving, as the pane's table draws it.
public struct ServerModelRow: Identifiable, Equatable {
    /// The id clients address it by, and the key `detach` takes.
    public let id: String
    /// The install this row came from, when the app opened it. Absent for a
    /// model attached by something else, which cannot happen today and is
    /// modelled honestly rather than force-unwrapped.
    public let displayName: String
    public let maxContext: UInt32
    public let expertCacheSlots: Int
    /// Whether this is also the model the Chat pane is talking to. Worth
    /// showing: detaching it here does NOT unload it, because the chat still
    /// holds its own session.
    public let isChatSession: Bool
    public let requestsServed: Int
    /// What this model is steering with, or `nil` when it is not.
    ///
    /// **READ BACK OFF THE SESSION, AND READ-ONLY HERE.** Steering resolves
    /// once at OPEN, and this server serves models that were already opened
    /// -- so it is a property of the model's load rather than of the server,
    /// and an editable control in this pane would silently re-open the model
    /// (tens of seconds, and a dropped KV cache) to take effect. Showing it
    /// is what lets an operator tell two served models apart when one is
    /// edited and the other is not.
    public let steeringSummary: String?
}

extension AppModel {
    /// The key a typed field actually starts a server with, or `nil` for no
    /// auth at all.
    ///
    /// Trimming is why this is worth naming: a field holding nothing but
    /// spaces resolves to `nil` and starts an UNAUTHENTICATED server, which
    /// is a legitimate reading of an empty field and is indistinguishable
    /// from a key that took unless something downstream says so. The Auth row
    /// is that something.
    ///
    /// `nonisolated` because it is: a pure function of its argument that
    /// touches no `AppModel` state and lives here for the name alone. Hopping
    /// to the main actor to trim a string would be an accommodation rather
    /// than a fact about the code.
    nonisolated static func serverAPIKey(from input: String) -> String? {
        let key = input.trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : key
    }
}

extension AppModel {

    // MARK: - Lifecycle

    /// Starts an in-process HTTP server, attaching the currently open
    /// `session` when there is one.
    ///
    /// **STARTS EVEN WITH NO MODEL LOADED**, which is the order the Server
    /// pane wants: the address exists, and can be shown and copied, before a
    /// user has decided what to serve. Generation routes answer 503 until
    /// something is attached, which is a state the pane names.
    ///
    /// Requests an OS-assigned port (`ServerOptions.port == 0`) unless the
    /// user pinned one. The bound ADDRESS -- host as well as port -- is read
    /// back from `TurboSparkServer.info()`; nothing here states one of its
    /// own.
    /// The guardrails a server started under `mode` runs with.
    ///
    /// **THE SERVED PATH IS A SECOND ENFORCEMENT POINT, and until this
    /// existed the app's guardrails setting did not reach it at all.**
    /// `ForgeGuardrailsEngine` runs in the agent loop, over a reply this app
    /// read itself; every HTTP client of this server bypasses that entirely
    /// and gets whatever the engine defaults to. A user who set "Always Off"
    /// and then pointed a client at this server got guardrails anyway, with
    /// nothing anywhere saying so.
    ///
    /// **`.select` RESOLVES TO ON, and that is forced rather than chosen.**
    /// It means "decide per project or per chat", and a server request has
    /// neither -- so there is no per-request answer to resolve, and the
    /// engine's own default is the honest fallback. `nonisolated` and static
    /// so it can be asserted without a model (Gotcha 26).
    nonisolated public static func serverGuardrails(
        from mode: AppGuardrailsMode
    ) -> ServerOptions.Guardrails {
        switch mode {
        case .alwaysOff: return .off
        case .alwaysOn, .select: return .on
        }
    }

    public func startServer() {
        guard server == nil, !serverBusy else { return }
        serverBusy = true
        Task {
            defer { serverBusy = false }
            do {
                let options = ServerOptions(
                    port: serverPinnedPort,
                    apiKey: Self.serverAPIKey(from: serverAPIKeyInput),
                    guardrails: Self.serverGuardrails(from: guardrailsMode)
                )
                let started = try TurboSparkServer.start(options: options)
                // Recorded from the OPTIONS the start actually used, so the
                // pane reports what is running rather than what the setting
                // says now.
                serverStartedGuardrails = options.guardrails
                // Attached AFTER the bind so a refused attach leaves a
                // stoppable server rather than a half-started one.
                if let session {
                    do {
                        let id = try started.attach(session)
                        serverAttachedSessions[id] = session
                    } catch {
                        started.stop()
                        throw error
                    }
                }
                // Read BEFORE publishing the handle. A throwing `info()` on a
                // successfully started server would otherwise leave one
                // running and serving behind a "Failed to start" toast and a
                // toggle showing on -- the UI disagreeing with the machine.
                let info: ServerInfo
                do {
                    info = try started.info()
                } catch {
                    started.stop()
                    throw error
                }
                // Honour a stop pressed while the bind was in flight, rather
                // than publishing a server the user has already switched off.
                if self.serverStopRequested {
                    self.serverStopRequested = false
                    started.stop()
                    self.serverAttachedSessions = [:]
                    showToast("Server stopped", style: .info)
                    return
                }
                self.server = started
                self.serverInfo = info
                self.serverMetrics = ServerMetricsStore()
                self.serverEventLog = []
                self.startServerPolling()
                let auth = info.authEnabled
                    ? "API key required"
                    : "no API key: any process on this machine can reach it"
                showToast(
                    "Server listening on \(info.host):\(info.port) (\(auth))",
                    style: .success
                )
            } catch {
                // **THE STOP REQUEST IS RESET ON FAILURE TOO** (state#53).
                // It is set by a Stop pressed during the bind and cleared
                // only on the success path, so a start that THREW left it
                // latched -- and the next successful start read it at the
                // publish point and stopped itself, reporting "Server
                // stopped" for a server the user had just asked for.
                self.serverStopRequested = false
                let msg = "Failed to start server: \(error.localizedDescription)"
                self.error = msg
                showToast(msg, style: .error)
            }
        }
    }

    /// Stops the running server, if any. Safe to call when none is running.
    ///
    /// This is what releases every attached model, so a host that stops the
    /// server is not leaking anything it attached.
    public func stopServer() {
        // **A STOP DURING A START USED TO VANISH.** `server` is published only
        // after the awaited bind, so this `guard` found nil and returned --
        // toggling on then off quickly left a server listening that the UI
        // showed as stopped. The request is recorded instead, and the start
        // path honours it as soon as it has a handle to stop.
        guard let server else {
            if serverBusy {
                serverStopRequested = true
            }
            return
        }
        stopServerPolling()
        // A final unbounded drain, so whatever the last tick missed still
        // reaches the console before the handle goes.
        ingestServerEvents(server.poll(max: 0))
        server.stop()
        self.server = nil
        self.serverInfo = nil
        self.serverAttachedSessions = [:]
        self.serverStartedGuardrails = nil
        showToast("Server stopped", style: .info)
    }

    // MARK: - Reading state

    /// Re-reads what the server reports and publishes it.
    ///
    /// **THE VIEW READS THE PUBLISHED SNAPSHOT, NEVER `info()`.** That call
    /// takes a lock, crosses the C ABI and decodes JSON, and a SwiftUI body
    /// runs far more often than a server changes. This is called when
    /// something actually changed it -- start, attach, detach -- and once a
    /// second by the poll timer for the uptime.
    public func refreshServerInfo() {
        guard let server else {
            serverInfo = nil
            serverInfoErrorReported = false
            return
        }
        do {
            serverInfo = try server.info()
            serverInfoErrorReported = false
        } catch {
            // **A TRANSIENT FAILURE IS NOT A SERVER WITH NO STATE**
            // (state#86). `try?` wrote nil, and this runs twice a second: one
            // failed read blanked the address, the port, the auth row and
            // every model row, which reads as the server having stopped. The
            // last good snapshot is the honest thing to keep showing, and the
            // error is latched so the banner appears once rather than at
            // 2 Hz.
            if !serverInfoErrorReported {
                serverInfoErrorReported = true
                self.error =
                    "Could not read the server's status: \(error.localizedDescription). The "
                    + "panel is showing the last reading."
            }
        }
    }

    /// The rows the pane's model table draws.
    public var serverModelRows: [ServerModelRow] {
        let served = Dictionary(
            grouping: serverMetrics.records.compactMap(\.servedModel), by: { $0 }
        ).mapValues(\.count)
        return (serverInfo?.models ?? []).map { id in
            let attached = serverAttachedSessions[id]
            return ServerModelRow(
                id: id,
                displayName: installed.first { Self.servedModelID(for: $0) == id }?.alias ?? id,
                // Read off the SESSION rather than restated: under automatic
                // sizing nothing was asked for, and these are what the KV
                // cache was actually allocated at (`swift/CLAUDE.md`
                // Gotcha 6).
                maxContext: attached.map { UInt32($0.info.maxContext) } ?? 0,
                expertCacheSlots: attached?.info.expertCacheSlots ?? 0,
                isChatSession: attached != nil && attached === session,
                requestsServed: served[id] ?? 0,
                steeringSummary: attached.flatMap { session in
                    session.info.steering.active ? session.info.steering.summary : nil
                })
        }
    }

    /// Installed models that are NOT already being served, which is what the
    /// pane's Load Model list offers.
    public var serverAttachableModels: [InstalledModel] {
        let attached = Set(serverInfo?.models ?? [])
        return installed.filter { !attached.contains(Self.servedModelID(for: $0)) }
    }

    /// The id a server serves this install under: its directory's own file
    /// name, which is what `crates/ffi` keys the registry on.
    ///
    /// **ONE DERIVATION, NOT TWO.** `serverAttachableModels` compared exact
    /// `lastPathComponent` while `serverModelRows` used `path.hasSuffix(id)`,
    /// and a suffix match is not a name match: an install at
    /// `.../mygemma4.gturbo` ends with `gemma4.gturbo`, so the row for an
    /// attached `gemma4.gturbo` was labelled with the WRONG install's alias
    /// while that install still appeared in the attachable list.
    static func servedModelID(for model: InstalledModel) -> String {
        (model.path as NSString).lastPathComponent
    }

    // MARK: - Polling

    /// Starts draining the server's event ring.
    ///
    /// **2 Hz, AND THE RATE IS A UI DECISION RATHER THAN A LIMIT.** The ring
    /// holds ~2,000 events, which is hundreds of requests, so nothing is lost
    /// at any polling rate a person would choose -- this one is picked so the
    /// console feels live without re-rendering the pane more often than a
    /// screen refresh could show.
    public func startServerPolling() {
        stopServerPolling()
        let timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.pollServerOnce()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        serverPollTimer = timer
    }

    public func stopServerPolling() {
        serverPollTimer?.invalidate()
        serverPollTimer = nil
    }

    private func pollServerOnce() {
        guard let server else { return }
        ingestServerEvents(server.poll(max: 256))
        // The uptime moves on its own, so the snapshot is refreshed on the
        // timer as well as on the events that change the model list.
        refreshServerInfo()
    }

    private func ingestServerEvents(_ batch: ServerEventBatch) {
        guard !batch.events.isEmpty || batch.dropped > 0 else { return }
        serverMetrics.ingest(batch)
        serverEventLog.append(contentsOf: batch.events)
        // The console's own window. Bounded for the reason the metrics store
        // is: a server left running overnight must not grow this without
        // limit, and a console nobody has scrolled to has nothing to lose.
        if serverEventLog.count > Self.serverEventLogCapacity {
            serverEventLog.removeFirst(serverEventLog.count - Self.serverEventLogCapacity)
        }
    }

    static let serverEventLogCapacity = 4_000
}
