import Foundation
import TurboSpark

/// The menu bar panel's reads of app state and its one write path.
///
/// Every value here is derived from state other panes already own; the panel
/// keeps none of its own, so it cannot disagree with the Server pane.
extension AppModel {
    var menuBarPhase: MenuBarServerPhase {
        let info = serverInfo
        let served = Set((info?.models ?? []) + (info?.imageModels ?? []) + (info?.audioModels ?? []))
        return MenuBarServerPhase.resolve(
            serverRunning: server != nil,
            serverBusy: serverBusy,
            servedModelCount: served.count,
            infoReadFailing: serverInfoErrorReported,
            startFailure: serverStartFailure,
            // Only asked when the in-app server is off: the daemon read touches
            // files and processes, and an in-app server answers the question.
            daemonEndpoint: server == nil ? menuBarDaemonEndpoint : nil)
    }

    /// The background daemon's address, when it is running.
    var menuBarDaemonEndpoint: String? {
        guard let daemon = try? TurboSparkDaemon.status(), daemon.running else { return nil }
        return daemon.endpoint ?? "http://127.0.0.1:\(daemon.port ?? 8080)/v1"
    }

    /// The address a client should use, when something is listening. Takes
    /// the phase the caller already resolved, so the daemon is read once.
    func menuBarAddress(for phase: MenuBarServerPhase) -> String? {
        if let info = serverInfo {
            return info.baseURL?.absoluteString ?? "http://\(info.host):\(info.port)"
        }
        if case .daemon(let endpoint) = phase { return endpoint }
        return nil
    }

    var menuBarSlotInput: MenuBarSlotInput {
        let info = serverInfo
        let image = Set(info?.imageModels ?? [])
        let audio = Set(info?.audioModels ?? [])
        let text = serverTextModelIDs.filter { !image.contains($0) && !audio.contains($0) }
        return MenuBarSlotInput(
            chatSessionName: session == nil ? nil : (selected?.alias ?? modelPathText),
            chatServedIDs: Set(serverAttachedSessions.filter { $0.value === session }.map(\.key)),
            servedText: text.map { id in
                MenuBarServedModel(
                    id: id,
                    name: installed.first { Self.servedModelID(for: $0) == id }?.alias ?? id)
            },
            embeddingIDs: serverEmbeddingModelIDs,
            audioIDs: info?.audioModels ?? [],
            speechAlias: server == nil ? nil : serverSpeechAttachedModel?.alias,
            imageSessionName: imageSession == nil
                ? nil
                : imageSessionPath.map { ($0 as NSString).lastPathComponent },
            imageServedIDs: info?.imageModels ?? [],
            audioWorkspaceAlias: audioWorkspace.loadedModelAlias,
            typeSafeRunning: typeSafeServer != nil)
    }

    var menuBarSlots: [MenuBarModelSlot] { MenuBarSlots.build(menuBarSlotInput) }

    var menuBarMemory: MenuBarMemoryReading {
        MenuBarMemoryReading(
            usedBytes: Self.currentPhysFootprintBytes() ?? 0,
            physicalBytes: ProcessInfo.processInfo.physicalMemory)
    }

    /// Whether the slot's release can run now, by the same predicate the
    /// owning pane disables its own control with.
    func menuBarCanUnload(_ action: MenuBarUnloadAction) -> Bool {
        switch action {
        case .chatSession: return canUnloadModel
        case .detach: return server != nil && !serverBusy
        case .speechModel, .servedImage: return server != nil
        case .imageSession: return canUnloadImageModel
        case .audioWorkspace: return audioWorkspace.canUnloadModel
        case .typeSafe: return !typeSafeBusy
        }
    }

    func menuBarUnload(_ action: MenuBarUnloadAction) {
        guard menuBarCanUnload(action) else { return }
        switch action {
        case .chatSession: unloadModel()
        case .detach(let id): detachModelFromServer(id: id)
        case .speechModel: detachAudioSpeechModelFromServer()
        case .imageSession: unloadImageModel()
        case .servedImage: detachImageModelFromServer()
        case .audioWorkspace: audioWorkspace.unloadModel()
        case .typeSafe: Task { await stopTypeSafeServer() }
        }
    }
}
