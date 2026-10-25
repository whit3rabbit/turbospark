import Foundation

/// The menu bar panel's one-glance server state.
///
/// Kept apart from the view so the mapping from app state to a status light
/// is a pure function the tests can pin, rather than a chain of `if`s inside
/// a `body` that only a screenshot could check.
enum MenuBarServerPhase: Equatable {
    case stopped
    case starting
    /// Running and busy: an embedding or audio attach holds `serverBusy`.
    case working
    /// The last start threw. Carries the engine's message.
    case failed(String)
    /// Running with nothing attached: reachable, but every request fails.
    case idle
    /// Running, but the last `info()` read failed and the panel is showing
    /// the previous snapshot.
    case degraded
    case running
    /// No in-app server, but the background daemon is up at this address.
    case daemon(endpoint: String)

    /// The light at the top of the panel. Stopped is gray, not red: a server
    /// the user turned off is not a fault.
    enum Level: Equatable {
        case off, ok, warning, error
    }

    var level: Level {
        switch self {
        case .stopped: return .off
        case .running, .daemon: return .ok
        case .starting, .working, .idle, .degraded: return .warning
        case .failed: return .error
        }
    }

    static func resolve(
        serverRunning: Bool, serverBusy: Bool, servedModelCount: Int,
        infoReadFailing: Bool, startFailure: String?, daemonEndpoint: String?
    ) -> Self {
        if serverRunning {
            if serverBusy { return .working }
            if infoReadFailing { return .degraded }
            return servedModelCount == 0 ? .idle : .running
        }
        // A start in flight wins over a stale failure from the last attempt.
        if serverBusy { return .starting }
        if let startFailure { return .failed(startFailure) }
        if let daemonEndpoint { return .daemon(endpoint: daemonEndpoint) }
        return .stopped
    }
}

/// The five kinds of model the panel groups residency by.
enum MenuBarModelKind: CaseIterable, Equatable {
    case chat, embedding, audio, image, decision

    var systemImage: String {
        switch self {
        case .chat: return "bubble.left"
        case .embedding: return "point.3.connected.trianglepath.dotted"
        case .audio: return "waveform"
        case .image: return "photo"
        case .decision: return "arrow.triangle.branch"
        }
    }
}

/// How one resident model is released. Each case maps to the app's existing
/// unload or detach call for that kind; the panel adds no release path of its
/// own.
enum MenuBarUnloadAction: Hashable {
    /// The Chat pane's session (`unloadModel`, which also stops serving it).
    case chatSession
    /// A server attachment with no other owner (`detachModelFromServer`).
    case detach(id: String)
    /// The served text-to-speech model (`detachAudioSpeechModelFromServer`).
    case speechModel
    /// The shared image pipeline (`unloadImageModel`).
    case imageSession
    /// A served image model whose pipeline is not open (`detachImageModelFromServer`).
    case servedImage
    /// The Audio workspace's resident model.
    case audioWorkspace
    /// The TypeSafe `openkindd` process.
    case typeSafe
}

/// One row of the panel's Loaded list.
struct MenuBarModelSlot: Identifiable, Equatable {
    let kind: MenuBarModelKind
    let name: String
    let unload: MenuBarUnloadAction
    /// Resident bytes for this model.
    ///
    /// STUB: always nil. The engine reports no per-model residency yet, and
    /// the size on disk is not a stand-in: a dense model's weights are mapped
    /// and mostly not charged to the footprint. The row shows a size only
    /// once this is measured.
    var residentBytes: UInt64?

    var id: String { "\(kind)|\(name)|\(unload)" }
}

/// A model the server attached, by the id it serves and the name to show.
struct MenuBarServedModel: Equatable {
    let id: String
    let name: String
}

/// What the Loaded list is built from, copied out of `AppModel` so the
/// derivation can be tested without one.
struct MenuBarSlotInput: Equatable {
    /// The Chat pane's loaded model, when a session is open.
    var chatSessionName: String?
    /// Server ids the chat session is attached under, so it is not listed twice.
    var chatServedIDs: Set<String> = []
    /// Text models the server is serving.
    var servedText: [MenuBarServedModel] = []
    var embeddingIDs: [String] = []
    var audioIDs: [String] = []
    var speechAlias: String?
    /// The shared image pipeline's model, when it is open.
    var imageSessionName: String?
    var imageServedIDs: [String] = []
    var audioWorkspaceAlias: String?
    var typeSafeRunning = false
}

enum MenuBarSlots {
    /// The `openkindd` daemon is one process whatever it loads, so it is
    /// listed under its executable's name.
    static let typeSafeProcessName = "openkindd"

    static func build(_ input: MenuBarSlotInput) -> [MenuBarModelSlot] {
        var slots: [MenuBarModelSlot] = []
        if let chat = input.chatSessionName {
            slots.append(MenuBarModelSlot(kind: .chat, name: chat, unload: .chatSession))
        }
        for model in input.servedText where !input.chatServedIDs.contains(model.id) {
            slots.append(MenuBarModelSlot(kind: .chat, name: model.name, unload: .detach(id: model.id)))
        }
        for id in input.embeddingIDs {
            slots.append(MenuBarModelSlot(kind: .embedding, name: id, unload: .detach(id: id)))
        }
        if let alias = input.audioWorkspaceAlias {
            slots.append(MenuBarModelSlot(kind: .audio, name: alias, unload: .audioWorkspace))
        }
        for id in input.audioIDs {
            slots.append(MenuBarModelSlot(kind: .audio, name: id, unload: .detach(id: id)))
        }
        if let speech = input.speechAlias, !input.audioIDs.contains(speech) {
            slots.append(MenuBarModelSlot(kind: .audio, name: speech, unload: .speechModel))
        }
        // The Images screen and the Image API share one resident pipeline, so
        // an open pipeline is the memory; a served id without one holds none
        // until a request reopens it.
        if let image = input.imageSessionName {
            slots.append(MenuBarModelSlot(kind: .image, name: image, unload: .imageSession))
        } else {
            for id in input.imageServedIDs {
                slots.append(MenuBarModelSlot(kind: .image, name: id, unload: .servedImage))
            }
        }
        if input.typeSafeRunning {
            slots.append(MenuBarModelSlot(kind: .decision, name: typeSafeProcessName, unload: .typeSafe))
        }
        return slots
    }
}

/// The panel's memory bar.
struct MenuBarMemoryReading: Equatable {
    /// This process's physical footprint, which every in-app model counts
    /// toward.
    let usedBytes: UInt64
    /// The bar's full scale: the model memory budget when one is set, else
    /// physical RAM.
    let capacityBytes: UInt64

    /// STUB: no model memory budget exists yet, so the bar scales against
    /// physical RAM. When the setting lands it replaces RAM here, and a load
    /// that would cross it can be refused before it starts.
    static let budgetBytes: UInt64? = nil

    init(usedBytes: UInt64, physicalBytes: UInt64, budgetBytes: UInt64? = Self.budgetBytes) {
        self.usedBytes = usedBytes
        self.capacityBytes = budgetBytes ?? physicalBytes
    }

    var fraction: Double {
        guard capacityBytes > 0 else { return 0 }
        return min(1, Double(usedBytes) / Double(capacityBytes))
    }

    static func format(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes), countStyle: .memory)
    }
}

enum MenuBarUptime {
    /// Verbatim and unit-only, like the status strip's readouts.
    static func text(_ seconds: UInt64) -> String {
        if seconds < 60 { return "\(seconds)s" }
        if seconds < 3600 { return "\(seconds / 60)m" }
        return "\(seconds / 3600)h \((seconds % 3600) / 60)m"
    }
}
