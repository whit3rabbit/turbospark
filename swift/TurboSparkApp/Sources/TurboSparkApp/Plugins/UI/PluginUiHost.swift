import Foundation

/// Persisted per-plugin opt-in: which plugins may show UI pages. Off for
/// every plugin until the person turns it on in the plugin's detail pane.
/// Keyed by plugin id (`name@origin`). The file is written through
/// `AppJSONStore`, which quarantines a corrupt file instead of overwriting
/// it, and a corrupt or missing file means nothing is opted in.
struct PluginUiOptInArchive: Codable, Equatable {
    var enabled: [String] = []
}

@MainActor
final class PluginUiOptInStore: ObservableObject {
    nonisolated static let fileName = "plugin_ui_optin.json"

    @Published private(set) var enabledIDs: Set<String>
    private let fileURL: URL?

    init(fileURL: URL? = AppStorageRoot.file(PluginUiOptInStore.fileName)) {
        self.fileURL = fileURL
        if let fileURL,
            let archive = AppJSONStore.load(PluginUiOptInArchive.self, from: fileURL, label: "Plugin UI opt-in")
        {
            enabledIDs = Set(archive.enabled)
        } else {
            enabledIDs = []
        }
    }

    func isEnabled(_ pluginID: String) -> Bool { enabledIDs.contains(pluginID) }

    func setEnabled(_ enabled: Bool, pluginID: String) {
        if enabled { enabledIDs.insert(pluginID) } else { enabledIDs.remove(pluginID) }
        guard let fileURL else { return }
        AppJSONStore.save(
            PluginUiOptInArchive(enabled: enabledIDs.sorted()), to: fileURL, label: "Plugin UI opt-in")
    }
}

/// Why `PluginUiHost.open` did not show a page.
enum PluginUiOpenRefusal: Error, Equatable {
    /// The plugin has not been opted into showing UI pages.
    case notOptedIn
    /// The plugin is disabled or no longer installed.
    case pluginUnavailable
    /// No such surface in the plugin's manifest.
    case noSuchSurface
    /// Every live page is in use, so none can be evicted for this one.
    case pageLimitReached
}

/// The page registry: identity to live session, the live-page cap, eviction,
/// and teardown. Identity is plugin id + surface id + session id (the chat
/// the page was opened from).
///
/// Memory: each page is a web content process in a Mac that also holds model
/// weights, so the cap is small. Opening past it closes the least recently
/// visible page that is not being interacted with; if every page is in use
/// the new one is refused with a named reason.
@MainActor
final class PluginUiHost: ObservableObject {
    /// Provisional. ZCode allows 64 on Electron; this app shares unified
    /// memory with model weights.
    static let maxLivePages = 8
    /// A page with a click or key press this recent, or a tool call in
    /// flight, is never evicted.
    static let interactingWindowSeconds: TimeInterval = 30

    struct Dependencies {
        var optIn: PluginUiOptInStore
        /// Resolves an enabled plugin by id for the current project, nil when
        /// it is disabled or gone.
        var plugin: (_ pluginID: String) -> LoadedPlugin?
        var invoker: (_ pluginName: String, _ sessionID: String) -> PageBridge.ToolInvoker
        var draftSink: @MainActor (String) -> Void
        var onSidePaneChange: @MainActor (UUID?) -> Void
        var now: () -> Date = Date.init
    }

    private let dependencies: Dependencies
    let widgetStates = PluginWidgetStateStore()
    let optIn: PluginUiOptInStore

    /// Live pages, least recently visible first.
    @Published private(set) var pages: [PluginPageSession] = []
    /// The page the side pane shows.
    @Published private(set) var sidePaneID: UUID?

    init(_ dependencies: Dependencies) {
        self.dependencies = dependencies
        self.optIn = dependencies.optIn
    }

    func session(id: UUID) -> PluginPageSession? { pages.first { $0.id == id } }

    func session(for identity: PluginPageIdentity) -> PluginPageSession? {
        pages.first { $0.identity == identity }
    }

    var sidePaneSession: PluginPageSession? { sidePaneID.flatMap(session(id:)) }

    private func isInteracting(_ session: PluginPageSession) -> Bool {
        if session.bridge.pendingToolCalls > 0 { return true }
        guard let last = session.bridge.lastUserGestureAt else { return false }
        return dependencies.now().timeIntervalSince(last) <= Self.interactingWindowSeconds
    }

    /// Opens (or re-shows) a surface's page in the side pane.
    func open(pluginID: String, surfaceID: String, sessionID: String) -> Result<PluginPageSession, PluginUiOpenRefusal> {
        guard dependencies.optIn.isEnabled(pluginID) else { return .failure(.notOptedIn) }
        guard let plugin = dependencies.plugin(pluginID) else { return .failure(.pluginUnavailable) }
        guard let surface = plugin.manifest.uiSurfaces.first(where: { $0.id == surfaceID }) else {
            return .failure(.noSuchSurface)
        }
        let identity = PluginPageIdentity(pluginID: pluginID, surfaceID: surfaceID, sessionID: sessionID)
        if let existing = session(for: identity) {
            show(existing)
            return .success(existing)
        }
        while pages.count >= Self.maxLivePages {
            // Least recently visible first; the page on screen is the most
            // recent by construction, and an interacting page is skipped.
            guard let victim = pages.first(where: { !isInteracting($0) && $0.id != sidePaneID }) else {
                return .failure(.pageLimitReached)
            }
            close(victim)
        }
        let environment = PluginPageEnvironment(
            invoker: dependencies.invoker(plugin.name, sessionID),
            draftSink: dependencies.draftSink,
            widgetStates: widgetStates,
            now: dependencies.now)
        let created = PluginPageSession(
            identity: identity, surface: surface, pluginName: plugin.name, root: plugin.directoryURL,
            environment: environment,
            declaredTools: PluginUiToolNames.declaredTools(pluginName: plugin.name, surface: surface))
        pages.append(created)
        created.load()
        show(created)
        return .success(created)
    }

    /// Marks a page as the most recently visible and puts it in the side pane.
    func show(_ session: PluginPageSession) {
        pages.removeAll { $0.id == session.id }
        pages.append(session)
        sidePaneID = session.id
        dependencies.onSidePaneChange(session.id)
    }

    /// Tears the page down and frees its slot. Widget state goes with it.
    func close(_ session: PluginPageSession) {
        session.tearDown()
        pages.removeAll { $0.id == session.id }
        if sidePaneID == session.id {
            sidePaneID = nil
            dependencies.onSidePaneChange(nil)
        }
    }

    func closePane() {
        if let current = sidePaneSession { close(current) }
    }

    /// Chat switch: pages belong to the chat they were opened from.
    func closePages(exceptSession sessionID: String) {
        for page in pages where page.identity.sessionID != sessionID { close(page) }
    }

    /// Plugin disabled, removed, or opted out.
    func closePages(forPlugin pluginID: String) {
        for page in pages where page.identity.pluginID == pluginID { close(page) }
    }

    /// After any plugin state change: a page whose plugin was disabled,
    /// removed or opted out must not outlive that decision.
    func reconcile() {
        for page in pages where dependencies.plugin(page.identity.pluginID) == nil
            || !dependencies.optIn.isEnabled(page.identity.pluginID)
        {
            close(page)
        }
    }

    /// App quit.
    func closeAll() {
        for page in pages { close(page) }
    }
}
