import Foundation
import Combine

struct BrowserRuntimeActionScope {
    let call: AppToolCall
    let project: AppProject?
    let permissionContext: BrowserPermissionContext
}

@MainActor
private final class BrowserCoordinatorTakeoverRelay {
    weak var coordinator: AppBrowserAutomationCoordinator?
}

/// Owns the browser engine and automation-session attachment for the app shell.
@MainActor
public final class AppBrowserAutomationCoordinator: ObservableObject {
    public let tabStore: BrowserTabStore
    public let engine: WebKitBrowserEngine
    public let automationSession: BrowserAutomationSession

    public private(set) var isPaneOpen = false
    private(set) var agentTabID: BrowserTabID?
    private var ownershipToken: BrowserAgentControlToken?
    private(set) var activeAction: BrowserRuntimeActionScope?

    public init(
        authorizeNavigation: @escaping BrowserNavigationAuthorizer
    ) {
        let tabStore = BrowserTabStore()
        let takeoverRelay = BrowserCoordinatorTakeoverRelay()
        let engine = WebKitBrowserEngine(
            tabStore: tabStore,
            authorizeNavigation: authorizeNavigation,
            onUserTakeover: { tabID in
                takeoverRelay.coordinator?.userTookOver(tabID: tabID)
            }
        )
        self.tabStore = tabStore
        self.engine = engine
        self.automationSession = BrowserAutomationSession(
            tabStore: tabStore,
            portFactory: { tabID in
                WebKitBrowserControlPort(engine: engine, tabID: tabID)
            }
        )
        takeoverRelay.coordinator = self
    }

    public func openPane() {
        isPaneOpen = true
        if tabStore.tabs.isEmpty {
            _ = engine.createTab(owner: .user)
        }
    }

    public func closePane() async {
        isPaneOpen = false
        activeAction = nil
        agentTabID = nil
        ownershipToken = nil
        await automationSession.detach()
    }

    public func paneWillClose() {
        isPaneOpen = false
    }

    func runtime(
        for call: AppToolCall,
        project: AppProject?,
        availability: BrowserToolAvailability
    ) -> BrowserToolRuntime? {
        guard availability.isEnabled else { return nil }
        let origin: BrowserOrigin?
        if call.name.lowercased() == "browser_navigate",
           let rawURL = call.arguments["url"],
           let url = URL(string: rawURL) {
            origin = BrowserOrigin(url: url)
        } else if let agentTabID,
                  let address = engine.tabStates[agentTabID]?.address ?? tabStore.tab(id: agentTabID)?.address,
                  let url = URL(string: address) {
            origin = BrowserOrigin(url: url)
        } else {
            origin = nil
        }
        guard origin != nil else { return nil }

        return BrowserToolRuntime(
            availability: availability,
            permissionContext: BrowserPermissionContext(origin: origin),
            performWithPermissionContext: { [weak self] command, context in
                guard let self else { throw BrowserControlError.engineCrashed }
                return try await self.perform(
                    command,
                    call: call,
                    project: project,
                    permissionContext: context
                )
            }
        )
    }

    public func userTookOver(tabID: BrowserTabID) {
        guard agentTabID == tabID else { return }
        agentTabID = nil
        ownershipToken = nil
        activeAction = nil
        let automationSession = self.automationSession
        Task { await automationSession.userTookOver(tabID: tabID) }
    }

    private func perform(
        _ command: BrowserControlCommand,
        call: AppToolCall,
        project: AppProject?,
        permissionContext: BrowserPermissionContext
    ) async throws -> BrowserControlResult {
        guard isPaneOpen else { throw BrowserAutomationSessionError.detached }
        // The user may have closed (or the engine lost) the agent tab. Release
        // the session's attachment to it BEFORE a replacement tab is created,
        // or attach(to:) fails with alreadyAttached on every later call.
        // Done ahead of the in-flight guard so no await sits between it and
        // the activeAction assignment.
        if let stale = agentTabID, tabStore.tab(id: stale)?.owner != .agent {
            agentTabID = nil
            ownershipToken = nil
            await automationSession.tabClosed(tabID: stale)
        }
        guard activeAction == nil else { throw BrowserAutomationSessionError.commandInFlight }
        let tabID = controlledTabID()
        let scope = BrowserRuntimeActionScope(
            call: call,
            project: project,
            permissionContext: permissionContext
        )
        activeAction = scope
        defer {
            if activeAction?.call.id == call.id { activeAction = nil }
        }

        let token: BrowserAgentControlToken
        if let ownershipToken {
            token = ownershipToken
        } else {
            token = try await automationSession.attach(to: tabID)
            ownershipToken = token
        }
        let timeout: TimeInterval?
        switch command {
        case let .navigate(_, _, value), let .click(_, value), let .type(_, _, _, value),
             let .pressKey(_, _, value):
            timeout = value
        case let .waitFor(_, value):
            timeout = value
        case .scroll, .screenshot, .readState, .setViewport:
            timeout = nil
        }
        return try await automationSession.perform(command, using: token, timeoutSeconds: timeout)
    }

    private func controlledTabID() -> BrowserTabID {
        if let agentTabID,
           tabStore.tab(id: agentTabID)?.owner == .agent {
            return agentTabID
        }
        let tabID = engine.createTab(owner: .agent, select: true)
        self.agentTabID = tabID
        ownershipToken = nil
        return tabID
    }
}
