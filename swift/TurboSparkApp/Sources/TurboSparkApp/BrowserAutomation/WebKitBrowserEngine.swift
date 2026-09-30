import AppKit
import Combine
import Foundation
import WebKit

public struct BrowserNavigationDestination: Equatable, Sendable {
    public let url: URL
    public let origin: BrowserOrigin

    public init?(url: URL) {
        guard let origin = BrowserOrigin(url: url) else { return nil }
        self.url = url
        self.origin = origin
    }
}

public enum BrowserNavigationCheckPhase: Equatable, Sendable {
    case explicitNavigation
    case mainFrameAction
    case formSubmission
    case redirect
    case mainFrameResponse
}

public struct BrowserNavigationAuthorizationRequest: Equatable, Sendable {
    public let tabID: BrowserTabID
    public let destination: BrowserOrigin
    public let source: BrowserOrigin?
    public let phase: BrowserNavigationCheckPhase

    public init(
        tabID: BrowserTabID,
        destination: BrowserOrigin,
        source: BrowserOrigin?,
        phase: BrowserNavigationCheckPhase
    ) {
        self.tabID = tabID
        self.destination = destination
        self.source = source
        self.phase = phase
    }
}

public enum BrowserNavigationAuthorizationDecision: Equatable, Sendable {
    case allow
    case deny
}

public struct BrowserEngineTabState: Equatable, Sendable {
    public let tabID: BrowserTabID
    public var address: String?
    public var title: String?
    public var progress: Double
    public var loadState: BrowserTabLoadState
    public var loadError: String?
    public var navigationError: BrowserControlError?

    public init(
        tabID: BrowserTabID,
        address: String?,
        title: String?,
        progress: Double = 0,
        loadState: BrowserTabLoadState = .idle,
        loadError: String? = nil,
        navigationError: BrowserControlError? = nil
    ) {
        self.tabID = tabID
        self.address = address
        self.title = title
        self.progress = progress
        self.loadState = loadState
        self.loadError = loadError
        self.navigationError = navigationError
    }
}

public enum WebKitBrowserEngineError: Error, Equatable {
    case noRecoverableAddress
    case unavailableWebView
}

public typealias BrowserNavigationAuthorizer = @MainActor (
    BrowserNavigationAuthorizationRequest
) async -> BrowserNavigationAuthorizationDecision

public typealias BrowserUserTakeoverHandler = @MainActor (BrowserTabID) -> Void

private final class BrowserTakeoverWebView: WKWebView {
    var onDirectInput: (() -> Void)?

    override func mouseDown(with event: NSEvent) {
        onDirectInput?()
        super.mouseDown(with: event)
    }

    override func rightMouseDown(with event: NSEvent) {
        onDirectInput?()
        super.rightMouseDown(with: event)
    }

    override func otherMouseDown(with event: NSEvent) {
        onDirectInput?()
        super.otherMouseDown(with: event)
    }

    override func keyDown(with event: NSEvent) {
        onDirectInput?()
        super.keyDown(with: event)
    }

    override func scrollWheel(with event: NSEvent) {
        onDirectInput?()
        super.scrollWheel(with: event)
    }
}

@MainActor
private final class BrowserNavigationPolicyGate {
    private var completion: ((Bool) -> Void)?

    init(completion: @escaping (Bool) -> Void) {
        self.completion = completion
    }

    func resolve(_ shouldAllow: Bool) {
        let completion = self.completion
        self.completion = nil
        completion?(shouldAllow)
    }
}

@MainActor
private final class ManagedBrowserTab {
    let webView: BrowserTakeoverWebView
    var observations: [NSKeyValueObservation] = []
    var snapshotService: DOMSnapshotService?
    var snapshotGeneration: UUID?
    var pendingResponseOrigins = Set<String>()
    var explicitNavigationURLs: [String: Int] = [:]
    var activeNavigationURL: String?

    init(webView: BrowserTakeoverWebView) {
        self.webView = webView
    }
}

/// Owns the one WebKit view and its navigation lifecycle for each managed tab ID.
@MainActor
public final class WebKitBrowserEngine: NSObject, ObservableObject {
    @Published public private(set) var tabStates: [BrowserTabID: BrowserEngineTabState] = [:]

    public let tabStore: BrowserTabStore

    private let authorizeNavigation: BrowserNavigationAuthorizer
    private let onUserTakeover: BrowserUserTakeoverHandler
    private var managedTabs: [BrowserTabID: ManagedBrowserTab] = [:]
    private var tabIDsByWebView: [ObjectIdentifier: BrowserTabID] = [:]
    private var pendingPolicyGates: [BrowserTabID: [UUID: BrowserNavigationPolicyGate]] = [:]

    public init(
        tabStore: BrowserTabStore,
        authorizeNavigation: @escaping BrowserNavigationAuthorizer,
        onUserTakeover: @escaping BrowserUserTakeoverHandler = { _ in }
    ) {
        self.tabStore = tabStore
        self.authorizeNavigation = authorizeNavigation
        self.onUserTakeover = onUserTakeover
        super.init()

        for tab in tabStore.tabs {
            attachWebView(to: tab)
        }
    }

    @discardableResult
    public func createTab(
        owner: BrowserTabOwner,
        address: String? = nil,
        title: String? = nil,
        select: Bool = true
    ) -> BrowserTabID {
        let tabID = tabStore.createTab(owner: owner, address: address, title: title, select: select)
        guard let tab = tabStore.tab(id: tabID) else { return tabID }
        attachWebView(to: tab)
        return tabID
    }

    public func webView(for tabID: BrowserTabID) -> WKWebView? {
        managedTabs[tabID]?.webView
    }

    /// Starts a validated HTTP(S) navigation. Agent destinations are authorized by the
    /// navigation delegate before the main-frame request is allowed to proceed.
    public func navigate(to url: URL, in tabID: BrowserTabID) throws {
        guard let destination = BrowserNavigationDestination(url: url) else {
            throw BrowserControlError.invalidInput(reason: .invalidAddress)
        }
        guard tabStore.tab(id: tabID) != nil,
              let managedTab = managedTabs[tabID] else {
            throw BrowserTabStoreError.tabNotFound(tabID)
        }

        try tabStore.startNavigation(in: tabID, to: destination.url.absoluteString)
        managedTab.pendingResponseOrigins.removeAll()
        managedTab.explicitNavigationURLs[destination.url.absoluteString, default: 0] += 1
        managedTab.activeNavigationURL = destination.url.absoluteString
        beginSnapshotNavigation(for: managedTab, origin: destination.origin)
        setNavigationStarted(tabID, destination: destination)

        guard managedTab.webView.load(URLRequest(url: destination.url)) != nil else {
            let error = BrowserControlError.navigationFailure(reason: .unknown)
            recordNavigationFailure(tabID, error: error, message: "WebKit could not start the navigation.")
            throw error
        }
    }

    /// Transfers an agent-owned tab to the user synchronously, before loading a user address.
    public func navigateAsUser(to url: URL, in tabID: BrowserTabID) throws {
        prepareForDirectUserInput(in: tabID)
        try navigate(to: url, in: tabID)
    }

    /// Called by the hosted WebView before forwarding a direct mouse, keyboard, or scroll event.
    public func prepareForDirectUserInput(in tabID: BrowserTabID) {
        guard let tab = tabStore.tab(id: tabID), tab.owner == .agent else { return }

        onUserTakeover(tabID)
        cancelPendingPolicyGates(for: tabID)
        try? tabStore.transferOwnershipToUser(of: tabID)
        refreshStoredTabState(tabID)
    }

    public func stopLoading(in tabID: BrowserTabID) {
        guard let managedTab = managedTabs[tabID] else { return }
        managedTab.webView.stopLoading()
        let message = "Navigation was stopped."
        if tabStore.tab(id: tabID)?.loadState == .loading {
            try? tabStore.failNavigation(in: tabID, reason: message)
        }
        updateState(for: tabID) { state in
            state.loadState = tabStore.tab(id: tabID)?.loadState ?? .failed(reason: message)
            state.loadError = message
            state.progress = 0
        }
    }

    public func closeTab(_ tabID: BrowserTabID) throws {
        guard let managedTab = managedTabs[tabID] else {
            throw BrowserTabStoreError.tabNotFound(tabID)
        }
        cancelPendingPolicyGates(for: tabID)
        managedTab.snapshotService?.invalidate()
        managedTab.observations.removeAll()
        managedTab.webView.stopLoading()
        managedTab.webView.navigationDelegate = nil
        managedTab.webView.uiDelegate = nil
        tabIDsByWebView.removeValue(forKey: ObjectIdentifier(managedTab.webView))
        managedTabs.removeValue(forKey: tabID)
        tabStates.removeValue(forKey: tabID)
        try tabStore.closeTab(tabID)
    }

    /// Recreates only the stored address after an explicit recovery request.
    public func recoverCrashedTab(_ tabID: BrowserTabID) throws {
        guard let tab = tabStore.tab(id: tabID) else {
            throw BrowserTabStoreError.tabNotFound(tabID)
        }
        guard tab.loadState == .crashed else {
            throw BrowserTabStoreError.invalidLoadTransition(tab.loadState)
        }
        guard let address = tab.address, let url = URL(string: address),
              BrowserNavigationDestination(url: url) != nil else {
            throw WebKitBrowserEngineError.noRecoverableAddress
        }

        try tabStore.restoreCrashedTab(tabID)
        updateState(for: tabID) { state in
            state.loadState = .restored
            state.loadError = nil
            state.navigationError = nil
            state.progress = 0
        }
        try navigate(to: url, in: tabID)
    }

    public func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let tabID = tabID(for: webView),
              let managedTab = managedTabs[tabID] else {
            decisionHandler(.cancel)
            return
        }
        guard let targetFrame = navigationAction.targetFrame else {
            // WKUIDelegate routes targetFrame == nil through BrowserTabStore.
            decisionHandler(.cancel)
            return
        }
        guard targetFrame.isMainFrame else {
            decisionHandler(.allow)
            return
        }
        guard let url = navigationAction.request.url,
              let destination = BrowserNavigationDestination(url: url) else {
            recordNavigationFailure(
                tabID,
                error: .invalidInput(reason: .invalidAddress),
                message: "The page requested a destination that is not an allowed HTTP(S) address."
            )
            decisionHandler(.cancel)
            return
        }

        let explicitPhase = consumeExplicitNavigationPhase(for: destination.url, in: managedTab)
        let phase: BrowserNavigationCheckPhase
        if let explicitPhase {
            phase = explicitPhase
        } else if navigationAction.navigationType == .formSubmitted {
            phase = .formSubmission
        } else if navigationAction.navigationType == .other,
                  tabStore.tab(id: tabID)?.loadState == .loading,
                  managedTab.activeNavigationURL != destination.url.absoluteString {
            phase = .redirect
        } else {
            phase = .mainFrameAction
        }
        let source = navigationAction.sourceFrame.request.url.flatMap(BrowserOrigin.init(url:))
        let request = BrowserNavigationAuthorizationRequest(
            tabID: tabID,
            destination: destination.origin,
            source: source,
            phase: phase
        )
        evaluatePolicy(
            request,
            tabID: tabID,
            webView: webView,
            gate: BrowserNavigationPolicyGate { [weak self, weak managedTab] shouldAllow in
                guard let self, let managedTab else { return }
                if shouldAllow {
                    managedTab.pendingResponseOrigins.insert(destination.origin.canonicalString)
                    managedTab.activeNavigationURL = destination.url.absoluteString
                    self.startLoadIfNeeded(tabID, destination: destination)
                    self.beginSnapshotNavigation(for: managedTab, origin: destination.origin)
                } else {
                    self.recordNavigationFailure(
                        tabID,
                        error: .blockedNavigation(origin: destination.origin.canonicalString),
                        message: "Navigation blocked by browser permission policy."
                    )
                }
                decisionHandler(shouldAllow ? .allow : .cancel)
            }
        )
    }

    public func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        guard navigationResponse.isForMainFrame else {
            decisionHandler(.allow)
            return
        }
        guard let tabID = tabID(for: webView),
              let managedTab = managedTabs[tabID],
              let url = navigationResponse.response.url,
              let destination = BrowserNavigationDestination(url: url) else {
            decisionHandler(.cancel)
            return
        }

        if tabStore.tab(id: tabID)?.owner == .agent,
           managedTab.pendingResponseOrigins.remove(destination.origin.canonicalString) != nil {
            managedTab.activeNavigationURL = destination.url.absoluteString
            startLoadIfNeeded(tabID, destination: destination)
            beginSnapshotNavigation(for: managedTab, origin: destination.origin)
            decisionHandler(.allow)
            return
        }

        let request = BrowserNavigationAuthorizationRequest(
            tabID: tabID,
            destination: destination.origin,
            source: webView.url.flatMap(BrowserOrigin.init(url:)),
            phase: .mainFrameResponse
        )
        evaluatePolicy(
            request,
            tabID: tabID,
            webView: webView,
            gate: BrowserNavigationPolicyGate { [weak self, weak managedTab] shouldAllow in
                guard let self, let managedTab else { return }
                if shouldAllow {
                    managedTab.pendingResponseOrigins.remove(destination.origin.canonicalString)
                    managedTab.activeNavigationURL = destination.url.absoluteString
                    self.startLoadIfNeeded(tabID, destination: destination)
                    self.beginSnapshotNavigation(for: managedTab, origin: destination.origin)
                } else {
                    self.recordNavigationFailure(
                        tabID,
                        error: .blockedNavigation(origin: destination.origin.canonicalString),
                        message: "Navigation blocked by browser permission policy."
                    )
                }
                decisionHandler(shouldAllow ? .allow : .cancel)
            }
        )
    }

    public func webView(
        _ webView: WKWebView,
        didStartProvisionalNavigation navigation: WKNavigation!
    ) {
        guard let tabID = tabID(for: webView) else { return }
        updateState(for: tabID) { state in
            state.progress = min(max(webView.estimatedProgress, 0), 1)
            state.loadError = nil
            state.navigationError = nil
        }
    }

    public func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
        guard let tabID = tabID(for: webView) else { return }
        if let managedTab = managedTabs[tabID], let generation = managedTab.snapshotGeneration {
            managedTab.snapshotService?.navigationCommitted(generation: generation)
        }
        refreshAddressAndTitle(for: tabID, webView: webView)
    }

    public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard let tabID = tabID(for: webView) else { return }
        if let managedTab = managedTabs[tabID], let generation = managedTab.snapshotGeneration {
            managedTab.snapshotService?.navigationCommitted(generation: generation)
        }
        if tabStore.tab(id: tabID)?.loadState == .loading {
            try? tabStore.completeNavigation(
                in: tabID,
                title: webView.title,
                address: webView.url?.absoluteString
            )
        }
        managedTabs[tabID]?.pendingResponseOrigins.removeAll()
        managedTabs[tabID]?.explicitNavigationURLs.removeAll()
        managedTabs[tabID]?.activeNavigationURL = nil
        updateState(for: tabID) { state in
            state.address = webView.url?.absoluteString ?? state.address
            state.title = webView.title
            state.progress = 1
            state.loadState = tabStore.tab(id: tabID)?.loadState ?? .loaded
            state.loadError = nil
            state.navigationError = nil
        }
    }

    public func webView(
        _ webView: WKWebView,
        didFail navigation: WKNavigation!,
        withError error: Error
    ) {
        recordWebKitFailure(error, for: webView)
    }

    public func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
    ) {
        recordWebKitFailure(error, for: webView)
    }

    public func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        guard let tabID = tabID(for: webView), let managedTab = managedTabs[tabID] else { return }
        cancelPendingPolicyGates(for: tabID)
        managedTab.pendingResponseOrigins.removeAll()
        managedTab.explicitNavigationURLs.removeAll()
        managedTab.activeNavigationURL = nil
        managedTab.snapshotGeneration = managedTab.snapshotService?.navigationStarted()
        try? tabStore.markCrashed(tabID)
        updateState(for: tabID) { state in
            state.loadState = .crashed
            state.progress = 0
            state.loadError = "The browser content process terminated."
            state.navigationError = .engineCrashed
        }
    }

    public func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        guard let sourceTabID = tabID(for: webView),
              let url = navigationAction.request.url else {
            return nil
        }
        _ = routeManagedPopup(from: sourceTabID, to: url)
        return nil
    }

    @discardableResult
    func routeManagedPopup(from sourceTabID: BrowserTabID, to url: URL) -> BrowserTabID? {
        guard let destination = BrowserNavigationDestination(url: url) else { return nil }
        let result: BrowserPopupRequestResult
        do {
            result = try tabStore.requestPopup(from: sourceTabID, address: destination.url.absoluteString)
        } catch {
            return nil
        }

        switch result {
        case .deniedAgentPopup:
            return nil
        case .opened(let popupTabID):
            guard let tab = tabStore.tab(id: popupTabID) else { return nil }
            attachWebView(to: tab)
            do {
                try navigate(to: destination.url, in: popupTabID)
                return popupTabID
            } catch {
                try? closeTab(popupTabID)
                return nil
            }
        }
    }

    private func attachWebView(to tab: BrowserTab) {
        guard managedTabs[tab.id] == nil else { return }
        let webView = BrowserTakeoverWebView(frame: .zero, configuration: WKWebViewConfiguration())
        webView.onDirectInput = { [weak self] in
            self?.prepareForDirectUserInput(in: tab.id)
        }
        webView.navigationDelegate = self
        webView.uiDelegate = self
        let managedTab = ManagedBrowserTab(webView: webView)
        managedTabs[tab.id] = managedTab
        tabIDsByWebView[ObjectIdentifier(webView)] = tab.id
        tabStates[tab.id] = BrowserEngineTabState(
            tabID: tab.id,
            address: tab.address,
            title: tab.title,
            loadState: tab.loadState
        )
        managedTab.observations = [
            webView.observe(\.estimatedProgress, options: [.new]) { [weak self] view, change in
                guard let progress = change.newValue else { return }
                Task { @MainActor [weak self] in
                    self?.updateState(for: tab.id) { $0.progress = min(max(progress, 0), 1) }
                    self?.refreshAddressAndTitle(for: tab.id, webView: view)
                }
            },
            webView.observe(\.title, options: [.new]) { [weak self] view, _ in
                Task { @MainActor [weak self] in
                    self?.refreshAddressAndTitle(for: tab.id, webView: view)
                }
            },
            webView.observe(\.url, options: [.new]) { [weak self] view, _ in
                Task { @MainActor [weak self] in
                    self?.refreshAddressAndTitle(for: tab.id, webView: view)
                }
            }
        ]
    }

    private func tabID(for webView: WKWebView) -> BrowserTabID? {
        tabIDsByWebView[ObjectIdentifier(webView)]
    }

    private func evaluatePolicy(
        _ request: BrowserNavigationAuthorizationRequest,
        tabID: BrowserTabID,
        webView: WKWebView,
        gate: BrowserNavigationPolicyGate
    ) {
        let requestID = UUID()
        let trackedGate = BrowserNavigationPolicyGate { [weak self, gate] shouldAllow in
            self?.pendingPolicyGates[tabID]?.removeValue(forKey: requestID)
            if self?.pendingPolicyGates[tabID]?.isEmpty == true {
                self?.pendingPolicyGates.removeValue(forKey: tabID)
            }
            gate.resolve(shouldAllow)
        }
        pendingPolicyGates[tabID, default: [:]][requestID] = trackedGate

        Task { @MainActor [weak self, trackedGate] in
            guard let self else {
                trackedGate.resolve(false)
                return
            }
            guard self.managedTabs[tabID]?.webView === webView,
                  let currentTab = self.tabStore.tab(id: tabID) else {
                self.completePolicyGate(tabID: tabID, requestID: requestID, allow: false)
                return
            }
            if currentTab.owner == .user {
                self.completePolicyGate(tabID: tabID, requestID: requestID, allow: true)
                return
            }

            let decision = await self.authorizeNavigation(request)
            let stillManaged = self.managedTabs[tabID]?.webView === webView
            let ownerIsUser = self.tabStore.tab(id: tabID)?.owner == .user
            self.completePolicyGate(
                tabID: tabID,
                requestID: requestID,
                allow: stillManaged && (ownerIsUser || decision == .allow)
            )
        }
    }

    private func completePolicyGate(tabID: BrowserTabID, requestID: UUID, allow: Bool) {
        guard let gate = pendingPolicyGates[tabID]?.removeValue(forKey: requestID) else { return }
        if pendingPolicyGates[tabID]?.isEmpty == true {
            pendingPolicyGates.removeValue(forKey: tabID)
        }
        gate.resolve(allow)
    }

    private func cancelPendingPolicyGates(for tabID: BrowserTabID) {
        let gates = pendingPolicyGates.removeValue(forKey: tabID)?.values ?? Dictionary<UUID, BrowserNavigationPolicyGate>().values
        for gate in gates {
            gate.resolve(false)
        }
    }

    private func consumeExplicitNavigationPhase(
        for url: URL,
        in managedTab: ManagedBrowserTab
    ) -> BrowserNavigationCheckPhase? {
        let key = url.absoluteString
        guard let count = managedTab.explicitNavigationURLs[key], count > 0 else { return nil }
        if count == 1 {
            managedTab.explicitNavigationURLs.removeValue(forKey: key)
        } else {
            managedTab.explicitNavigationURLs[key] = count - 1
        }
        return .explicitNavigation
    }

    private func startLoadIfNeeded(_ tabID: BrowserTabID, destination: BrowserNavigationDestination) {
        guard tabStore.tab(id: tabID)?.loadState != .loading else { return }
        guard (try? tabStore.startNavigation(in: tabID, to: destination.url.absoluteString)) != nil else { return }
        setNavigationStarted(tabID, destination: destination)
    }

    private func setNavigationStarted(_ tabID: BrowserTabID, destination: BrowserNavigationDestination) {
        updateState(for: tabID) { state in
            state.address = destination.url.absoluteString
            state.progress = 0
            state.loadState = .loading
            state.loadError = nil
            state.navigationError = nil
        }
    }

    private func beginSnapshotNavigation(for managedTab: ManagedBrowserTab, origin: BrowserOrigin) {
        if managedTab.snapshotService == nil {
            managedTab.snapshotService = DOMSnapshotService(
                webView: managedTab.webView,
                expectedOrigin: origin.canonicalString
            )
        }
        managedTab.snapshotGeneration = managedTab.snapshotService?.navigationStarted(
            expectedOrigin: origin.canonicalString
        )
    }

    private func recordNavigationFailure(
        _ tabID: BrowserTabID,
        error: BrowserControlError,
        message: String
    ) {
        if tabStore.tab(id: tabID)?.loadState == .loading {
            try? tabStore.failNavigation(in: tabID, reason: message)
        }
        updateState(for: tabID) { state in
            state.loadState = tabStore.tab(id: tabID)?.loadState ?? state.loadState
            state.loadError = message
            state.navigationError = error
            if case .loading = state.loadState {
                state.loadState = .failed(reason: message)
            }
            state.progress = 0
        }
    }

    private func recordWebKitFailure(_ error: Error, for webView: WKWebView) {
        guard let tabID = tabID(for: webView) else { return }
        let nsError = error as NSError
        let reason: BrowserNavigationFailureReason
        if nsError.code == NSURLErrorCancelled {
            reason = .cancelled
        } else if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorServerCertificateUntrusted {
            reason = .tls
        } else {
            reason = .transport
        }
        let message: String
        switch reason {
        case .cancelled:
            message = "Navigation was cancelled."
        case .tls:
            message = "The page could not establish a trusted TLS connection."
        case .serverResponse:
            message = "The server could not complete the request."
        case .transport:
            message = "The page could not be reached."
        case .unknown:
            message = "The page could not be loaded."
        }
        if tabStore.tab(id: tabID)?.loadState == .loading {
            try? tabStore.failNavigation(in: tabID, reason: message)
        }
        managedTabs[tabID]?.pendingResponseOrigins.removeAll()
        managedTabs[tabID]?.explicitNavigationURLs.removeAll()
        managedTabs[tabID]?.activeNavigationURL = nil
        updateState(for: tabID) { state in
            state.loadState = tabStore.tab(id: tabID)?.loadState ?? state.loadState
            state.progress = 0
            state.loadError = message
            if case .blockedNavigation = state.navigationError {
                return
            }
            state.navigationError = .navigationFailure(reason: reason)
        }
    }

    private func refreshAddressAndTitle(for tabID: BrowserTabID, webView: WKWebView) {
        guard managedTabs[tabID] != nil else { return }
        let address = webView.url?.absoluteString
        let title = webView.title
        updateState(for: tabID) { state in
            if let address { state.address = address }
            if let title { state.title = title }
        }
    }

    private func refreshStoredTabState(_ tabID: BrowserTabID) {
        guard let tab = tabStore.tab(id: tabID) else { return }
        updateState(for: tabID) { state in
            state.address = tab.address
            state.title = tab.title
            state.loadState = tab.loadState
        }
    }

    private func updateState(
        for tabID: BrowserTabID,
        _ update: (inout BrowserEngineTabState) -> Void
    ) {
        guard var state = tabStates[tabID] else { return }
        update(&state)
        tabStates[tabID] = state
    }
}

extension WebKitBrowserEngine: WKNavigationDelegate {
    // The delegate conformance is explicit so its callback surface remains visible to review.
}

extension WebKitBrowserEngine: WKUIDelegate {
    // The UI delegate conformance is explicit so popup routing stays at the engine boundary.
}
