import AppKit
import SwiftUI
import WebKit

// The web host for plugin pages (ZP-06). `ArtifactWebView` stays the host for
// artifacts; this sibling exists because a plugin page differs in three ways
// that cannot be parameters of a SwiftUI representable without weakening it:
// every byte is served through a scheme handler (so CSP and the file root are
// enforced by app code, not by WebKit file-read scopes), the bridge handler
// is installed, and the session outlives any one SwiftUI view (the host
// retains it while visible and tears it down on close). It reuses the one
// offline content-rule list, so there is a single definition of "no network".

/// Serves one page's declared files under `tsplugin://<token>/`.
@MainActor
final class PluginPageSchemeHandler: NSObject, WKURLSchemeHandler {
    private let token: String
    private let root: URL
    private let declared: Set<String>
    private let csp: String
    private let maxBytes: Int
    /// Called for every refused request with the normalized path (or the raw
    /// path when it did not normalize), so the session can name a failed
    /// page document and tests can read the refusals.
    var onRefusal: ((String, PluginPageError) -> Void)?
    private(set) var servedPaths: [String] = []

    init(
        token: String, root: URL, surface: PluginUiSurface,
        maxBytes: Int = PluginPageLimits.maxResourceBytes,
        cspOverrideForTests: String? = nil
    ) {
        self.token = token
        self.root = root
        self.declared = PluginPageResolver.declaredPaths(for: surface)
        self.csp = cspOverrideForTests ?? PluginPageCSP.header(for: surface)
        self.maxBytes = maxBytes
    }

    func webView(_ webView: WKWebView, start urlSchemeTask: any WKURLSchemeTask) {
        guard let url = urlSchemeTask.request.url else {
            urlSchemeTask.didFailWithError(URLError(.badURL))
            return
        }
        guard urlSchemeTask.request.httpMethod == nil || urlSchemeTask.request.httpMethod == "GET" else {
            respond(urlSchemeTask, url: url, status: 405, mime: "text/plain; charset=utf-8", body: Data("method not allowed".utf8))
            onRefusal?(url.path, .forbiddenPath)
            return
        }
        guard let path = PluginPageResolver.requestPath(from: url, token: token) else {
            respond(urlSchemeTask, url: url, status: 403, mime: "text/plain; charset=utf-8", body: Data("forbidden".utf8))
            onRefusal?(url.path, .forbiddenPath)
            return
        }
        switch PluginPageResolver.load(relativePath: path, root: root, declared: declared, maxBytes: maxBytes) {
        case .success(let page):
            servedPaths.append(path)
            respond(urlSchemeTask, url: url, status: 200, mime: page.mimeType, body: page.data)
        case .failure(let error):
            let status: Int
            switch error {
            case .notFound, .notDeclared: status = 404
            case .tooLarge: status = 413
            default: status = 403
            }
            respond(urlSchemeTask, url: url, status: status, mime: "text/plain; charset=utf-8", body: Data(error.code.utf8))
            onRefusal?(path, error)
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: any WKURLSchemeTask) {
        // Serving is synchronous inside `start`, so a task is never live
        // across calls and there is nothing to cancel.
    }

    private func respond(_ task: any WKURLSchemeTask, url: URL, status: Int, mime: String, body: Data) {
        let headers = [
            "Content-Type": mime,
            "Content-Length": String(body.count),
            "Content-Security-Policy": csp,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
            "Cross-Origin-Resource-Policy": "same-origin",
        ]
        guard let response = HTTPURLResponse(
            url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)
        else {
            task.didFailWithError(URLError(.cannotParseResponse))
            return
        }
        task.didReceive(response)
        task.didReceive(body)
        task.didFinish()
    }
}

/// A web view that records genuine user input. `WKScriptMessage` carries no
/// gesture flag and page script can fake any in-page signal, so the only
/// trustworthy "the person just clicked" is an AppKit event reaching this
/// view. The bridge reads `lastUserGestureAt` to gate tool calls and drafts.
final class PluginGestureWebView: WKWebView {
    var onUserGesture: (() -> Void)?

    override func mouseDown(with event: NSEvent) {
        onUserGesture?()
        super.mouseDown(with: event)
    }

    override func keyDown(with event: NSEvent) {
        onUserGesture?()
        super.keyDown(with: event)
    }

    /// No context menu: its items (open link, save image, reload) are
    /// navigation and download paths the sandbox denies elsewhere.
    override func willOpenMenu(_ menu: NSMenu, with event: NSEvent) {
        menu.removeAllItems()
    }
}

/// Everything a page session needs from the app. Closures, so tests inject
/// fakes and the production wiring stays in `PluginUiHost`.
struct PluginPageEnvironment {
    var invoker: PageBridge.ToolInvoker
    var draftSink: @MainActor (String) -> Void
    var widgetStates: PluginWidgetStateStore
    var now: () -> Date = Date.init
}

/// One live plugin page: its web view, its bridge, its failure state. The
/// host retains it while visible and calls `tearDown()` on close.
@MainActor
final class PluginPageSession: NSObject, ObservableObject, Identifiable, WKNavigationDelegate, WKUIDelegate {
    let id = UUID()
    let identity: PluginPageIdentity
    let surface: PluginUiSurface
    let pluginName: String
    let token: String
    let webView: PluginGestureWebView
    let bridge: PageBridge
    let schemeHandler: PluginPageSchemeHandler
    private let environment: PluginPageEnvironment

    @Published private(set) var failure: PluginPageError?
    @Published private(set) var isLoading = false
    private(set) var isTornDown = false
    private(set) var refusals: [(path: String, error: PluginPageError)] = []

    private let pageURL: URL
    private var initialLoadPending = false
    private var rulesInstalled = false
    private var reloadsAfterTermination = 0
    /// Set when the app cancels the page's own navigation, so the resulting
    /// `NSURLErrorCancelled` does not overwrite the typed cause.
    private var cancelledByPolicy = false

    init(
        identity: PluginPageIdentity,
        surface: PluginUiSurface,
        pluginName: String,
        root: URL,
        environment: PluginPageEnvironment,
        declaredTools: [String: String],
        cspOverrideForTests: String? = nil
    ) {
        self.identity = identity
        self.surface = surface
        self.pluginName = pluginName
        self.environment = environment
        let token = PluginPageScheme.makeToken()
        self.token = token
        // `resourceUri` was normalized by the parser; a surface built some
        // other way that fails normalization gets an unloadable URL and the
        // failure card instead of a crash.
        self.pageURL = PluginPageScheme.url(token: token, path: surface.resourceUri)
            ?? URL(string: "\(PluginPageScheme.name)://\(token)/invalid")!

        let handler = PluginPageSchemeHandler(
            token: token, root: root, surface: surface, cspOverrideForTests: cspOverrideForTests)
        schemeHandler = handler
        let bridge = PageBridge(
            identity: identity, pluginName: pluginName, pageTitle: surface.title, token: token,
            capabilities: Set(surface.capabilities), declaredTools: declaredTools,
            widgetStates: environment.widgetStates, invoker: environment.invoker,
            draftSink: environment.draftSink, now: environment.now)
        self.bridge = bridge

        let configuration = WKWebViewConfiguration()
        // A fresh non-persistent store per page: no cookies or storage on
        // disk, nothing shared with another view. Page origins also differ
        // (a per-load host token), so two live pages cannot see each other's
        // in-memory storage either.
        configuration.websiteDataStore = .nonPersistent()
        configuration.mediaTypesRequiringUserActionForPlayback = .all
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        configuration.setURLSchemeHandler(handler, forURLScheme: PluginPageScheme.name)
        let controller = configuration.userContentController
        controller.addScriptMessageHandler(bridge, contentWorld: .page, name: PageBridge.handlerName)
        controller.addUserScript(WKUserScript(
            source: PageBridgeScript.source, injectionTime: .atDocumentStart,
            forMainFrameOnly: true, in: .page))

        webView = PluginGestureWebView(frame: .zero, configuration: configuration)
        super.init()
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = false
        webView.allowsMagnification = false
        webView.setValue(false, forKey: "drawsBackground")
        webView.onUserGesture = { [weak self] in
            guard let self else { return }
            self.bridge.lastUserGestureAt = self.environment.now()
        }
        handler.onRefusal = { [weak self] path, error in
            guard let self else { return }
            self.refusals.append((path, error))
            // The page document itself was refused: show the card now. The
            // navigation-response delegate is not a reliable signal for a
            // custom scheme's error status, so the handler is the source.
            if path == self.surfaceDocumentPath { self.fail(error) }
        }
    }

    // MARK: - Lifecycle

    /// Loads (or reloads) the page's document. Waits for the offline rule
    /// list: loading earlier would give the page a window of unblocked
    /// subresource fetches, and `connect-src 'none'` is the second line, not
    /// the first.
    func load() {
        guard !isTornDown else { return }
        failure = nil
        isLoading = true
        cancelledByPolicy = false
        ArtifactContentRuleList.shared.withOfflineRules { [weak self] rules in
            guard let self, !self.isTornDown else { return }
            guard let rules else {
                self.fail(.sandboxUnavailable)
                return
            }
            if !self.rulesInstalled {
                self.webView.configuration.userContentController.add(rules)
                self.rulesInstalled = true
            }
            self.initialLoadPending = true
            self.webView.load(URLRequest(url: self.pageURL))
        }
    }

    /// The failure card's retry: a clean reload of the stored page. Widget
    /// state lives in the native store, so the page can read it back.
    func retry() {
        reloadsAfterTermination = 0
        load()
    }

    /// Frees everything the page held. Idempotent. Widget state goes with it
    /// (draft: cleared on pane close, chat switch and quit).
    func tearDown() {
        guard !isTornDown else { return }
        isTornDown = true
        webView.stopLoading()
        webView.navigationDelegate = nil
        webView.uiDelegate = nil
        webView.onUserGesture = nil
        let controller = webView.configuration.userContentController
        controller.removeScriptMessageHandler(forName: PageBridge.handlerName, contentWorld: .page)
        controller.removeAllUserScripts()
        controller.removeAllContentRuleLists()
        // WebKit refuses to clear a scheme handler (it raises), so the
        // handler is neutralized instead: no callbacks, and the web view
        // goes away with the session.
        schemeHandler.onRefusal = nil
        environment.widgetStates.clear(identity)
        webView.loadHTMLString("", baseURL: nil)
        webView.removeFromSuperview()
    }

    private func fail(_ error: PluginPageError) {
        isLoading = false
        // The first typed cause wins; a later generic failure is a symptom.
        if failure == nil { failure = error }
    }

    // MARK: - WKNavigationDelegate

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
    ) {
        let allowed = PluginPageNavigationPolicy.allows(
            url: navigationAction.request.url, pageURL: pageURL,
            isMainFrame: navigationAction.targetFrame?.isMainFrame ?? false,
            initialLoadPending: initialLoadPending)
        if allowed, navigationAction.request.url == pageURL { initialLoadPending = false }
        if !allowed { cancelledByPolicy = true }
        decisionHandler(allowed ? .allow : .cancel)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping @MainActor @Sendable (WKNavigationResponsePolicy) -> Void
    ) {
        if navigationResponse.isForMainFrame,
            let http = navigationResponse.response as? HTTPURLResponse, http.statusCode != 200
        {
            // A refused page document: name the cause from the handler's
            // record and show the card rather than the plain-text body.
            let cause = refusals.last(where: { $0.path == surfaceDocumentPath })?.error
                ?? .loadFailed("HTTP \(http.statusCode)")
            fail(cause)
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    private var surfaceDocumentPath: String {
        PluginPageResolver.normalizeRelativePath(surface.resourceUri) ?? surface.resourceUri
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        isLoading = false
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: any Error) {
        recordNavigationFailure(error)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: any Error) {
        recordNavigationFailure(error)
    }

    private func recordNavigationFailure(_ error: any Error) {
        let ns = error as NSError
        // Our own cancels (a denied navigation, a refused document) are not
        // load failures of the page; the typed cause is already recorded or
        // the page keeps rendering.
        if ns.domain == NSURLErrorDomain, ns.code == NSURLErrorCancelled, failure != nil || cancelledByPolicy {
            isLoading = false
            return
        }
        fail(.loadFailed(String(ns.localizedDescription.prefix(200))))
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        guard !isTornDown else { return }
        // WebKit reclaims a background web view's process under memory
        // pressure. One automatic reload from the stored page (the state is
        // in the native store); a second loss shows the card.
        if reloadsAfterTermination < 1 {
            reloadsAfterTermination += 1
            load()
        } else {
            fail(.processTerminated)
        }
    }

    // MARK: - WKUIDelegate (default-deny, like the browser engine's ZP-14)

    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        nil
    }

    func webView(
        _ webView: WKWebView,
        requestMediaCapturePermissionFor origin: WKSecurityOrigin,
        initiatedByFrame frame: WKFrameInfo,
        type: WKMediaCaptureType,
        decisionHandler: @escaping @MainActor @Sendable (WKPermissionDecision) -> Void
    ) {
        decisionHandler(.deny)
    }

    @available(macOS 27.0, *)
    @objc(webView:requestGeolocationPermissionForOrigin:initiatedByFrame:decisionHandler:)
    func webView(
        _ webView: WKWebView,
        requestGeolocationPermissionFor origin: WKSecurityOrigin,
        initiatedByFrame frame: WKFrameInfo,
        decisionHandler: @escaping @MainActor @Sendable (WKPermissionDecision) -> Void
    ) {
        decisionHandler(.deny)
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable () -> Void
    ) {
        completionHandler()
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
    ) {
        completionHandler(false)
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptTextInputPanelWithPrompt prompt: String,
        defaultText: String?,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable (String?) -> Void
    ) {
        completionHandler(nil)
    }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable ([URL]?) -> Void
    ) {
        completionHandler(nil)
    }
}

/// Hosts a session's web view inside SwiftUI. The session, not this
/// representable, owns the web view, so SwiftUI rebuilding the view does not
/// reload the page and the host's teardown is the only thing that frees it.
struct PluginPageWebView: NSViewRepresentable {
    @ObservedObject var session: PluginPageSession

    func makeNSView(context: Context) -> NSView {
        let container = NSView()
        attach(session.webView, to: container)
        return container
    }

    func updateNSView(_ container: NSView, context: Context) {
        if session.webView.superview !== container, !session.isTornDown {
            attach(session.webView, to: container)
        }
    }

    private func attach(_ webView: NSView, to container: NSView) {
        webView.removeFromSuperview()
        webView.frame = container.bounds
        webView.autoresizingMask = [.width, .height]
        container.addSubview(webView)
    }
}

/// The page card: the web view, or the bounded failure card with a retry.
struct PluginPageContentView: View {
    @ObservedObject var session: PluginPageSession

    var body: some View {
        if let failure = session.failure {
            PluginPageFailureCard(title: session.surface.title, error: failure) {
                session.retry()
            }
        } else {
            PluginPageWebView(session: session)
                .accessibilityLabel(Text("Plugin page \(session.surface.title)", bundle: .module))
        }
    }
}

/// Bounded error card for a page that could not load (ZP-17's deferred
/// contract). Says what happened in one sentence and offers a retry; never
/// shows the page's own text, and caps the technical detail.
struct PluginPageFailureCard: View {
    let title: String
    let error: PluginPageError
    let onRetry: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(.appWarning)
                    .accessibilityHidden(true)
                Text("This plugin page could not be shown", bundle: .module)
                    .themedFont(.small, weight: .semibold)
            }
            Text(verbatim: title)
                .themedFont(.small)
                .lineLimit(1)
                .truncationMode(.middle)
            Self.explanation(for: error)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(verbatim: error.code)
                .themedCode(.tiny)
                .foregroundStyle(.appSecondary)
            Button(action: onRetry) {
                Text("Try again", bundle: .module)
            }
            .appAction(.secondary, size: .small)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .accessibilityElement(children: .contain)
    }

    static func explanation(for error: PluginPageError) -> Text {
        switch error {
        case .processTerminated:
            return Text("The page stopped, usually because the Mac was short on memory. Closing other pages can help.", bundle: .module)
        case .sandboxUnavailable:
            return Text("The offline sandbox is unavailable, so the page was not loaded.", bundle: .module)
        case .tooLarge:
            return Text("A file of this page is over the size limit.", bundle: .module)
        case .notFound, .notDeclared, .forbiddenPath, .symlinkRefused, .unsupportedType, .unreadable:
            return Text("A file this page needs is missing, undeclared, or not allowed. The plugin's manifest decides what a page may load.", bundle: .module)
        case .loadFailed:
            return Text("The page failed to load.", bundle: .module)
        }
    }
}
