import SwiftUI
import WebKit

/// What the sandbox webview loads, and how much it may reach.
///
/// `.inline` is a chat fence's html: `loadHTMLString(baseURL: nil)` gives the
/// page an opaque origin with no file access at all, the WKWebView analogue
/// of a plain `sandbox="allow-scripts"` iframe. `.file` is a tool-written
/// `.html` artifact: `loadFileURL(allowingReadAccessTo:)` scopes every read
/// to that one folder, which is what makes sibling css/js/img work without
/// handing the page the rest of the disk.
public enum ArtifactWebDocument: Equatable {
    case file(page: URL, readAccessFolder: URL)
    case inline(html: String)

    /// The main-frame URL to expect in the navigation delegate, nil for
    /// inline (whose main frame is an inert `about:blank`).
    var expectedURL: URL? {
        switch self {
        case .file(let page, _): return page
        case .inline: return nil
        }
    }

    /// The folder a file page may read. Inline has none.
    var readAccessFolder: URL? {
        switch self {
        case .file(_, let folder): return folder
        case .inline: return nil
        }
    }

    /// Identity for the panel's `.id()`: a distinct document rebuilds the
    /// web view rather than reloading in place, so no page state leaks
    /// across artifacts.
    var identity: String {
        switch self {
        case .file(let page, _): return "file:\(page.path)"
        case .inline(let html): return "inline:\(html)"
        }
    }
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The sandboxed web renderer behind the artifact panel's `.html` arm.
///
/// The security posture, in the order it is enforced:
///
/// 1. **Ephemeral everything.** `.nonPersistent()` data store: no cookies,
///    no localStorage on disk, nothing shared with any other web view. This
///    stands in for unsloth studio's in-memory storage shims for an opaque
///    iframe origin.
/// 2. **Offline by default.** A `WKContentRuleList` blocks http/https/ws/wss
///    loads, which is the layer decidePolicyFor cannot see (subresource
///    fetches do not traverse it). The navigation delegate handles what
///    rules cannot bind tightly: main-frame navigations. Both gates are
///    independent; either alone holds the line for its own layer.
/// 3. **Scoped file access.** Only `.file` documents get a read scope at
///    all, and WebKit enforces the folder boundary itself.
///
/// Network is a per-content grant (`AppModel+ArtifactPanel`), and turning it
/// on rebuilds the view (identity includes the grant), so a granted page
/// cannot have loaded before its grant existed.
@MainActor
struct ArtifactWebView: NSViewRepresentable {
    let document: ArtifactWebDocument
    let networkAllowed: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.mediaTypesRequiringUserActionForPlayback = .all

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        context.coordinator.webView = webView

        context.coordinator.networkAllowed = networkAllowed
        load(document: document, networkAllowed: networkAllowed, in: webView, coordinator: context.coordinator)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        let coordinator = context.coordinator
        coordinator.networkAllowed = networkAllowed
        guard coordinator.loadedDocument != document || coordinator.loadedNetworkAllowed != networkAllowed else {
            return
        }
        load(document: document, networkAllowed: networkAllowed, in: webView, coordinator: coordinator)
    }

    private func load(
        document: ArtifactWebDocument,
        networkAllowed: Bool,
        in webView: WKWebView,
        coordinator: Coordinator
    ) {
        coordinator.loadedDocument = document
        coordinator.loadedNetworkAllowed = networkAllowed

        if networkAllowed {
            // The offline block list a previous load installed must go: it
            // is enforced in the network process, BELOW the navigation
            // delegate, so a granted page would still have every remote
            // load refused no matter what the delegate allows. This webview
            // only ever receives the one static list, so removing all is
            // removing exactly that.
            webView.configuration.userContentController.removeAllContentRuleLists()
            loadDocument(document, in: webView)
            return
        }

        // Content rules compile asynchronously, and loading before they are
        // installed would give the page a window in which subresource
        // fetches run unblocked. So the offline load WAITS for the rules.
        // The navigation delegate already covers main-frame navigations
        // while waiting; the panel shows a blank frame for that instant.
        ArtifactContentRuleList.shared.withOfflineRules { rules in
            guard let rules else {
                // Compilation failing means the load proceeds under the
                // navigation delegate alone. That delegate cannot see
                // subresource fetches, which is exactly why the rules
                // exist; a WebKit that cannot compile two static rules is
                // broken well beyond this panel.
                loadDocument(document, in: webView)
                return
            }
            webView.configuration.userContentController.add(rules)
            loadDocument(document, in: webView)
        }
    }

    private func loadDocument(_ document: ArtifactWebDocument, in webView: WKWebView) {
        switch document {
        case .file(let page, let folder):
            webView.loadFileURL(page, allowingReadAccessTo: folder)
        case .inline(let html):
            webView.loadHTMLString(html, baseURL: nil)
        }
    }

    // MARK: - Coordinator

    @MainActor
    final class Coordinator: NSObject, WKNavigationDelegate {
        var webView: WKWebView?
        var networkAllowed = false
        var loadedDocument: ArtifactWebDocument?
        var loadedNetworkAllowed = false

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            let policy = policy(for: navigationAction)
            decisionHandler(policy)
        }

        /// WebKit jetsam's a background web view and hands the page back as
        /// a blank frame; a preview that silently went blank reads as a
        /// broken page rather than a reclaimed one.
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            webView.reload()
        }

        private func policy(for navigationAction: WKNavigationAction) -> WKNavigationActionPolicy {
            guard let url = navigationAction.request.url else { return .cancel }
            let scheme = url.scheme?.lowercased()

            // The shell of an inline loadHTMLString document. Inert on its
            // own: anything it does next shows up here as its own
            // navigation and is judged on its merits.
            if scheme == "about" { return .allow }

            if scheme == "file" {
                // WebKit refuses file reads outside the load's own
                // allowingReadAccessTo scope, so the folder bound below is
                // the belt, not the wall. The trailing slash keeps the
                // prefix check from accepting a SIBLING whose name extends
                // the folder's ("/x/artifacts-secret" under "/x/artifacts").
                if let folder = loadedDocument?.readAccessFolder {
                    let folderPath = folder.standardizedFileURL.path
                    let path = url.standardizedFileURL.path
                    if path == folderPath || path.hasPrefix(folderPath + "/") {
                        return .allow
                    }
                }
                return .cancel
            }

            let isWebScheme = scheme == "http" || scheme == "https"
                || scheme == "ws" || scheme == "wss"
            if isWebScheme {
                // The grant covers every layer at once: main frame here,
                // subresources through the content rules. Without a grant
                // this arm is what stops a page self-navigating to the
                // network (unsloth's `allow-top-navigation` omission).
                return networkAllowed ? .allow : .cancel
            }

            return .cancel
        }
    }
}

/// The offline block list, compiled once per process and cached.
///
/// Subresource loads (img/script/fetch/XHR) never reach the navigation
/// delegate, so rules are the only lever that reaches them. Compilation is
/// asynchronous and the first offline load waits for it rather than opening
/// an unblocked window (`ArtifactWebView.load`).
@MainActor
final class ArtifactContentRuleList {
    static let shared = ArtifactContentRuleList()

    /// Deliberately over-broad: when a grant exists the rules are not
    /// installed at all, so anything matched here was going to be blocked
    /// anyway. Keeping every remote scheme in one place means the offline
    /// posture never depends on remembering a scheme here.
    private static let json = """
    [{"trigger":{"url-filter":"^https?:"},"action":{"type":"block"}},
     {"trigger":{"url-filter":"^wss?:"},"action":{"type":"block"}}]
    """

    private var cached: WKContentRuleList?
    private var isCompiling = false
    private var waiters: [(WKContentRuleList?) -> Void] = []

    func withOfflineRules(_ body: @escaping (WKContentRuleList?) -> Void) {
        if let cached {
            body(cached)
            return
        }
        waiters.append(body)
        guard !isCompiling else { return }
        isCompiling = true
        WKContentRuleListStore.default().compileContentRuleList(
            forIdentifier: "ArtifactPreviewOfflineBlock",
            encodedContentRuleList: Self.json
        ) { [weak self] rules, _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.cached = rules
                self.isCompiling = false
                let waiters = self.waiters
                self.waiters = []
                for waiter in waiters { waiter(rules) }
            }
        }
    }
}
