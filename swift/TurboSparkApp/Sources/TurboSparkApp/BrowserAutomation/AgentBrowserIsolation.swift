import Foundation
import WebKit

/// Content rules that keep an agent tab's page off the private network.
///
/// The navigation delegate authorizes frames only. `fetch`, XHR, images,
/// WebSockets and beacons never reach it, so a page on a granted public
/// origin could still POST to the TurboSpark listener, the OpenKind daemon,
/// a LAN router or a metadata endpoint. Content rules are enforced in the
/// network process and are the only lever that reaches those loads.
///
/// One static list, not one per project: a request to a private host is
/// blocked UNLESS the top document is itself on a private host. Reaching a
/// private top document already took an exact origin grant or a hard-gated
/// approval, so that page may load its own private subresources.
///
/// Limits: the rule regex has no alternation, so each host form is its own
/// rule, and numeric host spellings (`http://2130706433/`, octal, hex) cannot
/// be expressed at all. Frames with those hosts are still refused by the
/// navigation delegate. Subresources are not.
@MainActor
final class AgentBrowserNetworkRules {
    static let shared = AgentBrowserNetworkRules()

    /// Host prefixes, as rule regex, that name loopback, private, link-local
    /// or mDNS destinations.
    static let privateHostPatterns: [String] = [
        "localhost[:/]",
        "[^/]*\\.localhost[:/]",
        "[^/]*\\.local[:/]",
        // The named metadata hosts `AppToolSandbox.isPrivateOrMetadataHost` lists.
        "metadata\\.google\\.internal[:/]",
        "metadata[:/]",
        "instance-data[:/]",
        "127\\.",
        "0\\.0\\.0\\.0",
        "10\\.",
        "192\\.168\\.",
        "172\\.1[6-9]\\.",
        "172\\.2[0-9]\\.",
        "172\\.3[01]\\.",
        "169\\.254\\.",
        "\\[::1\\]",
        "\\[::\\]",
        "\\[f[cd][0-9a-f]*:",
        "\\[fe[89ab][0-9a-f]*:",
    ]

    /// The encoded rule list. Built rather than written out so every host form
    /// gets both schemes and both the plain and the userinfo spelling.
    static let encodedRules: String = {
        let topURLs = privateHostPatterns.map { "^https?://" + $0 }
        var rules: [[String: Any]] = []
        for scheme in ["https?", "wss?"] {
            for userinfo in ["", "[^/@]*@"] {
                for host in privateHostPatterns {
                    rules.append([
                        "trigger": [
                            "url-filter": "^\(scheme)://\(userinfo)\(host)",
                            "unless-top-url": topURLs,
                        ],
                        "action": ["type": "block"],
                    ])
                }
            }
        }
        let data = (try? JSONSerialization.data(withJSONObject: rules, options: [.sortedKeys])) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }()

    private var cached: WKContentRuleList?
    private var isCompiling = false
    private var waiters: [(WKContentRuleList?) -> Void] = []

    /// Calls `body` with the compiled list, or nil when compilation failed.
    func withRules(_ body: @escaping (WKContentRuleList?) -> Void) {
        if let cached {
            body(cached)
            return
        }
        waiters.append(body)
        guard !isCompiling else { return }
        isCompiling = true
        WKContentRuleListStore.default().compileContentRuleList(
            forIdentifier: "AgentBrowserPrivateNetworkBlock",
            encodedContentRuleList: Self.encodedRules
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

    /// The compiled list when it is ready, without waiting.
    var compiledRules: WKContentRuleList? { cached }
}

/// The website data store agent tabs are created with.
///
/// Separate from `WKWebsiteDataStore.default()`, which the user's own tabs
/// use, so an origin grant never hands the agent the user's signed-in
/// session. Persistent, so a site the user signs into inside an agent tab
/// stays signed in there. One store per profile, keyed by an id kept in that
/// profile's store root.
@MainActor
enum AgentBrowserDataStore {
    static let store: WKWebsiteDataStore = {
        if AppStorageRoot.isRunningTests { return .nonPersistent() }
        return WKWebsiteDataStore(forIdentifier: identifier())
    }()

    private static func identifier() -> UUID {
        let file = AppStorageRoot.file("agent_browser_store_id")
        if let text = try? String(contentsOf: file, encoding: .utf8),
           let id = UUID(uuidString: text.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return id
        }
        let id = UUID()
        try? id.uuidString.write(to: file, atomically: true, encoding: .utf8)
        return id
    }
}
