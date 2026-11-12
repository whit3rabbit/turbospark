import Foundation
import WebKit
import XCTest

@testable import TurboSparkApp

/// Agent browser tabs: their own data store, and no private-network
/// subresources from a public page.
@MainActor
final class AgentBrowserIsolationTests: XCTestCase {
    func testAgentTabsDoNotShareTheUsersDataStore() throws {
        let engine = WebKitBrowserEngine(tabStore: BrowserTabStore(), authorizeNavigation: { _ in .allow })
        let userTab = engine.createTab(owner: .user)
        let agentTab = engine.createTab(owner: .agent, select: false)

        let userStore = try XCTUnwrap(engine.webView(for: userTab)?.configuration.websiteDataStore)
        let agentStore = try XCTUnwrap(engine.webView(for: agentTab)?.configuration.websiteDataStore)
        XCTAssertTrue(userStore === WKWebsiteDataStore.default())
        XCTAssertFalse(
            agentStore === WKWebsiteDataStore.default(),
            "An origin grant must not hand the agent the user's cookies.")
    }

    func testRulesCompileAndCoverEveryHostFormForBothSchemes() async throws {
        let rules = await withCheckedContinuation { continuation in
            AgentBrowserNetworkRules.shared.withRules { continuation.resume(returning: $0) }
        }
        XCTAssertNotNil(rules, "WebKit refused the encoded rule list.")

        let data = Data(AgentBrowserNetworkRules.encodedRules.utf8)
        let decoded = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        let filters = decoded.compactMap { ($0["trigger"] as? [String: Any])?["url-filter"] as? String }
        XCTAssertEqual(filters.count, AgentBrowserNetworkRules.privateHostPatterns.count * 4)
        XCTAssertTrue(filters.contains("^wss?://127\\."))
        XCTAssertTrue(filters.contains("^https?://[^/@]*@169\\.254\\."))
    }

    /// The real gap: a page on a public origin beacons to loopback. The
    /// control load without the rules proves the fixture can see the hit.
    func testAPublicPageCannotReachLoopbackWithTheRulesInstalled() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let compiled = await withCheckedContinuation { continuation in
            AgentBrowserNetworkRules.shared.withRules { continuation.resume(returning: $0) }
        }
        let rules = try XCTUnwrap(compiled)
        func page(_ path: String) -> String {
            let target = server.url(path).absoluteString
            return """
            <!doctype html><html><body>
            <img src="\(target)?img">
            <script>fetch("\(target)?fetch", {mode: "no-cors"}).catch(() => {});</script>
            </body></html>
            """
        }
        // A public base URL, over http so mixed-content upgrading does not
        // hide the control's request.
        let publicBase = URL(string: "http://public.example/")

        let guarded = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        guarded.configuration.userContentController.add(rules)
        guarded.loadHTMLString(page("/blocked"), baseURL: publicBase)

        let control = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        control.loadHTMLString(page("/control"), baseURL: publicBase)

        let deadline = Date().addingTimeInterval(5)
        while server.requestCount(for: "/control") == 0 && Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        // Give the guarded page the same window plus a margin.
        try await Task.sleep(nanoseconds: 300_000_000)

        XCTAssertGreaterThan(server.requestCount(for: "/control"), 0, "The control never reached the fixture.")
        XCTAssertEqual(
            server.requestCount(for: "/blocked"), 0,
            "A public page's subresources reached loopback past the frame gate.")
    }

    func testOriginWithATrailingDotMatchesItsGrant() throws {
        let rooted = try XCTUnwrap(BrowserOrigin(origin: "https://docs.example."))
        let plain = try XCTUnwrap(BrowserOrigin(origin: "https://docs.example"))
        XCTAssertEqual(rooted, plain)
        XCTAssertNil(BrowserOrigin(origin: "https://."))
    }
}
