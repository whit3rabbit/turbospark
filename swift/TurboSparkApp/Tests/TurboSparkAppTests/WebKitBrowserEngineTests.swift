import Foundation
import XCTest
import WebKit

@testable import TurboSparkApp

@MainActor
final class WebKitBrowserEngineTests: XCTestCase {
    func testCrossOriginRedirectIsReauthorizedBeforeItCanCommit() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        var checks: [BrowserNavigationAuthorizationRequest] = []
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { request in
                checks.append(request)
                return request.destination.host == "127.0.0.1" ? .allow : .deny
            }
        )
        let tabID = engine.createTab(owner: .agent)

        try engine.navigate(to: server.url("/redirect"), in: tabID)

        let denied = await waitUntil {
            guard let state = engine.tabStates[tabID] else { return false }
            return state.navigationError == .blockedNavigation(
                origin: "http://localhost:\(server.port)"
            )
        }
        XCTAssertTrue(denied, "The cross-origin redirect should be denied before its page commits.")
        XCTAssertTrue(checks.contains {
            $0.destination.canonicalString == "http://localhost:\(server.port)" && $0.phase == .redirect
        })
        XCTAssertEqual(server.requestCount(for: "/landing"), 0)
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .failed(reason: "Navigation blocked by browser permission policy."))
    }

    func testCrossOriginFormSubmissionIsCheckedAtTheWebKitActionBoundary() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        var checks: [BrowserNavigationAuthorizationRequest] = []
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { request in
                checks.append(request)
                return request.destination.host == "127.0.0.1" ? .allow : .deny
            }
        )
        let tabID = engine.createTab(owner: .agent)
        try engine.navigate(to: server.url("/form"), in: tabID)
        let initialLoadCompleted = await waitUntil { engine.tabStates[tabID]?.loadState == .loaded }
        XCTAssertTrue(initialLoadCompleted)

        _ = try await engine.webView(for: tabID)?.evaluateJavaScript(
            "document.getElementById('cross-origin').requestSubmit()"
        )

        let denied = await waitUntil {
            engine.tabStates[tabID]?.navigationError == .blockedNavigation(
                origin: "http://localhost:\(server.port)"
            )
        }
        XCTAssertTrue(denied)
        XCTAssertTrue(checks.contains { $0.phase == .formSubmission })
        XCTAssertEqual(server.requestCount(for: "/submitted"), 0)
        XCTAssertEqual(engine.webView(for: tabID)?.url?.host, "127.0.0.1")
    }

    func testApprovedCrossOriginRedirectCommitsAfterDestinationApproval() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        var checks: [BrowserNavigationAuthorizationRequest] = []
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { request in
                checks.append(request)
                return .allow
            }
        )
        let tabID = engine.createTab(owner: .agent)

        try engine.navigate(to: server.url("/redirect"), in: tabID)

        let navigationCompleted = await waitUntil { engine.tabStates[tabID]?.loadState == .loaded }
        XCTAssertTrue(navigationCompleted)
        XCTAssertEqual(engine.webView(for: tabID)?.url?.host, "localhost")
        XCTAssertTrue(checks.contains {
            $0.destination.canonicalString == "http://localhost:\(server.port)" && $0.phase == .redirect
        })
        XCTAssertGreaterThan(server.requestCount(for: "/landing"), 0)
    }

    func testDirectUserNavigationTransfersOwnershipBeforeLoadingAndBypassesAgentGate() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        var authorizationChecks = 0
        var takeoverCallbacks = 0
        var takeoverSawAgentOwnership = false
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in
                authorizationChecks += 1
                return .deny
            },
            onUserTakeover: { id in
                takeoverCallbacks += 1
                takeoverSawAgentOwnership = store.tab(id: id)?.owner == .agent
            }
        )
        let tabID = engine.createTab(owner: .agent)

        try engine.navigateAsUser(to: URL(string: "http://localhost:\(server.port)/ok")!, in: tabID)

        XCTAssertEqual(takeoverCallbacks, 1)
        XCTAssertTrue(takeoverSawAgentOwnership, "Cancellation callback runs before the store releases agent ownership.")
        XCTAssertEqual(store.tab(id: tabID)?.owner, .user)
        let userLoadCompleted = await waitUntil { engine.tabStates[tabID]?.loadState == .loaded }
        XCTAssertTrue(userLoadCompleted)
        XCTAssertEqual(authorizationChecks, 0)
        XCTAssertEqual(engine.tabStates[tabID]?.progress, 1)
    }

    func testUserPopupCreatesManagedTabAndAgentPopupCreatesNone() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let userStore = BrowserTabStore()
        let userEngine = WebKitBrowserEngine(
            tabStore: userStore,
            authorizeNavigation: { _ in .deny }
        )
        let userTabID = userEngine.createTab(owner: .user)
        let userPopupID = try XCTUnwrap(
            userEngine.routeManagedPopup(from: userTabID, to: server.url("/ok"))
        )
        XCTAssertEqual(userStore.tabs.count, 2)
        let popup = try XCTUnwrap(userStore.tab(id: userPopupID))
        XCTAssertEqual(popup.owner, .user)
        XCTAssertNotNil(userEngine.webView(for: popup.id))
        XCTAssertTrue(userEngine.webView(for: userTabID) === userEngine.webView(for: userTabID))
        XCTAssertFalse(userEngine.webView(for: popup.id) === userEngine.webView(for: userTabID))
        let userPopupLoaded = await waitUntil { userEngine.tabStates[popup.id]?.loadState == .loaded }
        XCTAssertTrue(userPopupLoaded)
        XCTAssertEqual(userEngine.webView(for: popup.id)?.url?.host, "127.0.0.1")

        let agentStore = BrowserTabStore()
        let agentEngine = WebKitBrowserEngine(
            tabStore: agentStore,
            authorizeNavigation: { _ in .allow }
        )
        let agentTabID = agentEngine.createTab(owner: .agent)
        XCTAssertNil(agentEngine.routeManagedPopup(from: agentTabID, to: server.url("/ok")))
        XCTAssertEqual(agentStore.tabs.count, 1)
        XCTAssertNil(agentEngine.tabStates.values.first { $0.tabID != agentTabID })
    }

    func testLoadFailureProgressCrashAndExplicitRecovery() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in .allow }
        )
        let failingTab = engine.createTab(owner: .user)
        try engine.navigate(to: server.url("/drop"), in: failingTab)

        let failedLoadRecorded = await waitUntil {
            guard case .failed = engine.tabStates[failingTab]?.loadState else { return false }
            return engine.tabStates[failingTab]?.loadError != nil
        }
        XCTAssertTrue(failedLoadRecorded)

        let recoveryTab = engine.createTab(owner: .user)
        try engine.navigate(to: server.url("/ok"), in: recoveryTab)
        let initialRecoveryPageLoaded = await waitUntil { engine.tabStates[recoveryTab]?.loadState == .loaded }
        XCTAssertTrue(initialRecoveryPageLoaded)
        XCTAssertEqual(engine.tabStates[recoveryTab]?.progress, 1)
        XCTAssertEqual(engine.tabStates[recoveryTab]?.title, "Ok")
        _ = try await engine.webView(for: recoveryTab)?.evaluateJavaScript("window.recoveryMarker = 'old page'")
        let webView = try XCTUnwrap(engine.webView(for: recoveryTab))
        engine.webViewWebContentProcessDidTerminate(webView)
        XCTAssertEqual(store.tab(id: recoveryTab)?.loadState, .crashed)

        try engine.recoverCrashedTab(recoveryTab)

        let recoveryLoadCompleted = await waitUntil { engine.tabStates[recoveryTab]?.loadState == .loaded }
        XCTAssertTrue(recoveryLoadCompleted)
        let marker = try await webView.evaluateJavaScript("window.recoveryMarker || null") as? String
        XCTAssertNil(marker, "Recovery reloads the stored address and does not restore prior page state.")
        XCTAssertEqual(engine.tabStates[recoveryTab]?.progress, 1)
    }

    func testNavigationRejectsNonHTTPAndUserInfoAndNormalizesExactOrigin() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        var checkedOrigin: BrowserOrigin?
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { request in
                checkedOrigin = request.destination
                return .allow
            }
        )
        let tabID = engine.createTab(owner: .agent)

        XCTAssertThrowsError(try engine.navigate(to: URL(string: "file:///tmp/page.html")!, in: tabID))
        XCTAssertThrowsError(try engine.navigate(to: URL(string: "http://user:secret@127.0.0.1:\(server.port)/ok")!, in: tabID))
        try engine.navigate(
            to: URL(string: "HTTP://LOCALHOST:\(server.port)/ok")!,
            in: tabID
        )

        let normalizedNavigationCompleted = await waitUntil { engine.tabStates[tabID]?.loadState == .loaded }
        XCTAssertTrue(normalizedNavigationCompleted)
        XCTAssertEqual(checkedOrigin?.canonicalString, "http://localhost:\(server.port)")
        XCTAssertEqual(
            BrowserNavigationDestination(url: URL(string: "HTTP://Example.COM:80/path")!)?.origin.canonicalString,
            "http://example.com"
        )
        XCTAssertEqual(
            BrowserNavigationDestination(url: URL(string: "HTTPS://Example.COM:443/path")!)?.origin.canonicalString,
            "https://example.com"
        )
        XCTAssertNil(BrowserNavigationDestination(url: URL(string: "javascript:alert(1)")!))
        XCTAssertNil(BrowserNavigationDestination(url: URL(string: "https://user@example.com/path")!))
    }

    private func waitUntil(
        timeout: TimeInterval = 5,
        condition: @MainActor () -> Bool
    ) async -> Bool {
        let attempts = Int(timeout / 0.025)
        for _ in 0..<attempts {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        return condition()
    }
}
