import Foundation
import XCTest
import WebKit

@testable import TurboSparkApp

@MainActor
final class WebKitBrowserEngineTests: XCTestCase {
    func testScreenshotUsesRequestedBackgroundTabsWebView() async throws {
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in .allow }
        )
        let visibleTabID = engine.createTab(owner: .user)
        let backgroundTabID = engine.createTab(owner: .agent, select: false)
        let visibleWebView = try XCTUnwrap(engine.webView(for: visibleTabID))
        let backgroundWebView = try XCTUnwrap(engine.webView(for: backgroundTabID))
        let expectedImage = makeScreenshotImage(hasContent: true)
        var capturedWebView: WKWebView?

        engine.screenshotSnapshotter = { webView in
            capturedWebView = webView
            return expectedImage
        }

        let image = try await engine.captureScreenshot(for: backgroundTabID)

        XCTAssertTrue(capturedWebView === backgroundWebView)
        XCTAssertFalse(capturedWebView === visibleWebView)
        XCTAssertEqual(store.activeTabID, visibleTabID)
        XCTAssertTrue(image === expectedImage)
    }

    func testScreenshotRetriesBlankCaptureBeforeReturningCurrentImage() async throws {
        let engine = makeScreenshotEngine()
        let blankImage = makeScreenshotImage(hasContent: false)
        let currentImage = makeScreenshotImage(hasContent: true)
        var captureCount = 0
        engine.screenshotSnapshotter = { _ in
            captureCount += 1
            return captureCount == 1 ? blankImage : currentImage
        }

        let image = try await engine.captureScreenshot(for: engine.tabStore.tabs[0].id)

        XCTAssertEqual(captureCount, 2)
        XCTAssertTrue(image === currentImage)
    }

    func testScreenshotReturnsTypedErrorAfterBoundedBlankRetries() async throws {
        let engine = makeScreenshotEngine()
        let blankImage = makeScreenshotImage(hasContent: false)
        var captureCount = 0
        engine.screenshotSnapshotter = { _ in
            captureCount += 1
            return blankImage
        }

        do {
            _ = try await engine.captureScreenshot(for: engine.tabStore.tabs[0].id)
            XCTFail("A blank screenshot must not be returned as a successful capture.")
        } catch let error as WebKitBrowserEngineError {
            XCTAssertEqual(error, .screenshotUnavailable)
        }

        XCTAssertEqual(captureCount, 3)
    }

    func testScreenshotRetriesWhenTheTabNavigatesDuringCapture() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let engine = makeScreenshotEngine()
        let tabID = try XCTUnwrap(engine.tabStore.tabs.first?.id)
        try engine.navigate(to: server.url("/ok"), in: tabID)
        let staleImage = makeScreenshotImage(hasContent: true)
        let currentImage = makeScreenshotImage(hasContent: true, marker: .systemBlue)
        var captureCount = 0
        engine.screenshotSnapshotter = { webView in
            captureCount += 1
            if captureCount == 1 {
                try engine.navigate(to: server.url("/landing"), in: tabID)
                return staleImage
            }
            XCTAssertTrue(webView === engine.webView(for: tabID))
            return currentImage
        }

        let image = try await engine.captureScreenshot(for: tabID)

        XCTAssertEqual(captureCount, 2)
        XCTAssertTrue(image === currentImage)
    }

    func testScreenshotReturnsTypedErrorAfterThreeStaleCaptures() async throws {
        let engine = makeScreenshotEngine()
        let tabID = try XCTUnwrap(engine.tabStore.tabs.first?.id)
        let image = makeScreenshotImage(hasContent: true)
        var captureCount = 0
        engine.screenshotSnapshotter = { _ in
            captureCount += 1
            engine.stopLoading(in: tabID)
            return image
        }

        do {
            _ = try await engine.captureScreenshot(for: tabID)
            XCTFail("A screenshot captured across three tab revisions must fail.")
        } catch let error as WebKitBrowserEngineError {
            XCTAssertEqual(error, .screenshotUnavailable)
        }

        XCTAssertEqual(captureCount, 3)
    }

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

    func testNativeDialogsAutomaticallyAcceptAllKinds() throws {
        let (engine, tabID) = makeDialogEngine(policy: .autoAccept)
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }

        for kind in BrowserJavaScriptDialogKind.allCases {
            var callbackDecision: BrowserDialogDecision?
            _ = engine.handleJavaScriptDialog(
                tabID: tabID,
                kind: kind,
                message: "private page message",
                defaultText: "suggested"
            ) { callbackDecision = $0 }

            XCTAssertEqual(callbackDecision, .accept(promptText: kind == .prompt ? "suggested" : nil))
            XCTAssertEqual(events.last?.kind, kind)
            XCTAssertEqual(events.last?.message, "private page message")
            XCTAssertEqual(events.last?.resolution, .accepted)
        }
        XCTAssertTrue(engine.pendingDialogs.isEmpty)
    }

    func testNativeDialogsAutomaticallyDismissAllKinds() throws {
        let (engine, tabID) = makeDialogEngine(policy: .autoDismiss)
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }

        for kind in BrowserJavaScriptDialogKind.allCases {
            var callbackDecision: BrowserDialogDecision?
            _ = engine.handleJavaScriptDialog(
                tabID: tabID,
                kind: kind,
                message: "private page message",
                defaultText: "suggested"
            ) { callbackDecision = $0 }

            XCTAssertEqual(callbackDecision, .dismiss)
            XCTAssertEqual(events.last?.kind, kind)
            XCTAssertEqual(events.last?.resolution, .dismissed)
        }
        XCTAssertTrue(engine.pendingDialogs.isEmpty)
    }

    func testAskPolicyLeavesEachWebKitCallbackPendingUntilAppDecision() throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        var callbackDecisions: [BrowserDialogDecision] = []
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }
        var requestIDs: [UUID] = []

        for kind in BrowserJavaScriptDialogKind.allCases {
            let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
                tabID: tabID,
                kind: kind,
                message: "ask \(kind.rawValue)",
                defaultText: "suggested"
            ) { callbackDecisions.append($0) })
            requestIDs.append(requestID)
        }

        XCTAssertEqual(engine.pendingDialogs.map(\.kind), BrowserJavaScriptDialogKind.allCases)
        XCTAssertTrue(callbackDecisions.isEmpty, "The engine must return while app-owned approval is pending.")
        XCTAssertTrue(events.isEmpty)

        for (requestID, kind) in zip(requestIDs, BrowserJavaScriptDialogKind.allCases) {
            XCTAssertTrue(engine.resolvePendingDialog(
                requestID,
                decision: .accept(promptText: kind == .prompt ? "entered" : nil)
            ))
        }

        XCTAssertEqual(callbackDecisions, BrowserJavaScriptDialogKind.allCases.map {
            .accept(promptText: $0 == .prompt ? "entered" : nil)
        })
        XCTAssertEqual(events.map(\.resolution), Array(repeating: .accepted, count: 3))
        XCTAssertTrue(engine.pendingDialogs.isEmpty)
    }

    func testWebKitConfirmDelegateBridgesAskDecisionBackToJavaScript() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        try engine.navigate(to: server.url("/ok"), in: tabID)
        let pageLoaded = await waitUntil { engine.tabStates[tabID]?.loadState == .loaded }
        XCTAssertTrue(pageLoaded)
        let webView = try XCTUnwrap(engine.webView(for: tabID))
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }

        let confirmation = Task { try await webView.evaluateJavaScript("window.confirm('confirm text')") }
        let requestAppeared = await waitUntil {
            engine.pendingDialogs.first?.message == "confirm text"
        }
        XCTAssertTrue(requestAppeared)
        let request = try XCTUnwrap(engine.pendingDialogs.first)
        XCTAssertTrue(engine.resolvePendingDialog(request.id, decision: .accept(promptText: nil)))

        let confirmationResult = try await confirmation.value as? Bool
        XCTAssertEqual(confirmationResult, true)
        XCTAssertEqual(events.single?.message, "confirm text")
        XCTAssertEqual(events.single?.resolution, .accepted)
    }

    func testAskPolicyCanDismissEveryDialogKindFromAppOwnedUI() throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        var callbackDecisions: [BrowserDialogDecision] = []
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }

        for kind in BrowserJavaScriptDialogKind.allCases {
            let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
                tabID: tabID,
                kind: kind,
                message: "dismiss \(kind.rawValue)"
            ) { callbackDecisions.append($0) })
            XCTAssertTrue(engine.resolvePendingDialog(requestID, decision: .dismiss))
        }

        XCTAssertEqual(callbackDecisions, Array(repeating: .dismiss, count: 3))
        XCTAssertEqual(events.map(\.resolution), Array(repeating: .dismissed, count: 3))
        XCTAssertTrue(engine.pendingDialogs.isEmpty)
    }

    func testDialogResolutionCompletesCallbackExactlyOnceAndRejectsLateDecision() throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        var callbackCount = 0
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }
        let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
            tabID: tabID,
            kind: .confirm,
            message: "continue?"
        ) { _ in callbackCount += 1 })

        XCTAssertTrue(engine.resolvePendingDialog(requestID, decision: .accept(promptText: nil)))
        XCTAssertFalse(engine.resolvePendingDialog(requestID, decision: .dismiss))
        XCTAssertFalse(engine.cancelPendingDialog(requestID, reason: .cancelled))
        XCTAssertEqual(callbackCount, 1)
        XCTAssertEqual(events.count, 1)
    }

    func testExplicitDialogCancellationSafelyDismissesAndRejectsLateDecision() throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        var callbackDecisions: [BrowserDialogDecision] = []
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }
        let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
            tabID: tabID,
            kind: .prompt,
            message: "name?",
            defaultText: "guest"
        ) { callbackDecisions.append($0) })

        XCTAssertTrue(engine.cancelPendingDialog(requestID, reason: .cancelled))
        XCTAssertFalse(engine.resolvePendingDialog(requestID, decision: .accept(promptText: "late")))
        XCTAssertEqual(callbackDecisions, [.dismiss])
        XCTAssertEqual(events.single?.resolution, .cancelled)
    }

    func testDialogTimeoutSafelyDismissesAndRejectsLateDecision() async throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask, timeout: 0.01)
        var callbackDecisions: [BrowserDialogDecision] = []
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }
        let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
            tabID: tabID,
            kind: .confirm,
            message: "continue?"
        ) { callbackDecisions.append($0) })

        let timedOut = await waitUntil { events.single?.resolution == .timedOut }
        XCTAssertTrue(timedOut)
        XCTAssertFalse(engine.resolvePendingDialog(requestID, decision: .accept(promptText: nil)))
        XCTAssertEqual(callbackDecisions, [.dismiss])
    }

    func testClosingTabSafelyDismissesItsPendingDialog() throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        var callbackDecisions: [BrowserDialogDecision] = []
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }
        let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
            tabID: tabID,
            kind: .alert,
            message: "notice"
        ) { callbackDecisions.append($0) })

        try engine.closeTab(tabID)

        XCTAssertFalse(engine.resolvePendingDialog(requestID, decision: .accept(promptText: nil)))
        XCTAssertEqual(callbackDecisions, [.dismiss])
        XCTAssertEqual(events.single?.resolution, .tabClosed)
    }

    func testDirectTakeoverSafelyDismissesPendingDialog() throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        var callbackDecisions: [BrowserDialogDecision] = []
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }
        let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
            tabID: tabID,
            kind: .confirm,
            message: "continue?"
        ) { callbackDecisions.append($0) })

        engine.prepareForDirectUserInput(in: tabID)

        XCTAssertFalse(engine.resolvePendingDialog(requestID, decision: .accept(promptText: nil)))
        XCTAssertEqual(callbackDecisions, [.dismiss])
        XCTAssertEqual(events.single?.resolution, .userTakeover)
    }

    func testContentProcessFailureSafelyDismissesPendingDialog() throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        var callbackDecisions: [BrowserDialogDecision] = []
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }
        let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
            tabID: tabID,
            kind: .prompt,
            message: "name?"
        ) { callbackDecisions.append($0) })
        let webView = try XCTUnwrap(engine.webView(for: tabID))

        engine.webViewWebContentProcessDidTerminate(webView)

        XCTAssertFalse(engine.resolvePendingDialog(requestID, decision: .accept(promptText: "late")))
        XCTAssertEqual(callbackDecisions, [.dismiss])
        XCTAssertEqual(events.single?.resolution, .engineFailure)
    }

    func testNavigationFailureSafelyDismissesPendingDialog() throws {
        let (engine, tabID) = makeDialogEngine(policy: .ask)
        var callbackDecisions: [BrowserDialogDecision] = []
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }
        let requestID = try XCTUnwrap(engine.handleJavaScriptDialog(
            tabID: tabID,
            kind: .alert,
            message: "notice"
        ) { callbackDecisions.append($0) })
        let webView = try XCTUnwrap(engine.webView(for: tabID))

        engine.webView(webView, didFail: nil, withError: NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut))

        XCTAssertFalse(engine.resolvePendingDialog(requestID, decision: .accept(promptText: nil)))
        XCTAssertEqual(callbackDecisions, [.dismiss])
        XCTAssertEqual(events.single?.resolution, .engineFailure)
    }

    private func makeDialogEngine(
        policy: BrowserDialogPolicy,
        timeout: TimeInterval = 60
    ) -> (engine: WebKitBrowserEngine, tabID: BrowserTabID) {
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in .allow },
            dialogPolicyProvider: { policy },
            dialogDecisionTimeout: timeout
        )
        let tabID = engine.createTab(owner: .agent)
        return (engine, tabID)
    }

    private func makeScreenshotEngine() -> WebKitBrowserEngine {
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in .allow }
        )
        _ = engine.createTab(owner: .agent)
        return engine
    }

    private func makeScreenshotImage(
        hasContent: Bool,
        marker: NSColor = .black
    ) -> NSImage {
        let width = 32
        let height = 32
        let representation = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        )!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: representation)
        NSColor.white.setFill()
        NSBezierPath(rect: NSRect(x: 0, y: 0, width: width, height: height)).fill()
        if hasContent {
            marker.setFill()
            NSBezierPath(rect: NSRect(x: 8, y: 8, width: 8, height: 8)).fill()
        }
        NSGraphicsContext.restoreGraphicsState()

        let image = NSImage(size: NSSize(width: width, height: height))
        image.addRepresentation(representation)
        return image
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

private extension Array {
    var single: Element? { count == 1 ? first : nil }
}
