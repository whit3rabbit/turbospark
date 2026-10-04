import XCTest
@testable import TurboSparkApp

@MainActor
final class BrowserPaneViewTests: XCTestCase {
    func testPaneCreatesSelectsAndClosesUserTabs() throws {
        let store = BrowserTabStore()
        let engine = makeEngine(store)
        let pane = BrowserPaneModel(engine: engine)

        let firstTab = pane.createTab()
        let secondTab = pane.createTab()

        XCTAssertEqual(pane.selectedTabID, secondTab)
        XCTAssertEqual(store.activeTabID, secondTab)
        XCTAssertEqual(store.tab(id: firstTab)?.owner, .user)
        XCTAssertEqual(store.tab(id: secondTab)?.owner, .user)

        pane.closeTab(firstTab)
        XCTAssertNil(store.tab(id: firstTab))
        XCTAssertEqual(pane.selectedTabID, secondTab)

        pane.closeSelectedTab()
        XCTAssertNil(store.tab(id: secondTab))
        XCTAssertNil(pane.selectedTabID)
    }

    func testPaneCanViewUserTabWhileAgentOwnsBackgroundTab() throws {
        let store = BrowserTabStore()
        let visibleTab = store.createTab(owner: .user)
        let agentTab = store.createTab(owner: .agent, select: false)
        let engine = makeEngine(store)
        _ = try store.acquireAgentControl(for: agentTab)
        let pane = BrowserPaneModel(engine: engine)

        pane.selectTab(agentTab)
        XCTAssertTrue(pane.isAgentControlled(tabID: agentTab))
        pane.selectTab(visibleTab)

        XCTAssertFalse(pane.isAgentControlled(tabID: visibleTab))
        XCTAssertEqual(store.agentControlledTabID, agentTab)
        XCTAssertEqual(store.activeTabID, visibleTab)
        XCTAssertEqual(store.tab(id: visibleTab)?.owner, .user)
    }

    func testDirectNavigationTransfersAgentTabAndStopUpdatesState() throws {
        let store = BrowserTabStore()
        var takeoverCount = 0
        let engine = makeEngine(store) { _ in takeoverCount += 1 }
        let agentTab = engine.createTab(owner: .agent)
        _ = try store.acquireAgentControl(for: agentTab)
        let pane = BrowserPaneModel(
            engine: engine,
            externalBrowserOpener: { _ in }
        )
        pane.updateAddressDraft("https://example.test/path")

        XCTAssertTrue(pane.submitAddress())
        XCTAssertEqual(takeoverCount, 1)
        XCTAssertEqual(store.tab(id: agentTab)?.owner, .user)
        XCTAssertEqual(pane.selectedTabState?.loadState, .loading)

        pane.stopLoading()
        XCTAssertNotEqual(pane.selectedTabState?.loadState, .loading)
    }

    func testViewportInputTransfersTheSelectedAgentTabToTheUser() throws {
        let store = BrowserTabStore()
        var takeoverCount = 0
        let engine = makeEngine(store) { _ in takeoverCount += 1 }
        let agentTab = engine.createTab(owner: .agent)
        _ = try store.acquireAgentControl(for: agentTab)
        let pane = BrowserPaneModel(engine: engine)

        pane.prepareForViewportInput()

        XCTAssertEqual(takeoverCount, 1)
        XCTAssertEqual(store.tab(id: agentTab)?.owner, .user)
        XCTAssertNil(store.agentControlledTabID)
    }

    func testOpenInSystemBrowserUsesSelectedTabAddress() {
        let store = BrowserTabStore()
        _ = store.createTab(owner: .user, address: "https://example.test/path")
        let engine = makeEngine(store)
        var openedURL: URL?
        let pane = BrowserPaneModel(
            engine: engine,
            externalBrowserOpener: { openedURL = $0 }
        )

        XCTAssertTrue(pane.openInSystemBrowser())
        XCTAssertEqual(openedURL?.absoluteString, "https://example.test/path")
    }

    func testRecoveringCrashedTabPreservesTheTabSet() throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        let engine = makeEngine(store)
        let crashedTab = engine.createTab(owner: .user)
        let remainingTab = engine.createTab(owner: .user, select: false)
        try engine.navigate(to: server.url("/ok"), in: crashedTab)
        let crashedWebView = try XCTUnwrap(engine.webView(for: crashedTab))
        engine.webViewWebContentProcessDidTerminate(crashedWebView)

        let pane = BrowserPaneModel(engine: engine)
        pane.selectTab(crashedTab)

        XCTAssertTrue(pane.retryOrRecover())
        XCTAssertEqual(store.tabs.map(\.id), [crashedTab, remainingTab])
        XCTAssertEqual(store.tab(id: crashedTab)?.loadState, .loading)
        XCTAssertEqual(store.tab(id: remainingTab)?.loadState, .idle)
        XCTAssertNotNil(engine.webView(for: remainingTab))
    }

    func testAddressBarRejectsNonHTTPAndUserInfoURLs() {
        let store = BrowserTabStore()
        let engine = makeEngine(store)
        let pane = BrowserPaneModel(engine: engine)
        let tabID = pane.createTab()

        pane.updateAddressDraft("file:///etc/passwd")
        XCTAssertFalse(pane.submitAddress())
        XCTAssertTrue(pane.isAddressInvalid)

        pane.updateAddressDraft("http://user:secret@example.test/")
        XCTAssertFalse(pane.submitAddress())
        XCTAssertTrue(pane.isAddressInvalid)
        XCTAssertEqual(store.tab(id: tabID)?.address, nil)
    }

    func testTabPresentationShowsBoundedProgressAndLoadFailures() {
        let tabID = BrowserTabID()
        let loading = BrowserEngineTabState(
            tabID: tabID,
            address: "https://example.test",
            title: nil,
            progress: 1.5,
            loadState: .loading
        )
        let failed = BrowserEngineTabState(
            tabID: tabID,
            address: "https://example.test",
            title: nil,
            loadState: .failed(reason: "Connection refused")
        )
        let crashed = BrowserEngineTabState(
            tabID: tabID,
            address: "https://example.test",
            title: nil,
            loadState: .crashed,
            loadError: "The browser content process terminated."
        )

        XCTAssertEqual(BrowserTabView.presentation(for: loading), .loading(progress: 1))
        XCTAssertEqual(
            BrowserTabView.presentation(for: failed),
            .failed(message: "Connection refused")
        )
        XCTAssertEqual(
            BrowserTabView.presentation(for: crashed),
            .crashed(message: "The browser content process terminated.")
        )
    }

    private func makeEngine(
        _ store: BrowserTabStore,
        onUserTakeover: @escaping BrowserUserTakeoverHandler = { _ in }
    ) -> WebKitBrowserEngine {
        WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in .allow },
            onUserTakeover: onUserTakeover
        )
    }
}
