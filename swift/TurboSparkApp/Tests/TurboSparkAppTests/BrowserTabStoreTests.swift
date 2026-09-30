import XCTest

@testable import TurboSparkApp

final class BrowserTabStoreTests: XCTestCase {
    func testTabIdentityAndSelectionRemainStableAcrossNavigationAndClose() throws {
        let store = BrowserTabStore()
        let first = store.createTab(owner: .user, address: "https://one.example")
        let second = store.createTab(owner: .user, address: "https://two.example", select: false)
        let third = store.createTab(owner: .user, address: "https://three.example", select: false)

        XCTAssertEqual(store.activeTabID, first)
        try store.selectTab(second)
        try store.startNavigation(in: second, to: "https://two.example/next")
        try store.completeNavigation(in: second, title: "Two", address: "https://two.example/next")

        XCTAssertEqual(store.tab(id: second)?.id, second)
        XCTAssertEqual(store.tab(id: second)?.loadState, .loaded)
        XCTAssertEqual(store.activeTabID, second)

        try store.closeTab(second)
        XCTAssertEqual(store.activeTabID, third, "Closing an active tab selects the tab to its right when available.")

        try store.closeTab(third)
        XCTAssertEqual(store.activeTabID, first, "Closing the last tab selects the nearest remaining tab to the left.")

        try store.closeTab(first)
        XCTAssertNil(store.activeTabID)
        XCTAssertTrue(store.tabs.isEmpty)
    }

    func testAgentOwnedBackgroundTabRemainsIndependentOfVisibleTab() throws {
        let store = BrowserTabStore()
        let userTab = store.createTab(owner: .user, address: "https://user.example")
        let agentTab = store.createTab(owner: .agent, address: "https://agent.example", select: false)
        let controlToken = try store.acquireAgentControl(for: agentTab)

        try store.selectTab(userTab)

        XCTAssertEqual(store.activeTabID, userTab)
        XCTAssertEqual(try store.agentControlledTab(for: controlToken), agentTab)
        XCTAssertEqual(store.tab(id: userTab)?.owner, .user)
        XCTAssertEqual(store.tab(id: agentTab)?.owner, .agent)
        XCTAssertEqual(store.tab(id: agentTab)?.address, "https://agent.example")
    }

    func testAgentControlIsUniqueAndReusesTheCurrentTabToken() throws {
        let store = BrowserTabStore()
        let controlledTab = store.createTab(owner: .agent, address: "https://controlled.example")
        let backgroundTab = store.createTab(owner: .agent, address: "https://background.example", select: false)
        let userTab = store.createTab(owner: .user, address: "https://user.example", select: false)

        let token = try store.acquireAgentControl(for: controlledTab)

        XCTAssertEqual(try store.acquireAgentControl(for: controlledTab), token)
        XCTAssertThrowsError(try store.acquireAgentControl(for: backgroundTab)) { error in
            XCTAssertEqual(error as? BrowserTabStoreError, .agentControlAlreadyAssigned(controlledTab))
        }
        XCTAssertThrowsError(try store.acquireAgentControl(for: userTab)) { error in
            XCTAssertEqual(error as? BrowserTabStoreError, .tabNotAgentOwned(userTab))
        }
        XCTAssertEqual(try store.agentControlledTab(for: token), controlledTab)
    }

    func testUserTakeoverRejectsTheFormerAgentControlToken() throws {
        let store = BrowserTabStore()
        let tabID = store.createTab(owner: .agent, address: "https://agent.example")
        let token = try store.acquireAgentControl(for: tabID)

        try store.transferOwnershipToUser(of: tabID)

        XCTAssertEqual(store.tab(id: tabID)?.owner, .user)
        XCTAssertNil(store.agentControlledTabID)
        XCTAssertThrowsError(try store.agentControlledTab(for: token)) { error in
            XCTAssertEqual(error as? BrowserTabStoreError, .invalidAgentControlToken)
        }
    }

    func testClosingControlledTabRejectsItsFormerAgentControlToken() throws {
        let store = BrowserTabStore()
        let controlledTab = store.createTab(owner: .agent, address: "https://controlled.example")
        let remainingTab = store.createTab(owner: .agent, address: "https://remaining.example", select: false)
        let token = try store.acquireAgentControl(for: controlledTab)

        try store.closeTab(controlledTab)

        XCTAssertNil(store.agentControlledTabID)
        XCTAssertThrowsError(try store.agentControlledTab(for: token)) { error in
            XCTAssertEqual(error as? BrowserTabStoreError, .invalidAgentControlToken)
        }
        let replacementToken = try store.acquireAgentControl(for: remainingTab)
        XCTAssertNotEqual(token, replacementToken)
        XCTAssertEqual(try store.agentControlledTab(for: replacementToken), remainingTab)
    }

    func testLoadFailureCrashAndExplicitMetadataRestoreTransitions() throws {
        let store = BrowserTabStore()
        let tabID = store.createTab(owner: .user, address: "https://example.com/start", title: "Start")

        try store.startNavigation(in: tabID, to: "https://example.com/loaded")
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .loading)

        try store.completeNavigation(in: tabID, title: "Loaded", address: "https://example.com/loaded")
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .loaded)

        try store.startNavigation(in: tabID, to: "https://example.com/fails")
        try store.failNavigation(in: tabID, reason: "network unavailable")
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .failed(reason: "network unavailable"))

        try store.startNavigation(in: tabID, to: "https://example.com/retry")
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .loading)
        try store.markCrashed(tabID)
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .crashed)

        try store.restoreCrashedTab(tabID)
        let restoredTab = try XCTUnwrap(store.tab(id: tabID))
        XCTAssertEqual(restoredTab.id, tabID)
        XCTAssertEqual(restoredTab.address, "https://example.com/retry")
        XCTAssertEqual(restoredTab.title, "Loaded")
        XCTAssertEqual(restoredTab.loadState, .restored)
        XCTAssertNotEqual(restoredTab.loadState, .loaded, "Metadata recovery must not claim page state was restored.")

        try store.startNavigation(in: tabID, to: "https://example.com/retry")
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .loading)
    }

    func testIllegalStateTransitionsAreRejectedWithoutChangingTheTab() throws {
        let store = BrowserTabStore()
        let tabID = store.createTab(owner: .user, address: "https://example.com")

        XCTAssertThrowsError(try store.completeNavigation(in: tabID, title: "Unexpected"))
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .idle)

        try store.markCrashed(tabID)
        XCTAssertThrowsError(try store.startNavigation(in: tabID, to: "https://example.com/retry"))
        XCTAssertEqual(store.tab(id: tabID)?.loadState, .crashed)
    }

    func testUserPopupCreatesAndSelectsAManagedUserTab() throws {
        let store = BrowserTabStore()
        let source = store.createTab(owner: .user, address: "https://source.example")

        let result = try store.requestPopup(from: source, address: "https://popup.example")

        guard case let .opened(tabID) = result else {
            return XCTFail("A popup from a user-owned tab should enter the managed tab set.")
        }
        XCTAssertEqual(store.tabs.count, 2)
        XCTAssertEqual(store.activeTabID, tabID)
        XCTAssertEqual(store.tab(id: tabID)?.owner, .user)
        XCTAssertEqual(store.tab(id: tabID)?.address, "https://popup.example")
    }

    func testAgentPopupIsDeniedWithoutCreatingOrSelectingATab() throws {
        let store = BrowserTabStore()
        let userTab = store.createTab(owner: .user, address: "https://user.example")
        let agentTab = store.createTab(owner: .agent, address: "https://agent.example", select: false)
        try store.selectTab(userTab)
        let originalTabs = store.tabs

        let result = try store.requestPopup(from: agentTab, address: "https://unmanaged.example")

        XCTAssertEqual(result, .deniedAgentPopup)
        XCTAssertEqual(store.tabs, originalTabs)
        XCTAssertEqual(store.activeTabID, userTab)
        XCTAssertNil(store.tabs.first(where: { $0.address == "https://unmanaged.example" }))
    }

    func testExplicitUserTakeoverAllowsManagedPopupFromFormerAgentTab() throws {
        let store = BrowserTabStore()
        let tabID = store.createTab(owner: .agent, address: "https://agent.example")

        try store.transferOwnershipToUser(of: tabID)
        let result = try store.requestPopup(from: tabID, address: "https://user-popup.example")

        guard case let .opened(popupID) = result else {
            return XCTFail("A popup after explicit user takeover should use the managed user-tab path.")
        }
        XCTAssertEqual(store.tab(id: tabID)?.owner, .user)
        XCTAssertEqual(store.tab(id: popupID)?.owner, .user)
    }
}
