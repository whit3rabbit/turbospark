import Foundation

public struct BrowserTabID: Codable, Hashable, RawRepresentable, Sendable {
    public let rawValue: UUID

    public init(rawValue: UUID) {
        self.rawValue = rawValue
    }

    public init() {
        self.init(rawValue: UUID())
    }
}

public enum BrowserTabOwner: String, Codable, Equatable, Sendable {
    case user
    case agent
}

/// Opaque authority for the single tab currently controlled by browser automation.
public struct BrowserAgentControlToken: Hashable, Sendable {
    fileprivate let rawValue: UUID

    fileprivate init() {
        self.rawValue = UUID()
    }
}

public enum BrowserTabLoadState: Codable, Equatable, Sendable {
    case idle
    case loading
    case loaded
    case failed(reason: String)
    case crashed
    /// Tab metadata was recovered explicitly. This does not claim the page state survived.
    case restored
}

public struct BrowserTab: Codable, Equatable, Identifiable, Sendable {
    public let id: BrowserTabID
    public var title: String?
    public var address: String?
    public var loadState: BrowserTabLoadState
    public var owner: BrowserTabOwner

    public init(
        id: BrowserTabID = BrowserTabID(),
        title: String? = nil,
        address: String? = nil,
        loadState: BrowserTabLoadState = .idle,
        owner: BrowserTabOwner = .user
    ) {
        self.id = id
        self.title = title
        self.address = address
        self.loadState = loadState
        self.owner = owner
    }
}

public enum BrowserTabStoreError: Error, Equatable, Sendable {
    case tabNotFound(BrowserTabID)
    case tabNotAgentOwned(BrowserTabID)
    case agentControlAlreadyAssigned(BrowserTabID)
    case invalidAgentControlToken
    case invalidLoadTransition(BrowserTabLoadState)
}

public enum BrowserPopupRequestResult: Equatable, Sendable {
    case opened(tabID: BrowserTabID)
    case deniedAgentPopup
}

/// Stores tab metadata and coordinates managed tab lifecycle requests.
/// WebView creation and navigation remain the responsibility of the browser engine.
public final class BrowserTabStore: @unchecked Sendable {
    private struct AgentControlLease {
        let tabID: BrowserTabID
        let token: BrowserAgentControlToken
    }

    // The store is read by the BrowserAutomationSession actor (cooperative
    // pool) while WebKit delegate callbacks and user actions mutate it on the
    // main thread. A Swift Array read during an in-place write is undefined
    // behavior, and the lease check-then-act must be atomic, so every access
    // goes through this recursive lock (recursive because requestPopup calls
    // createTab).
    private let lock = NSRecursiveLock()
    private var tabsStorage: [BrowserTab] = []
    private var activeTabIDStorage: BrowserTabID?
    private var agentControlLease: AgentControlLease?

    public var tabs: [BrowserTab] { lock.withLock { tabsStorage } }
    public var activeTabID: BrowserTabID? { lock.withLock { activeTabIDStorage } }
    public var agentControlledTabID: BrowserTabID? { lock.withLock { agentControlLease?.tabID } }

    public init() {}

    /// Creates a metadata record. The first tab becomes active even when `select` is false.
    @discardableResult
    public func createTab(
        owner: BrowserTabOwner,
        address: String? = nil,
        title: String? = nil,
        select: Bool = true
    ) -> BrowserTabID {
        lock.withLock {
            let tab = BrowserTab(title: title, address: address, owner: owner)
            tabsStorage.append(tab)
            if select || activeTabIDStorage == nil {
                activeTabIDStorage = tab.id
            }
            return tab.id
        }
    }

    public func tab(id: BrowserTabID) -> BrowserTab? {
        lock.withLock { tabsStorage.first { $0.id == id } }
    }

    public func selectTab(_ tabID: BrowserTabID) throws {
        try lock.withLock {
            guard tabsStorage.contains(where: { $0.id == tabID }) else {
                throw BrowserTabStoreError.tabNotFound(tabID)
            }
            activeTabIDStorage = tabID
        }
    }

    /// Closing the active tab selects the next tab in order, or the previous tab if it was last.
    public func closeTab(_ tabID: BrowserTabID) throws {
        try lock.withLock {
            guard let index = tabsStorage.firstIndex(where: { $0.id == tabID }) else {
                throw BrowserTabStoreError.tabNotFound(tabID)
            }

            if agentControlLease?.tabID == tabID {
                agentControlLease = nil
            }
            tabsStorage.remove(at: index)
            guard activeTabIDStorage == tabID else { return }

            if tabsStorage.isEmpty {
                activeTabIDStorage = nil
            } else {
                activeTabIDStorage = tabsStorage[min(index, tabsStorage.count - 1)].id
            }
        }
    }

    /// Records a requested navigation. The engine performs the actual load and reports its result.
    public func startNavigation(in tabID: BrowserTabID, to address: String) throws {
        try lock.withLock {
            let index = try indexOfTab(tabID)
            guard tabsStorage[index].loadState != .crashed else {
                throw BrowserTabStoreError.invalidLoadTransition(tabsStorage[index].loadState)
            }
            tabsStorage[index].address = address
            tabsStorage[index].loadState = .loading
        }
    }

    public func completeNavigation(
        in tabID: BrowserTabID,
        title: String? = nil,
        address: String? = nil
    ) throws {
        try lock.withLock {
            let index = try indexOfTab(tabID)
            guard tabsStorage[index].loadState == .loading else {
                throw BrowserTabStoreError.invalidLoadTransition(tabsStorage[index].loadState)
            }
            if let title {
                tabsStorage[index].title = title
            }
            if let address {
                tabsStorage[index].address = address
            }
            tabsStorage[index].loadState = .loaded
        }
    }

    public func failNavigation(in tabID: BrowserTabID, reason: String) throws {
        try lock.withLock {
            let index = try indexOfTab(tabID)
            guard tabsStorage[index].loadState == .loading else {
                throw BrowserTabStoreError.invalidLoadTransition(tabsStorage[index].loadState)
            }
            tabsStorage[index].loadState = .failed(reason: reason)
        }
    }

    /// Records content-process termination without constructing or navigating a WebView.
    public func markCrashed(_ tabID: BrowserTabID) throws {
        try lock.withLock {
            let index = try indexOfTab(tabID)
            tabsStorage[index].loadState = .crashed
        }
    }

    /// Explicitly recovers the tab record after a crash. The engine must load the address again.
    public func restoreCrashedTab(_ tabID: BrowserTabID) throws {
        try lock.withLock {
            let index = try indexOfTab(tabID)
            guard tabsStorage[index].loadState == .crashed else {
                throw BrowserTabStoreError.invalidLoadTransition(tabsStorage[index].loadState)
            }
            tabsStorage[index].loadState = .restored
        }
    }

    /// Transfers the tab to direct user control before accepting user-driven navigation or popups.
    public func transferOwnershipToUser(of tabID: BrowserTabID) throws {
        try lock.withLock {
            let index = try indexOfTab(tabID)
            tabsStorage[index].owner = .user
            if agentControlLease?.tabID == tabID {
                agentControlLease = nil
            }
        }
    }

    /// Acquires the store's single agent-control slot for an agent-owned tab.
    /// Reacquiring the current tab returns its existing token without changing ownership.
    public func acquireAgentControl(for tabID: BrowserTabID) throws -> BrowserAgentControlToken {
        try lock.withLock {
            let index = try indexOfTab(tabID)
            guard tabsStorage[index].owner == .agent else {
                throw BrowserTabStoreError.tabNotAgentOwned(tabID)
            }

            if let currentLease = agentControlLease {
                guard currentLease.tabID == tabID else {
                    throw BrowserTabStoreError.agentControlAlreadyAssigned(currentLease.tabID)
                }
                return currentLease.token
            }

            let token = BrowserAgentControlToken()
            agentControlLease = AgentControlLease(tabID: tabID, token: token)
            return token
        }
    }

    /// Resolves a token only while it still owns the current agent-controlled tab.
    public func agentControlledTab(for token: BrowserAgentControlToken) throws -> BrowserTabID {
        try lock.withLock {
            guard let currentLease = agentControlLease, currentLease.token == token else {
                throw BrowserTabStoreError.invalidAgentControlToken
            }
            return currentLease.tabID
        }
    }

    /// Routes user popups into managed tabs and refuses agent popups without creating a tab.
    public func requestPopup(
        from sourceTabID: BrowserTabID,
        address: String? = nil
    ) throws -> BrowserPopupRequestResult {
        try lock.withLock {
            let sourceIndex = try indexOfTab(sourceTabID)
            guard tabsStorage[sourceIndex].owner == .user else {
                return .deniedAgentPopup
            }

            return .opened(tabID: createTab(owner: .user, address: address))
        }
    }

    // Callers must hold `lock`.
    private func indexOfTab(_ tabID: BrowserTabID) throws -> Int {
        guard let index = tabsStorage.firstIndex(where: { $0.id == tabID }) else {
            throw BrowserTabStoreError.tabNotFound(tabID)
        }
        return index
    }
}
