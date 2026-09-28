import Foundation

/// Backend-neutral operation names exposed to automation callers.
public enum BrowserControlCommandKind: String, CaseIterable, Codable, Hashable, Sendable {
    case navigate
    case click
    case type
    case pressKey
    case scroll
    case screenshot
    case readState
    case waitFor
    case setViewport
}

public enum BrowserControlProtocol {
    /// Each entry is the complete command vocabulary for that protocol version. Add a new
    /// snapshot and advance currentVersion when adding a command; preserve earlier snapshots.
    public static let commandKindsByVersion: [Int: [BrowserControlCommandKind]] = [
        1: [
            .navigate, .click, .type, .pressKey, .scroll,
            .screenshot, .readState, .waitFor, .setViewport
        ]
    ]

    /// Explicitly names the latest registered command snapshot.
    public static let currentVersion = 1

    public static func commandKinds(forVersion version: Int) -> [BrowserControlCommandKind] {
        commandKindsByVersion[version] ?? []
    }
}

public enum BrowserLoadState: String, Codable, Hashable, Sendable {
    case started
    case committed
    case finished
}

public enum BrowserScrollDirection: String, Codable, Hashable, Sendable {
    case up
    case down
    case left
    case right
}

public enum BrowserSnapshotScope: String, Codable, Hashable, Sendable {
    case interactiveElements
    case visibleText
    case pageStructure
}

public enum BrowserElementWaitCondition: String, Codable, Hashable, Sendable {
    case exists
    case visible
    case hidden
    case enabled
}

public enum BrowserWaitTarget: Codable, Equatable, Sendable {
    case loadState(BrowserLoadState)
    case element(reference: String, condition: BrowserElementWaitCondition)
}

/// A closed, versioned set of browser operations. Dialog resolution stays in app-owned UI.
public enum BrowserControlCommand: Sendable, Equatable {
    case navigate(url: String, waitUntil: BrowserLoadState, timeoutSeconds: Double?)
    case click(reference: String, timeoutSeconds: Double?)
    case type(reference: String, text: String, submit: Bool, timeoutSeconds: Double?)
    case pressKey(reference: String?, key: String, timeoutSeconds: Double?)
    case scroll(reference: String?, direction: BrowserScrollDirection, amount: Int)
    case screenshot(fullPage: Bool)
    case readState(scope: BrowserSnapshotScope)
    case waitFor(target: BrowserWaitTarget, timeoutSeconds: Double)
    case setViewport(width: Int?, height: Int?, zoom: Double?)

    public var kind: BrowserControlCommandKind {
        switch self {
        case .navigate: .navigate
        case .click: .click
        case .type: .type
        case .pressKey: .pressKey
        case .scroll: .scroll
        case .screenshot: .screenshot
        case .readState: .readState
        case .waitFor: .waitFor
        case .setViewport: .setViewport
        }
    }
}

public struct BrowserControlRequest: Sendable, Equatable {
    public let id: UUID
    public let command: BrowserControlCommand

    public init(id: UUID = UUID(), command: BrowserControlCommand) {
        self.id = id
        self.command = command
    }
}
