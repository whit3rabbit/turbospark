import Foundation

public enum BrowserBackendCommandAvailability: String, Codable, Hashable, Sendable {
    case supported
    case unsupported
}

public struct BrowserBackendManifest: Codable, Equatable, Sendable {
    public let backendIdentifier: String
    public let commandSurfaceVersion: Int
    public let commandSupport: [BrowserControlCommandKind: BrowserBackendCommandAvailability]

    public init(
        backendIdentifier: String,
        commandSurfaceVersion: Int = BrowserControlProtocol.currentVersion,
        supportedCommands: Set<BrowserControlCommandKind>
    ) {
        self.backendIdentifier = backendIdentifier
        self.commandSurfaceVersion = commandSurfaceVersion
        self.commandSupport = Dictionary(
            uniqueKeysWithValues: BrowserControlCommandKind.allCases.map { kind in
                (kind, supportedCommands.contains(kind) ? .supported : .unsupported)
            }
        )
    }

    public func support(for command: BrowserControlCommandKind) -> BrowserBackendCommandAvailability {
        commandSupport[command] ?? .unsupported
    }
}

public struct BrowserNavigationResult: Codable, Equatable, Sendable {
    public let url: String
    public let reachedState: BrowserLoadState

    public init(url: String, reachedState: BrowserLoadState) {
        self.url = url
        self.reachedState = reachedState
    }
}

public struct BrowserElementActionResult: Codable, Equatable, Sendable {
    public let reference: String

    public init(reference: String) {
        self.reference = reference
    }
}

public struct BrowserKeyPressResult: Codable, Equatable, Sendable {
    public let reference: String?
    public let key: String

    public init(reference: String?, key: String) {
        self.reference = reference
        self.key = key
    }
}

public struct BrowserScrollResult: Codable, Equatable, Sendable {
    public let reference: String?
    public let direction: BrowserScrollDirection
    public let amount: Int

    public init(reference: String?, direction: BrowserScrollDirection, amount: Int) {
        self.reference = reference
        self.direction = direction
        self.amount = amount
    }
}

public struct BrowserScreenshotResult: Codable, Equatable, Sendable {
    public let captureID: UUID
    public let width: Int
    public let height: Int
    public let fullPage: Bool

    public init(captureID: UUID, width: Int, height: Int, fullPage: Bool) {
        self.captureID = captureID
        self.width = width
        self.height = height
        self.fullPage = fullPage
    }
}

public struct BrowserSnapshotResult: Codable, Equatable, Sendable {
    public let scope: BrowserSnapshotScope
    public let url: String?
    public let title: String?
    public let snapshot: String
    public let truncated: Bool

    public init(
        scope: BrowserSnapshotScope,
        url: String?,
        title: String?,
        snapshot: String,
        truncated: Bool
    ) {
        self.scope = scope
        self.url = url
        self.title = title
        self.snapshot = snapshot
        self.truncated = truncated
    }
}

public struct BrowserWaitResult: Codable, Equatable, Sendable {
    public let target: BrowserWaitTarget

    public init(target: BrowserWaitTarget) {
        self.target = target
    }
}

public struct BrowserViewportResult: Codable, Equatable, Sendable {
    public let width: Int?
    public let height: Int?
    public let zoom: Double?

    public init(width: Int?, height: Int?, zoom: Double?) {
        self.width = width
        self.height = height
        self.zoom = zoom
    }
}

/// The result payload is closed and has a distinct case for each command kind.
public enum BrowserControlValue: Codable, Equatable, Sendable {
    case navigated(BrowserNavigationResult)
    case clicked(BrowserElementActionResult)
    case typed(BrowserElementActionResult)
    case keyPressed(BrowserKeyPressResult)
    case scrolled(BrowserScrollResult)
    case screenshot(BrowserScreenshotResult)
    case state(BrowserSnapshotResult)
    case waitCompleted(BrowserWaitResult)
    case viewportSet(BrowserViewportResult)

    public var commandKind: BrowserControlCommandKind {
        switch self {
        case .navigated: .navigate
        case .clicked: .click
        case .typed: .type
        case .keyPressed: .pressKey
        case .scrolled: .scroll
        case .screenshot: .screenshot
        case .state: .readState
        case .waitCompleted: .waitFor
        case .viewportSet: .setViewport
        }
    }
}

public struct BrowserControlResult: Codable, Equatable, Sendable {
    public let requestID: UUID
    public let commandKind: BrowserControlCommandKind
    public let value: BrowserControlValue
    public let durationMilliseconds: Int
    /// Transient PNG bytes for a screenshot command. Deliberately excluded from Codable output.
    public let screenshotPNGData: Data?

    private enum CodingKeys: String, CodingKey {
        case requestID, commandKind, value, durationMilliseconds
    }

    public init(
        request: BrowserControlRequest,
        value: BrowserControlValue,
        durationMilliseconds: Int,
        screenshotPNGData: Data? = nil
    ) {
        precondition(
            request.command.kind == value.commandKind,
            "A browser result value must match its request command kind."
        )
        self.requestID = request.id
        self.commandKind = request.command.kind
        self.value = value
        self.durationMilliseconds = durationMilliseconds
        self.screenshotPNGData = screenshotPNGData
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        requestID = try container.decode(UUID.self, forKey: .requestID)
        commandKind = try container.decode(BrowserControlCommandKind.self, forKey: .commandKind)
        value = try container.decode(BrowserControlValue.self, forKey: .value)
        durationMilliseconds = try container.decode(Int.self, forKey: .durationMilliseconds)
        screenshotPNGData = nil
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(requestID, forKey: .requestID)
        try container.encode(commandKind, forKey: .commandKind)
        try container.encode(value, forKey: .value)
        try container.encode(durationMilliseconds, forKey: .durationMilliseconds)
    }
}

public enum BrowserNavigationFailureReason: String, Codable, Hashable, Sendable {
    case transport
    case tls
    case serverResponse
    case cancelled
    case unknown
}

public enum BrowserInputErrorReason: String, Codable, Hashable, Sendable {
    case invalidAddress
    case missingReference
    case invalidKey
    case invalidScrollAmount
    case invalidViewport
    case invalidTimeout
    case emptyText
}

public enum BrowserControlErrorKind: String, Codable, Hashable, Sendable {
    case staleReference
    case timeout
    case unsupported
    case blockedNavigation
    case navigationFailure
    case engineCrashed
    case invalidInput
}

public enum BrowserControlError: Error, Codable, Equatable, Sendable {
    case staleReference(reference: String)
    case timeout
    case unsupported(command: BrowserControlCommandKind)
    case blockedNavigation(origin: String)
    case navigationFailure(reason: BrowserNavigationFailureReason)
    case engineCrashed
    case invalidInput(reason: BrowserInputErrorReason)

    public var kind: BrowserControlErrorKind {
        switch self {
        case .staleReference: .staleReference
        case .timeout: .timeout
        case .unsupported: .unsupported
        case .blockedNavigation: .blockedNavigation
        case .navigationFailure: .navigationFailure
        case .engineCrashed: .engineCrashed
        case .invalidInput: .invalidInput
        }
    }
}

public protocol BrowserControlPort: AnyObject, Sendable {
    static var backendIdentifier: String { get }
    static var supportedCommands: Set<BrowserControlCommandKind> { get }

    func perform(_ request: BrowserControlRequest) async throws -> BrowserControlResult

    /// Samples the requested load state or element condition without waiting.
    func isWaitConditionSatisfied(_ target: BrowserWaitTarget, commandID: UUID) async throws -> Bool

    /// Cancellation is idempotent. Repeating a command ID preserves its cancelled state.
    func cancel(commandID: UUID) async
}
