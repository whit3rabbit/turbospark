import Foundation

// The contract between the app and the computer-use helper. Both link this
// module, so they cannot disagree about a message's shape. Coordinates
// are global screen points with a top-left origin, the space CGEvent and the
// AX frame attributes use.

public enum ComputerUseProtocol {
    /// Bumped whenever a command or result changes shape. A helper that
    /// reports another version gets no tools at all (fail closed).
    public static let currentVersion = 1
}

public enum ComputerUseCommandKind: String, Codable, CaseIterable, Hashable, Sendable {
    case listApps
    case snapshot
    case screenshot
    case click
    case type
    case key
    case scroll
    case openApp
}

public struct ComputerUseBackendManifest: Codable, Equatable, Sendable {
    public let backendIdentifier: String
    public let commandSurfaceVersion: Int
    public let supportedCommands: Set<ComputerUseCommandKind>

    public init(
        backendIdentifier: String,
        commandSurfaceVersion: Int = ComputerUseProtocol.currentVersion,
        supportedCommands: Set<ComputerUseCommandKind>
    ) {
        self.backendIdentifier = backendIdentifier
        self.commandSurfaceVersion = commandSurfaceVersion
        self.supportedCommands = supportedCommands
    }

    public var isCurrent: Bool { commandSurfaceVersion == ComputerUseProtocol.currentVersion }

    public func supports(_ kind: ComputerUseCommandKind) -> Bool {
        isCurrent && supportedCommands.contains(kind)
    }
}

public struct ComputerUseRect: Codable, Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

public enum ComputerUseMouseButton: String, Codable, Sendable {
    case left
    case right
}

public enum ComputerUseScrollDirection: String, Codable, CaseIterable, Sendable {
    case up
    case down
    case left
    case right
}

/// Where an input lands. A reference comes from the latest snapshot of the
/// app; a point is already mapped from screenshot pixels to screen points.
public enum ComputerUseTarget: Codable, Equatable, Sendable {
    case reference(String)
    case point(x: Double, y: Double)
}

public enum ComputerUseCommand: Codable, Equatable, Sendable {
    case listApps
    case snapshot(bundleID: String)
    case screenshot(bundleID: String)
    case click(bundleID: String, target: ComputerUseTarget, button: ComputerUseMouseButton, count: Int)
    case type(bundleID: String, text: String)
    case key(bundleID: String, combo: String)
    case scroll(bundleID: String, target: ComputerUseTarget?, direction: ComputerUseScrollDirection, amount: Int)
    case openApp(bundleID: String)

    public var kind: ComputerUseCommandKind {
        switch self {
        case .listApps: return .listApps
        case .snapshot: return .snapshot
        case .screenshot: return .screenshot
        case .click: return .click
        case .type: return .type
        case .key: return .key
        case .scroll: return .scroll
        case .openApp: return .openApp
        }
    }

    /// The app a command acts on; nil only for `listApps`.
    public var bundleID: String? {
        switch self {
        case .listApps: return nil
        case .snapshot(let id), .screenshot(let id), .type(let id, _), .key(let id, _), .openApp(let id):
            return id
        case .click(let id, _, _, _), .scroll(let id, _, _, _):
            return id
        }
    }
}

public struct ComputerUseRunningApp: Codable, Equatable, Sendable {
    public let bundleID: String
    public let name: String
    public let isFrontmost: Bool

    public init(bundleID: String, name: String, isFrontmost: Bool) {
        self.bundleID = bundleID
        self.name = name
        self.isFrontmost = isFrontmost
    }
}

/// One capture of an app's window. `windowFrame` is in screen points; the
/// PNG may be larger (Retina) and is downscaled again before the model sees it.
public struct ComputerUseCapture: Codable, Equatable, Sendable {
    public let pngData: Data
    public let windowFrame: ComputerUseRect

    public init(pngData: Data, windowFrame: ComputerUseRect) {
        self.pngData = pngData
        self.windowFrame = windowFrame
    }
}

public enum ComputerUseResultValue: Codable, Equatable, Sendable {
    case apps([ComputerUseRunningApp])
    /// A bounded, secure-value-masked AX tree rendered as text with refs.
    case snapshot(String)
    case screenshot(ComputerUseCapture)
    case acknowledged(String)
}

/// The helper connection. The app's implementation is
/// `HelperComputerUsePort`; tests use a fake.
public protocol ComputerUseControlPort: Sendable {
    func perform(_ command: ComputerUseCommand) async throws -> ComputerUseResultValue
    /// Re-arms input after the user's kill switch. Called only when the user
    /// approves a new app grant.
    func resumeAfterStop() async
}

extension ComputerUseControlPort {
    public func resumeAfterStop() async {}
}
