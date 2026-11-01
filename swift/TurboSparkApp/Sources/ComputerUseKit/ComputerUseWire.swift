import Foundation

/// The macOS grants the helper holds. Neither is ever requested at launch.
public enum ComputerUsePermissionKind: String, Codable, CaseIterable, Sendable {
    case accessibility
    case screenRecording
}

public struct ComputerUseHelperStatus: Codable, Equatable, Sendable {
    public let accessibilityGranted: Bool
    public let screenRecordingGranted: Bool

    public init(accessibilityGranted: Bool, screenRecordingGranted: Bool) {
        self.accessibilityGranted = accessibilityGranted
        self.screenRecordingGranted = screenRecordingGranted
    }
}

/// App to helper. `hello` must come first on a connection; the helper drops
/// a connection whose first message is anything else.
public enum ComputerUseRequest: Codable, Equatable, Sendable {
    case hello(token: String, protocolVersion: Int)
    case status
    /// Shows the macOS prompt (or opens System Settings once it was
    /// declined). The app sends this only from a user's click.
    case requestPermission(ComputerUsePermissionKind)
    case command(ComputerUseCommand)
    /// Re-arms input after the kill switch. Sent when the user starts a new
    /// grant, never by the model.
    case resume
    case shutdown
}

public enum ComputerUseErrorKind: String, Codable, Sendable {
    case unauthorized
    case permissionMissing
    case appNotRunning
    case notFrontmost
    case staleReference
    case outsideWindow
    case invalidInput
    case stoppedByUser
    case failed
}

/// Helper to app. One response per request, in order.
public enum ComputerUseResponse: Codable, Equatable, Sendable {
    case welcome(manifest: ComputerUseBackendManifest, status: ComputerUseHelperStatus)
    case status(ComputerUseHelperStatus)
    case result(ComputerUseResultValue)
    case error(kind: ComputerUseErrorKind, message: String)
    case ok
}

/// Names and paths both sides must agree on.
public enum ComputerUseHelperLayout {
    public static let bundleIdentifier = "com.whit3rabbit.turbospark.computeruse"
    public static let bundleName = "TurboSpark Computer Use.app"
    public static let executableName = "TurboSparkComputerUse"
    /// Relative to the main app bundle.
    public static let relativeBundlePath = "Contents/Helpers/\(bundleName)"
    public static let socketFileName = "s"
    public static let tokenFileName = "token"
    /// Marks events the helper posts, so its kill switch ignores them.
    public static let syntheticEventMarker: Int64 = 0x5453_4355
}

/// A helper refusal or failure, carried to the tool result verbatim.
public struct ComputerUsePortError: Error, Equatable, Sendable {
    public let kind: ComputerUseErrorKind
    public let message: String

    public init(kind: ComputerUseErrorKind, message: String) {
        self.kind = kind
        self.message = message
    }
}
