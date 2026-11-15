import XCTest
@testable import TurboSparkApp

/// OC-07: the welcome flow and splash never ask macOS for anything, never
/// reach for a credential, and never touch the network. A source scan in the
/// `RecordingPermissionsTests` style, over every file in `App/FirstRun/`.
/// A transitive prompt (a called helper that asks) is not visible to a scan;
/// the flow calls only the coordinator, the store and shared controls, which
/// the scan below also bounds.
final class FirstRunNoPermissionPromptTests: XCTestCase {
    private static let forbidden = [
        // macOS privacy prompts (RecordingPermissions owns these).
        "requestAccess(", "CGRequestScreenCaptureAccess", "SCShareableContent",
        "AVCaptureDevice", "AXIsProcessTrusted", "RecordingPermissions",
        // Notification authorization (OC-02 asks from its Allow button only).
        "UNUserNotificationCenter", "requestAuthorization", "AppNotificationCenter",
        // Credentials and the Keychain.
        "SecItem", "Keychain", "ServerKeychain", "TypeSafeKeychain",
        // Network.
        "URLSession", "URLRequest", "NWConnection", "WKWebView",
        // A download started from the flow would be a model the user did not
        // choose.
        "installModel", "installRepo", "enqueueModelDownload",
    ]

    private func flowSources() throws -> [(name: String, lines: [String])] {
        let folder = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/App/FirstRun")
        let files = try FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "swift" }
        return try files.map {
            ($0.lastPathComponent, try String(contentsOf: $0, encoding: .utf8).components(separatedBy: "\n"))
        }
    }

    func testFlowSourcesNeverCallPromptNetworkOrCredentialAPIs() throws {
        let sources = try flowSources()
        // The scan must see the files it protects, or it passes on nothing.
        let names = Set(sources.map(\.name))
        for expected in [
            "FirstRunCoordinator.swift", "FirstRunFlowView.swift", "FirstRunStepViews.swift",
            "SplashOverlayView.swift", "FirstRunHost.swift",
        ] {
            XCTAssertTrue(names.contains(expected), "scan lost \(expected)")
        }
        var violations: [String] = []
        for (name, lines) in sources {
            for (index, line) in lines.enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                if code.hasPrefix("//") || code.hasPrefix("///") { continue }
                for api in Self.forbidden where code.contains(api) {
                    violations.append("\(name):\(index + 1) uses \(api)")
                }
            }
        }
        XCTAssertEqual(violations, [])
    }

    /// The flow starts from a view's `onAppear` through the coordinator, which
    /// takes plain values: it cannot reach `RecordingPermissions` or the
    /// engine even by accident.
    func testCoordinatorTakesValuesNotTheModel() throws {
        let sources = try flowSources()
        let coordinator = try XCTUnwrap(sources.first { $0.name == "FirstRunCoordinator.swift" })
        let text = coordinator.lines.joined(separator: "\n")
        XCTAssertFalse(text.contains("AppModel"))
    }
}
