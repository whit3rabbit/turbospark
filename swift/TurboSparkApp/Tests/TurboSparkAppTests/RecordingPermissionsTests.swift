import AVFoundation
import XCTest
@testable import TurboSparkApp

/// Fake TCC state. Every system call goes through it, so these tests never
/// read or prompt for the developer's real microphone or screen grants.
@MainActor
private final class FakeGrants {
    var microphone: AVAuthorizationStatus = .notDetermined
    var screen = false
    var grantMicrophoneOnRequest = true
    var microphoneRequests = 0
    var screenRequests = 0
    var opened: [MacPrivacyPane] = []

    var dependencies: RecordingPermissionDependencies {
        var deps = RecordingPermissionDependencies()
        deps.microphoneStatus = { self.microphone }
        deps.requestMicrophone = {
            self.microphoneRequests += 1
            self.microphone = self.grantMicrophoneOnRequest ? .authorized : .denied
            return self.grantMicrophoneOnRequest
        }
        deps.systemAudioGranted = { self.screen }
        // macOS shows its own alert and returns false at once on a first
        // request, so a request alone never grants screen capture.
        deps.requestSystemAudio = { self.screenRequests += 1; return false }
        deps.openSettings = { self.opened.append($0) }
        return deps
    }
}

final class RecordingPermissionsTests: XCTestCase {
    @MainActor func testStatusMapping() {
        let fake = FakeGrants()
        let permissions = RecordingPermissions(dependencies: fake.dependencies)
        XCTAssertEqual(permissions.state(.microphone), .notDetermined)
        XCTAssertEqual(permissions.state(.systemAudio), .notDetermined)
        for (status, expected) in [(AVAuthorizationStatus.authorized, RecordingPermissionState.granted),
                                   (.denied, .denied), (.restricted, .restricted)] {
            fake.microphone = status
            permissions.refresh()
            XCTAssertEqual(permissions.state(.microphone), expected)
        }
        fake.screen = true
        permissions.refresh()
        XCTAssertEqual(permissions.state(.systemAudio), .granted)
    }

    @MainActor func testNeededAndMissingFollowSourceSelection() {
        XCTAssertEqual(RecordingPermission.needed(microphone: true, systemAudio: true), [.microphone, .systemAudio])
        XCTAssertEqual(RecordingPermission.needed(microphone: false, systemAudio: true), [.systemAudio])
        XCTAssertEqual(RecordingPermission.needed(microphone: false, systemAudio: false), [])
        let fake = FakeGrants()
        fake.microphone = .authorized
        let permissions = RecordingPermissions(dependencies: fake.dependencies)
        XCTAssertEqual(permissions.missing([.microphone, .systemAudio]), [.systemAudio])
        XCTAssertEqual(permissions.missing([.microphone]), [])
    }

    /// Screen capture preflight cannot tell "never asked" from "refused", and
    /// macOS prompts only once. After one request the next step must be
    /// System Settings, not a second request that shows nothing.
    @MainActor func testSystemAudioRefusalMovesToSystemSettings() async {
        let fake = FakeGrants()
        let permissions = RecordingPermissions(dependencies: fake.dependencies)
        await permissions.request(.systemAudio)
        XCTAssertEqual(fake.screenRequests, 1)
        XCTAssertEqual(permissions.state(.systemAudio), .denied)
        XCTAssertTrue(fake.opened.isEmpty, "the first request must not also open Settings")
        await permissions.request(.systemAudio)
        XCTAssertEqual(fake.screenRequests, 1)
        XCTAssertEqual(fake.opened, [.screenRecording])
    }

    @MainActor func testRequestAllPromptsInOrderAndNeverOpensSettingsTwice() async {
        let fake = FakeGrants()
        let permissions = RecordingPermissions(dependencies: fake.dependencies)
        await permissions.requestAll([.microphone, .systemAudio])
        XCTAssertEqual(fake.microphoneRequests, 1)
        XCTAssertEqual(fake.screenRequests, 1)
        XCTAssertEqual(permissions.state(.microphone), .granted)
        XCTAssertEqual(permissions.state(.systemAudio), .denied)
        XCTAssertTrue(fake.opened.isEmpty)
        await permissions.requestAll([.microphone, .systemAudio])
        XCTAssertEqual(fake.microphoneRequests, 1)
        XCTAssertEqual(fake.opened, [.screenRecording])
        fake.screen = true
        permissions.refresh()
        XCTAssertTrue(permissions.missing([.microphone, .systemAudio]).isEmpty)
    }

    @MainActor func testDeniedMicrophoneOpensSettingsAndStops() async {
        let fake = FakeGrants()
        fake.microphone = .denied
        let permissions = RecordingPermissions(dependencies: fake.dependencies)
        await permissions.requestAll([.microphone, .systemAudio])
        XCTAssertEqual(fake.microphoneRequests, 0)
        XCTAssertEqual(fake.screenRequests, 0, "a step that needs Settings ends the pass")
        XCTAssertEqual(fake.opened, [.microphone])
    }

    @MainActor func testRestrictedGrantIsLeftAlone() async {
        let fake = FakeGrants()
        fake.microphone = .restricted
        let permissions = RecordingPermissions(dependencies: fake.dependencies)
        await permissions.request(.microphone)
        XCTAssertEqual(fake.microphoneRequests, 0)
        XCTAssertTrue(fake.opened.isEmpty)
        XCTAssertEqual(permissions.state(.microphone), .restricted)
    }

    /// The app never asks on its own: building the model and reading status,
    /// as the Record panel does on appear and on every activation, must not
    /// reach a request.
    @MainActor func testReadingStatusNeverPrompts() {
        let fake = FakeGrants()
        let permissions = RecordingPermissions(dependencies: fake.dependencies)
        permissions.refresh()
        _ = permissions.missing([.microphone, .systemAudio])
        _ = permissions.fallback([.microphone, .systemAudio])
        XCTAssertEqual(fake.microphoneRequests, 0)
        XCTAssertEqual(fake.screenRequests, 0)
        XCTAssertTrue(fake.opened.isEmpty)
    }

    /// Declining one source must leave a way to record with the other.
    @MainActor func testFallbackOffersOnlyAllowedSources() {
        let fake = FakeGrants()
        fake.microphone = .denied
        fake.screen = true
        let permissions = RecordingPermissions(dependencies: fake.dependencies)
        let both: [RecordingPermission] = [.microphone, .systemAudio]
        XCTAssertEqual(permissions.fallback(both)?.allowed, [.systemAudio])
        XCTAssertEqual(permissions.fallback(both)?.declined, [.microphone])
        XCTAssertNil(permissions.fallback([.microphone]), "nothing allowed to fall back to")
        fake.microphone = .authorized
        permissions.refresh()
        XCTAssertNil(permissions.fallback(both), "everything allowed, nothing to decline")
    }

    /// Only `RecordingPermissions` may call the APIs that show a macOS
    /// prompt, and every ScreenCaptureKit content query (which prompts when
    /// access is missing) sits behind the access check. A new call site that
    /// asks on its own fails here instead of shipping a surprise prompt.
    func testOnlyRecordingPermissionsCallsTheMacOSPrompts() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        let files = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
        XCTAssertFalse(files.isEmpty)
        var violations: [String] = []
        var guardedQueries = 0
        for file in files {
            let lines = try String(contentsOf: file, encoding: .utf8).components(separatedBy: "\n")
            for (index, line) in lines.enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                if code.hasPrefix("//") { continue }
                let name = file.lastPathComponent
                for api in ["requestAccess(for:", "CGRequestScreenCaptureAccess("] where code.contains(api) {
                    if name != "RecordingPermissions.swift" { violations.append("\(name):\(index + 1) calls \(api)") }
                }
                // The Accessibility grant belongs to the computer-use helper,
                // never to this app, whose identity changes every update.
                if code.contains("AXIsProcessTrustedWithOptions(") {
                    violations.append("\(name):\(index + 1) prompts for Accessibility")
                }
                if code.contains("SCShareableContent.") {
                    if index > 0, lines[index - 1].contains("systemAudioAccessAllowed()") { guardedQueries += 1 }
                    else { violations.append("\(name):\(index + 1) queries ScreenCaptureKit without the access check") }
                }
            }
        }
        XCTAssertEqual(violations, [])
        XCTAssertGreaterThan(guardedQueries, 0, "the scan must see the guarded queries it protects")
    }

    func testPrivacyPaneLinks() {
        XCTAssertEqual(MacPrivacyPane.microphone.url?.absoluteString,
                       "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
        XCTAssertEqual(MacPrivacyPane.screenRecording.url?.absoluteString,
                       "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
    }
}
