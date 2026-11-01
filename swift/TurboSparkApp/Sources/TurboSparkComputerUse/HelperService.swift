import AppKit
import ApplicationServices
import ComputerUseKit
import CoreGraphics

/// Handles one request at a time on the main actor. Every input checks,
/// right before posting, that the target app is frontmost and that the point
/// lands in one of its windows, so a focus change cannot redirect input.
@MainActor
final class HelperService {
    nonisolated static let manifest = ComputerUseBackendManifest(
        backendIdentifier: "turbospark-computer-use-helper",
        supportedCommands: Set(ComputerUseCommandKind.allCases))

    private let snapshotter = AXSnapshotter()
    private let capturer = WindowCapturer()
    private let input = InputSynthesizer()
    private var stopped = false
    /// The kill switch listens only shortly after input, while the agent is
    /// visibly driving, so the user's ordinary Escape presses do not disarm it.
    private var lastInputAt: Date = .distantPast
    private var keyMonitor: Any?
    var onStoppedChange: ((Bool) -> Void)?

    func installKillSwitch() {
        keyMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard event.keyCode == 0x35 else { return }
            let marker = event.cgEvent?.getIntegerValueField(.eventSourceUserData) ?? 0
            guard marker != ComputerUseHelperLayout.syntheticEventMarker else { return }
            Task { @MainActor in
                guard let self, Date().timeIntervalSince(self.lastInputAt) < 10 else { return }
                self.stop()
            }
        }
    }

    func stop() {
        stopped = true
        onStoppedChange?(true)
    }

    func welcome() async -> ComputerUseResponse {
        .welcome(manifest: Self.manifest, status: status())
    }

    func status() -> ComputerUseHelperStatus {
        // Status reads never prompt.
        ComputerUseHelperStatus(
            accessibilityGranted: AXIsProcessTrusted(),
            screenRecordingGranted: CGPreflightScreenCaptureAccess())
    }

    func handle(_ request: ComputerUseRequest) async -> ComputerUseResponse {
        switch request {
        case .hello:
            return .error(kind: .invalidInput, message: "Already connected.")
        case .status:
            return .status(status())
        case .requestPermission(let kind):
            requestPermission(kind)
            return .status(status())
        case .resume:
            stopped = false
            onStoppedChange?(false)
            return .ok
        case .shutdown:
            return .ok
        case .command(let command):
            do {
                return .result(try await perform(command))
            } catch let error as HelperError {
                return .error(kind: error.kind, message: error.message)
            } catch {
                return .error(kind: .failed, message: "The action failed.")
            }
        }
    }

    /// The only place the helper asks macOS for a grant. The app sends this
    /// request only from the user's Allow click in Settings.
    private func requestPermission(_ kind: ComputerUsePermissionKind) {
        switch kind {
        case .accessibility:
            guard !AXIsProcessTrusted() else { return }
            let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
            if !AXIsProcessTrustedWithOptions(options) {
                openPrivacyPane("Privacy_Accessibility")
            }
        case .screenRecording:
            guard !CGPreflightScreenCaptureAccess() else { return }
            if !CGRequestScreenCaptureAccess() {
                openPrivacyPane("Privacy_ScreenCapture")
            }
        }
    }

    private func openPrivacyPane(_ anchor: String) {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)") {
            NSWorkspace.shared.open(url)
        }
    }

    private func perform(_ command: ComputerUseCommand) async throws -> ComputerUseResultValue {
        if case .listApps = command { return .apps(runningApps()) }
        guard let bundleID = command.bundleID else { throw HelperError(.invalidInput, "Missing app.") }

        if case .openApp = command {
            try await openApp(bundleID)
            return .acknowledged("Opened \(bundleID).")
        }
        let app = try runningApp(bundleID)

        switch command {
        case .snapshot:
            try requireAccessibility()
            return .snapshot(try snapshotter.snapshot(app: app))
        case .screenshot:
            guard CGPreflightScreenCaptureAccess() else {
                throw HelperError(.permissionMissing, "Screen Recording is not granted to TurboSpark Computer Use.")
            }
            return .screenshot(try await capturer.capture(pid: app.processIdentifier))
        default:
            break
        }

        // Everything below posts input.
        guard !stopped else {
            throw HelperError(.stoppedByUser, "The user stopped agent control. Ask them before continuing.")
        }
        try requireAccessibility()
        try await input.bringToFront(app)
        lastInputAt = Date()

        switch command {
        case .click(_, let target, let button, let count):
            if case .reference(let ref) = target, button == .left, count == 1,
               let element = snapshotter.element(bundleID: bundleID, reference: ref),
               AXUIElementPerformAction(element, kAXPressAction as CFString) == .success
            {
                return .acknowledged("Pressed \(ref).")
            }
            let point = try resolvePoint(target, bundleID: bundleID)
            try input.click(at: point, pid: app.processIdentifier, button: button, count: count)
            return .acknowledged("Clicked.")
        case .type(_, let text):
            try input.type(text, app: app)
            return .acknowledged("Typed \(text.count) characters.")
        case .key(_, let combo):
            guard let parsed = ComputerUseKeyCombo.parse(combo) else {
                throw HelperError(.invalidInput, "Unknown key \(combo).")
            }
            try input.press(parsed, app: app)
            return .acknowledged("Pressed \(combo).")
        case .scroll(_, let target, let direction, let amount):
            let point = try target.map { try resolvePoint($0, bundleID: bundleID) }
                ?? input.frontWindowCenter(pid: app.processIdentifier)
            try input.scroll(at: point, pid: app.processIdentifier, direction: direction, lines: amount)
            return .acknowledged("Scrolled \(direction.rawValue).")
        default:
            throw HelperError(.invalidInput, "Unsupported command.")
        }
    }

    private func resolvePoint(_ target: ComputerUseTarget, bundleID: String) throws -> CGPoint {
        switch target {
        case .point(let x, let y):
            return CGPoint(x: x, y: y)
        case .reference(let ref):
            guard let element = snapshotter.element(bundleID: bundleID, reference: ref),
                  let frame = AXSnapshotter.frame(of: element)
            else {
                throw HelperError(.staleReference, "Reference \(ref) is stale. Take a new computer_snapshot.")
            }
            return CGPoint(x: frame.midX, y: frame.midY)
        }
    }

    private func requireAccessibility() throws {
        guard AXIsProcessTrusted() else {
            throw HelperError(.permissionMissing, "Accessibility is not granted to TurboSpark Computer Use.")
        }
    }

    private func runningApps() -> [ComputerUseRunningApp] {
        let front = NSWorkspace.shared.frontmostApplication?.processIdentifier
        return NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular }
            .compactMap { app in
                guard let id = app.bundleIdentifier else { return nil }
                return ComputerUseRunningApp(
                    bundleID: id,
                    name: app.localizedName ?? id,
                    isFrontmost: app.processIdentifier == front)
            }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private func runningApp(_ bundleID: String) throws -> NSRunningApplication {
        guard let app = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)
            .first(where: { !$0.isTerminated })
        else {
            throw HelperError(.appNotRunning, "\(bundleID) is not running. Use computer_open_app.")
        }
        return app
    }

    private func openApp(_ bundleID: String) async throws {
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) else {
            throw HelperError(.appNotRunning, "\(bundleID) is not installed.")
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        _ = try await NSWorkspace.shared.openApplication(at: url, configuration: configuration)
    }
}

struct HelperError: Error {
    let kind: ComputerUseErrorKind
    let message: String

    init(_ kind: ComputerUseErrorKind, _ message: String) {
        self.kind = kind
        self.message = message
    }
}
