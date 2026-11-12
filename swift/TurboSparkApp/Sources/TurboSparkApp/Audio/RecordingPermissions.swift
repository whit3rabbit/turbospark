import AppKit
import AVFoundation
import CoreGraphics
import SwiftUI

/// A macOS privacy grant a recording source depends on. The microphone uses
/// AVFoundation's grant; app and system audio come through ScreenCaptureKit,
/// which macOS gates behind the screen recording grant even when only audio
/// is captured.
enum RecordingPermission: CaseIterable, Identifiable, Hashable, Sendable {
    case microphone
    case systemAudio

    var id: Self { self }

    var symbol: String {
        switch self {
        case .microphone: return "mic.fill"
        case .systemAudio: return "speaker.wave.2.fill"
        }
    }

    var title: Text {
        switch self {
        case .microphone: return Text("Microphone", bundle: .module)
        // Same label as the Record panel's source toggle.
        case .systemAudio: return Text("App or system audio", bundle: .module)
        }
    }

    var detail: Text {
        switch self {
        case .microphone:
            return Text("Records your voice from the selected input.", bundle: .module)
        case .systemAudio:
            return Text("Records audio from meeting apps or the whole system. macOS lists this under screen recording.", bundle: .module)
        }
    }

    var privacyPane: MacPrivacyPane {
        switch self {
        case .microphone: return .microphone
        case .systemAudio: return .screenRecording
        }
    }

    /// The grants the current source selection needs, in request order.
    static func needed(microphone: Bool, systemAudio: Bool) -> [RecordingPermission] {
        (microphone ? [.microphone] : []) + (systemAudio ? [.systemAudio] : [])
    }
}

enum RecordingPermissionState: Equatable, Sendable {
    /// macOS has not asked yet. Requesting shows the system prompt.
    case notDetermined
    case granted
    /// Turned off. Only System Settings can change it, macOS will not prompt again.
    case denied
    /// Blocked by a configuration profile or Screen Time. The user cannot change it here.
    case restricted
}

/// The system calls behind `RecordingPermissions`, replaceable in tests so a
/// test run never touches the developer's real TCC grants.
struct RecordingPermissionDependencies {
    var microphoneStatus: () -> AVAuthorizationStatus = { AVCaptureDevice.authorizationStatus(for: .audio) }
    var requestMicrophone: () async -> Bool = { await AVCaptureDevice.requestAccess(for: .audio) }
    var systemAudioGranted: () -> Bool = { CGPreflightScreenCaptureAccess() }
    var requestSystemAudio: () -> Bool = { CGRequestScreenCaptureAccess() }
    var openSettings: (MacPrivacyPane) -> Void = { pane in
        if let url = pane.url { NSWorkspace.shared.open(url) }
    }
}

/// Reads and requests the recording grants. Reading never prompts, so the
/// Record panel can check status at any time; only an explicit user action
/// calls a request.
@MainActor
final class RecordingPermissions: ObservableObject {
    /// One instance for the app. TCC state is per process, and the record of
    /// a screen capture request this launch must outlive the view that made
    /// it, or a second Allow click would request again and show nothing.
    static let shared = RecordingPermissions()

    @Published private(set) var states: [RecordingPermission: RecordingPermissionState] = [:]
    @Published private(set) var isRequesting = false
    private let dependencies: RecordingPermissionDependencies
    /// Screen capture has no "not determined" status: preflight only says
    /// granted or not. macOS shows its prompt once per app, so after one
    /// request this launch a refusal is treated as denied and the next step is
    /// System Settings rather than a request that silently does nothing.
    private var requestedSystemAudio = false

    init(dependencies: RecordingPermissionDependencies = RecordingPermissionDependencies()) {
        self.dependencies = dependencies
        refresh()
    }

    func state(_ permission: RecordingPermission) -> RecordingPermissionState {
        states[permission] ?? .notDetermined
    }

    func missing(_ needed: [RecordingPermission]) -> [RecordingPermission] {
        needed.filter { state($0) != .granted }
    }

    /// Declining one grant is a supported path, not a dead end: when some
    /// needed grants are in and others are not, a recording can still go
    /// ahead with the allowed sources. Nil when there is nothing to fall
    /// back to (all allowed, or none).
    func fallback(_ needed: [RecordingPermission]) -> (allowed: [RecordingPermission], declined: [RecordingPermission])? {
        let declined = missing(needed)
        let allowed = needed.filter { state($0) == .granted }
        guard !declined.isEmpty, !allowed.isEmpty else { return nil }
        return (allowed, declined)
    }

    /// Called on appear and whenever the app becomes active, because the user
    /// changes a denied grant in System Settings and then switches back.
    func refresh() {
        let microphone: RecordingPermissionState
        switch dependencies.microphoneStatus() {
        case .authorized: microphone = .granted
        case .denied: microphone = .denied
        case .restricted: microphone = .restricted
        case .notDetermined: microphone = .notDetermined
        @unknown default: microphone = .notDetermined
        }
        let systemAudio: RecordingPermissionState = dependencies.systemAudioGranted()
            ? .granted : (requestedSystemAudio ? .denied : .notDetermined)
        states = [.microphone: microphone, .systemAudio: systemAudio]
    }

    /// The one next step for a grant: prompt when macOS still can, open the
    /// System Settings pane when it cannot, nothing when it is out of reach.
    func request(_ permission: RecordingPermission) async {
        guard !isRequesting else { return }
        switch state(permission) {
        case .granted, .restricted:
            return
        case .denied:
            dependencies.openSettings(permission.privacyPane)
            return
        case .notDetermined:
            isRequesting = true
            defer { isRequesting = false; refresh() }
            switch permission {
            case .microphone:
                _ = await dependencies.requestMicrophone()
            case .systemAudio:
                requestedSystemAudio = true
                _ = dependencies.requestSystemAudio()
            }
        }
    }

    /// Requests each missing grant in order and stops at the first one that
    /// needs System Settings, so one click never opens Settings twice.
    func requestAll(_ needed: [RecordingPermission]) async {
        for permission in missing(needed) {
            let before = state(permission)
            await request(permission)
            if before != .notDetermined || state(permission) != .granted { return }
        }
    }
}
