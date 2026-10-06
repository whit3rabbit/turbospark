import AppKit
import AVFoundation
import CoreAudio
import Foundation

/// A source the app-audio capture can record.
struct SystemAudioSource: Identifiable, Equatable, Sendable {
    /// `0` is "all system audio"; otherwise the app's process ID.
    let pid: pid_t
    let name: String
    let bundleIdentifier: String?
    /// Whether Core Audio currently lists the app as an audio client. An app
    /// that has never opened an audio stream has no process object to tap.
    let hasAudioProcess: Bool

    var id: pid_t { pid }
    var isSystemWide: Bool { pid == 0 }

    static func allSystemAudio(name: String) -> SystemAudioSource {
        SystemAudioSource(pid: 0, name: name, bundleIdentifier: nil, hasAudioProcess: true)
    }
}

/// Records another app's output (or everything but TurboSpark) through a
/// Core Audio process tap, macOS 14.2+.
///
/// Process taps rather than ScreenCaptureKit: ScreenCaptureKit would ask for
/// Screen Recording permission to capture sound, which is the wrong consent
/// for an audio-only feature. Rather than BlackHole or another loopback
/// driver: those need a kernel-adjacent install and are GPL. The tap is a
/// private aggregate device that exists only while recording.
///
/// The tap writes float WAV through the same `CaptureSink` as the
/// microphone, so the engine decodes both identically.
@MainActor
final class SystemAudioCaptureService: ObservableObject {
    static let shared = SystemAudioCaptureService()

    enum CaptureError: LocalizedError {
        case unsupportedOS
        case noAudioProcess(String)
        case coreAudio(String, OSStatus)

        var errorDescription: String? {
            switch self {
            case .unsupportedOS:
                return String(localized: "Requires macOS 14.2 or later.", bundle: .module)
            case .noAudioProcess(let name):
                return String(
                    localized: "\(name) is not playing audio. Start playback in it, then try again.",
                    bundle: .module)
            case .coreAudio(let step, let status):
                // Formatted first: an OSStatus interpolated into a localized
                // key would pick an integer specifier per platform width.
                let detail = "\(step), OSStatus \(status)"
                return String(localized: "App audio capture failed (\(detail)).", bundle: .module)
            }
        }
    }

    @Published private(set) var isRecording = false
    @Published private(set) var source: SystemAudioSource?
    @Published private(set) var elapsed: TimeInterval = 0
    @Published private(set) var levels: [Float] = []

    private var session: AnyObject?
    /// The model a running capture attaches to when it hits the length cap.
    private weak var model: AppModel?
    private var outputURL: URL?
    private var startedAt: Date?
    private var clock: Task<Void, Never>?

    /// Running apps with a regular activation policy, TurboSpark excluded,
    /// with "All system audio" first.
    func availableSources() -> [SystemAudioSource] {
        var sources = [SystemAudioSource.allSystemAudio(
            name: String(localized: "All system audio", bundle: .module))]
        guard #available(macOS 14.2, *) else { return sources }
        let ownPID = ProcessInfo.processInfo.processIdentifier
        let apps = NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular && $0.processIdentifier != ownPID }
            .sorted { ($0.localizedName ?? "") < ($1.localizedName ?? "") }
        for app in apps {
            sources.append(SystemAudioSource(
                pid: app.processIdentifier,
                name: app.localizedName ?? app.bundleIdentifier ?? "\(app.processIdentifier)",
                bundleIdentifier: app.bundleIdentifier,
                hasAudioProcess: ProcessTapSession.processObject(for: app.processIdentifier) != nil))
        }
        return sources
    }

    func start(source: SystemAudioSource, model: AppModel) throws {
        guard !isRecording else { return }
        guard #available(macOS 14.2, *) else { throw CaptureError.unsupportedOS }
        let url = AudioEngineBridge.temporaryURL(extension: "wav")
        let tap = try ProcessTapSession(source: source, outputURL: url) { [weak self] level in
            Task { @MainActor in self?.push(level: level) }
        }
        session = tap
        self.model = model
        outputURL = url
        self.source = source
        levels = Array(repeating: 0, count: AudioCaptureService.levelHistory)
        elapsed = 0
        startedAt = Date()
        isRecording = true
        startClock()
    }

    /// Stops and returns the finished WAV.
    @discardableResult
    func stop() -> URL? {
        guard isRecording else { return nil }
        tearDown()
        let url = outputURL
        outputURL = nil
        return url
    }

    func cancel() {
        guard isRecording else { return }
        tearDown()
        if let outputURL { try? FileManager.default.removeItem(at: outputURL) }
        outputURL = nil
    }

    /// Stops and attaches the recording to the selected chat's draft.
    func finish(into model: AppModel) {
        guard let url = stop() else { return }
        let chatID = model.selectedChatID
        Task {
            let outcome = await AttachmentImporter.importDocuments(
                [url], into: model, chatID: chatID)
            try? FileManager.default.removeItem(at: url)
            if let error = outcome.errorText {
                model.showToast(error, style: .error)
            }
        }
    }

    private func tearDown() {
        if #available(macOS 14.2, *) {
            (session as? ProcessTapSession)?.invalidate()
        }
        session = nil
        clock?.cancel()
        clock = nil
        isRecording = false
        source = nil
    }

    private func push(level: Float) {
        guard isRecording else { return }
        levels.append(level)
        if levels.count > AudioCaptureService.levelHistory {
            levels.removeFirst(levels.count - AudioCaptureService.levelHistory)
        }
    }

    private func startClock() {
        clock?.cancel()
        let limit = TimeInterval(AudioPreferences.maxClipSeconds)
        clock = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 200_000_000)
                guard let self, let startedAt = self.startedAt, self.isRecording else { return }
                self.elapsed = Date().timeIntervalSince(startedAt)
                if self.elapsed >= limit {
                    // Finish, not stop: a capture that reaches the cap is
                    // attached like any other, never silently dropped.
                    if let model = self.model {
                        self.finish(into: model)
                    } else {
                        self.cancel()
                    }
                    return
                }
            }
        }
    }
}

/// One live tap: the tap object, its private aggregate device, the IO proc
/// writing into a `CaptureSink`. `invalidate()` tears all three down in
/// reverse order; it is idempotent.
@available(macOS 14.2, *)
private final class ProcessTapSession {
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    /// Optional so every stored property has a value from the first line of
    /// `init`: a failure after the tap exists can then call `invalidate()`,
    /// which a partially initialized object could not, and leaked the tap
    /// and the aggregate device.
    private var sink: CaptureSink?
    private let queue = DispatchQueue(label: "com.turbospark.process-tap", qos: .userInitiated)

    init(
        source: SystemAudioSource, outputURL: URL,
        onLevel: @escaping @Sendable (Float) -> Void
    ) throws {
        let description: CATapDescription
        if source.isSystemWide {
            // Everything except TurboSpark itself, so read-aloud and clip
            // playback never record back into the capture.
            let own = Self.processObject(for: ProcessInfo.processInfo.processIdentifier)
            description = CATapDescription(
                stereoGlobalTapButExcludeProcesses: own.map { [NSNumber(value: $0)] } ?? [])
        } else {
            guard let process = Self.processObject(for: source.pid) else {
                throw SystemAudioCaptureService.CaptureError.noAudioProcess(source.name)
            }
            description = CATapDescription(stereoMixdownOfProcesses: [NSNumber(value: process)])
        }
        description.uuid = UUID()
        description.isPrivate = true
        description.muteBehavior = .unmuted
        description.name = "TurboSpark capture"

        var tap = AudioObjectID(kAudioObjectUnknown)
        var status = AudioHardwareCreateProcessTap(description, &tap)
        guard status == noErr else {
            throw SystemAudioCaptureService.CaptureError.coreAudio("create tap", status)
        }
        tapID = tap

        // The tap's own stream format: what the IO proc will hand us.
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var formatAddress = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        status = AudioObjectGetPropertyData(tap, &formatAddress, 0, nil, &size, &asbd)
        guard status == noErr, let format = AVAudioFormat(streamDescription: &asbd) else {
            let failure = status
            Self.destroyTap(tap)
            throw SystemAudioCaptureService.CaptureError.coreAudio("read tap format", failure)
        }

        // The default output device clocks the aggregate; the tap rides it
        // with drift compensation. Without a clock source the IO proc never
        // fires.
        guard let outputUID = Self.defaultOutputDeviceUID() else {
            Self.destroyTap(tap)
            throw SystemAudioCaptureService.CaptureError.coreAudio(
                "find output device", OSStatus(kAudioHardwareBadDeviceError))
        }
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "TurboSpark capture",
            kAudioAggregateDeviceUIDKey: "com.turbospark.capture.\(UUID().uuidString)",
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapDriftCompensationKey: true,
                    kAudioSubTapUIDKey: description.uuid.uuidString,
                ]
            ],
        ]
        var device = AudioObjectID(kAudioObjectUnknown)
        status = AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &device)
        guard status == noErr else {
            Self.destroyTap(tap)
            throw SystemAudioCaptureService.CaptureError.coreAudio("create aggregate device", status)
        }
        aggregateID = device

        let file: AVAudioFile
        do {
            file = try AVAudioFile(
                forWriting: outputURL, settings: AudioCaptureService.wavSettings(for: format),
                commonFormat: .pcmFormatFloat32, interleaved: format.isInterleaved)
        } catch {
            invalidate()
            throw error
        }
        let sink = CaptureSink(file: file, onLevel: onLevel)
        self.sink = sink

        var proc: AudioDeviceIOProcID?
        status = AudioDeviceCreateIOProcIDWithBlock(&proc, device, queue) { _, input, _, _, _ in
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: format, bufferListNoCopy: input, deallocator: nil)
            else { return }
            sink.consume(buffer)
        }
        guard status == noErr, let proc else {
            invalidate()
            throw SystemAudioCaptureService.CaptureError.coreAudio("create IO proc", status)
        }
        procID = proc
        status = AudioDeviceStart(device, proc)
        guard status == noErr else {
            invalidate()
            throw SystemAudioCaptureService.CaptureError.coreAudio("start device", status)
        }
    }

    deinit {
        invalidate()
    }

    func invalidate() {
        if aggregateID != kAudioObjectUnknown {
            if let procID {
                AudioDeviceStop(aggregateID, procID)
                AudioDeviceDestroyIOProcID(aggregateID, procID)
            }
            AudioHardwareDestroyAggregateDevice(aggregateID)
        }
        procID = nil
        aggregateID = AudioObjectID(kAudioObjectUnknown)
        if tapID != kAudioObjectUnknown {
            Self.destroyTap(tapID)
        }
        tapID = AudioObjectID(kAudioObjectUnknown)
        sink?.close()
    }

    private static func destroyTap(_ tap: AudioObjectID) {
        AudioHardwareDestroyProcessTap(tap)
    }

    /// UID of the current default output device.
    static func defaultOutputDeviceUID() -> String? {
        var address = AudioObjectPropertyAddress(
            // DefaultOutputDevice, not DefaultSystemOutputDevice: the
            // latter is the alert-sound device.
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var device = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &device) == noErr,
            device != kAudioObjectUnknown
        else { return nil }
        address.mSelector = kAudioDevicePropertyDeviceUID
        var uid: Unmanaged<CFString>?
        size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &uid) == noErr,
            let uid
        else { return nil }
        return uid.takeRetainedValue() as String
    }

    /// The Core Audio process object for `pid`, or nil when the process has
    /// never opened an audio stream.
    static func processObject(for pid: pid_t) -> AudioObjectID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var qualifier = pid
        var object = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address,
            UInt32(MemoryLayout<pid_t>.size), &qualifier, &size, &object)
        guard status == noErr, object != kAudioObjectUnknown else { return nil }
        return object
    }
}
