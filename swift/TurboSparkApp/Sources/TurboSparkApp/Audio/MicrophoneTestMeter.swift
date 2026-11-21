import AVFoundation
import AudioToolbox
import SwiftUI

/// A short-lived input tap that only measures level, so a user can confirm
/// the chosen microphone hears them before a take or a call.
///
/// Privacy: it CHECKS microphone access and never asks for it. Without a
/// grant it opens nothing and returns false; the macOS prompt belongs to
/// `RecordingPermissions` alone (swift/AGENTS.md). Samples are reduced to a
/// level on the audio thread and dropped: nothing is kept or written.
@MainActor
final class MicrophoneLevelProbe: ObservableObject {
    @Published private(set) var isRunning = false
    /// 0...1 on the decibel scale of `MicrophoneLevel.fraction`, with a short
    /// fall so speech reads as a steady bar rather than flicker.
    @Published private(set) var level: Double = 0
    @Published private(set) var failed = false

    private var engine: AVAudioEngine?
    private var configurationObserver: NSObjectProtocol?

    @discardableResult
    func start(deviceID: UInt32?) -> Bool {
        stop()
        failed = false
        guard AudioCapture.microphoneAccessAllowed() else { return false }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        if var device = deviceID {
            guard let unit = input.audioUnit,
                  AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                       &device, UInt32(MemoryLayout<UInt32>.size)) == noErr
            else { failed = true; return false }
        }
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0, format.commonFormat == .pcmFormatFloat32 else {
            failed = true
            return false
        }
        input.installTap(onBus: 0, bufferSize: 2048, format: format, block: Self.tap { [weak self] value in
            Task { @MainActor in self?.receive(value) }
        })
        engine.prepare()
        do { try engine.start() } catch {
            input.removeTap(onBus: 0)
            failed = true
            return false
        }
        self.engine = engine
        // An unplugged or reconfigured device ends the test instead of
        // leaving a dead tap behind.
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.stop() }
        }
        isRunning = true
        return true
    }

    func stop() {
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
        configurationObserver = nil
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        isRunning = false
        level = 0
    }

    private func receive(_ value: Double) {
        guard isRunning else { return }
        // About 23 buffers a second: fall 0.04 per buffer, rise at once.
        level = max(value, level - 0.04)
    }

    /// Built outside the main actor so the audio thread never hops to it
    /// just to compute a number.
    nonisolated private static func tap(_ deliver: @escaping @Sendable (Double) -> Void) -> AVAudioNodeTapBlock {
        { buffer, _ in
            guard let channels = buffer.floatChannelData, buffer.frameLength > 0 else { return }
            // Channel 0 is enough to show the device hears you.
            let samples = Array(UnsafeBufferPointer(start: channels[0], count: Int(buffer.frameLength)))
            deliver(MicrophoneLevel.fraction(rms: MicrophoneLevel.rms(samples)))
        }
    }
}

/// "Test microphone" with a level bar. The test is opt-in per press, runs
/// only while this view is on screen, and stops when it disappears, when the
/// device changes to one it cannot open, or when `blocked` (a call or a take
/// holds the microphone). Shown only when microphone access is already
/// granted; otherwise `accessHint` (if any) explains where access is asked.
@MainActor
struct MicrophoneTestMeter: View {
    /// nil tests the system default input.
    let deviceID: UInt32?
    var blocked = false
    var accessHint: Text?

    @StateObject private var probe = MicrophoneLevelProbe()
    @ObservedObject private var permissions = RecordingPermissions.shared
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @Environment(\.appTheme) private var theme
    @State private var shown: Double = 0

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    private var granted: Bool { permissions.state(.microphone) == .granted }
    private var hears: Bool { MicrophoneLevel.hearsInput(fraction: shown) }

    var body: some View {
        Group {
            if granted {
                content
            } else if let accessHint {
                accessHint
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear { permissions.refresh() }
        .onDisappear { probe.stop() }
        .onChange(of: deviceID) { _, device in
            if probe.isRunning { probe.start(deviceID: device) }
        }
        .onChange(of: blocked) { _, isBlocked in
            if isBlocked { probe.stop() }
        }
        .onChange(of: granted) { _, isGranted in
            if !isGranted { probe.stop() }
        }
        // Reduce Motion samples slowly so the bar steps instead of streaming.
        .onReceive(Timer.publish(every: reduceMotion ? 0.25 : 0.06, on: .main, in: .common).autoconnect()) { _ in
            let next = probe.isRunning ? probe.level : 0
            if next != shown { shown = next }
        }
    }

    private var content: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Button {
                    if probe.isRunning { probe.stop() } else { probe.start(deviceID: deviceID) }
                } label: {
                    Label {
                        probe.isRunning ? Text("Stop test", bundle: .module) : Text("Test microphone", bundle: .module)
                    } icon: {
                        Image(systemName: probe.isRunning ? "stop.fill" : "waveform")
                    }
                }
                .appAction(.secondary, size: .small)
                .disabled(blocked)
                if probe.isRunning {
                    // Words carry the state, so it never rests on the bar alone.
                    (hears ? Text("Picking you up", bundle: .module) : Text("Say something to test", bundle: .module))
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .accessibilityHidden(true)
                }
            }
            if probe.isRunning {
                AppProgressBar(
                    value: shown,
                    label: Text("Input level", bundle: .module),
                    valueText: hears ? Text("Picking you up", bundle: .module) : Text("Say something to test", bundle: .module),
                    size: .compact,
                    animated: false)
                    .frame(minWidth: 120, idealWidth: 220 * AppMetrics.textScale(theme), maxWidth: 320 * AppMetrics.textScale(theme))
                Text("Nothing is recorded while you test.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            if probe.failed {
                Label {
                    Text("The microphone could not be opened. Choose another input or try again.", bundle: .module)
                        .fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill").accessibilityHidden(true)
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
        }
    }
}
