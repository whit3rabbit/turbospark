import AppKit
import AVFoundation
import AudioToolbox
import CoreAudio
import CoreMedia
import Foundation
import ScreenCaptureKit

struct AudioCaptureTimeline {
    let start: Double
    private var pausedAt: Double?
    private var pausedDuration: Double = 0
    init(start: Double) { self.start = start }
    mutating func pause(at time: Double) { if pausedAt == nil { pausedAt = time } }
    mutating func resume(at time: Double) {
        if let pausedAt { pausedDuration += max(0, time - pausedAt); self.pausedAt = nil }
    }
    func elapsed(at time: Double) -> Double { max(0, (pausedAt ?? time) - start - pausedDuration) }
    func offset(at time: Double) -> Double? {
        guard pausedAt == nil, time.isFinite, time >= start else { return nil }
        return max(0, time - start - pausedDuration)
    }
}

final class AudioCaptureBudget: @unchecked Sendable {
    let limit: Int
    private let lock = NSLock()
    private var bytes = 0
    init(limit: Int) { self.limit = limit }
    var pendingBytes: Int { lock.lock(); defer { lock.unlock() }; return bytes }
    func reserve(_ count: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard count >= 0, count <= limit - bytes else { return false }
        bytes += count
        return true
    }
    func release(_ count: Int) { lock.lock(); bytes = max(0, bytes - max(0, count)); lock.unlock() }
}

/// Capture owns no UI or model state. The sink must commit each chunk synchronously
/// and must never wait for MainActor, so profile lock can safely drain it.
final class AudioCapture: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    struct Microphone: Identifiable, Sendable { var id: UInt32; var name: String }
    struct Application: Identifiable, Sendable { var id: Int32; var name: String }
    enum SystemSource: Equatable, Sendable { case none, all, application(Int32) }
    struct Configuration: Sendable {
        var microphoneID: UInt32? = nil
        var capturesMicrophone = true
        var systemSource: SystemSource = .none
    }
    enum Event: Sendable {
        case level(source: String, value: Float)
        case failure(String)
        case interruption(String)
    }
    enum CaptureError: LocalizedError {
        case permission, unavailable, noSource, alreadyRecording, invalidFormat
        var errorDescription: String? {
            switch self {
            case .permission: return String(localized: "Microphone access is off. Allow TurboSpark in System Settings > Privacy & Security > Microphone, then try again.", bundle: .module)
            case .unavailable: return String(localized: "The selected audio source is unavailable. Select an available microphone or application.", bundle: .module)
            case .noSource: return String(localized: "Select a microphone or system audio source before recording.", bundle: .module)
            case .alreadyRecording: return String(localized: "A recording is already active. Stop it before starting another.", bundle: .module)
            case .invalidFormat: return String(localized: "The audio source has an unsupported format. Choose another input device.", bundle: .module)
            }
        }
    }

    private let lock = NSLock()
    private let persistence = DispatchQueue(label: "app.turbospark.audio.capture.persistence", qos: .userInitiated)
    private let events = DispatchQueue(label: "app.turbospark.audio.capture.events")
    private let systemQueue = DispatchQueue(label: "app.turbospark.audio.capture.system", qos: .userInitiated)
    private let budget = AudioCaptureBudget(limit: 8 * 1024 * 1024)
    private let onChunk: @Sendable (AudioCaptureChunk) throws -> Void
    private let onEvent: @Sendable (Event) -> Void
    private var accepting = false
    private var starting = false
    private var generation: UInt64 = 0
    private var timeline: AudioCaptureTimeline?
    private var stoppedElapsed: Double = 0
    private var engine: AVAudioEngine?
    private var stream: SCStream?
    private var configurationObserver: NSObjectProtocol?
    private var applicationObserver: NSObjectProtocol?
    private var terminalFailure: String?
    // Accessed only on the persistence queue.
    private var accumulators: [String: Chunker] = [:]
    private var writeFailed = false

    init(onChunk: @escaping @Sendable (AudioCaptureChunk) throws -> Void, onEvent: @escaping @Sendable (Event) -> Void) {
        self.onChunk = onChunk
        self.onEvent = onEvent
    }

    static func requestMicrophonePermission() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }

    static func applications() async throws -> [Application] {
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: false)
        return content.applications.filter { $0.processID != ProcessInfo.processInfo.processIdentifier }.map { Application(id: $0.processID, name: $0.applicationName) }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    static func microphones() throws -> [Microphone] {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else { throw CaptureError.unavailable }
        var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else { throw CaptureError.unavailable }
        return ids.compactMap { id in
            var streams = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams, mScope: kAudioDevicePropertyScopeInput, mElement: kAudioObjectPropertyElementMain)
            var bytes: UInt32 = 0
            guard AudioObjectGetPropertyDataSize(id, &streams, 0, nil, &bytes) == noErr, bytes > 0 else { return nil }
            var property = AudioObjectPropertyAddress(mSelector: kAudioObjectPropertyName, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
            var name: Unmanaged<CFString>?
            bytes = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            guard AudioObjectGetPropertyData(id, &property, 0, nil, &bytes, &name) == noErr, let name else { return nil }
            return Microphone(id: id, name: name.takeUnretainedValue() as String)
        }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    func start(configuration: Configuration) async throws {
        guard let generation = try reserveStart() else { throw CaptureError.alreadyRecording }
        var newStream: SCStream?
        do {
            guard configuration.capturesMicrophone || configuration.systemSource != .none else { throw CaptureError.noSource }
            if configuration.systemSource != .none {
                let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: false)
                guard let display = content.displays.first else { throw CaptureError.unavailable }
                let filter: SCContentFilter
                switch configuration.systemSource {
                case .none: throw CaptureError.noSource
                case .all:
                    filter = SCContentFilter(display: display, excludingApplications: content.applications.filter { $0.processID == ProcessInfo.processInfo.processIdentifier }, exceptingWindows: [])
                case .application(let pid):
                    guard pid != ProcessInfo.processInfo.processIdentifier, let application = content.applications.first(where: { $0.processID == pid }) else { throw CaptureError.unavailable }
                    filter = SCContentFilter(display: display, including: [application], exceptingWindows: [])
                }
                let config = SCStreamConfiguration()
                config.capturesAudio = true
                config.excludesCurrentProcessAudio = true
                config.sampleRate = 48_000
                config.channelCount = 2
                config.width = 2
                config.height = 2
                config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
                config.showsCursor = false
                let stream = SCStream(filter: filter, configuration: config, delegate: self)
                try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: systemQueue)
                newStream = stream
            }
            let newEngine = configuration.capturesMicrophone ? try makeMicrophoneEngine(device: configuration.microphoneID) : nil
            persistence.sync { accumulators.removeAll(); writeFailed = false }
            guard activate(engine: newEngine, stream: newStream, generation: generation, source: configuration.systemSource) else {
                newEngine?.inputNode.removeTap(onBus: 0)
                newEngine?.stop()
                throw CancellationError()
            }
            try newEngine?.start()
            if let newStream { try await newStream.startCapture() }
            guard isAccepting else { throw CancellationError() }
        } catch {
            _ = stopAcceptingAndDrain(expectedGeneration: generation)
            if let newStream {
                try? await newStream.stopCapture()
                clearStream(newStream)
            }
            throw error
        }
    }

    var failureMessage: String? { lock.lock(); defer { lock.unlock() }; return terminalFailure }

    func currentElapsed() -> Double {
        lock.lock(); defer { lock.unlock() }
        return timeline?.elapsed(at: Self.now) ?? stoppedElapsed
    }

    func pause() {
        lock.lock()
        if accepting { timeline?.pause(at: Self.now) }
        lock.unlock()
        persistence.async { [self] in flushAll() }
    }

    func resume() {
        lock.lock(); if accepting { timeline?.resume(at: Self.now) }; lock.unlock()
    }

    /// This is the profile lifecycle barrier. No sink call is possible after it
    /// returns. ScreenCaptureKit's asynchronous shutdown can finish afterward.
    @discardableResult func stopAcceptingAndDrain() -> Double {
        stopAcceptingAndDrain(expectedGeneration: nil) ?? 0
    }

    private func stopAcceptingAndDrain(expectedGeneration: UInt64?) -> Double? {
        lock.lock()
        if let expectedGeneration, generation != expectedGeneration { lock.unlock(); return nil }
        accepting = false
        starting = false
        generation &+= 1
        stoppedElapsed = timeline?.elapsed(at: Self.now) ?? stoppedElapsed
        timeline = nil
        let engine = self.engine
        self.engine = nil
        let observer = configurationObserver
        configurationObserver = nil
        let applicationObserver = self.applicationObserver
        self.applicationObserver = nil
        lock.unlock()
        if let observer { NotificationCenter.default.removeObserver(observer) }
        if let applicationObserver { NSWorkspace.shared.notificationCenter.removeObserver(applicationObserver) }
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        persistence.sync { flushAll() }
        return stoppedElapsed
    }

    @discardableResult func stop() async -> Double {
        let duration = stopAcceptingAndDrain()
        if let stream = takeStream() { try? await stream.stopCapture() }
        return duration
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fail(error.localizedDescription, interruption: true)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid, CMSampleBufferDataIsReady(sampleBuffer),
              let description = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description),
              let format = AVAudioFormat(streamDescription: asbd) else { return }
        let frames = CMSampleBufferGetNumSamples(sampleBuffer)
        guard format.sampleRate.isFinite, (8_000...192_000).contains(format.sampleRate),
              frames > 0, frames <= Int(format.sampleRate * 5), (1...8).contains(format.channelCount),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else { fail(String(localized: "System audio returned an invalid buffer. Stop and select the source again.", bundle: .module)); return }
        buffer.frameLength = AVAudioFrameCount(frames)
        let result = CMSampleBufferCopyPCMDataIntoAudioBufferList(sampleBuffer, at: 0, frameCount: Int32(frames), into: buffer.mutableAudioBufferList)
        guard result == noErr else { fail(String(localized: "System audio could not be read. Stop and select the source again.", bundle: .module)); return }
        let hostTime = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        enqueue(buffer, hostTime: hostTime, source: "system")
    }

    private static var now: Double { AVAudioTime.seconds(forHostTime: mach_absolute_time()) }
    private var isAccepting: Bool { lock.lock(); defer { lock.unlock() }; return accepting }
    private func reserveStart() throws -> UInt64? {
        lock.lock(); defer { lock.unlock() }
        // A cancelled starter must not mint a fresh generation after teardown.
        guard !Task.isCancelled else { throw CancellationError() }
        guard !starting, !accepting, stream == nil else { return nil }
        starting = true
        generation &+= 1
        return generation
    }
    private func activate(engine: AVAudioEngine?, stream: SCStream?, generation: UInt64, source: SystemSource) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard starting, self.generation == generation else { return false }
        self.engine = engine
        self.stream = stream
        timeline = AudioCaptureTimeline(start: Self.now)
        stoppedElapsed = 0
        terminalFailure = nil
        accepting = true
        starting = false
        if let engine {
            configurationObserver = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
                self?.fail(String(localized: "The microphone configuration changed. Saved audio is safe. Select the input again and start a new recording.", bundle: .module), interruption: true)
            }
        }
        if case .application(let pid) = source {
            applicationObserver = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didTerminateApplicationNotification, object: nil, queue: nil) { [weak self] notification in
                guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication,
                      app.processIdentifier == pid else { return }
                self?.fail(String(localized: "The recorded application closed. Saved audio is safe. Select an available source and start a new recording.", bundle: .module), interruption: true)
            }
        }
        return true
    }
    private func clearStream(_ candidate: SCStream) {
        lock.lock(); defer { lock.unlock() }
        if stream === candidate { stream = nil }
    }
    private func takeStream() -> SCStream? { lock.lock(); defer { lock.unlock() }; let result = stream; stream = nil; return result }

    private func makeMicrophoneEngine(device: UInt32?) throws -> AVAudioEngine {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { throw CaptureError.permission }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        if var device {
            guard let unit = input.audioUnit,
                  AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &device, UInt32(MemoryLayout<UInt32>.size)) == noErr else { throw CaptureError.unavailable }
        }
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0, format.channelCount <= 8,
              format.commonFormat == .pcmFormatFloat32 else { throw CaptureError.invalidFormat }
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, time in
            guard time.isHostTimeValid else { self?.fail(String(localized: "Microphone timestamps are unavailable. Choose another input device.", bundle: .module)); return }
            self?.enqueue(buffer, hostTime: AVAudioTime.seconds(forHostTime: time.hostTime), source: "microphone")
        }
        engine.prepare()
        return engine
    }

    private func enqueue(_ buffer: AVAudioPCMBuffer, hostTime: Double, source: String) {
        lock.lock()
        guard accepting, let offset = timeline?.offset(at: hostTime) else { lock.unlock(); return }
        let bytes = Int(buffer.frameLength) * Int(buffer.format.channelCount) * MemoryLayout<Float>.size
        guard budget.reserve(bytes) else {
            lock.unlock()
            fail(String(localized: "Recording stopped because storage could not keep up. Saved chunks are safe. Free disk space or choose a faster storage volume.", bundle: .module))
            return
        }
        // Queue insertion and acceptance shutdown share this lock. Otherwise a
        // callback could enqueue plaintext after the profile drain returned.
        do {
            let samples = try AudioMediaIO.interleavedSamples(buffer)
            let rate = buffer.format.sampleRate
            let channels = Int(buffer.format.channelCount)
            persistence.async { [self] in
                defer { budget.release(bytes) }
                guard !writeFailed else { return }
                do {
                    let accumulator = accumulators[source] ?? Chunker(source: source)
                    accumulators[source] = accumulator
                    try accumulator.append(samples, sampleRate: rate, channels: channels, offset: offset, emit: onChunk)
                    let level = samples.isEmpty ? 0 : sqrt(samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(samples.count))
                    emit(.level(source: source, value: min(1, level)))
                } catch { writeFailed = true; fail(error.localizedDescription, reportWhenStopped: true) }
            }
            lock.unlock()
        } catch { budget.release(bytes); lock.unlock(); fail(error.localizedDescription) }
    }

    private func emit(_ event: Event) {
        let localized: Event
        switch event {
        case .level: localized = event
        case .failure(let message): localized = .failure(Bundle.module.localizedString(forKey: message, value: message, table: nil))
        case .interruption(let message): localized = .interruption(Bundle.module.localizedString(forKey: message, value: message, table: nil))
        }
        events.async { [onEvent] in onEvent(localized) }
    }
    private func fail(_ message: String, interruption: Bool = false, reportWhenStopped: Bool = false) {
        lock.lock()
        let wasAccepting = accepting
        if wasAccepting || reportWhenStopped { terminalFailure = Bundle.module.localizedString(forKey: message, value: message, table: nil) }
        accepting = false
        timeline?.pause(at: Self.now)
        lock.unlock()
        if wasAccepting || reportWhenStopped { emit(interruption ? .interruption(message) : .failure(message)) }
    }
    private func flushAll() {
        guard !writeFailed else { return }
        do { for accumulator in accumulators.values { try accumulator.flush(emit: onChunk) } }
        catch { writeFailed = true; fail(error.localizedDescription, reportWhenStopped: true) }
    }

    final class Chunker {
        let source: String
        var samples: [Float] = []
        var sampleRate: Double = 0
        var channels = 0
        var offset: Double = 0
        var lastEnd: Double = 0
        init(source: String) { self.source = source }
        func append(_ packet: [Float], sampleRate: Double, channels: Int, offset: Double, emit: (AudioCaptureChunk) throws -> Void) throws {
            guard sampleRate.isFinite, (8_000...192_000).contains(sampleRate), channels > 0, channels <= 8,
                  packet.count % channels == 0, packet.allSatisfy(\.isFinite) else { throw CaptureError.invalidFormat }
            let expected = self.offset + (self.sampleRate > 0 ? Double(samples.count / max(1, self.channels)) / self.sampleRate : 0)
            if self.sampleRate != sampleRate || self.channels != channels || (!samples.isEmpty && abs(offset - expected) > 0.05) { try flush(emit: emit) }
            self.sampleRate = sampleRate
            self.channels = channels
            var cursor = 0
            if samples.isEmpty {
                // Late packets after a clock discontinuity cannot overlap audio
                // that has already been committed for this source.
                let overlap = max(0, lastEnd - offset)
                cursor = min(packet.count, Int((overlap * sampleRate).rounded(.up)) * channels)
                self.offset = max(offset, lastEnd)
            }
            let capacity = Int(sampleRate * 5) * channels
            while cursor < packet.count {
                let amount = min(capacity - samples.count, packet.count - cursor)
                samples.append(contentsOf: packet[cursor..<(cursor + amount)])
                cursor += amount
                if samples.count == capacity { try flush(emit: emit) }
            }
        }
        func flush(emit: (AudioCaptureChunk) throws -> Void) throws {
            guard !samples.isEmpty else { return }
            let count = Int64(samples.count / channels)
            let chunk = AudioCaptureChunk(wav: try AudioMediaIO.wav(samples: samples, sampleRate: sampleRate, channels: channels), sampleRate: sampleRate, channels: channels, frameCount: count, offset: offset, source: source)
            try emit(chunk)
            lastEnd = offset + Double(count) / sampleRate
            offset = lastEnd
            samples.removeAll(keepingCapacity: true)
        }
    }
}
