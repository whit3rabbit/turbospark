import AVFoundation
import Foundation

/// Hands capture packets to live transcription without ever blocking the
/// capture path. Capture calls `offer` under its own lock on the audio
/// thread, so `offer` does O(1) work under a short lock and drops a packet
/// when the reader has fallen behind. Saved audio is unaffected: capture
/// persists every packet separately, and the final pass reads that.
final class AudioLiveFeed: @unchecked Sendable {
    struct Packet: Sendable {
        let samples: [Float]
        let sampleRate: Double
        let channels: Int
        /// Seconds on the recording timeline, pauses removed.
        let offset: Double
        let source: String
    }

    private let lock = NSLock()
    private var packets: [Packet] = []
    private var bufferedSeconds: [String: Double] = [:]
    private var droppedSinceDrain = 0
    let limitSeconds: Double

    init(limitSeconds: Double = 10) { self.limitSeconds = limitSeconds }

    /// Never waits. Returns false when the packet was dropped.
    @discardableResult
    func offer(_ samples: [Float], sampleRate: Double, channels: Int, offset: Double, source: String) -> Bool {
        guard sampleRate > 0, channels > 0, !samples.isEmpty else { return false }
        let seconds = Double(samples.count / channels) / sampleRate
        lock.lock(); defer { lock.unlock() }
        let buffered = bufferedSeconds[source, default: 0]
        guard buffered + seconds <= limitSeconds else { droppedSinceDrain += 1; return false }
        bufferedSeconds[source] = buffered + seconds
        packets.append(Packet(samples: samples, sampleRate: sampleRate, channels: channels, offset: offset, source: source))
        return true
    }

    /// Everything offered since the last drain, in arrival order, and how
    /// many packets were dropped meanwhile.
    func drain() -> (packets: [Packet], dropped: Int) {
        lock.lock(); defer { lock.unlock() }
        let out = (packets, droppedSinceDrain)
        packets = []; bufferedSeconds = [:]; droppedSinceDrain = 0
        return out
    }
}

/// Converts one source's interleaved packets to 16 kHz mono.
protocol AudioLiveResampling {
    mutating func mono16k(_ interleaved: [Float], sampleRate: Double, channels: Int) -> [Float]
}

/// `AVAudioConverter` kept across packets, so the filter state carries over
/// packet edges instead of clicking at each one.
struct AudioLiveConverter: AudioLiveResampling {
    private var converter: AVAudioConverter?
    private var inputRate: Double = 0
    private let output = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!

    mutating func mono16k(_ interleaved: [Float], sampleRate: Double, channels: Int) -> [Float] {
        let frames = interleaved.count / max(1, channels)
        guard frames > 0 else { return [] }
        // Average channels first: speech is the same in each, and mono is
        // what the model takes.
        var mono = [Float](repeating: 0, count: frames)
        for frame in 0..<frames {
            var sum: Float = 0
            for channel in 0..<channels { sum += interleaved[frame * channels + channel] }
            mono[frame] = sum / Float(channels)
        }
        if sampleRate == 16_000 { return mono }
        if converter == nil || inputRate != sampleRate {
            guard let input = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false) else { return [] }
            converter = AVAudioConverter(from: input, to: output)
            inputRate = sampleRate
        }
        guard let converter,
              let source = AVAudioPCMBuffer(pcmFormat: converter.inputFormat, frameCapacity: AVAudioFrameCount(frames)) else { return [] }
        source.frameLength = AVAudioFrameCount(frames)
        mono.withUnsafeBufferPointer { source.floatChannelData![0].update(from: $0.baseAddress!, count: frames) }
        let capacity = AVAudioFrameCount(Double(frames) * 16_000 / sampleRate) + 64
        guard let target = AVAudioPCMBuffer(pcmFormat: output, frameCapacity: capacity) else { return [] }
        var fed = false
        var error: NSError?
        converter.convert(to: target, error: &error) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return source
        }
        guard error == nil, let data = target.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(target.frameLength)))
    }
}

/// Places each source's 16 kHz mono audio on the shared recording timeline
/// and sums the sources, as `AudioMediaIO.streamMix` does for saved audio,
/// so live text hears what the final pass will. Output is contiguous from
/// time zero: a gap in a source (a dropped packet) is silence.
struct AudioLiveMixer<Resampler: AudioLiveResampling> {
    static var sampleRate: Int { 16_000 }
    /// A source more than this far behind the newest is treated as silent,
    /// so one that stalls (ScreenCaptureKit delivers in bursts) cannot hold
    /// live text back.
    let maxLagSamples: Int
    /// Arrival jitter inside which a packet counts as contiguous.
    let toleranceSamples: Int
    private let makeResampler: () -> Resampler
    private var sources: [String: SourceState] = [:]
    private(set) var emitted = 0

    private struct SourceState {
        var resampler: Resampler
        var start: Int?
        var pending: [Float] = []
        var end: Int { (start ?? 0) + pending.count }
    }

    init(maxLagSeconds: Double = 0.5, toleranceSeconds: Double = 0.05, makeResampler: @escaping () -> Resampler) {
        maxLagSamples = Int(maxLagSeconds * 16_000)
        toleranceSamples = Int(toleranceSeconds * 16_000)
        self.makeResampler = makeResampler
    }

    mutating func add(_ packet: AudioLiveFeed.Packet) {
        var state = sources[packet.source] ?? SourceState(resampler: makeResampler())
        var mono = state.resampler.mono16k(packet.samples, sampleRate: packet.sampleRate, channels: packet.channels)
        guard !mono.isEmpty, packet.offset.isFinite, packet.offset >= 0 else { sources[packet.source] = state; return }
        var at = Int((packet.offset * 16_000).rounded())
        if let start = state.start {
            let expected = start + state.pending.count
            if abs(at - expected) <= toleranceSamples {
                at = expected
            } else if at < expected {
                // Overlaps audio this source already gave: keep only the new part.
                mono.removeFirst(min(mono.count, expected - at)); at = expected
            }
        }
        // Audio behind the output position (this source was skipped as
        // stalled) is gone; drop it before placing the rest.
        if at < emitted {
            let late = min(mono.count, emitted - at)
            mono.removeFirst(late); at += late
        }
        if let start = state.start, !state.pending.isEmpty {
            let expected = start + state.pending.count
            if at > expected { state.pending.append(contentsOf: repeatElement(0, count: at - expected)) }
        } else {
            // Nothing buffered: start where this packet lands; the mixer
            // treats the time before it as silence from this source.
            state.start = at; state.pending = []
        }
        state.pending.append(contentsOf: mono)
        sources[packet.source] = state
    }

    /// Mixed audio up to what every active source has delivered. `flush`
    /// emits everything buffered, for the end of a recording.
    mutating func drain(flush: Bool = false) -> [Float] {
        let ends = sources.values.filter { $0.start != nil }.map(\.end)
        guard let newest = ends.max() else { return [] }
        let until = flush ? newest : (ends.filter { $0 >= newest - maxLagSamples }.min() ?? newest)
        guard until > emitted else { return [] }
        var out = [Float](repeating: 0, count: until - emitted)
        for key in Array(sources.keys) {
            guard var state = sources[key], let start = state.start else { continue }
            let from = max(start, emitted), to = min(state.end, until)
            if to > from {
                for i in from..<to { out[i - emitted] += state.pending[i - start] }
            }
            // Drop what is now behind the output position.
            let consumed = min(state.pending.count, max(0, until - start))
            state.pending.removeFirst(consumed)
            state.start = start + consumed
            sources[key] = state
        }
        emitted = until
        return out
    }
}
