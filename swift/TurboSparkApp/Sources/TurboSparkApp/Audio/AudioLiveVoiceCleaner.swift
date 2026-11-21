import Foundation
import TurboSpark

/// The calls voice cleaning needs from a live isolation stream, so tests can
/// drive the cleaner without the model.
protocol AudioLiveEnhancing: AnyObject, Sendable {
    func push(_ pcm: [Float]) throws -> AudioEnhancePush
    func flush() throws -> [Float]
    func close()
}
extension AudioEnhanceStream: AudioLiveEnhancing {}

/// "Reduce background noise" for live text: the microphone's packets pass
/// through a live isolation stream before the mixer, so live text hears the
/// cleaned voice. Other sources (the system audio tap) pass untouched.
///
/// Only live text sees the result. Capture persists its own copy of every
/// packet before the feed gets one, so the saved recording and the final
/// pass stay the original, uncleaned audio.
///
/// Timing: the stream releases sample `i` of its output for sample `i` of
/// its input, about 63 ms plus compute later. Each released sample is put
/// back at the timeline offset of the input it came from, so live word
/// times stay on the recording timeline; the words only arrive that much
/// later.
///
/// Runs on the transcriber's serial queue, never on a capture thread (the
/// feed in between drops rather than blocks). When it cannot keep up it
/// stops for the rest of the take and live text gets the raw microphone.
struct AudioLiveVoiceCleaner {
    enum Fallback: Equatable {
        /// Pushes ran late for longer than `lateLimitSeconds` of audio in a
        /// row, or the queue backed up.
        case fellBehind
        case failed(String)
    }

    static let sampleRate = Double(AudioEnhanceStream.sampleRate)
    let source: String
    let lateLimitSeconds: Double
    /// Microphone audio waiting in one drain past which the queue counts as
    /// backed up (normally a pump sees about 100 ms).
    let backlogLimitSeconds: Double
    private(set) var stream: AudioLiveEnhancing?
    private(set) var gaveUp = false
    private var resampler = AudioLiveConverter(outputRate: AudioLiveVoiceCleaner.sampleRate)
    /// Pushed input not yet released: its timeline offset and 48 kHz frames.
    private var spans: [(offset: Double, frames: Int)] = []
    private var lateRun = 0.0

    init(source: String = "microphone", lateLimitSeconds: Double = 1, backlogLimitSeconds: Double = 2) {
        self.source = source; self.lateLimitSeconds = lateLimitSeconds; self.backlogLimitSeconds = backlogLimitSeconds
    }

    /// Starts cleaning with an opened stream. Until then the microphone
    /// passes raw. Refused (closed) after a fallback.
    mutating func attach(_ stream: AudioLiveEnhancing) {
        guard !gaveUp, self.stream == nil else { stream.close(); return }
        self.stream = stream
    }

    /// The packets to mix: cleaned microphone voice, everything else as
    /// given. Returns a fallback the first time cleaning stops.
    mutating func process(_ packets: [AudioLiveFeed.Packet], dropped: Int) -> (packets: [AudioLiveFeed.Packet], fallback: Fallback?) {
        guard stream != nil else { return (packets, nil) }
        let micSeconds = packets.filter { $0.source == source }
            .reduce(0.0) { $0 + Double($1.samples.count / max(1, $1.channels)) / max(1, $1.sampleRate) }
        if dropped > 0 || micSeconds > backlogLimitSeconds {
            // Behind before this batch even started: do not add to it.
            return (finishStream() + packets, .fellBehind)
        }
        var out: [AudioLiveFeed.Packet] = []
        var fallback: Fallback?
        for packet in packets {
            guard packet.source == source, let stream, fallback == nil else { out.append(packet); continue }
            let mono = resampler.mono(packet.samples, sampleRate: packet.sampleRate, channels: packet.channels)
            guard !mono.isEmpty else { continue }
            do {
                let push = try stream.push(mono)
                spans.append((packet.offset, mono.count))
                out += place(push.voice)
                lateRun = push.late ? lateRun + Double(mono.count) / Self.sampleRate : 0
                if lateRun > lateLimitSeconds {
                    out += finishStream()
                    fallback = .fellBehind
                }
            } catch {
                // The stream's state is unknown: drop it, and give live text
                // this packet raw (the mixer drops any overlap).
                close()
                gaveUp = true
                out.append(packet)
                fallback = .failed(error.localizedDescription)
            }
        }
        return (out, fallback)
    }

    /// The voice still held back, at the end of a take.
    mutating func finish() -> [AudioLiveFeed.Packet] {
        guard stream != nil else { return [] }
        return finishStream()
    }

    mutating func close() {
        stream?.close(); stream = nil
        spans = []
    }

    /// Flushes the tail, closes the stream, and cleans nothing more this take.
    private mutating func finishStream() -> [AudioLiveFeed.Packet] {
        let tail = (try? stream?.flush()) ?? []
        let packets = place(tail)
        close()
        gaveUp = true
        return packets
    }

    /// Released voice back on the timeline, split at the input spans it came from.
    private mutating func place(_ voice: [Float]) -> [AudioLiveFeed.Packet] {
        var packets: [AudioLiveFeed.Packet] = []
        var cursor = 0
        while cursor < voice.count, !spans.isEmpty {
            let span = spans[0]
            let take = min(span.frames, voice.count - cursor)
            packets.append(AudioLiveFeed.Packet(samples: Array(voice[cursor..<(cursor + take)]), sampleRate: Self.sampleRate,
                                                channels: 1, offset: span.offset, source: source))
            cursor += take
            if take == span.frames {
                spans.removeFirst()
            } else {
                spans[0] = (span.offset + Double(take) / Self.sampleRate, span.frames - take)
            }
        }
        return packets
    }
}
