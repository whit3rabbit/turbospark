import Foundation
import MLX

/// Isolates voices at 48 kHz, offline for best quality or streaming.
///
/// Not reentrant: one isolator (and its network) serves one call at a time.
/// Confine it to one actor or queue; separate isolators may run concurrently.
public final class VoiceIsolator {
    public enum Mode: String, Sendable { case offline, streaming }

    public static let streamPieceSeconds = 0.5
    public static let streamWindowFrames = 600
    public static let streamCrossfadeFrames = 200
    /// Frames per block when `isolate` renders a clip through the stream; the
    /// output does not depend on the block size, larger blocks amortize the
    /// per-block dispatches.
    public static let renderBlockFrames = 25

    public let network: Network
    public let mode: Mode
    public var blockFrames: Int
    public var cacheFrames: Int?
    public var windowFrames: Int?
    public var crossfadeFrames: Int?
    public var offlinePasses: Int

    /// - Parameters:
    ///   - network: a loaded network (offline checkpoint for `.offline`, causal
    ///     for `.streaming`).
    ///   - lookaheadFrames: streaming look-ahead (one of the trained positions).
    ///   - windowFrames: streaming attention window with a crossfaded second
    ///     lane (600 frames by default, 0 turns it off).
    /// Loads a checkpoint and builds an isolator for it.
    ///
    /// - Parameters:
    ///   - url: a converted MLX checkpoint (`model_fp32` / `model_fp16`) or a
    ///     released PyTorch `.safetensors`.
    ///   - dtype: compute precision for an unquantized checkpoint; float16 by
    ///     default, `nil` keeps the stored precision. Quantized checkpoints
    ///     compute in their scales' precision.
    public convenience init(contentsOf url: URL, mode: Mode = .offline,
                            dtype: DType? = .float16, lookaheadFrames: Int? = nil) throws {
        try self.init(network: Network(contentsOf: url, dtype: dtype), mode: mode,
                      lookaheadFrames: lookaheadFrames)
    }

    public init(network: Network, mode: Mode = .offline, lookaheadFrames: Int? = nil,
                blockFrames: Int = 4, cacheFrames: Int? = nil, windowFrames: Int? = nil,
                crossfadeFrames: Int? = nil, offlinePasses: Int? = nil) throws {
        let config = network.config
        if mode == .offline && config.causal {
            throw VoiceIsolatorError.invalidArgument("this checkpoint only streams")
        }
        if mode == .streaming && !config.causal {
            throw VoiceIsolatorError.invalidArgument("this checkpoint has no streaming mode")
        }
        if mode == .streaming { try network.setLookahead(lookaheadFrames ?? config.lookaheadFrames) }
        let window = windowFrames ?? Self.streamWindowFrames
        let crossfade = crossfadeFrames ?? Self.streamCrossfadeFrames
        if window > 0 && !(0 < crossfade && crossfade < window) {
            throw VoiceIsolatorError.invalidArgument("crossfadeFrames must be in (0, windowFrames)")
        }
        self.network = network
        self.mode = mode
        self.blockFrames = blockFrames
        self.cacheFrames = cacheFrames
        self.windowFrames = window > 0 ? window : nil
        self.crossfadeFrames = window > 0 ? crossfade : nil
        self.offlinePasses = offlinePasses ?? config.offlinePasses
    }

    public var sampleRate: Int { network.config.sampleRate }

    /// The streaming algorithmic latency in samples: one analysis window plus
    /// the look-ahead frames.
    public var latencySamples: Int {
        network.config.nFFT + network.config.hopLength * network.lookaheadFrames
    }

    /// A streaming session (`.streaming` mode only).
    public func stream(blockFrames: Int? = nil, cacheFrames: Int? = nil, windowFrames: Int? = nil,
                       crossfadeFrames: Int? = nil) throws -> IsolationStream {
        guard mode == .streaming else {
            throw VoiceIsolatorError.invalidArgument("stream() needs .streaming mode")
        }
        let window = windowFrames ?? (self.windowFrames ?? 0)
        let crossfade = crossfadeFrames ?? (self.crossfadeFrames ?? Self.streamCrossfadeFrames)
        return try IsolationStream(network: network, blockFrames: blockFrames ?? self.blockFrames,
                          cacheFrames: cacheFrames ?? self.cacheFrames,
                          windowFrames: window > 0 ? window : nil,
                          crossfadeFrames: window > 0 ? crossfade : nil)
    }

    /// Isolates the voice of mono samples; returns float32 samples at the
    /// input rate unless `outputSampleRate` is given.
    ///
    /// - Parameters:
    ///   - progress: called with the completed fraction (0...1) after every
    ///     render window (offline) or half-second piece (streaming).
    /// - Throws: `CancellationError` between windows when the calling task is
    ///   cancelled.
    public func isolate(_ samples: [Float], sampleRate: Int, outputSampleRate: Int? = nil,
                        chunkSeconds: Double? = nil, overlapSeconds: Double? = nil,
                        progress: ((Double) -> Void)? = nil) throws -> [Float]
    {
        guard !samples.isEmpty else {
            throw VoiceIsolatorError.invalidArgument("the audio is empty")
        }
        guard samples.allSatisfy({ $0.isFinite }) else {
            throw VoiceIsolatorError.invalidArgument("the audio contains NaN or infinite samples")
        }
        let x = MLXArray(samples, [samples.count])
        let a = DSP.resample(x, from: sampleRate, to: self.sampleRate)
        eval(a)
        var y: MLXArray
        if mode == .streaming {
            let session = try stream(blockFrames: max(blockFrames, Self.renderBlockFrames))
            let input = a.asArray(Float.self)
            let piece = Int(Self.streamPieceSeconds * Double(self.sampleRate))
            var out: [Float] = []
            out.reserveCapacity(input.count)
            var i = 0
            while i < input.count {
                try Task.checkCancellation()
                out += try session.push(Array(input[i ..< min(input.count, i + piece)]))
                i += piece
                progress?(Double(min(i, input.count)) / Double(input.count))
            }
            out += try session.flush()
            y = MLXArray(out, [out.count])
        } else {
            let passes = max(1, offlinePasses)
            y = a
            for pass in 0 ..< passes {
                y = try enhance(network, y, chunkSeconds: chunkSeconds,
                                overlapSeconds: overlapSeconds) { done in
                    progress?((Double(pass) + done) / Double(passes))
                }
            }
        }
        y = DSP.fitLength(y, a.dim(0))
        let outRate = outputSampleRate ?? sampleRate
        y = DSP.resample(y, from: self.sampleRate, to: outRate)
        if outRate == sampleRate { y = DSP.fitLength(y, samples.count) }
        eval(y)
        return y.asArray(Float.self)
    }
}

/// The render window and its crossfade in samples (crossfade <= half).
public func windowSamples(chunkSeconds: Double, overlapSeconds: Double, sampleRate: Int,
                          minSamples: Int = 1) throws -> (Int, Int) {
    guard chunkSeconds.isFinite, overlapSeconds.isFinite else {
        throw VoiceIsolatorError.invalidArgument("chunk and overlap seconds must be finite")
    }
    let chunk = Int(chunkSeconds * Double(sampleRate))
    guard chunk >= minSamples else {
        throw VoiceIsolatorError.invalidArgument("chunkSeconds=\(chunkSeconds) is too short")
    }
    guard overlapSeconds >= 0 else {
        throw VoiceIsolatorError.invalidArgument("overlapSeconds must not be negative")
    }
    return (chunk, min(Int(overlapSeconds * Double(sampleRate)), chunk / 2))
}

/// float32 linspace evaluated as PyTorch's CPU kernel does.
func torchLinspace(_ start: Float, _ end: Float, _ steps: Int) -> [Float] {
    if steps == 1 { return [start] }
    let step = (end - start) / Float(steps - 1)
    let half = steps / 2
    return (0 ..< steps).map { i in
        i < half ? start + step * Float(i) : end - step * Float(steps - i - 1)
    }
}

/// (start, weight) of every window of a clip longer than one window.
func chunkPlan(n: Int, chunk: Int, overlap: Int) -> [(Int, [Float])] {
    var starts = Array(stride(from: 0, to: n - chunk, by: chunk - overlap))
    starts.append(n - chunk)
    var plan: [(Int, [Float])] = []
    for (index, start) in starts.enumerated() {
        var w = [Float](repeating: 1, count: chunk)
        if index > 0 {
            let fadeIn = starts[index - 1] + chunk - overlap - start
            for i in 0 ..< min(fadeIn, chunk) { w[i] = 0 }
            if overlap > 0 {
                let ramp = torchLinspace(0, 1, overlap)
                for i in 0 ..< overlap where fadeIn + i < chunk { w[fadeIn + i] = ramp[i] }
            }
        }
        if overlap > 0 && start + chunk < n {
            let ramp = torchLinspace(1, 0, overlap)
            for i in 0 ..< overlap { w[chunk - overlap + i] = ramp[i] }
        }
        plan.append((start, w))
    }
    return plan
}

/// Isolates mono audio [samples] of any length offline: windows of
/// `chunkSeconds` overlap by `overlapSeconds` with a linear crossfade, the last
/// window ending at the clip's end.
///
/// `progress` receives the completed fraction after each window; a cancelled
/// calling task stops the render with `CancellationError` between windows.
public func enhance(_ net: Network, _ audio: MLXArray, chunkSeconds: Double? = nil,
                    overlapSeconds: Double? = nil,
                    progress: ((Double) -> Void)? = nil) throws -> MLXArray {
    let config = net.config
    let (chunk, overlap) = try windowSamples(
        chunkSeconds: chunkSeconds ?? config.chunkSeconds,
        overlapSeconds: overlapSeconds ?? config.overlapSeconds,
        sampleRate: config.sampleRate, minSamples: config.hopLength)
    let n = audio.dim(0)
    if n <= chunk {
        try Task.checkCancellation()
        let out = net(audio.reshaped([1, n]))[0]
        eval(out)
        progress?(1)
        return out
    }
    var total = [Float](repeating: 0, count: n)
    var weight = [Float](repeating: 0, count: n)
    let plan = chunkPlan(n: n, chunk: chunk, overlap: overlap)
    for (index, (start, w)) in plan.enumerated() {
        try Task.checkCancellation()
        defer { progress?(Double(index + 1) / Double(plan.count)) }
        let y = net(audio[start ..< (start + chunk)].reshaped([1, chunk]))
        eval(y)
        let values = y.asArray(Float.self)
        for i in 0 ..< chunk {
            total[start + i] += values[i] * w[i]
            weight[start + i] += w[i]
        }
    }
    for i in 0 ..< n { total[i] /= max(weight[i], 1e-6) }
    return MLXArray(total, [n])
}

/// Mixes interleaved or planar channels to mono float32 by averaging.
public func toMono(_ channels: [[Float]]) -> [Float] {
    guard let first = channels.first else { return [] }
    if channels.count == 1 { return first }
    var out = [Float](repeating: 0, count: first.count)
    for ch in channels {
        for i in 0 ..< out.count { out[i] += ch[i] }
    }
    let n = Float(channels.count)
    return out.map { $0 / n }
}
