import Foundation
import MLX

/// One lane's time-attention past for one attention block: keys and values
/// [rows, heads, capacity, headDim], a key rotated once at its slot (a slot's
/// index is its rotary position).
final class AttentionState {
    var keys: MLXArray?
    var values: MLXArray?
    var end = 0
}

final class MambaState {
    var carried: (MLXArray, MLXArray)?
    init(_ carried: (MLXArray, MLXArray)? = nil) { self.carried = carried }
}

enum LaneState {
    case attention(AttentionState)
    case mamba(MambaState)
}

/// Shared by every attention block: the visible past, the buffer capacity,
/// the rotary tables of every slot and the masks.
final class AttentionCache {
    let frames: Int, capacity: Int, window: Int
    let cos: MLXArray, sin: MLXArray
    private var masks: [Int: MLXArray] = [:]

    init(frames: Int, capacity: Int, headDim: Int, window: Int) {
        self.frames = frames
        self.capacity = capacity
        self.window = window
        (cos, sin) = ropeTables(frames: capacity, headDim: headDim)
    }

    /// Rotary tables of slots [start, stop): float32 for the head-split
    /// kernel, the activation dtype where MLX ops rotate.
    func tables(_ start: Int, _ stop: Int, dtype: DType = .float32) -> (MLXArray, MLXArray) {
        (cos[start ..< stop].asType(dtype), sin[start ..< stop].asType(dtype))
    }

    func mask(frames: Int, past: Int) -> MLXArray {
        let key = frames * 1_000_003 + past
        if let m = masks[key] { return m }
        var values = [Bool](repeating: false, count: frames * (past + frames))
        for q in 0 ..< frames {
            let query = past + q
            for p in 0 ..< (past + frames) {
                var ok = p <= query
                if window > 0 { ok = ok && p > query - window }
                values[q * (past + frames) + p] = ok
            }
        }
        if masks.count > 256 { masks.removeAll() }
        let m = MLXArray(values, [frames, past + frames])
        masks[key] = m
        return m
    }
}

/// Runs a causal network at its streaming position over audio pieces.
///
/// `push` takes mono float samples at the model rate and returns the samples
/// that became final; `flush` ends the stream and returns the rest. Every
/// block of up to `blockFrames` new frames runs once with the state carried
/// (time attention over a cache of rotated keys, the time Mamba's convolution
/// tail and scan state, the STFT's overlap-add). The latency is the analysis
/// window plus the look-ahead frames.
///
/// With `windowFrames` and `crossfadeFrames`, the time attention never looks
/// back across a window boundary: a second lane starts `crossfadeFrames`
/// before each boundary with the Mamba states shared and an empty attention
/// cache, the lanes' decoder outputs blend linearly over the crossfade, and
/// the new lane carries on from the boundary.
///
/// Not reentrant: push from one actor or queue at a time.
public final class IsolationStream {
    public let blockFrames: Int
    public let cacheFrames: Int
    public let lookahead: Int
    public let windowFrames: Int?
    public let crossfadeFrames: Int?
    public private(set) var boundaries: [Int] = []
    let net: Network
    let nFFT: Int, hop: Int
    var states: [LaneState]
    var statesB: [LaneState]?
    let attention: AttentionCache
    let windowValues: [Float]
    let windowSquared: [Float]
    var audio: [Float] = []
    var audioBase = 0
    var frames = 0
    var emitted = 0
    var raw: [MLXArray] = []
    var spectra: [MLXArray] = []
    var pendingBase = 0
    var ola: [Float]
    var envelope: [Float]
    var olaBase = 0
    var released = 0
    var closed = false

    init(network: Network, blockFrames: Int = 4, cacheFrames: Int? = nil,
         windowFrames: Int? = nil, crossfadeFrames: Int? = nil) throws {
        guard !network.offline else {
            throw VoiceIsolatorError.invalidArgument("streaming needs a causal checkpoint")
        }
        guard blockFrames >= 1 else {
            throw VoiceIsolatorError.invalidArgument("process at least one frame per block")
        }
        if (windowFrames == nil) != (crossfadeFrames == nil) {
            throw VoiceIsolatorError.invalidArgument("windowFrames and crossfadeFrames go together")
        }
        if let w = windowFrames, let c = crossfadeFrames, !(0 < c && c < w) {
            throw VoiceIsolatorError.invalidArgument("crossfadeFrames must be in (0, windowFrames)")
        }
        let config = network.config
        net = network
        self.blockFrames = blockFrames
        let defaultCache = config.attentionWindowFrames > 0
            ? config.attentionWindowFrames
            : max(1, Int((config.chunkSeconds * Double(config.sampleRate)
                / Double(config.hopLength)).rounded(.toNearestOrEven)))
        self.cacheFrames = cacheFrames ?? defaultCache
        lookahead = network.lookaheadFrames
        nFFT = config.nFFT
        hop = config.hopLength
        self.windowFrames = windowFrames
        self.crossfadeFrames = crossfadeFrames
        states = network.blocks.map {
            $0 is DualPathAttention ? .attention(AttentionState()) : .mamba(MambaState())
        }
        let capacity: Int
        if let w = windowFrames, let c = crossfadeFrames, w + c <= self.cacheFrames + blockFrames {
            capacity = w + c
        } else {
            capacity = self.cacheFrames + max(self.cacheFrames / 2, blockFrames)
        }
        attention = AttentionCache(frames: self.cacheFrames, capacity: capacity,
                                   headDim: config.headDim, window: config.attentionWindowFrames)
        eval(network.window)
        windowValues = network.window.asArray(Float.self)
        windowSquared = windowValues.map { $0 * $0 }
        ola = [Float](repeating: 0, count: nFFT)
        envelope = [Float](repeating: 0, count: nFFT)
    }

    /// Input samples that must follow an output sample before its release.
    public var latencySamples: Int { nFFT + hop * lookahead }

    // MARK: analysis

    /// Spectra [count, 1, bins] of frames from `first`: each frame's window of
    /// the audio, zeros where there is none.
    func analyze(first: Int, count: Int) -> MLXArray {
        let start = first * hop - nFFT / 2
        var segments = [Float](repeating: 0, count: count * nFFT)
        let total = audioBase + audio.count
        for f in 0 ..< count {
            let s0 = start + f * hop
            for j in 0 ..< nFFT {
                let at = s0 + j
                if at >= audioBase && at < total {
                    segments[f * nFFT + j] = audio[at - audioBase] * windowValues[j]
                }
            }
        }
        let frames = MLXArray(segments, [1, count, nFFT])
        return rfft(frames, stream: .cpu).transposed(1, 0, 2)
    }

    // MARK: backbone

    private func openSlots(_ state: AttentionState, rows: Int, heads: Int, size: Int,
                           frames: Int, dtype: DType) -> Int {
        if state.end + frames > attention.capacity { compact(state) }
        reserve(state, slots: state.end + frames, rows: rows, heads: heads, size: size,
                dtype: dtype)
        return state.end
    }

    private func compact(_ state: AttentionState) {
        let keep = min(state.end, attention.frames)
        let shift = state.end - keep
        guard shift > 0, let keys = state.keys, let values = state.values else { return }
        let (cos, sin) = attention.tables(shift, shift + 1, dtype: keys.dtype)
        let kept = keys[0..., 0..., shift ..< state.end]
        keys[0..., 0..., ..<keep] = applyRope(kept, cos, -sin)
        values[0..., 0..., ..<keep] = values[0..., 0..., shift ..< state.end]
        state.end = keep
    }

    private func reserve(_ state: AttentionState, slots: Int, rows: Int, heads: Int, size: Int,
                         dtype: DType) {
        let held = state.keys?.dim(2) ?? 0
        if slots <= held { return }
        let shape = [rows, heads, min(attention.capacity, max(slots, 2 * held, 64)), size]
        let keys = MLXArray.zeros(shape, dtype: dtype)
        let values = MLXArray.zeros(shape, dtype: dtype)
        if state.end > 0, let oldK = state.keys, let oldV = state.values {
            keys[0..., 0..., ..<state.end] = oldK[0..., 0..., ..<state.end]
            values[0..., 0..., ..<state.end] = oldV[0..., 0..., ..<state.end]
        }
        state.keys = keys
        state.values = values
    }

    private func attendSlots(_ state: AttentionState, query: MLXArray, key: MLXArray,
                             value: MLXArray, first: Int, scale: Float) -> MLXArray {
        let frames = key.dim(2)
        state.keys![0..., 0..., first ..< (first + frames)] = key
        state.values![0..., 0..., first ..< (first + frames)] = value
        state.end = first + frames
        let past = min(first, attention.frames)
        return MLXFast.scaledDotProductAttention(
            queries: query, keys: state.keys![0..., 0..., (first - past) ..< state.end],
            values: state.values![0..., 0..., (first - past) ..< state.end], scale: scale,
            mask: .array(attention.mask(frames: frames, past: past)))
    }

    private func attend(_ block: DualPathAttention, _ tokens: MLXArray, batch: Int, bands: Int,
                        frames: Int, _ lanes: [AttentionState]) -> MLXArray {
        let attn = block.time
        let layout = SeqLayout.time(rows: batch * bands, frames: frames)
        let per = layout.rows / lanes.count
        let firsts = lanes.map {
            openSlots($0, rows: per, heads: attn.heads, size: attn.headDim, frames: frames,
                      dtype: tokens.dtype)
        }
        let output: MLXArray
        let value: MLXArray
        if lanes.count == 1 {
            let (cos, sin) = attention.tables(firsts[0], firsts[0] + frames)
            let (q, k, v) = attn.headsOf(tokens, layout, cos: cos, sin: sin)
            value = v
            output = attendSlots(lanes[0], query: q, key: k, value: v, first: firsts[0],
                                 scale: attn.scale)
        } else {
            let (q, k, v) = attn.headsOf(tokens, layout)
            value = v
            var outs: [MLXArray] = []
            for (i, state) in lanes.enumerated() {
                let (cos, sin) = attention.tables(firsts[i], firsts[i] + frames, dtype: q.dtype)
                let rows = (i * per) ..< ((i + 1) * per)
                outs.append(attendSlots(
                    state, query: applyRope(q[rows], cos, sin), key: applyRope(k[rows], cos, sin),
                    value: v[rows], first: firsts[i], scale: attn.scale))
            }
            output = concatenated(outs, axis: 0)
        }
        let y = attn.finish(tokens, output, value, layout)
        return block.band(y, .band(batch: batch, bands: bands, frames: frames))
    }

    private func scan(_ block: DualPathMamba, _ tokens: MLXArray, batch: Int, bands: Int,
                      frames: Int, _ lanes: [MambaState]) -> MLXArray {
        let layout = SeqLayout.time(rows: batch * bands, frames: frames)
        var carried: (MLXArray, MLXArray)? = nil
        if let first = lanes[0].carried {
            if lanes.count == 1 {
                carried = first
            } else {
                carried = (concatenated(lanes.map { $0.carried!.0 }, axis: 0),
                           concatenated(lanes.map { $0.carried!.1 }, axis: 0))
            }
        }
        let (y, state) = block.time(tokens, layout, state: carried, returnState: true)
        let per = layout.rows / lanes.count
        for (i, lane) in lanes.enumerated() {
            if lanes.count == 1 {
                lane.carried = state
            } else {
                let rows = (i * per) ..< ((i + 1) * per)
                lane.carried = (state!.0[rows], state!.1[rows])
            }
        }
        return block.freq(y, .band(batch: batch, bands: bands, frames: frames)).0
    }

    private func backbone(_ tokens: MLXArray, batch: Int, frames: Int,
                          lanes: [[LaneState]]) -> MLXArray {
        var x = tokens
        var b = batch
        if lanes.count > 1 {
            x = concatenated(Array(repeating: tokens, count: lanes.count), axis: 0)
            b *= lanes.count
        }
        for (index, block) in net.blocks.enumerated() {
            if let attn = block as? DualPathAttention {
                let ls = lanes.map { lane -> AttentionState in
                    if case .attention(let s) = lane[index] { return s }
                    fatalError("lane state mismatch")
                }
                x = attend(attn, x, batch: b, bands: net.bands, frames: frames, ls)
            } else if let mamba = block as? DualPathMamba {
                let ls = lanes.map { lane -> MambaState in
                    if case .mamba(let s) = lane[index] { return s }
                    fatalError("lane state mismatch")
                }
                x = scan(mamba, x, batch: b, bands: net.bands, frames: frames, ls)
            }
        }
        return x
    }

    private func cloneStates(_ states: [LaneState]) -> [LaneState] {
        states.map {
            switch $0 {
            case .mamba(let m): return .mamba(MambaState(m.carried))
            case .attention: return .attention(AttentionState())
            }
        }
    }

    private func nextCut() -> Int {
        guard let w = windowFrames, let c = crossfadeFrames else { return frames + blockFrames }
        let start = frames - frames % w
        let crossfadeStart = start + w - c
        return frames < crossfadeStart ? crossfadeStart : start + w
    }

    /// Backbone and decoder over spectra [count, 1, bins]: raw [count, 1, bins, 4].
    private func runBlock(_ spectra: MLXArray) throws -> MLXArray {
        let count = spectra.dim(0)
        let tokens = net.encode(spectra.transposed(1, 0, 2))
        guard let w = windowFrames, let c = crossfadeFrames else {
            let raw = net.decode(backbone(tokens, batch: 1, frames: count, lanes: [states]),
                                 batch: 1, frames: count)
            return raw.transposed(1, 0, 2, 3)
        }
        let first = frames % w
        let crossfadeStart = w - c
        if first + count > w || (first < crossfadeStart && crossfadeStart < first + count) {
            throw VoiceIsolatorError.invalidArgument("a block crossed a crossfade start or boundary")
        }
        if first >= crossfadeStart && statesB == nil { statesB = cloneStates(states) }
        if first < crossfadeStart {
            let raw = net.decode(backbone(tokens, batch: 1, frames: count, lanes: [states]),
                                 batch: 1, frames: count)
            return raw.transposed(1, 0, 2, 3)
        }
        let both = net.decode(
            backbone(tokens, batch: 1, frames: count, lanes: [states, statesB!]), batch: 2,
            frames: count)
        let weights = (first ..< (first + count)).map { f -> Float in
            min(max(Float(f - crossfadeStart + 1) / Float(c), 0), 1)
        }
        let weight = MLXArray(weights, [1, count, 1, 1])
        let raw = both[0 ..< 1] * (1 - weight) + both[1 ..< 2] * weight
        if first + count == w {
            boundaries.append(frames + count)
            states = statesB!
            statesB = nil
        }
        return raw.transposed(1, 0, 2, 3)
    }

    private func stateArrays() -> [MLXArray] {
        var out: [MLXArray] = []
        for lane in [states, statesB ?? []] {
            for s in lane {
                switch s {
                case .attention(let a):
                    if let k = a.keys, let v = a.values { out += [k, v] }
                case .mamba(let m):
                    if let (t, h) = m.carried { out += [t, h] }
                }
            }
        }
        return out
    }

    // MARK: synthesis

    /// Windowed time frames [count * nFFT] of the corrected spectra.
    private func framesOf(_ raw: MLXArray, _ spectrum: MLXArray) -> [Float] {
        let corrected = net.correction(raw, spectrum)
        let frames = irfft(corrected, n: nFFT, stream: .cpu)
        eval(frames)
        var values = frames.asArray(Float.self)
        for i in 0 ..< values.count { values[i] *= windowValues[i % nFFT] }
        return values
    }

    private func grow(_ needed: Int) {
        if needed > ola.count {
            ola += [Float](repeating: 0, count: needed - ola.count)
            envelope += [Float](repeating: 0, count: needed - envelope.count)
        }
    }

    private func synthesizeOne() {
        let index = emitted
        let last = frames - 1
        let source = index <= last ? min(index + lookahead, last) : index
        let frame = framesOf(raw[source - pendingBase], spectra[index - pendingBase])
        let start = index * hop - olaBase
        grow(start + nFFT)
        for j in 0 ..< nFFT {
            ola[start + j] += frame[j]
            envelope[start + j] += windowSquared[j]
        }
        emitted += 1
    }

    private func synthesizeReady() {
        let count = frames - lookahead - emitted
        guard count > 0 else { return }
        let first = emitted - pendingBase
        let rawBlock = stacked(Array(raw[(first + lookahead) ..< (first + lookahead + count)]))
        let spectrum = stacked(Array(spectra[first ..< (first + count)]))
        let values = framesOf(rawBlock, spectrum)
        let start = emitted * hop - olaBase
        grow(start + (count - 1) * hop + nFFT)
        for k in 0 ..< count {
            let at = start + k * hop
            for j in 0 ..< nFFT {
                ola[at + j] += values[k * nFFT + j]
                envelope[at + j] += windowSquared[j]
            }
        }
        emitted += count
    }

    private func advance(to limit: Int) throws {
        while frames < limit {
            let count = min(blockFrames, limit - frames, nextCut() - frames)
            let spectraBlock = analyze(first: frames, count: count)
            let rawBlock = try runBlock(spectraBlock)
            eval([rawBlock, spectraBlock] + stateArrays())
            for i in 0 ..< count {
                raw.append(rawBlock[i])
                spectra.append(spectraBlock[i])
            }
            frames += count
        }
        synthesizeReady()
        while emitted + lookahead < frames { synthesizeOne() }
        trim()
    }

    private func trim() {
        let drop = emitted - pendingBase
        if drop > 0 {
            raw.removeFirst(min(drop, raw.count))
            spectra.removeFirst(min(drop, spectra.count))
        }
        pendingBase = emitted
        let keepFrom = released - olaBase
        if keepFrom > 0 {
            ola.removeFirst(keepFrom)
            envelope.removeFirst(keepFrom)
            olaBase = released
        }
        let neededFrom = min(released, max(0, frames * hop - nFFT / 2))
        let dropAudio = neededFrom - audioBase
        if dropAudio > 0 {
            audio.removeFirst(dropAudio)
            audioBase = neededFrom
        }
    }

    private func release(upTo upto: Int) -> [Float] {
        let total = audioBase + audio.count
        let start = released
        let stop = min(upto, total)
        guard stop > start else { return [] }
        let offset = nFFT / 2 - olaBase
        var out = [Float](repeating: 0, count: stop - start)
        for i in 0 ..< out.count {
            let s = start + i
            let delta = ola[s + offset] / max(envelope[s + offset], 1e-11)
            out[i] = audio[s - audioBase] + delta
        }
        released = stop
        return out
    }

    /// Feeds new samples; returns the output samples that became final.
    public func push(_ samples: [Float]) throws -> [Float] {
        guard !closed else { throw VoiceIsolatorError.invalidArgument("the stream was flushed") }
        audio += samples
        let total = audioBase + audio.count
        let computable = (total - nFFT / 2) >= 0 ? (total - nFFT / 2) / hop + 1 : 0
        try advance(to: max(0, computable))
        let out = release(upTo: emitted * hop - nFFT / 2)
        trim()
        return out
    }

    /// Ends the stream and returns every remaining output sample.
    public func flush() throws -> [Float] {
        if closed { return [] }
        closed = true
        let total = audioBase + audio.count
        try advance(to: total / hop + 1)
        while emitted < frames { synthesizeOne() }
        return release(upTo: total)
    }
}
