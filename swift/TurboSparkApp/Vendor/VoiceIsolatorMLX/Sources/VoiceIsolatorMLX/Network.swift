import Foundation
import MLX

let layerNormEps: Float = 1e-5

/// `y = x W^T + b` for a float32 or 8-bit affine-quantized `W`; `W` may carry
/// leading batch axes ([n, out, in]) that broadcast against `x`.
final class Linear {
    let weight: MLXArray
    var bias: MLXArray?
    let scales: MLXArray?
    let biases: MLXArray?
    let weightT: MLXArray?
    let groupSize: Int
    let bits: Int

    init(_ arrays: [String: MLXArray], _ prefix: String, _ quant: Quantization?) throws {
        guard let weight = arrays["\(prefix).weight"] else {
            throw VoiceIsolatorError.invalidCheckpoint("missing \(prefix).weight")
        }
        self.weight = weight
        bias = arrays["\(prefix).bias"]
        scales = arrays["\(prefix).scales"]
        biases = arrays["\(prefix).biases"]
        groupSize = quant?.groupSize ?? 0
        bits = quant?.bits ?? 0
        weightT = scales == nil ? weight.swappedAxes(-1, -2) : nil
    }

    func matmul(_ x: MLXArray) -> MLXArray {
        if let scales {
            // quantizedMatmul (renamed quantizedMM in mlx-swift 0.30) keeps
            // the package building on mlx-swift 0.29 through 0.32
            return quantizedMatmul(x, weight, scales: scales, biases: biases, transpose: true,
                                   groupSize: groupSize, bits: bits)
        }
        return MLX.matmul(x, weightT!)
    }

    /// `c + x W^T` (the bias is not added).
    func addmm(_ c: MLXArray, _ x: MLXArray) -> MLXArray {
        if scales == nil, x.ndim == 2, weight.ndim == 2 {
            return addMM(c, x, weightT!)
        }
        return c + matmul(x)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let bias else { return matmul(x) }
        return addmm(bias, x)
    }
}

final class LayerNorm {
    let weight: MLXArray
    let bias: MLXArray

    init(_ arrays: [String: MLXArray], _ prefix: String) throws {
        guard let w = arrays["\(prefix).weight"], let b = arrays["\(prefix).bias"] else {
            throw VoiceIsolatorError.invalidCheckpoint("missing \(prefix)")
        }
        weight = w
        bias = b
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: layerNormEps)
    }
}

/// Exact (erf) GELU.
func gelu(_ x: MLXArray) -> MLXArray { Kernels.gelu(x) }

/// Cosine and sine [frames, headDim / 2] of rotary positions: the float32
/// inverse frequencies and angles PyTorch forms, cosine and sine evaluated in
/// float64 and rounded.
func ropeTables(frames: Int, headDim: Int) -> (MLXArray, MLXArray) {
    let half = headDim / 2
    let inv = (0 ..< half).map { i -> Float in
        let exponent = -Float(i) / Float(half)
        return Float(pow(10000.0, Double(exponent)))
    }
    var cosv = [Float](repeating: 0, count: frames * half)
    var sinv = [Float](repeating: 0, count: frames * half)
    for p in 0 ..< frames {
        for i in 0 ..< half {
            let angle = Double(Float(p) * inv[i])
            cosv[p * half + i] = Float(Foundation.cos(angle))
            sinv[p * half + i] = Float(Foundation.sin(angle))
        }
    }
    return (MLXArray(cosv, [frames, half]), MLXArray(sinv, [frames, half]))
}

/// Rotates the halves of `x` [..., frames, headDim] by the tables.
func applyRope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis: -1)
}

/// A pre-norm transformer layer along one sequence axis.
final class Attention {
    let norm1: LayerNorm, norm2: LayerNorm
    let qkv: Linear, proj: Linear, fc1: Linear, fc2: Linear
    let heads: Int, headDim: Int
    let scale: Float
    let exclusive: Bool
    let window: Int

    init(_ arrays: [String: MLXArray], _ prefix: String, _ config: Config, _ quant: Quantization?,
         exclusive: Bool) throws {
        norm1 = try LayerNorm(arrays, "\(prefix).norm1")
        qkv = try Linear(arrays, "\(prefix).qkv", quant)
        proj = try Linear(arrays, "\(prefix).proj", quant)
        norm2 = try LayerNorm(arrays, "\(prefix).norm2")
        fc1 = try Linear(arrays, "\(prefix).fc1", quant)
        fc2 = try Linear(arrays, "\(prefix).fc2", quant)
        heads = config.numHeads
        headDim = config.headDim
        scale = Float(1.0 / Double(config.headDim).squareRoot())
        self.exclusive = exclusive
        window = config.attentionWindowFrames
    }

    func headsOf(_ x: MLXArray, _ layout: SeqLayout, cos: MLXArray? = nil, sin: MLXArray? = nil)
        -> (MLXArray, MLXArray, MLXArray)
    {
        Kernels.splitHeads(qkv(norm1(x)), layout, heads: heads, headDim: headDim, cos: cos,
                           sin: sin)
    }

    func finish(_ x: MLXArray, _ output: MLXArray, _ value: MLXArray, _ layout: SeqLayout)
        -> MLXArray
    {
        let merged = Kernels.mergeHeads(output, layout, value: exclusive ? value : nil)
        let y = proj.addmm(x + proj.bias!, merged)
        let h = gelu(fc1(norm2(y)))
        return fc2.addmm(y + fc2.bias!, h)
    }

    func mask(frames: Int, causal: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if !causal { return .none }
        if window > 0 && window < frames {
            let pos = MLXArray(0 ..< frames)
            let lag = pos.expandedDimensions(axis: 1) - pos.expandedDimensions(axis: 0)
            return .array((lag .>= 0) .&& (lag .< window))
        }
        return .causal
    }

    func callAsFunction(_ x: MLXArray, _ layout: SeqLayout, cos: MLXArray? = nil,
                        sin: MLXArray? = nil, causal: Bool = false) -> MLXArray {
        let (q, k, v) = headsOf(x, layout, cos: cos, sin: sin)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale,
            mask: mask(frames: layout.length, causal: causal))
        return finish(x, out, v, layout)
    }
}

/// Selective state-space layers along one axis: forward only, or forward and
/// reverse with their output projections folded together.
final class MambaAxis {
    let inU: Linear, inZ: Linear, xProj: Linear, out: Linear, skip: Linear
    let convW: MLXArray, convB: MLXArray, dtW: MLXArray, dtB: MLXArray, A: MLXArray, D: MLXArray
    let norm: LayerNorm
    let outDType: DType
    let inner: Int, taps: Int

    init(_ arrays: [String: MLXArray], _ prefix: String, _ config: Config, _ quant: Quantization?)
        throws
    {
        func get(_ key: String) throws -> MLXArray {
            guard let a = arrays["\(prefix).\(key)"] else {
                throw VoiceIsolatorError.invalidCheckpoint("missing \(prefix).\(key)")
            }
            return a
        }
        inU = try Linear(arrays, "\(prefix).in_u", quant)
        inZ = try Linear(arrays, "\(prefix).in_z", quant)
        convW = try get("conv.weight")
        convB = try get("conv.bias")
        xProj = try Linear(arrays, "\(prefix).x_proj", quant)
        dtW = try get("dt.weight")
        dtB = try get("dt.bias")
        A = try get("A")
        D = try get("D")
        out = try Linear(arrays, "\(prefix).out", quant)
        skip = try Linear(arrays, "\(prefix).skip", quant)
        norm = try LayerNorm(arrays, "\(prefix).norm")
        outDType = out.scales?.dtype ?? .float32
        inner = config.inner
        taps = config.mambaConv
    }

    /// `x + norm(projection)` over tokens [tokens, width]; with a carried
    /// `(tail, ssm)` state (forward-only layers) and `returnState`, also the
    /// state after the last position.
    func callAsFunction(_ x: MLXArray, _ layout: SeqLayout, state: (MLXArray, MLXArray)? = nil,
                        returnState: Bool = false) -> (MLXArray, (MLXArray, MLXArray)?) {
        let u = inU(x)
        let uc = Kernels.convSiLU(u, weight: convW, bias: convB, layout, history: state?.0)
        let dbc = xProj(uc)
        let (y, ssm) = Kernels.selectiveScan(
            uc: uc, z: inZ(x), dbc: dbc, A: A, D: D, wDt: dtW, bDt: dtB, layout,
            state: state?.1, returnState: returnState, outDType: outDType)
        // the projection before the norm reaches ~1e5: float32 in every mode
        let x32 = x.asType(.float32)
        let result = x + norm(out.addmm(skip.addmm(out.bias!, x32), y)).asType(x.dtype)
        guard returnState else { return (result, nil) }
        let k = taps - 1
        let rows = u.reshaped([layout.rows, layout.length, inner])
        let history = state?.0 ?? MLXArray.zeros([layout.rows, k, inner], dtype: u.dtype)
        let joined = concatenated([history, rows], axis: 1)
        let tail = joined[0..., (joined.dim(1) - k)...]
        return (result, (tail, ssm!))
    }
}

/// One forward pass's geometry. Every sequence layer is independent across
/// its other axis, so a layer runs over groups of rows (bands along time,
/// frames across bands) of at most `maxTokens` tokens; each group is scheduled
/// as soon as its graph is built and the group before it is awaited, so at
/// most two groups' intermediates are allocated.
final class Pass {
    let batch: Int, bands: Int, frames: Int, width: Int
    let cos: MLXArray?, sin: MLXArray?
    let causal: Bool
    let maxTokens: Int

    init(batch: Int, bands: Int, frames: Int, width: Int, cos: MLXArray?, sin: MLXArray?,
         causal: Bool, maxTokens: Int?) {
        self.batch = batch
        self.bands = bands
        self.frames = frames
        self.width = width
        self.cos = cos
        self.sin = sin
        self.causal = causal
        self.maxTokens = maxTokens ?? batch * bands * frames
    }

    private func pipeline(_ outs: [MLXArray]) {
        asyncEval(outs[outs.count - 1])
        if outs.count > 1 { eval(outs[outs.count - 2]) }
    }

    func alongTime(_ x: MLXArray, _ fn: (MLXArray, SeqLayout) -> MLXArray) -> MLXArray {
        let rows = batch * bands
        let per = max(1, maxTokens / frames)
        if per >= rows { return fn(x, .time(rows: rows, frames: frames)) }
        var outs: [MLXArray] = []
        var r0 = 0
        while r0 < rows {
            let r1 = min(rows, r0 + per)
            outs.append(fn(x[(r0 * frames) ..< (r1 * frames)], .time(rows: r1 - r0, frames: frames)))
            pipeline(outs)
            r0 = r1
        }
        return concatenated(outs, axis: 0)
    }

    func alongBands(_ x: MLXArray, _ fn: (MLXArray, SeqLayout) -> MLXArray) -> MLXArray {
        let per = max(1, maxTokens / (batch * bands))
        if per >= frames { return fn(x, .band(batch: batch, bands: bands, frames: frames)) }
        let x4 = x.reshaped([batch, bands, frames, width])
        var outs: [MLXArray] = []
        var f0 = 0
        while f0 < frames {
            let f1 = min(frames, f0 + per)
            let chunk = x4[0..., 0..., f0 ..< f1].reshaped([-1, width])
            let y = fn(chunk, .band(batch: batch, bands: bands, frames: f1 - f0))
            outs.append(y.reshaped([batch, bands, f1 - f0, width]))
            pipeline(outs)
            f0 = f1
        }
        return concatenated(outs, axis: 2).reshaped([-1, width])
    }
}

protocol Block: AnyObject {
    func callAsFunction(_ x: MLXArray, _ p: Pass) -> MLXArray
}

final class DualPathAttention: Block {
    let time: Attention
    let band: Attention

    init(_ arrays: [String: MLXArray], _ prefix: String, _ config: Config, _ quant: Quantization?)
        throws
    {
        time = try Attention(arrays, "\(prefix).time", config, quant,
                             exclusive: config.exclusiveTime)
        band = try Attention(arrays, "\(prefix).band", config, quant,
                             exclusive: config.exclusiveBand)
    }

    func callAsFunction(_ x: MLXArray, _ p: Pass) -> MLXArray {
        let t = p.alongTime(x) { c, layout in
            self.time(c, layout, cos: p.cos, sin: p.sin, causal: p.causal)
        }
        return p.alongBands(t) { c, layout in self.band(c, layout) }
    }
}

final class DualPathMamba: Block {
    let time: MambaAxis
    let freq: MambaAxis

    init(_ arrays: [String: MLXArray], _ prefix: String, _ config: Config, _ quant: Quantization?)
        throws
    {
        time = try MambaAxis(arrays, "\(prefix).time", config, quant)
        freq = try MambaAxis(arrays, "\(prefix).freq", config, quant)
    }

    func callAsFunction(_ x: MLXArray, _ p: Pass) -> MLXArray {
        let t = p.alongTime(x) { c, layout in self.time(c, layout).0 }
        return p.alongBands(t) { c, layout in self.freq(c, layout).0 }
    }
}

/// The correction's elementwise part: the bounded complex mask and the
/// bounded residual applied to the spectrum's real and imaginary parts.
func correctionParts(_ a: [MLXArray]) -> [MLXArray] {
    let raw = a[0], sr = a[1], si = a[2], mag = a[3]
    let mr = 2 * tanh(raw[.ellipsis, 0])
    let mi = 2 * tanh(raw[.ellipsis, 1])
    let r0 = tanh(raw[.ellipsis, 2])
    let r1 = tanh(raw[.ellipsis, 3])
    let norm = maximum(sqrt(r0 * r0 + r1 * r1), Float(1))
    let cr = sr * mr - si * mi + (r0 / norm) * mag
    let ci = sr * mi + si * mr + (r1 / norm) * mag
    return [stacked([cr, ci], axis: -1)]
}

/// The isolator network: a band-split dual-path model over the STFT. The
/// token stream is one `[batch * bands * frames, width]` matrix for the whole
/// backbone.
///
/// Not reentrant: confine a network to one actor or queue. Its weights are
/// read-only after loading; the per-call state lives in each pass.
public final class Network {
    /// Tokens per sequence-layer chunk (activation memory bound).
    public static let defaultMaxTokens = 4096

    public let config: Config
    public let quantization: Quantization?
    /// Activation dtype: float32, float16 or bfloat16, following the weights
    /// (quantized layers compute in their scales' dtype).
    public let dtype: DType
    public let bands: Int
    public var maxTokens: Int? = Network.defaultMaxTokens
    public private(set) var lookaheadFrames: Int
    let layout: BandLayout
    let window: MLXArray
    let gather: MLXArray, valid: MLXArray, counts: MLXArray, keep: MLXArray
    let encNormW: MLXArray, encNormB: MLXArray, encoder: Linear, encoderBias: MLXArray
    let lookaheadEmbedding: MLXArray?
    var lookaheadBias: MLXArray?
    let blocks: [Block]
    let decScale: MLXArray, decShift: MLXArray, decNormW: MLXArray, decNormB: MLXArray
    let decFc1: Linear, decFc1B: MLXArray, decFc2: Linear, decFc2B: MLXArray
    private var ropeCache: [Int: (MLXArray, MLXArray)] = [:]

    public init(_ w: LoadedWeights) throws {
        let config = w.config
        let arrays = w.arrays
        let quant = w.quantization
        func get(_ key: String) throws -> MLXArray {
            guard let a = arrays[key] else {
                throw VoiceIsolatorError.invalidCheckpoint("missing \(key)")
            }
            return a
        }
        self.config = config
        quantization = quant
        dtype = (arrays["blocks.0.time.qkv.scales"] ?? arrays["blocks.0.time.qkv.weight"])?.dtype
            ?? .float32
        layout = BandLayout(config: config, inMultiple: w.inMultiple)
        bands = layout.edges.count
        window = arrays["stft.window"] ?? DSP.hannWindow(config.nFFT)
        gather = MLXArray(layout.gather, [layout.gather.count])
        valid = MLXArray(layout.valid, [bands, layout.inPad])
        counts = MLXArray(layout.counts, [bands, 1])
        keep = MLXArray(layout.keep, [layout.keep.count])
        encNormW = try get("encoder.norm.weight")
        encNormB = try get("encoder.norm.bias")
        encoder = try Linear(arrays, "encoder", quant)
        encoder.bias = nil
        encoderBias = try get("encoder.bias").expandedDimensions(axis: 1)
        lookaheadEmbedding = arrays["lookahead_embedding"]
        var blocks: [Block] = []
        for (i, kind) in config.blockTypes.enumerated() {
            if kind == "transformer" {
                blocks.append(try DualPathAttention(arrays, "blocks.\(i)", config, quant))
            } else {
                blocks.append(try DualPathMamba(arrays, "blocks.\(i)", config, quant))
            }
        }
        self.blocks = blocks
        decScale = try get("decoder.scale")
        decShift = try get("decoder.shift")
        decNormW = try get("decoder.norm.weight").expandedDimensions(axis: 1)
        decNormB = try get("decoder.norm.bias").expandedDimensions(axis: 1)
        decFc1 = try Linear(arrays, "decoder.fc1", quant)
        decFc1B = try get("decoder.fc1.bias").expandedDimensions(axis: 1)
        decFc2 = try Linear(arrays, "decoder.fc2", quant)
        decFc2B = try get("decoder.fc2.bias").expandedDimensions(axis: 1)
        lookaheadFrames = config.causal ? config.lookaheadFrames : 0
        if !config.lookaheadFramesChoices.isEmpty {
            try setLookahead(config.lookaheadFrames)
        }
        eval(Array(arrays.values))
    }

    /// Loads a converted MLX file or a released PyTorch checkpoint.
    ///
    /// `dtype` (.float32, .float16 or .bfloat16) sets the compute precision of
    /// an unquantized checkpoint, casting its weights as needed; `nil` keeps
    /// the stored precision, and quantized checkpoints ignore it. The STFT,
    /// the scan state, the Mamba output projections, softmax and norm
    /// statistics stay float32 in every precision.
    public convenience init(contentsOf url: URL, dtype: DType? = nil) throws {
        var w = try Weights.load(url)
        if let dtype, w.quantization == nil {
            w = LoadedWeights(config: w.config, arrays: Weights.cast(w.arrays, to: dtype),
                              quantization: nil, inMultiple: w.inMultiple)
        }
        try self.init(w)
    }

    public var offline: Bool { !config.causal }

    /// Moves a causal checkpoint to one of its trained look-ahead positions.
    public func setLookahead(_ frames: Int) throws {
        let choices = config.lookaheadFramesChoices
        if choices.isEmpty {
            guard frames == config.lookaheadFrames else {
                throw VoiceIsolatorError.invalidArgument("this checkpoint has a fixed look-ahead")
            }
            return
        }
        guard let index = choices.firstIndex(of: frames) else {
            throw VoiceIsolatorError.invalidArgument("look-ahead \(frames) is not among \(choices)")
        }
        lookaheadFrames = frames
        lookaheadBias = lookaheadEmbedding![index]
    }

    func rope(_ frames: Int) -> (MLXArray, MLXArray) {
        if let t = ropeCache[frames] { return t }
        if ropeCache.count > 16 { ropeCache.removeAll() }
        let t = ropeTables(frames: frames, headDim: config.headDim)
        ropeCache[frames] = t
        return t
    }

    public func stft(_ audio: MLXArray) -> MLXArray {
        DSP.stft(audio, window: window, nFFT: config.nFFT, hop: config.hopLength)
    }

    /// Tokens [batch * bands * frames, width] of spectra [batch, frames, bins].
    func encode(_ spectrum: MLXArray) -> MLXArray {
        let batch = spectrum.dim(0)
        let frames = spectrum.dim(1)
        let bins = spectrum.dim(2)
        var comps = spectrum.view(dtype: .float32).reshaped([batch, frames, 2 * bins])
        comps = MLX.padded(comps, widths: [0, 0, [0, 1]])
        var x = take(comps, gather, axis: 2).reshaped([batch, frames, bands, -1])
        let mean = x.sum(axis: -1, keepDims: true) / counts
        let centered = (x - mean) * valid
        let variance = (centered * centered).sum(axis: -1, keepDims: true) / counts
        x = centered * rsqrt(variance + layerNormEps)
        x = x * encNormW + encNormB
        x = x.transposed(0, 2, 1, 3).asType(dtype)
        var bias = encoderBias
        if let lookaheadBias { bias = bias + lookaheadBias }
        return encoder.addmm(bias, x).reshaped([-1, config.width])
    }

    func makePass(batch: Int, frames: Int, causal: Bool) -> Pass {
        let (cos, sin) = rope(frames)
        return Pass(batch: batch, bands: bands, frames: frames, width: config.width, cos: cos,
                    sin: sin, causal: causal, maxTokens: maxTokens)
    }

    func backbone(_ tokens: MLXArray, batch: Int, frames: Int, causal: Bool) -> MLXArray {
        let p = makePass(batch: batch, frames: frames, causal: causal)
        var x = tokens
        for block in blocks { x = block(x, p) }
        return x
    }

    /// Raw decoder output [batch, frames, bins, 4] of the tokens.
    func decode(_ tokens: MLXArray, batch: Int, frames: Int) -> MLXArray {
        var x = tokens.reshaped([batch, bands, frames, config.width])
        x = x * decScale + decShift
        x = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: layerNormEps)
        x = x * decNormW + decNormB
        x = gelu(decFc1.addmm(decFc1B, x))
        x = decFc2.addmm(decFc2B, x).asType(.float32)
        x = x.transposed(0, 2, 1, 3).reshaped([batch, frames, -1])
        x = take(x, keep, axis: 2)
        return x.reshaped([batch, frames, config.bins, 4])
    }

    /// The complex change of the spectrum that the raw output encodes.
    func correction(_ raw: MLXArray, _ spectrum: MLXArray) -> MLXArray {
        let mag = Float(config.correctionBound) * abs(spectrum)
        let parts = correctionParts([raw, spectrum.realPart(), spectrum.imaginaryPart(), mag])[0]
        return parts.view(dtype: .complex64)[.ellipsis, 0]
    }

    /// Isolates float32 mono audio [batch, samples] in one pass.
    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let batch = audio.dim(0)
        let spectrum = stft(audio)
        let frames = spectrum.dim(1)
        let tokens = backbone(encode(spectrum), batch: batch, frames: frames, causal: !offline)
        var raw = decode(tokens, batch: batch, frames: frames)
        if lookaheadFrames > 0 {
            let index = minimum(MLXArray(0 ..< frames) + lookaheadFrames, frames - 1)
            raw = take(raw, index, axis: 1)
        }
        let delta = DSP.istft(correction(raw, spectrum), window: window, nFFT: config.nFFT,
                              hop: config.hopLength, length: audio.dim(1))
        return audio + delta
    }
}
