import Accelerate
import Foundation
import MLX

/// Quantization of a converted checkpoint: affine, float32 scales and biases.
public struct Quantization: Codable, Sendable, Equatable {
    public var bits: Int
    public var groupSize: Int

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }

    public init(bits: Int = 8, groupSize: Int = 32) {
        self.bits = bits
        self.groupSize = groupSize
    }
}

/// A checkpoint in the MLX layout: config, arrays and their quantization.
public struct LoadedWeights {
    public let config: Config
    public let arrays: [String: MLXArray]
    public let quantization: Quantization?
    public let inMultiple: Int
}

/// Checkpoint loading and conversion. The converted layout packs the ragged
/// per-band encoders and decoder heads into padded batched tensors, stacks the
/// Mamba directions, folds the Mamba output projections (in float64) and the
/// load-time constants, and optionally stores the linear layers as 8-bit
/// affine-quantized weights. It is the same file format the Python port writes.
public enum Weights {
    public static let format = "voice-isolator-mlx"
    public static let formatVersion = 3
    static let statePrefixes = ["_forward_module.", "module.", "model.", "isolator."]
    static let trainingOnlyPrefixes = ["discriminator.", "disc.", "ema.", "optimizer.",
                                       "scheduler.", "enhance_decoder.", "decision_head."]
    static let quantizedLayers: Set<String> = ["qkv", "proj", "fc1", "fc2", "in_u", "in_z",
                                               "x_proj", "out", "skip", "encoder"]

    /// Loads a converted MLX file, or converts a released checkpoint in memory.
    public static func load(_ url: URL) throws -> LoadedWeights {
        let (arrays, meta) = try loadArraysAndMetadata(url: url)
        if meta["format"] == format {
            guard Int(meta["format_version"] ?? "") == formatVersion else {
                throw VoiceIsolatorError.invalidCheckpoint("unsupported format version")
            }
            let config = try Config.parse(json: meta["config"] ?? "{}")
            var quant: Quantization? = nil
            if let q = meta["quantization"], q != "null", let data = q.data(using: .utf8) {
                quant = try JSONDecoder().decode(Quantization.self, from: data)
            }
            return LoadedWeights(config: config, arrays: arrays, quantization: quant,
                                 inMultiple: Int(meta["in_pad_multiple"] ?? "1") ?? 1)
        }
        guard let json = meta["config"] else {
            throw VoiceIsolatorError.invalidCheckpoint("safetensors metadata lacks 'config'")
        }
        let config = try Config.parse(json: json)
        let packed = try convertState(config: config, state: arrays, inMultiple: 1)
        return LoadedWeights(config: config, arrays: packed, quantization: nil, inMultiple: 1)
    }

    /// Weights kept float32 in half-precision checkpoints: the scan's
    /// parameters, the Mamba output projections (their pre-norm outputs exceed
    /// float16's range) and the STFT window.
    static let float32Only = [".A", ".D", ".dt.weight", ".dt.bias", ".out.weight", ".out.bias",
                              ".skip.weight", "stft.window"]

    /// Casts floating-point weights to `dtype`, except `float32Only`.
    public static func cast(_ arrays: [String: MLXArray], to dtype: DType) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in arrays {
            let isFloat = [DType.float32, .float16, .bfloat16].contains(v.dtype)
            let keep = !isFloat || float32Only.contains(where: { k.hasSuffix($0) })
            out[k] = keep ? v : v.asType(dtype)
        }
        eval(Array(out.values))
        return out
    }

    /// Converts a released checkpoint to an MLX file: 8-bit when
    /// `quantization`, else float32 or `dtype` (.float16 / .bfloat16).
    public static func convert(src: URL, dst: URL, quantization: Quantization?,
                               dtype: DType? = nil) throws {
        let (arrays, meta) = try loadArraysAndMetadata(url: src)
        guard let json = meta["config"] else {
            throw VoiceIsolatorError.invalidCheckpoint("safetensors metadata lacks 'config'")
        }
        let config = try Config.parse(json: json)
        let inMultiple = quantization?.groupSize ?? 1
        var packed = try convertState(config: config, state: arrays, inMultiple: inMultiple)
        if let quantization {
            packed = quantize(packed, quantization)
        } else if let dtype {
            packed = cast(packed, to: dtype)
        }
        var qjson = "null"
        if let quantization {
            qjson = "{\"bits\": \(quantization.bits), \"group_size\": \(quantization.groupSize)}"
        }
        let metadata = [
            "format": format, "format_version": "\(formatVersion)", "config": config.json(),
            "quantization": qjson, "in_pad_multiple": "\(inMultiple)",
        ]
        try save(arrays: packed, metadata: metadata, url: dst)
    }

    /// 8-bit affine quantization of the linear layers' matrices.
    public static func quantize(_ arrays: [String: MLXArray], _ q: Quantization)
        -> [String: MLXArray]
    {
        var out: [String: MLXArray] = [:]
        for (name, array) in arrays {
            if name.hasSuffix(".weight"), array.ndim >= 2,
                array.dim(-1) % q.groupSize == 0
            {
                let prefix = String(name.dropLast(".weight".count))
                let layer = prefix.split(separator: ".").last.map(String.init) ?? prefix
                if quantizedLayers.contains(layer) {
                    let (wq, scales, biases) = quantized(array, groupSize: q.groupSize,
                                                         bits: q.bits)
                    out["\(prefix).weight"] = wq
                    out["\(prefix).scales"] = scales
                    out["\(prefix).biases"] = biases!
                    continue
                }
            }
            out[name] = array
        }
        eval(Array(out.values))
        return out
    }

    // MARK: - Conversion of the released layout

    private struct State {
        var arrays: [String: MLXArray]

        mutating func pop(_ key: String) throws -> MLXArray {
            guard let a = arrays.removeValue(forKey: key) else {
                throw VoiceIsolatorError.invalidCheckpoint("missing tensor \(key)")
            }
            return a.asType(.float32)
        }

        mutating func floats(_ key: String) throws -> [Float] {
            floatsOf(try pop(key))
        }

        func floatsOf(_ a: MLXArray) -> [Float] {
            let f = a.asType(.float32)
            eval(f)
            return f.asArray(Float.self)
        }
    }

    static func stripPrefixes(_ state: [String: MLXArray]) -> [String: MLXArray] {
        var state = state
        for p in statePrefixes where state.keys.contains(where: { $0.hasPrefix(p) }) {
            var next: [String: MLXArray] = [:]
            for (k, v) in state { next[k.hasPrefix(p) ? String(k.dropFirst(p.count)) : k] = v }
            state = next
        }
        return state
    }

    /// `a @ b` in float64 (row-major [m, k] x [k, n]).
    static func dgemm(_ a: [Double], _ b: [Double], m: Int, k: Int, n: Int) -> [Double] {
        var c = [Double](repeating: 0, count: m * n)
        a.withUnsafeBufferPointer { pa in
            b.withUnsafeBufferPointer { pb in
                c.withUnsafeMutableBufferPointer { pc in
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n),
                                Int32(k), 1.0, pa.baseAddress!, Int32(k), pb.baseAddress!,
                                Int32(n), 0.0, pc.baseAddress!, Int32(n))
                }
            }
        }
        return c
    }

    static func convertState(config: Config, state raw: [String: MLXArray], inMultiple: Int)
        throws -> [String: MLXArray]
    {
        var state = State(arrays: stripPrefixes(raw))
        let layout = BandLayout(config: config, inMultiple: inMultiple)
        let bands = layout.edges.count
        let width = config.width
        let pad = layout.inPad
        var out: [String: MLXArray] = [:]

        func move(_ dst: String, _ src: String) throws { out[dst] = try state.pop(src) }
        func linear(_ dst: String, _ src: String, bias: Bool = true) throws {
            try move("\(dst).weight", "\(src).weight")
            if bias { try move("\(dst).bias", "\(src).bias") }
        }

        // encoders, the band embedding folded into their biases
        var normW = [Float](repeating: 0, count: bands * pad)
        var normB = [Float](repeating: 0, count: bands * pad)
        var encW = [Float](repeating: 0, count: bands * width * pad)
        var encB = [Float](repeating: 0, count: bands * width)
        let embedding = try state.floats("band_embedding")
        for (band, (s, e)) in layout.edges.enumerated() {
            let n = 2 * (e - s)
            let w = try state.floats("encoders.\(band).0.weight")
            let b = try state.floats("encoders.\(band).0.bias")
            let lw = try state.floats("encoders.\(band).1.weight")
            let lb = try state.floats("encoders.\(band).1.bias")
            for j in 0 ..< n {
                normW[band * pad + j] = w[j]
                normB[band * pad + j] = b[j]
            }
            for o in 0 ..< width {
                for j in 0 ..< n { encW[(band * width + o) * pad + j] = lw[o * n + j] }
                encB[band * width + o] = lb[o] + embedding[band * width + o]
            }
        }
        out["encoder.norm.weight"] = MLXArray(normW, [bands, pad])
        out["encoder.norm.bias"] = MLXArray(normB, [bands, pad])
        out["encoder.weight"] = MLXArray(encW, [bands, width, pad])
        out["encoder.bias"] = MLXArray(encB, [bands, width])
        if !config.lookaheadFramesChoices.isEmpty {
            try move("lookahead_embedding", "lookahead_embedding.weight")
        }

        for (i, kind) in config.blockTypes.enumerated() {
            let src = "blocks.\(i)"
            if kind == "transformer" {
                let t = "\(src).time_transformer"
                try linear("\(src).time.norm1", "\(t).norm1")
                try linear("\(src).time.qkv", "\(t).attn.qkv")
                try linear("\(src).time.proj", "\(t).attn.proj")
                try linear("\(src).time.norm2", "\(t).norm2")
                try linear("\(src).time.fc1", "\(t).ffn.0")
                try linear("\(src).time.fc2", "\(t).ffn.3")
                let b = "\(src).band_transformer"
                try linear("\(src).band.norm1", "\(b).norm1")
                try move("\(src).band.qkv.weight", "\(b).self_attn.in_proj_weight")
                try move("\(src).band.qkv.bias", "\(b).self_attn.in_proj_bias")
                try linear("\(src).band.proj", "\(b).self_attn.out_proj")
                try linear("\(src).band.norm2", "\(b).norm2")
                try linear("\(src).band.fc1", "\(b).linear1")
                try linear("\(src).band.fc2", "\(b).linear2")
            } else {
                for axis in ["time", "freq"] {
                    try mambaAxis(&out, "\(src).\(axis)", &state, "\(src).\(axis)_mamba", config)
                }
            }
        }

        // decoder: FiLM controls [1, 0] folded to a scale and shift
        let condW = try state.floats("denoise_decoder.condition.weight")
        let condB = try state.floats("denoise_decoder.condition.bias")
        let film = (0 ..< 2 * width).map { condW[$0 * 2] + condB[$0] }
        out["decoder.scale"] = MLXArray((0 ..< width).map { Float(1) + film[$0] }, [width])
        out["decoder.shift"] = MLXArray(Array(film[width...]), [width])
        let outPad = layout.outPad
        var dnW = [Float](), dnB = [Float](), f1W = [Float](), f1B = [Float]()
        var f2W = [Float](repeating: 0, count: bands * outPad * width)
        var f2B = [Float](repeating: 0, count: bands * outPad)
        for (band, (s, e)) in layout.edges.enumerated() {
            let h = "denoise_decoder.heads.\(band)"
            dnW += try state.floats("\(h).0.weight")
            dnB += try state.floats("\(h).0.bias")
            f1W += try state.floats("\(h).1.weight")
            f1B += try state.floats("\(h).1.bias")
            let w2 = try state.floats("\(h).3.weight")
            let b2 = try state.floats("\(h).3.bias")
            let n = 4 * (e - s)
            for o in 0 ..< n {
                for j in 0 ..< width { f2W[(band * outPad + o) * width + j] = w2[o * width + j] }
                f2B[band * outPad + o] = b2[o]
            }
        }
        out["decoder.norm.weight"] = MLXArray(dnW, [bands, width])
        out["decoder.norm.bias"] = MLXArray(dnB, [bands, width])
        out["decoder.fc1.weight"] = MLXArray(f1W, [bands, width, width])
        out["decoder.fc1.bias"] = MLXArray(f1B, [bands, width])
        out["decoder.fc2.weight"] = MLXArray(f2W, [bands, outPad, width])
        out["decoder.fc2.bias"] = MLXArray(f2B, [bands, outPad])
        out["stft.window"] = DSP.hannWindow(config.nFFT)

        let spare = state.arrays.keys.filter { key in
            !trainingOnlyPrefixes.contains(where: { key.hasPrefix($0) })
        }
        if !spare.isEmpty {
            throw VoiceIsolatorError.invalidCheckpoint("unconverted keys: \(spare.prefix(8))")
        }
        eval(Array(out.values))
        return out
    }

    /// One Mamba axis: directions stacked, output projections folded:
    /// `norm(output_proj(cat_d(out_proj_d(y_d) + x)))` equals
    /// `norm(sum_d (P_d O_d) y_d + (sum_d P_d) x + b)`.
    private static func mambaAxis(_ out: inout [String: MLXArray], _ dst: String,
                                  _ state: inout State, _ src: String, _ config: Config) throws
    {
        var blocks = ["\(src).forward_blocks"]
        if state.arrays["\(src).backward_blocks.in_proj.weight"] != nil {
            blocks.append("\(src).backward_blocks")
        }
        let inner = config.inner
        let width = config.width
        var inU: [MLXArray] = [], inZ: [MLXArray] = [], conv: [MLXArray] = []
        var convB: [MLXArray] = [], xProj: [MLXArray] = [], dtW: [MLXArray] = []
        var dtB: [MLXArray] = [], A: [MLXArray] = [], D: [MLXArray] = []
        var outs: [[Double]] = []
        for b in blocks {
            let w = try state.pop("\(b).in_proj.weight")
            inU.append(w[..<inner])
            inZ.append(w[inner...])
            conv.append(try state.pop("\(b).conv1d.weight")[0..., 0, 0...])
            convB.append(try state.pop("\(b).conv1d.bias"))
            xProj.append(try state.pop("\(b).x_proj.weight"))
            dtW.append(try state.pop("\(b).dt_proj.weight"))
            dtB.append(try state.pop("\(b).dt_proj.bias"))
            let aLog = try state.pop("\(b).A_log")
            let shape = aLog.shape
            A.append(MLXArray(state.floatsOf(aLog).map { Float(-Foundation.exp(Double($0))) },
                              shape))
            D.append(try state.pop("\(b).D"))
            outs.append(try state.floats("\(b).out_proj.weight").map(Double.init))
        }
        out["\(dst).in_u.weight"] = concatenated(inU, axis: 0)
        out["\(dst).in_z.weight"] = concatenated(inZ, axis: 0)
        out["\(dst).conv.weight"] = stacked(conv)
        out["\(dst).conv.bias"] = stacked(convB)
        out["\(dst).x_proj.weight"] = stacked(xProj)
        out["\(dst).dt.weight"] = stacked(dtW)
        out["\(dst).dt.bias"] = stacked(dtB)
        out["\(dst).A"] = stacked(A)
        out["\(dst).D"] = stacked(D)
        let proj = try state.floats("\(src).output_proj.weight").map(Double.init)
        let projIn = proj.count / width
        let dirs = blocks.count
        var fold = [Float](repeating: 0, count: width * dirs * inner)
        var skip = [Double](repeating: 0, count: width * width)
        for d in 0 ..< dirs {
            var part = [Double](repeating: 0, count: width * width)
            for r in 0 ..< width {
                for c in 0 ..< width {
                    part[r * width + c] = proj[r * projIn + d * width + c]
                    skip[r * width + c] += proj[r * projIn + d * width + c]
                }
            }
            let product = dgemm(part, outs[d], m: width, k: width, n: inner)
            for r in 0 ..< width {
                for c in 0 ..< inner {
                    fold[r * dirs * inner + d * inner + c] = Float(product[r * inner + c])
                }
            }
        }
        out["\(dst).out.weight"] = MLXArray(fold, [width, dirs * inner])
        out["\(dst).out.bias"] = try state.pop("\(src).output_proj.bias")
        out["\(dst).skip.weight"] = MLXArray(skip.map { Float($0) }, [width, width])
        out["\(dst).norm.weight"] = try state.pop("\(src).norm.weight")
        out["\(dst).norm.bias"] = try state.pop("\(src).norm.bias")
    }
}
