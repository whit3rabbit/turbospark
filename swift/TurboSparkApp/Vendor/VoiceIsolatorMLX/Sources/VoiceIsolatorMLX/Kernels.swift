import Foundation
import MLX

/// Mapping of a sequence axis onto the canonical token order
/// `[batch, bands, frames]`:
///
///     token = (row / inner) * outerStride + (row % inner) * innerStride
///             + position * seqStride
///
/// so no layer transposes the token stream between the time and band axes.
public struct SeqLayout {
    public let rows: Int
    public let length: Int
    public let tokens: Int
    let params: MLXArray

    init(rows: Int, length: Int, inner: Int, outerStride: Int, innerStride: Int,
         seqStride: Int) {
        self.rows = rows
        self.length = length
        self.tokens = rows * length
        self.params = MLXArray(
            [Int32(length), Int32(inner), Int32(outerStride), Int32(innerStride),
             Int32(seqStride), Int32(rows * length)], [6])
    }

    /// Rows of consecutive tokens: (batch, band) pairs along time.
    public static func time(rows: Int, frames: Int) -> SeqLayout {
        SeqLayout(rows: rows, length: frames, inner: 1, outerStride: frames,
                  innerStride: 0, seqStride: 1)
    }

    /// (batch, frame) rows across the bands of `[batch, bands, frames]`.
    public static func band(batch: Int, bands: Int, frames: Int) -> SeqLayout {
        SeqLayout(rows: batch * frames, length: bands, inner: frames,
                  outerStride: bands * frames, innerStride: 1, seqStride: frames)
    }
}

/// The custom Metal kernels (sources shared with the Python port).
enum Kernels {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [String: MLXFast.MLXFastKernel] = [:]

    static func metalType(_ dtype: DType) -> String {
        switch dtype {
        case .float16: return "half"
        case .bfloat16: return "bfloat16_t"
        default: return "float"
        }
    }

    /// The kernel `name` for activations of `dtype` (`{T}` in the source; `{Y}`
    /// is `outDType`, default `dtype`); arithmetic is float32 in every case.
    private static func build(_ name: String, inputs: [String], outputs: [String],
                              source: String, header: String = "", dtype: DType = .float32,
                              outDType: DType? = nil) -> MLXFast.MLXFastKernel {
        let source = source.replacingOccurrences(of: "{T}", with: metalType(dtype))
            .replacingOccurrences(of: "{Y}", with: metalType(outDType ?? dtype))
        let name = "\(name)_\(metalType(dtype))_\(metalType(outDType ?? dtype))"
        lock.lock()
        defer { lock.unlock() }
        if let kernel = cache[name] { return kernel }
        let kernel = MLXFast.metalKernel(
            name: name, inputNames: inputs, outputNames: outputs, source: source,
            header: KernelSources.token + header, ensureRowContiguous: true)
        cache[name] = kernel
        return kernel
    }

    /// Exact GELU, `0.5 x (1 + erf(x / sqrt(2)))`, in one pass with float32
    /// arithmetic (a kernel rather than a `compile` fusion, so it does not
    /// depend on the compiled-function locking of a given mlx-swift release).
    static func gelu(_ x: MLXArray) -> MLXArray {
        let kernel = build("vi_gelu", inputs: ["x"], outputs: ["y"], source: KernelSources.gelu,
                           header: KernelSources.erf, dtype: x.dtype)
        return kernel([x], grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
                      outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }

    /// q, k, v [rows, heads, L, headDim] of qkv [tokens, 3 * width]; q and k
    /// rotated by the RoPE tables (indexed by position) when given.
    static func splitHeads(_ qkv: MLXArray, _ layout: SeqLayout, heads: Int, headDim: Int,
                           cos: MLXArray? = nil, sin: MLXArray? = nil)
        -> (MLXArray, MLXArray, MLXArray)
    {
        let kernel: MLXFast.MLXFastKernel
        let inputs: [MLXArray]
        if let cos, let sin {
            kernel = build("vi_split_heads_rope", inputs: ["qkv", "cos_t", "sin_t", "params"],
                           outputs: ["q", "k", "v"],
                           source: KernelSources.split.replacingOccurrences(
                               of: "{rope}", with: KernelSources.rope), dtype: qkv.dtype)
            inputs = [qkv, cos, sin, layout.params]
        } else {
            kernel = build("vi_split_heads", inputs: ["qkv", "params"], outputs: ["q", "k", "v"],
                           source: KernelSources.split.replacingOccurrences(of: "{rope}", with: ""),
                           dtype: qkv.dtype)
            inputs = [qkv, layout.params]
        }
        let shape = [layout.rows, heads, layout.length, headDim]
        let out = kernel(
            inputs, template: [("H", heads), ("D", headDim)],
            grid: (headDim / 2, heads, layout.tokens), threadGroup: (headDim / 2, heads, 1),
            outputShapes: [shape, shape, shape], outputDTypes: [qkv.dtype, qkv.dtype, qkv.dtype])
        return (out[0], out[1], out[2])
    }

    /// [rows, H, L, D] back to [tokens, H * D]; exclusive self attention when
    /// `value` is given.
    static func mergeHeads(_ o: MLXArray, _ layout: SeqLayout, value: MLXArray? = nil) -> MLXArray {
        let heads = o.dim(1)
        let headDim = o.dim(3)
        let exclusive = value != nil
        let kernel = build(
            exclusive ? "vi_merge_heads_excl" : "vi_merge_heads",
            inputs: exclusive ? ["o", "v", "params"] : ["o", "params"], outputs: ["out"],
            source: KernelSources.merge.replacingOccurrences(
                of: "{exclusive}", with: exclusive ? KernelSources.exclusive : ""), dtype: o.dtype)
        let inputs = exclusive ? [o, value!, layout.params] : [o, layout.params]
        return kernel(
            inputs, template: [("H", heads), ("D", headDim)],
            grid: (32, heads, layout.tokens), threadGroup: (32, heads, 1),
            outputShapes: [[layout.tokens, heads * headDim]], outputDTypes: [o.dtype])[0]
    }

    /// SiLU of the depthwise causal (direction 0) / anti-causal (direction 1)
    /// convolution of u [tokens, dirs * C]; returns uc [dirs, tokens, C].
    static func convSiLU(_ u: MLXArray, weight: MLXArray, bias: MLXArray, _ layout: SeqLayout,
                         history: MLXArray? = nil) -> MLXArray {
        let dirs = weight.dim(0)
        let channels = weight.dim(1)
        let taps = weight.dim(2)
        let hist = history != nil
        let kernel = build(
            "vi_conv_silu_\(hist ? 1 : 0)",
            inputs: ["u", "weight", "bias", "params"] + (hist ? ["history"] : []),
            outputs: ["uc"],
            source: KernelSources.conv.replacingOccurrences(
                of: "{history}", with: hist ? KernelSources.convHistory : ""), dtype: u.dtype)
        var inputs = [u, weight, bias, layout.params]
        if let history { inputs.append(history) }
        let runs = (layout.length + 7) / 8
        return kernel(
            inputs, template: [("KW", taps), ("DIRS", dirs)],
            grid: (channels, layout.rows * runs, dirs), threadGroup: (min(channels, 256), 1, 1),
            outputShapes: [[dirs, layout.tokens, channels]], outputDTypes: [u.dtype])[0]
    }

    /// Gated selective scans of every direction (direction 1 in reverse); the
    /// step size is computed in the kernel from the dt rows of `dbc`.
    static func selectiveScan(
        uc: MLXArray, z: MLXArray, dbc: MLXArray, A: MLXArray, D: MLXArray, wDt: MLXArray,
        bDt: MLXArray, _ layout: SeqLayout, state: MLXArray? = nil, returnState: Bool = false,
        outDType: DType? = nil
    ) -> (MLXArray, MLXArray?) {
        let outDType = outDType ?? uc.dtype
        let dirs = A.dim(0)
        let channels = A.dim(1)
        let states = A.dim(2)
        let rank = wDt.dim(2)
        let carryIn = state != nil
        var inputs = ["uc", "z", "dbc", "A", "D", "Wdt", "bdt", "params"]
        if carryIn { inputs.append("h0") }
        let source = KernelSources.scan
            .replacingOccurrences(
                of: "{h_init}", with: carryIn ? "h0[(size_t(row) * C + c) * S + s]" : "0.0f")
            .replacingOccurrences(of: "{h_store}", with: returnState ? KernelSources.scanStore : "")
        let kernel = build(
            "vi_scan_\(carryIn ? 1 : 0)\(returnState ? 1 : 0)", inputs: inputs,
            outputs: returnState ? ["y", "h_out"] : ["y"], source: source,
            header: KernelSources.softplus, dtype: uc.dtype, outDType: outDType)
        var args = [uc, z, dbc, A, D, wDt, bDt, layout.params]
        if let state { args.append(state) }
        var shapes = [[layout.tokens, dirs * channels]]
        if returnState { shapes.append([layout.rows, channels, states]) }
        let out = kernel(
            args, template: [("S", states), ("R", rank), ("DIRS", dirs)],
            grid: (channels, layout.rows, dirs), threadGroup: (min(channels, 256), 1, 1),
            outputShapes: shapes, outputDTypes: Array([outDType, DType.float32][..<shapes.count]))
        return (out[0], returnState ? out[1] : nil)
    }
}
