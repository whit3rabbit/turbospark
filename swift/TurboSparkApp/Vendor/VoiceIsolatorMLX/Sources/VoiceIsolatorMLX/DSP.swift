import Foundation
import MLX

/// Signal processing: STFT, inverse STFT and Kaiser-windowed sinc resampling.
///
/// The FFTs run on MLX's CPU stream, which uses pocketfft like PyTorch's CPU
/// backend and gives bit-identical spectra for identical frames.
public enum DSP {
    /// Periodic Hann window; PyTorch's float32 bits for n = 2048, otherwise
    /// evaluated in float64 and rounded.
    public static func hannWindow(_ n: Int) -> MLXArray {
        if n == 2048 {
            let text = TorchWindow.hann2048Base64.filter { !$0.isWhitespace }
            if let data = Data(base64Encoded: text), data.count == 4 * n {
                let values = data.withUnsafeBytes { raw in
                    (0 ..< n).map { Float(bitPattern: UInt32(littleEndian: raw.load(
                        fromByteOffset: 4 * $0, as: UInt32.self))) }
                }
                return MLXArray(values, [n])
            }
        }
        let values = (0 ..< n).map { k in
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(k) / Double(n)))
        }
        return MLXArray(values, [n])
    }

    public static func frameCount(samples: Int, hop: Int) -> Int { 1 + samples / hop }

    /// Centered STFT of audio [batch, samples]: complex [batch, frames, bins].
    /// Reflect padding by nFFT / 2 (zeros when the clip is not longer).
    public static func stft(_ audio: MLXArray, window: MLXArray, nFFT: Int, hop: Int) -> MLXArray {
        let batch = audio.dim(0)
        let n = audio.dim(1)
        let pad = nFFT / 2
        let s = StreamOrDevice.cpu
        let padded: MLXArray
        if n > pad {
            let left = audio[0..., 1 ..< (pad + 1)][0..., .stride(by: -1)]
            let right = audio[0..., (n - pad - 1) ..< (n - 1)][0..., .stride(by: -1)]
            padded = concatenated([left, audio, right], axis: 1, stream: s)
        } else {
            padded = MLX.padded(audio, widths: [0, [pad, pad]], stream: s)
        }
        let frames = frameCount(samples: n, hop: hop)
        let segments = asStrided(padded, [batch, frames, nFFT], strides: [n + 2 * pad, hop, 1],
                                 stream: s)
        return rfft(multiply(segments, window, stream: s), stream: s)
    }

    /// Sums frames [batch, count, n] placed `hop` apart: [batch, length].
    static func overlapAdd(_ frames: MLXArray, hop: Int, stream: StreamOrDevice = .cpu) -> MLXArray {
        let batch = frames.dim(0)
        let count = frames.dim(1)
        let n = frames.dim(2)
        let pieces = (n + hop - 1) / hop
        let full = MLX.padded(frames, widths: [0, 0, [0, pieces * hop - n]], stream: stream)
            .reshaped([batch, count, pieces, hop], stream: stream)
        var out: MLXArray? = nil
        for j in 0 ..< pieces {
            let part = MLX.padded(full[0..., 0..., j], widths: [0, [j, pieces - 1 - j], 0],
                                  stream: stream)
            out = out.map { add($0, part, stream: stream) } ?? part
        }
        return out!.reshaped([batch, (count + pieces - 1) * hop], stream: stream)
    }

    nonisolated(unsafe) private static var envelopes: [String: MLXArray] = [:]
    private static let envelopeLock = NSLock()

    /// Overlap-added squared window [length] of `frames` frames.
    static func windowEnvelope(_ window: MLXArray, frames: Int, hop: Int) -> MLXArray {
        let key = "\(frames)/\(hop)/\(window.dim(0))"
        envelopeLock.lock()
        defer { envelopeLock.unlock() }
        if let env = envelopes[key] { return env }
        let w2 = (window * window).reshaped([1, 1, window.dim(0)])
        let env = overlapAdd(broadcast(w2, to: [1, frames, window.dim(0)]), hop: hop)[0]
        eval(env)
        if envelopes.count > 64 { envelopes.removeAll() }
        envelopes[key] = env
        return env
    }

    /// Inverse of `stft` (torch.istft semantics): audio [batch, length].
    public static func istft(_ spectrum: MLXArray, window: MLXArray, nFFT: Int, hop: Int,
                             length: Int) -> MLXArray {
        let s = StreamOrDevice.cpu
        let frames = multiply(irfft(spectrum, n: nFFT, stream: s), window, stream: s)
        let ola = overlapAdd(frames, hop: hop, stream: s)
        let env = windowEnvelope(window, frames: frames.dim(1), hop: hop)
        let start = nFFT / 2
        return divide(ola[0..., start ..< (start + length)], env[start ..< (start + length)],
                      stream: s)
    }

    // MARK: - Resampling (torchaudio kaiser_best)

    static let lowpassFilterWidth = 64.0
    static let rolloff = 0.9475937167399596
    static let beta = 14.769656459379492

    /// Modified Bessel function of the first kind, order 0 (power series).
    static func besselI0(_ x: Double) -> Double {
        var sum = 1.0
        var term = 1.0
        let q = x * x / 4
        var k = 1.0
        while true {
            term *= q / (k * k)
            sum += term
            if term < sum * 1e-17 { break }
            k += 1
        }
        return sum
    }

    /// torchaudio's sinc_interp_kaiser kernel [new, taps] in float64.
    static func sincKernel(orig: Int, new: Int) -> (kernel: [Float], taps: Int, width: Int) {
        let base = Double(min(orig, new)) * rolloff
        let width = Int((lowpassFilterWidth * Double(orig) / base).rounded(.up))
        let taps = 2 * width + orig
        let i0Beta = besselI0(beta)
        var kernel = [Float](repeating: 0, count: new * taps)
        for phase in 0 ..< new {
            for j in 0 ..< taps {
                var t = (-Double(phase) / Double(new) + Double(j - width) / Double(orig)) * base
                t = Swift.max(-lowpassFilterWidth, Swift.min(lowpassFilterWidth, t))
                let r = t / lowpassFilterWidth
                let window = besselI0(beta * (1 - r * r).squareRoot()) / i0Beta
                let x = t * Double.pi
                let sinc = x == 0 ? 1.0 : Foundation.sin(x) / x
                kernel[phase * taps + j] = Float(sinc * window * (base / Double(orig)))
            }
        }
        return (kernel, taps, width)
    }

    nonisolated(unsafe) private static var kernels: [String: (MLXArray, Int)] = [:]
    private static let kernelLock = NSLock()

    static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

    /// Resamples [..., samples] from `origFreq` to `newFreq` (identity when
    /// equal). The polyphase filter runs as a strided matmul over at most
    /// `maxFrames` input frames at a time.
    public static func resample(_ audio: MLXArray, from origFreq: Int, to newFreq: Int,
                                maxFrames: Int = 1 << 16) -> MLXArray {
        if origFreq == newFreq { return audio }
        let g = gcd(origFreq, newFreq)
        let orig = origFreq / g
        let new = newFreq / g
        let key = "\(orig)/\(new)"
        kernelLock.lock()
        if kernels[key] == nil {
            let (k, taps, width) = sincKernel(orig: orig, new: new)
            kernels[key] = (MLXArray(k, [new, taps]), width)
        }
        let (kernel, width) = kernels[key]!
        kernelLock.unlock()
        let shape = audio.shape
        let length = shape.last!
        var x = audio.reshaped([-1, length]).asType(.float32)
        let rows = x.dim(0)
        x = MLX.padded(x, widths: [0, [width, width + orig]])
        let taps = kernel.dim(1)
        let frames = length / orig + 1
        let total = x.dim(1)
        var outs: [MLXArray] = []
        var f0 = 0
        while f0 < frames {
            let f1 = Swift.min(frames, f0 + maxFrames)
            let seg = asStrided(x, [rows, f1 - f0, taps], strides: [total, orig, 1],
                                offset: f0 * orig)
            outs.append(matmul(seg, kernel.T))
            f0 = f1
        }
        let y = (outs.count == 1 ? outs[0] : concatenated(outs, axis: 1)).reshaped([rows, -1])
        let target = Int((Double(new) * Double(length) / Double(orig)).rounded(.up))
        var outShape = Array(shape.dropLast())
        outShape.append(target)
        return y[0..., ..<target].reshaped(outShape)
    }

    /// Trims the last axis to `n` samples or pads it with trailing zeros.
    public static func fitLength(_ x: MLXArray, _ n: Int) -> MLXArray {
        let have = x.shape.last!
        if have >= n { return x[.ellipsis, ..<n] }
        var widths = [IntOrPair](repeating: 0, count: x.ndim)
        widths[x.ndim - 1] = [0, n - have]
        return MLX.padded(x, widths: widths)
    }
}
