import Foundation
import MLX
import XCTest

@testable import VoiceIsolatorMLX

final class DSPTests: XCTestCase {
    func testWindowIsThePeriodicHann() {
        let w = DSP.hannWindow(2048).asArray(Float.self)

        var err: Double = 0
        for k in 0 ..< 2048 {
            let exact = 0.5 - 0.5 * cos(2 * Double.pi * Double(k) / 2048)
            err = max(err, abs(Double(w[k]) - exact))
        }

        XCTAssertEqual(w[0], 0)
        XCTAssertEqual(w[1024], 1)
        XCTAssertLessThan(err, 1e-6)
    }

    func testSTFTRoundTrip() {
        for length in [1000, 3 * 48000 + 17] {
            let x = MLXRandom.uniform(low: -0.5, high: 0.5, [1, length])
            let window = DSP.hannWindow(2048)

            let spectrum = DSP.stft(x, window: window, nFFT: 2048, hop: 480)
            let y = DSP.istft(spectrum, window: window, nFFT: 2048, hop: 480, length: length)

            XCTAssertEqual(spectrum.shape, [1, 1 + length / 480, 1025])
            XCTAssertLessThan(abs(y - x).max().item(Float.self), 1e-6, "length \(length)")
        }
    }

    private func tone(_ f: Double, rate: Int, seconds: Double = 2) -> MLXArray {
        let n = Int(seconds * Double(rate))
        return MLXArray((0 ..< n).map { Float(0.5 * sin(2 * .pi * f * Double($0) / Double(rate))) }, [n])
    }

    /// Mean power of the middle 80% relative to a 0.5-amplitude tone, dB.
    private func level(_ y: MLXArray) -> Double {
        let v = y.asArray(Float.self)
        let lo = v.count / 10, hi = v.count - v.count / 10
        var e = 0.0
        for i in lo ..< hi { e += Double(v[i]) * Double(v[i]) }
        return 10 * log10(e / Double(hi - lo) / 0.125)
    }

    func testResamplerPassesThePassbandAndRejectsAliases() {
        let pass = DSP.resample(tone(7000, rate: 48000), from: 48000, to: 16000)
        let alias = DSP.resample(tone(12000, rate: 48000), from: 48000, to: 16000)
        let up = DSP.resample(tone(7000, rate: 16000), from: 16000, to: 48000)

        XCTAssertEqual(pass.dim(0), 32000)
        XCTAssertEqual(up.dim(0), 96000)
        XCTAssertEqual(level(pass), 0, accuracy: 0.01)
        XCTAssertLessThan(level(alias), -100)
        XCTAssertEqual(level(up), 0, accuracy: 0.01)
    }

    func testResampleLengthAndIdentity() {
        let x = MLXRandom.uniform(low: -0.5, high: 0.5, [44100])

        XCTAssertEqual(DSP.resample(x, from: 44100, to: 48000).dim(0), 48000)
        XCTAssertEqual(DSP.resample(x, from: 44100, to: 44100).dim(0), 44100)
    }

    func testBandLayoutCoversEveryBinOnce() {
        let config = Config()
        let edges = config.bandEdges()
        let layout = BandLayout(config: config)

        XCTAssertEqual(edges.count, 46)
        XCTAssertEqual(edges.first?.0, 0)
        XCTAssertEqual(edges.last?.1, 1025)
        for (a, b) in zip(edges, edges.dropFirst()) { XCTAssertEqual(a.1, b.0) }
        XCTAssertEqual(layout.keep.count, 4 * 1025)
    }

    func testRenderWindowsCrossfadeToUnitWeight() {
        let n = 30 * 48000
        var total = [Float](repeating: 0, count: n)

        for (start, w) in chunkPlan(n: n, chunk: 8 * 48000, overlap: 2 * 48000) {
            for i in 0 ..< w.count { total[start + i] += w[i] }
        }

        XCTAssertEqual(total.min()!, 1, accuracy: 1e-6)
        XCTAssertEqual(total.max()!, 1, accuracy: 1e-6)
    }

    func testMonoAveragesChannels() {
        XCTAssertEqual(toMono([[1, 0.5], [0, -0.5]]), [0.5, 0])
    }
}
