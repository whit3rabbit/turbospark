import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// Larger families: SenseVoice, Paraformer, Parakeet ja, PocketTTS.
func runBig(modelsRoot: URL, audioDir: URL, outDir: URL) async {
    let fm = FileManager.default
    func p(_ s: String) { print(s); fflush(stdout) }
    func read(_ name: String) throws -> [Float] {
        let file = try AVAudioFile(forReading: audioDir.appendingPathComponent(name))
        let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: buf)
        return Array(UnsafeBufferPointer(start: buf.floatChannelData![0], count: Int(buf.frameLength)))
    }
    func to16k(_ x: [Float], from rate: Double) throws -> [Float] {
        let src = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1)!
        let dst = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1)!
        let inBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(x.count))!
        inBuf.frameLength = inBuf.frameCapacity
        x.withUnsafeBufferPointer { inBuf.floatChannelData![0].update(from: $0.baseAddress!, count: x.count) }
        let conv = AVAudioConverter(from: src, to: dst)!
        let outBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: AVAudioFrameCount(Double(x.count) * 16_000 / rate) + 1024)!
        var fed = false
        var err: NSError?
        conv.convert(to: outBuf, error: &err) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return inBuf
        }
        if let err { throw err }
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
    }
    func clock<T>(_ label: String, _ body: () async throws -> T) async rethrows -> T {
        let t0 = Date()
        let r = try await body()
        p(String(format: "  %@: %.3f s", label, Date().timeIntervalSince(t0)))
        return r
    }

    var sense: SenseVoiceManager?
    p("SenseVoice int8")
    do {
        let models = try await clock("load") {
            try SenseVoiceModels.load(from: modelsRoot.appendingPathComponent("sensevoice"), precision: .int8)
        }
        let m = SenseVoiceManager(models: models)
        sense = m
        for clip in ["a.wav", "zh.wav", "ja.wav"] {
            let samples = try read(clip)
            let text = try await clock("transcribe \(clip)") { try await m.transcribe(audio: samples) }
            p("    \(clip): \(text)")
        }
    } catch { p("  SENSEVOICE FAIL: \(error)") }

    p("Paraformer zh int8")
    do {
        let models = try await clock("load") {
            try ParaformerModels.load(from: modelsRoot.appendingPathComponent("paraformer"), precision: .int8)
        }
        let m = ParaformerManager(models: models)
        let text = try await clock("transcribe zh.wav") { try await m.transcribe(audio: try read("zh.wav")) }
        p("    zh.wav: \(text)")
    } catch { p("  PARAFORMER FAIL: \(error)") }

    p("Parakeet ja (TDT)")
    do {
        let d = modelsRoot.appendingPathComponent("parakeet-ja")
        let cfg = AsrModels.defaultConfiguration()
        p("  computeUnits=\(cfg.computeUnits.rawValue)")
        let cpu = MLModelConfiguration()
        cpu.computeUnits = .cpuOnly
        let t0 = Date()
        let pre = try MLModel(contentsOf: d.appendingPathComponent("Preprocessor.mlmodelc"), configuration: cpu)
        let enc = try MLModel(contentsOf: d.appendingPathComponent("Encoder.mlmodelc"), configuration: cfg)
        let dec = try MLModel(contentsOf: d.appendingPathComponent("Decoderv2.mlmodelc"), configuration: cfg)
        let joint = try MLModel(contentsOf: d.appendingPathComponent("Jointerv2.mlmodelc"), configuration: cfg)
        p(String(format: "  load: %.3f s", Date().timeIntervalSince(t0)))
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: d.appendingPathComponent("vocab.json")))
        var vocab: [Int: String] = [:]
        if let arr = json as? [String] { for (i, t) in arr.enumerated() { vocab[i] = t } }
        if let dict = json as? [String: String] { for (k, v) in dict { if let i = Int(k) { vocab[i] = v } } }
        let models = AsrModels(
            encoder: enc, preprocessor: pre, decoder: dec, joint: joint,
            configuration: cfg, vocabulary: vocab, version: .tdtJa)
        let asr = AsrManager(config: .default, models: models)
        var state = TdtDecoderState.make(decoderLayers: AsrModelVersion.tdtJa.decoderLayers)
        let samples = try read("ja.wav")
        let r = try await clock("transcribe ja.wav") { try await asr.transcribe(samples, decoderState: &state) }
        p("    ja.wav: \(r.text) (tokens with timing: \(r.tokenTimings?.count ?? 0))")
    } catch { p("  PARAKEET JA FAIL: \(error)") }

    p("PocketTTS english v2.1 fp16 gpu")
    do {
        let m = PocketTtsManager(language: .english, directory: modelsRoot.appendingPathComponent("pocket"), precision: .fp16, placement: .gpu)
        try await clock("initialize") { try await m.initialize() }
        let text = "Good morning. This sentence was synthesized on device with Core ML."
        let r = try await clock("first synth") { try await m.synthesizeDetailed(text: text) }
        _ = try await clock("second synth") { try await m.synthesizeDetailed(text: text) }
        let rms = sqrt(r.samples.reduce(0) { $0 + $1 * $1 } / Float(max(1, r.samples.count)))
        p("  samples=\(r.samples.count) frames=\(r.frameCount) finite=\(r.samples.allSatisfy(\.isFinite)) rms=\(rms)")
        try r.audio.write(to: outDir.appendingPathComponent("pocket.wav"))
        if let sense {
            let back = try await sense.transcribe(audio: try to16k(r.samples, from: 24_000))
            p("  round trip via SenseVoice: \(back)")
        }
    } catch { p("  POCKET FAIL: \(error)") }

    if let sense, let file = try? AVAudioFile(forReading: outDir.appendingPathComponent("kokoro.wav")) {
        let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try? file.read(into: buf)
        let x = Array(UnsafeBufferPointer(start: buf.floatChannelData![0], count: Int(buf.frameLength)))
        if let back = try? await sense.transcribe(audio: try to16k(x, from: file.processingFormat.sampleRate)) {
            p("Kokoro round trip via SenseVoice: \(back)")
        }
    }
    _ = fm
}
