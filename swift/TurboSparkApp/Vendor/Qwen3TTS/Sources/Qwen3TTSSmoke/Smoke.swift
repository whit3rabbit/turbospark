import AVFoundation
import Foundation
import MLX
import Qwen3TTS

@main struct Smoke {
    static func main() async throws {
        let args = CommandLine.arguments
        if args.count > 1 && args[1] == "breeze" { return try await breeze(Array(args.dropFirst(2))) }
        guard args.count >= 3 else {
            throw NSError(domain: "Usage: Qwen3TTSSmoke MODEL OUTPUT.wav [REFERENCE_AUDIO] [REFERENCE_TEXT] [TARGET_TEXT] [VOICE]", code: 1)
        }
        let folder = URL(fileURLWithPath: args[1])
        let descriptor = try Qwen3TTSCheckpoint(data: Data(contentsOf: folder.appendingPathComponent("config.json")))
        let reference = args.count > 3 && !args[3].isEmpty ? try readReference(URL(fileURLWithPath: args[3])) : nil
        let transcript = args.count > 4 && !args[4].isEmpty ? args[4] : nil
        let text = args.count > 5 ? args[5] : "The sky is clear. We are ready for the next flight."
        let voice = descriptor.variant == .base ? nil : args.count > 6 ? args[6] : descriptor.variant == .voiceDesign ? "A calm adult male voice with a warm low pitch, speaking clear English at a relaxed pace." : "ryan"
        let start = Date()
        let model = try await Qwen3TTSModel.fromModelDirectory(folder)
        MLXRandom.seed(42)
        let samples = try model.synthesize(text: text, voice: voice, reference: reference, referenceText: transcript,
            language: "en", parameters: GenerateParameters(maxTokens: 512, temperature: 0.7))
        guard !samples.isEmpty, samples.allSatisfy(\.isFinite), samples.contains(where: { abs($0) > 1e-5 }) else {
            throw Qwen3TTSError.invalidRequest
        }
        let format = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        let output = try AVAudioFile(forWriting: URL(fileURLWithPath: args[2]), settings: format.settings)
        try output.write(from: buffer)
        let rms = sqrt(samples.reduce(Double(0)) { $0 + Double($1) * Double($1) } / Double(samples.count))
        print("SMOKE variant=\(descriptor.variant.rawValue) samples=\(samples.count) seconds=\(Double(samples.count)/24000) rms=\(rms) elapsed=\(Date().timeIntervalSince(start)) output=\(args[2])")
    }

    /// `Qwen3TTSSmoke breeze MODEL OUT.wav TEXT [--ref AUDIO --ref-text T]
    /// [--instruct I] [--cfg S] [--speaker N] [--greedy] [--stream] [--max N]`
    static func breeze(_ args: [String]) async throws {
        guard args.count >= 3 else {
            throw NSError(domain: "Usage: Qwen3TTSSmoke breeze MODEL OUT.wav TEXT [--ref AUDIO --ref-text T] [--instruct I] [--cfg S] [--speaker N] [--greedy] [--stream] [--max N]", code: 1)
        }
        func option(_ name: String) -> String? { args.firstIndex(of: name).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
        var request = BreezeTTSRequest(text: args[2], speaker: Int(option("--speaker") ?? "0") ?? 0, instruction: option("--instruct"),
                                       referenceText: option("--ref-text"), cfgScale: option("--cfg").flatMap(Float.init),
                                       maxFramesPerSegment: Int(option("--max") ?? "750") ?? 750, seed: 42)
        if let ref = option("--ref") { request.referenceAudio = try readReference(URL(fileURLWithPath: ref)) }
        if args.contains("--greedy") { request.temperature = 0 }
        let start = Date()
        let model = try await BreezeTTSModel.fromModelDirectory(URL(fileURLWithPath: args[0]))
        let loaded = Date()
        var firstAudio: Date?
        let result = try model.generate(request, onAudio: args.contains("--stream") ? { _ in if firstAudio == nil { firstAudio = Date() } } : nil)
        let samples = result.samples
        guard !samples.isEmpty, samples.allSatisfy(\.isFinite), samples.contains(where: { abs($0) > 1e-5 }) else {
            throw BreezeTTSError.invalidRequest("generated audio is empty or silent")
        }
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(result.sampleRate), channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        let output = try AVAudioFile(forWriting: URL(fileURLWithPath: args[1]), settings: format.settings)
        try output.write(from: buffer)
        let rms = sqrt(samples.reduce(Double(0)) { $0 + Double($1) * Double($1) } / Double(samples.count))
        let ttfa = firstAudio.map { String(format: "%.3f", $0.timeIntervalSince(loaded)) } ?? "n/a"
        print("BREEZE frames=\(result.frames) limit=\(result.hitFrameLimit) seconds=\(Double(samples.count) / Double(result.sampleRate)) rms=\(rms) load=\(loaded.timeIntervalSince(start)) gen=\(Date().timeIntervalSince(loaded)) ttfa=\(ttfa) peakGB=\(Double(Memory.peakMemory) / 1e9) output=\(args[1])")
        print("BREEZE first_frames=\(args.contains("--all-codes") ? Array(result.codes) : Array(result.codes.prefix(3)))")
    }

    static func readReference(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard file.length > 0, Double(file.length) / file.processingFormat.sampleRate <= 30 else { throw Qwen3TTSError.invalidReference }
        let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: input)
        let format = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!
        let converter = AVAudioConverter(from: file.processingFormat, to: format)!
        converter.downmix = file.processingFormat.channelCount > 1
        let capacity = AVAudioFrameCount(ceil(Double(file.length) * 24_000 / file.processingFormat.sampleRate)) + 256
        let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity)!
        var fed = false; var error: NSError?
        let result = converter.convert(to: output, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true; status.pointee = .haveData; return input
        }
        if let error { throw error }
        guard result != .error else { throw Qwen3TTSError.invalidReference }
        return Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
    }
}
