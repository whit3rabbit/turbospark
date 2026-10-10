import AVFoundation
import Foundation
import MLX
import Qwen3TTS

@main struct Smoke {
    static func main() async throws {
        let args = CommandLine.arguments
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
