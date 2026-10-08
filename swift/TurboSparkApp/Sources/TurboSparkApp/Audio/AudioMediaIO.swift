import AVFoundation
import Foundation

struct AudioCaptureChunk: Sendable {
    var wav: Data
    var sampleRate: Double
    var channels: Int
    var frameCount: Int64
    var offset: Double
    var source: String
}

enum AudioMediaIO {
    enum MediaError: LocalizedError {
        case invalidPCM, tooLong, conversion, noAudio, overlappingChunks
        var errorDescription: String? {
            switch self {
            case .invalidPCM: return String(localized: "Audio samples or their format are invalid. Choose another audio file.", bundle: .module)
            case .tooLong: return String(localized: "This audio window is too large. Process the recording in smaller sections.", bundle: .module)
            case .conversion: return String(localized: "The audio format could not be converted. Try a WAV or M4A file.", bundle: .module)
            case .noAudio: return String(localized: "There is no audio to play or export.", bundle: .module)
            case .overlappingChunks: return String(localized: "Saved chunks from the same source overlap. Export each track separately.", bundle: .module)
            }
        }
    }

    /// The same bounded Float WAV representation is used for capture and model output.
    static func wav(samples: [Float], sampleRate: Double, channels: Int) throws -> Data {
        guard sampleRate.isFinite, (8_000...192_000).contains(sampleRate),
              channels > 0, channels <= 8, samples.count % channels == 0,
              samples.count <= (Int(UInt32.max) - 36) / 4,
              samples.allSatisfy(\.isFinite) else { throw MediaError.invalidPCM }
        var data = Data(capacity: 44 + samples.count * 4)
        func append<T: FixedWidthInteger>(_ value: T) {
            var bytes = value.littleEndian
            withUnsafeBytes(of: &bytes) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: "RIFF".utf8); append(UInt32(36 + samples.count * 4))
        data.append(contentsOf: "WAVEfmt ".utf8); append(UInt32(16))
        append(UInt16(3)); append(UInt16(channels)); append(UInt32(sampleRate.rounded()))
        append(UInt32(sampleRate.rounded()) * UInt32(channels * 4))
        append(UInt16(channels * 4)); append(UInt16(32))
        data.append(contentsOf: "data".utf8); append(UInt32(samples.count * 4))
        for sample in samples { append(sample.bitPattern) }
        return data
    }

    static func importFile(_ url: URL, cancel: () throws -> Void = {}, consume: (AudioCaptureChunk) throws -> Void) throws {
        try cancel()
        let scope = url.startAccessingSecurityScopedResource()
        defer { if scope { url.stopAccessingSecurityScopedResource() } }
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = file.processingFormat
        guard file.length > 0, (8_000...192_000).contains(format.sampleRate),
              (1...8).contains(format.channelCount) else { throw MediaError.invalidPCM }
        let capacity = AVAudioFrameCount(format.sampleRate * 5)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { throw MediaError.conversion }
        while file.framePosition < file.length {
            try Task.checkCancellation(); try cancel()
            let start = file.framePosition
            try file.read(into: buffer, frameCount: capacity)
            guard buffer.frameLength > 0 else { throw MediaError.conversion }
            let samples = try interleavedSamples(buffer)
            try cancel()
            try consume(AudioCaptureChunk(wav: wav(samples: samples, sampleRate: format.sampleRate, channels: Int(format.channelCount)), sampleRate: format.sampleRate, channels: Int(format.channelCount), frameCount: Int64(buffer.frameLength), offset: Double(start) / format.sampleRate, source: "import"))
        }
    }

    static func interleavedSamples(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        guard buffer.format.commonFormat == .pcmFormatFloat32,
              let pointers = buffer.floatChannelData else { throw MediaError.invalidPCM }
        let count = Int(buffer.frameLength), channels = Int(buffer.format.channelCount)
        guard channels > 0, channels <= 8, count <= 192_000 * 5 else { throw MediaError.tooLong }
        if buffer.format.isInterleaved {
            return Array(UnsafeBufferPointer(start: pointers[0], count: count * channels))
        }
        var samples = [Float](repeating: 0, count: count * channels)
        for channel in 0..<channels {
            for frame in 0..<count { samples[frame * channels + channel] = pointers[channel][frame] }
        }
        return samples
    }

    /// Inference is always a bounded read; entire meetings are never assembled in memory.
    static func readMono16k(_ clips: [AudioLibraryClip], store: ManagedAssetStore = .shared, start: Double, duration: Double, cancel: () throws -> Void = {}) throws -> [Float] {
        try readMono16k(clips, start: start, duration: duration, cancel: cancel, resolve: store.materializedURL(for:))
    }

    static func readMono16k(_ clips: [AudioLibraryClip], start: Double, duration: Double, cancel: () throws -> Void = {}, resolve: (String) throws -> URL) throws -> [Float] {
        guard start.isFinite, start >= 0, duration.isFinite, duration > 0, duration <= 30 else { throw MediaError.tooLong }
        var output: [Float] = []
        output.reserveCapacity(Int(ceil(duration * 16_000)))
        try streamMix(clips, resolve: resolve, sampleRate: 16_000, channels: 1, start: start, duration: duration, cancel: cancel) { samples in output.append(contentsOf: samples) }
        return output
    }

    /// Render the common timeline in short blocks. Same-source chunks stay sequential;
    /// overlapping microphone and system tracks are mixed at their captured offsets.
    static func render(_ clips: [AudioLibraryClip], store: ManagedAssetStore = .shared, to url: URL, sampleRate: Double? = nil, cancel: () throws -> Void = {}) throws {
        try cancel()
        guard let first = clips.first else { throw MediaError.noAudio }
        let rate = sampleRate ?? first.sampleRate
        let channels = min(8, clips.map(\.channels).max() ?? 1)
        let duration = clips.map { $0.offset + $0.duration }.max() ?? 0
        guard duration > 0, duration.isFinite, duration <= 86_400 else { throw MediaError.invalidPCM }
        let file = try AVAudioFile(forWriting: url, settings: [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate, AVNumberOfChannelsKey: channels, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsNonInterleaved: false], commonFormat: .pcmFormatFloat32, interleaved: false)
        do {
            try streamMix(clips, resolve: store.materializedURL(for:), sampleRate: rate, channels: channels, start: 0, duration: duration, cancel: cancel) { samples in
                guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(samples.count / channels)), let pointers = buffer.floatChannelData else { throw MediaError.conversion }
                buffer.frameLength = buffer.frameCapacity
                for channel in 0..<channels { for frame in 0..<Int(buffer.frameLength) { pointers[channel][frame] = samples[frame * channels + channel] } }
                try file.write(from: buffer)
            }
        } catch {
            try? FileManager.default.removeItem(at: url)
            throw error
        }
    }

    static func encodeM4A(from source: URL, to destination: URL) async throws {
        let asset = AVURLAsset(url: source)
        guard let exporter = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleM4A) else { throw MediaError.conversion }
        exporter.outputURL = destination
        exporter.outputFileType = .m4a
        await withTaskCancellationHandler(operation: { await exporter.export() }, onCancel: { exporter.cancelExport() })
        guard exporter.status == .completed else {
            try? FileManager.default.removeItem(at: destination)
            throw exporter.error ?? MediaError.conversion
        }
    }

    private static func streamMix(_ clips: [AudioLibraryClip], resolve: (String) throws -> URL, sampleRate: Double, channels: Int, start: Double, duration: Double, cancel: () throws -> Void, consume: ([Float]) throws -> Void) throws {
        guard sampleRate.isFinite, (8_000...192_000).contains(sampleRate), (1...8).contains(channels) else { throw MediaError.invalidPCM }
        guard !clips.isEmpty else { throw MediaError.noAudio }
        for clip in clips {
            guard clip.offset.isFinite, clip.offset >= 0, clip.sampleRate.isFinite,
                  (8_000...192_000).contains(clip.sampleRate), clip.frameCount > 0,
                  (1...8).contains(clip.channels) else { throw MediaError.invalidPCM }
        }
        let selected = clips.filter { $0.offset < start + duration && $0.offset + $0.duration > start }.sorted { $0.offset < $1.offset }
        var lastEnd: [String: Double] = [:]
        for clip in selected {
            if let end = lastEnd[clip.source], clip.offset < end - 0.02 { throw MediaError.overlappingChunks }
            lastEnd[clip.source] = clip.offset + clip.duration
        }
        var readers: [UUID: ResampledReader] = [:]
        // Match clip boundaries so floating-point duration error adds no trailing frame.
        let total = Int64((duration * sampleRate).rounded())
        var frame: Int64 = 0
        while frame < total {
            try Task.checkCancellation(); try cancel()
            let count = Int(min(8192, total - frame))
            let from = start + Double(frame) / sampleRate
            let to = from + Double(count) / sampleRate
            var mix = [Float](repeating: 0, count: count * channels)
            for clip in selected where clip.offset < to && clip.offset + clip.duration > from {
                let begin = max(0, Int(((clip.offset - from) * sampleRate).rounded()))
                let end = min(count, Int(((clip.offset + clip.duration - from) * sampleRate).rounded()))
                guard end > begin else { continue }
                let reader: ResampledReader
                if let existing = readers[clip.id] { reader = existing }
                else {
                    try cancel()
                    reader = try ResampledReader(url: resolve(clip.assetReference), sampleRate: sampleRate, channels: channels, start: max(0, from - clip.offset), expected: clip)
                    readers[clip.id] = reader
                }
                let samples = try reader.read(frames: end - begin)
                for index in samples.indices { mix[begin * channels + index] += samples[index] }
            }
            // Keep Float WAV headroom intact. A renderer must not normalize each block
            // independently, which would audibly pump a meeting's volume.
            try cancel()
            try consume(mix)
            frame += Int64(count)
            readers = readers.filter { id, _ in selected.contains { $0.id == id && $0.offset + $0.duration > to } }
        }
    }

    private final class ResampledReader {
        let file: AVAudioFile
        let converter: AVAudioConverter
        let outputFormat: AVAudioFormat
        var ended = false
        init(url: URL, sampleRate: Double, channels: Int, start: Double, expected: AudioLibraryClip) throws {
            file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
            guard file.length == expected.frameCount, file.processingFormat.sampleRate == expected.sampleRate,
                  Int(file.processingFormat.channelCount) == expected.channels else { throw MediaError.invalidPCM }
            file.framePosition = AVAudioFramePosition(min(Double(file.length), (start * file.processingFormat.sampleRate).rounded(.down)))
            guard let output = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: AVAudioChannelCount(channels), interleaved: false), let converter = AVAudioConverter(from: file.processingFormat, to: output) else { throw MediaError.conversion }
            self.converter = converter
            outputFormat = output
            converter.primeMethod = .normal
            converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        }
        func read(frames: Int) throws -> [Float] {
            guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: AVAudioFrameCount(frames)) else { throw MediaError.conversion }
            var readError: Error?
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError) { count, status in
                if self.ended || self.file.framePosition >= self.file.length { status.pointee = .endOfStream; return nil }
                guard let input = AVAudioPCMBuffer(pcmFormat: self.file.processingFormat, frameCapacity: max(1, min(count, 65_536))) else { status.pointee = .noDataNow; return nil }
                do {
                    try self.file.read(into: input, frameCount: input.frameCapacity)
                    self.ended = input.frameLength == 0
                    status.pointee = self.ended ? .endOfStream : .haveData
                    return self.ended ? nil : input
                } catch { readError = error; status.pointee = .endOfStream; return nil }
            }
            if let readError { throw readError }
            if let conversionError { throw conversionError }
            guard status != .error else { throw MediaError.conversion }
            return try interleavedSamples(output)
        }
    }
}
