import AVFoundation
import Foundation
import MLX
import Qwen3TTS
import TurboSpark

extension AppModel {
    public func attachAudioSpeechModelToServer(_ model: AudioInstalledRecord) {
        if isMoving(path: model.path) {
            announceModelBeingMoved()
            return
        }
        guard let server else { return }
        do {
            try server.attachAudioSpeechModel(id: model.alias)
            serverSpeechAttachedModel = model
            serverSpeechProgress = "Ready"
            refreshServerInfo()
        } catch {
            showToast(
                AppStrings.localized("Could not serve audio speech model: \(error.localizedDescription)"),
                style: .error)
        }
    }

    public func detachAudioSpeechModelFromServer() {
        guard let server else { return }
        cancelServerSpeechJobs()
        do {
            try server.detachAudioSpeechModel()
            serverSpeechAttachedModel = nil
            serverSpeechProgress = "Idle"
            refreshServerInfo()
        } catch {
            showToast(
                AppStrings.localized("Could not detach audio speech model: \(error.localizedDescription)"),
                style: .error)
        }
    }

    func cancelServerSpeechJobs() {
        for task in serverSpeechTasks.values { task.cancel() }
        serverSpeechTasks.removeAll()
    }

    func handleServerSpeechEvents(_ server: TurboSparkServer) {
        for event in server.pollAudioSpeechEvents() {
            switch event {
            case let .cancel(id):
                serverSpeechTasks.removeValue(forKey: id)?.cancel()
            case let .start(id, request):
                guard let model = serverSpeechAttachedModel, model.alias == request.model else {
                    try? server.completeAudioSpeechRequest(id: id, samples: nil, error: "speech model was detached")
                    continue
                }
                serverSpeechTasks[id] = Task { [weak self] in
                    guard let self else { return }
                    await self.runServerSpeechRequest(id: id, request: request, model: model, server: server)
                }
            }
        }
    }

    private func runServerSpeechRequest(
        id: UInt64, request: ServerSpeechRequest,
        model: AudioInstalledRecord, server: TurboSparkServer
    ) async {
        defer { serverSpeechTasks[id] = nil }
        var permit: NativeHeavyWorkPermit?
        while permit == nil {
            if Task.isCancelled {
                try? server.completeAudioSpeechRequest(id: id, samples: nil, error: "speech generation was cancelled")
                return
            }
            permit = try? NativeHeavyWorkPermit.tryAcquire()
            if permit == nil { try? await Task.sleep(for: .milliseconds(50)) }
        }
        defer { permit?.release() }

        do {
            try Task.checkCancellation()
            // The synthesizer is opened from the folder here, per request. A model that
            // is attached and then moved answers with the reason, and is not opened.
            if isMoving(path: model.path) {
                throw TurboSparkError(code: .open, message: ModelMoveText.modelBeingMoved())
            }
            let directory = URL(fileURLWithPath: model.path, isDirectory: true)

            // Decode reference audio if provided (base64)
            var referenceSamples: [Float]?
            if let refAudioB64 = request.referenceAudio, let refData = Data(base64Encoded: refAudioB64) {
                referenceSamples = try Self.decodeAudioDataToMono24k(refData)
            }

            let voiceArg: String?
            if let instruction = request.instruction, !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                voiceArg = request.voice.isEmpty ? instruction : "\(request.voice),\(instruction)"
            } else if !request.voice.isEmpty && request.voice != "clone" {
                voiceArg = request.voice
            } else {
                voiceArg = nil
            }

            serverSpeechProgress = "Loading speech model"
            var ttsModel: Qwen3TTSModel?
            defer {
                ttsModel = nil
                MLX.Memory.clearCache()
            }
            ttsModel = try await Qwen3TTSModel.fromModelDirectory(directory)
            try Task.checkCancellation()

            serverSpeechProgress = "Synthesizing"
            var tokenCount = 0
            let lang = request.language ?? "auto"
            let generated = try ttsModel!.synthesize(
                text: request.text,
                voice: voiceArg,
                reference: referenceSamples,
                referenceText: request.referenceText,
                language: lang,
                parameters: GenerateParameters(maxTokens: 2048, temperature: 0.7),
                onToken: { _ in
                    tokenCount += 1
                    if tokenCount % 12 == 0 {
                        self.serverSpeechProgress = "Synthesizing (\(tokenCount) tokens)"
                    }
                }
            )
            try Task.checkCancellation()
            guard !generated.isEmpty, generated.allSatisfy(\.isFinite) else {
                throw TurboSparkError(code: .generate, message: "Synthesizer produced empty or non-finite audio")
            }

            try server.completeAudioSpeechRequest(id: id, samples: generated, sampleRate: 24_000)
            serverSpeechProgress = "Ready"
        } catch is CancellationError {
            try? server.completeAudioSpeechRequest(id: id, samples: nil, error: "speech generation was cancelled")
            serverSpeechProgress = "Cancelled"
        } catch {
            try? server.completeAudioSpeechRequest(id: id, samples: nil, error: error.localizedDescription)
            serverSpeechProgress = "Failed: \(error.localizedDescription)"
        }
    }

    private static func decodeAudioDataToMono24k(_ data: Data) throws -> [Float] {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try data.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let file = try AVAudioFile(forReading: tempURL)
        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false) else {
            throw AudioMediaIO.MediaError.conversion
        }
        guard let converter = AVAudioConverter(from: file.processingFormat, to: targetFormat) else {
            throw AudioMediaIO.MediaError.conversion
        }
        let frameCount = AVAudioFrameCount(Double(file.length) * (24_000.0 / max(1.0, file.processingFormat.sampleRate)))
        guard let buffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: max(1, frameCount)) else {
            throw AudioMediaIO.MediaError.conversion
        }

        var convError: NSError?
        converter.convert(to: buffer, error: &convError) { packetCount, outStatus in
            guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: packetCount) else {
                outStatus.pointee = .endOfStream
                return nil
            }
            do {
                try file.read(into: inputBuffer, frameCount: packetCount)
                outStatus.pointee = .haveData
                return inputBuffer
            } catch {
                outStatus.pointee = .endOfStream
                return nil
            }
        }
        if let convError { throw convError }
        guard let channelData = buffer.floatChannelData else {
            throw AudioMediaIO.MediaError.invalidPCM
        }
        return Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
    }
}
