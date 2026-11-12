import Foundation
import Metal
import MLX
import TurboSpark

extension AudioWorkspaceController {
    /// The resident FluidAudio recognizer for a catalog install. Like the Rust
    /// and Qwen sessions it replaces whatever model is resident, checks every
    /// byte of the install before loading, and admits the load only within the
    /// audio memory budget.
    func residentFluidTranscriber(for profile: AudioProfile, family: FluidAudioFamily) async throws -> any FluidAudioTranscriber {
        idleTask?.cancel()
        // Only the catalog's verified files reach FluidAudio, never a folder
        // the user picked: the signature check stops a trap, not wrong weights.
        guard let path = installedPaths[profile.identity.alias] else {
            throw AudioWorkspaceError.message(AppStrings.localized("Download this model first."))
        }
        let key = "fluidaudio:\(path)"
        if sessionKey == key, let fluidTranscriber { return fluidTranscriber }
        if let previous = session { await Task.detached { previous.close() }.value }
        session = nil
        modelAccessURL?.stopAccessingSecurityScopedResource(); modelAccessURL = nil
        if qwenModel != nil { qwenModel = nil; MLX.Memory.clearCache() }
        fluidTranscriber = nil; sessionKey = nil; loadedModelAlias = nil

        let budget = min(ProcessInfo.processInfo.physicalMemory * 3 / 4, MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
        guard profile.residentMemoryBytes <= budget else {
            throw AudioWorkspaceError.message(AppStrings.localized("This checkpoint exceeds the available audio model memory budget."))
        }
        let token = epoch
        let opening = AudioModelOpening()
        opening.onProgress = { [weak self, weak opening] fraction in
            Task { @MainActor in
                guard let self, let opening, self.valid(token), self.opening === opening else { return }
                self.status = AppStrings.localized("Verifying")
                self.progress = fraction
            }
        }
        self.opening = opening
        do {
            guard let openToken = opening.token else { throw CancellationError() }
            let verified = try await Task.detached {
                try opening.perform {
                    try AudioCatalog.verify(profile.identity, token: openToken) { progress in
                        guard let total = progress.total, total > 0 else { return }
                        opening.onProgress?(min(1, Double(progress.completed) / Double(total)))
                    }
                }
            }.value
            guard verified == path else { throw CancellationError() }
            status = AppStrings.localized("Compiling for the Neural Engine. The first load can take a few minutes.")
            progress = nil
            // Off the main actor: the first load compiles for the Neural
            // Engine and can take minutes.
            let loaded = try await FluidAudioEngine.shared.transcriber(for: family, installRoot: URL(fileURLWithPath: path, isDirectory: true))
            self.opening = nil
            guard valid(token), !Task.isCancelled else { throw CancellationError() }
            fluidTranscriber = loaded; sessionKey = key; loadedModelAlias = profile.identity.alias
            return loaded
        } catch {
            if self.opening === opening { self.opening = nil }
            progress = nil
            throw error
        }
    }

    /// Transcribes a saved recording with a FluidAudio recognizer through the
    /// workspace's window loop. Window length comes from the adapter. Windows
    /// overlap only when the model reports token times, because only then can
    /// the overlap be cut without repeating words.
    @discardableResult func transcribeWithFluidAudio(item: AudioLibraryItem, into id: UUID, profile: AudioProfile, family: FluidAudioFamily,
                                  source: AudioTranscriptRevision.Source, token: UUID) async throws -> Transcription {
        let transcriber = try await residentFluidTranscriber(for: profile, family: family)
        // 10 ms under the adapter's limit: the PCM reader rounds a window's
        // sample count up, and the adapter refuses one sample over.
        let length = transcriber.maxWindowSamples.map { Double($0) / 16_000 - 0.01 } ?? 30
        let windows = AudioTranscriptionPlan.windows(duration: item.duration, length: length,
                                                     overlap: transcriber.reportsTokenTimes ? 2 : 0, skipsCoveredTail: true)
        let languages = profile.capabilities.languages
        let language = languages.count == 1 ? languages[0] : nil
        return try await transcribe(item: item, into: id, windows: windows, source: source, token: token) { pcm, window in
            try await transcriber.transcribe(pcm).transcript(in: window, language: language)
        }
    }
}
