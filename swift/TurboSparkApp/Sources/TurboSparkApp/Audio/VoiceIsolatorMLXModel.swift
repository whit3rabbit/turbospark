import Foundation
import MLX
import os
import TurboSpark
import VoiceIsolatorMLX

/// The offline Voice Isolator on the vendored MLX port
/// (Vendor/VoiceIsolatorMLX/UPSTREAM.md).
///
/// The catalog row stays a Rust row: the CLI, the FFI, the realtime
/// checkpoint and live streams run the Rust engine. Only this app's offline
/// enhancement (Clean Up and Clean reference) comes here, because the Rust
/// Metal path is per-call and takes about 14 s for 10 s of audio on an
/// M4 Max, where this port takes about 1 s.
final class VoiceIsolatorMLXModel: @unchecked Sendable {
    nonisolated static let alias = "voice-isolator-v1"
    nonisolated static let family = "voice_isolator"
    nonisolated static let weightFile = "denoiser.safetensors"
    /// The rate the Rust enhancement contract runs and returns at.
    nonisolated static let sampleRate = 48_000

    /// Whether this engine runs the profile. The offline checkpoint only:
    /// the realtime file streams, and its offline render stays on Rust.
    nonisolated static func handles(_ profile: AudioProfile) -> Bool {
        profile.identity.task == .enhancement && profile.family == family && profile.identity.alias == alias
    }

    private let isolator: VoiceIsolator
    /// `VoiceIsolator` is not reentrant; one render at a time.
    private let lock = NSLock()

    /// Loads the released checkpoint and converts it in memory, computing in
    /// float16 (the library default; upstream measures its SI-SDR within
    /// 0.02 dB of float32).
    init(directory: URL) throws {
        isolator = try VoiceIsolator(contentsOf: directory.appendingPathComponent(Self.weightFile))
    }

    /// The Rust enhancement contract: 48 kHz mono in, the voice out at 48 kHz
    /// and the input's length, so input minus voice is the removed part.
    /// Both callers read 48 kHz and the library keeps the length, so the two
    /// refusals are invariants and reuse an existing string.
    func enhance(_ request: AudioRequest, pcm: [Float],
                 progress: @escaping @Sendable (AudioProgress) -> Void) async throws -> AudioResult {
        guard request.inputSampleRate == UInt32(Self.sampleRate) else {
            throw AudioWorkspaceError.message(AppStrings.localized("Stopped unexpectedly"))
        }
        // Detached so the render leaves the main actor; the handler forwards
        // a cancel, which the library honours between render windows.
        let work = Task.detached(priority: .userInitiated) { [self] () throws -> [Float] in
            lock.lock(); defer { lock.unlock() }
            return try isolator.isolate(pcm, sampleRate: Self.sampleRate) { done in
                progress(AudioProgress(stage: "enhance", completed: Int((done * 1000).rounded()), total: 1000))
            }
        }
        let voice = try await withTaskCancellationHandler { try await work.value } onCancel: { work.cancel() }
        try Task.checkCancellation()
        guard voice.count == pcm.count else {
            throw AudioWorkspaceError.message(AppStrings.localized("Stopped unexpectedly"))
        }
        return AudioResult(pcm: AudioPCM(format: AudioPCMFormat(sampleRate: UInt32(Self.sampleRate), channels: 1),
                                         samples: voice))
    }
}

/// The slow steps of a resident MLX Voice Isolator load. Tests replace them
/// to load without weights or to make the MLX step fail.
struct AudioVoiceIsolatorLoadDependencies {
    var openToken: @Sendable () -> AudioOpenToken? = { try? AudioOpenToken() }
    var verify: @Sendable (AudioProfileIdentity, AudioOpenToken, @escaping @Sendable (AudioProgress) -> Void) throws -> String = {
        try AudioCatalog.verify($0, token: $1, progress: $2)
    }
    var load: @Sendable (URL) throws -> VoiceIsolatorMLXModel = { try VoiceIsolatorMLXModel(directory: $0) }
}

/// The MLX step failed, as opposed to finding or verifying the install.
/// Only this falls back to the Rust session, which would refuse a missing
/// or corrupt install the same way.
struct VoiceIsolatorMLXLoadFailure: Error {
    let underlying: Error
}

private let voiceIsolatorLog = Logger(subsystem: "com.turbospark.app", category: "VoiceIsolatorMLX")

extension AudioWorkspaceController {
    /// The resident MLX isolator for an offline Voice Isolator profile, or
    /// nil when another engine runs it: a different profile, or an MLX load
    /// that failed, in which case the caller opens the Rust session as
    /// before. Finding, moving and verification errors propagate.
    func offlineVoiceIsolator(for profile: AudioProfile, source: AudioLibraryItem? = nil) async throws -> VoiceIsolatorMLXModel? {
        guard VoiceIsolatorMLXModel.handles(profile) else { return nil }
        do {
            return try await residentVoiceIsolator(for: profile, source: source)
        } catch let failure as VoiceIsolatorMLXLoadFailure {
            voiceIsolatorLog.error("MLX load failed, using the Rust session: \(String(describing: failure.underlying), privacy: .public)")
            return nil
        }
    }

    /// Makes the MLX isolator the one resident audio model, as
    /// `residentQwenSession` does for Qwen3-TTS: a managed install loads only
    /// after its bytes verify, a chosen folder through its bookmark.
    func residentVoiceIsolator(for profile: AudioProfile, source: AudioLibraryItem? = nil) async throws -> VoiceIsolatorMLXModel {
        idleTask?.cancel()
        var url: URL?
        let directory: URL
        let path: String
        if let bookmark = try modelBookmark(for: profile, source: source) {
            var stale = false
            let resolved = try URL(resolvingBookmarkData: bookmark, options: .withSecurityScope, relativeTo: nil, bookmarkDataIsStale: &stale)
            guard !stale else { throw AudioWorkspaceError.message(AppStrings.localized("The model folder moved. Choose it again.")) }
            _ = resolved.startAccessingSecurityScopedResource(); url = resolved; directory = resolved; path = resolved.path
        } else if let installed = installedPaths[profile.identity.alias] {
            path = installed
            directory = URL(fileURLWithPath: installed, isDirectory: true)
        } else {
            throw AudioWorkspaceError.message(AppStrings.localized("Download this model or choose its checkpoint folder."))
        }
        if modelMoving?(path) == true {
            url?.stopAccessingSecurityScopedResource()
            throw AudioWorkspaceError.message(ModelMoveText.modelBeingMoved())
        }
        let key = "voice_isolator_mlx:\(path)"
        if sessionKey == key, let voiceIsolatorModel {
            url?.stopAccessingSecurityScopedResource()
            return voiceIsolatorModel
        }
        if let previous = session { await Task.detached { previous.close() }.value }
        session = nil
        modelAccessURL?.stopAccessingSecurityScopedResource(); modelAccessURL = nil
        qwenModel = nil; voiceIsolatorModel = nil; fluidTranscriber = nil; sessionKey = nil; loadedModelAlias = nil
        MLX.Memory.clearCache()

        let token = epoch
        do {
            let opening = AudioModelOpening(token: voiceIsolatorLoadDependencies.openToken())
            opening.onProgress = { [weak self, weak opening] fraction in
                Task { @MainActor in
                    guard let self, let opening, self.valid(token), self.opening === opening else { return }
                    self.status = AppStrings.localized("Verifying")
                    self.progress = fraction
                }
            }
            self.opening = opening
            let verifies = url == nil
            let verify = voiceIsolatorLoadDependencies.verify
            let verified = try await Task.detached {
                try opening.perform { () throws -> String in
                    guard verifies else { return path }
                    guard let openToken = opening.token else { throw CancellationError() }
                    return try verify(profile.identity, openToken) { progress in
                        guard let total = progress.total, total > 0 else { return }
                        opening.onProgress?(min(1, Double(progress.completed) / Double(total)))
                    }
                }
            }.value
            guard verified == path else { throw CancellationError() }

            self.status = AppStrings.localized("Loading model")
            let load = voiceIsolatorLoadDependencies.load
            let loaded = try await Task.detached { () throws -> VoiceIsolatorMLXModel in
                try Task.checkCancellation()
                do { return try load(directory) } catch { throw VoiceIsolatorMLXLoadFailure(underlying: error) }
            }.value

            self.opening = nil
            self.progress = nil
            guard valid(token), !Task.isCancelled else { throw CancellationError() }
            voiceIsolatorModel = loaded
            sessionKey = key
            loadedModelAlias = profile.identity.alias
            modelAccessURL = url
            return loaded
        } catch {
            opening = nil
            progress = nil
            url?.stopAccessingSecurityScopedResource()
            throw error
        }
    }
}
