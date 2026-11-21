import Foundation
import TurboSpark

/// Runs one enhancement job for Clean reference: the profile, the request and
/// its 48 kHz mono PCM in, the engine's result out, reporting progress.
typealias AudioReferenceCleaningRunner = @MainActor (
    AudioProfile, AudioRequest, [Float], @escaping @Sendable (AudioProgress) -> Void
) async throws -> AudioResult

/// What Clean reference reads and sends: only the trimmed region of the
/// source, never the whole take, so the cleaned copy is exactly what the
/// user chose to clone from.
struct AudioReferenceCleaningPlan: Equatable, Sendable {
    let request: AudioRequest
    let clips: [AudioLibraryClip]
    let start: Double
    let duration: Double
}

/// Clean reference on the Voice Cloning page: speech enhancement over the
/// reference region, saved as a new `.enhancement` take that then becomes the
/// reference. The source take is never modified, and consent is cleared
/// because the cleaned take is a new reference the user has not confirmed.
extension AudioWorkspaceController {
    /// Enhancement installs Clean reference uses, best first: the Voice
    /// Isolator, its realtime variant, then MossFormer2.
    nonisolated static let referenceCleaningAliases = ["voice-isolator-v1", "voice-isolator-v1-realtime", "mossformer2-se-48k-fp16"]

    /// The preferred installed enhancement profile this Mac can run through a
    /// Rust session, or nil when none is (or the runtime cannot run
    /// enhancement at all), which hides the action.
    nonisolated static func referenceCleaningProfile(
        in profiles: [AudioProfile], runnable: Bool = AudioTask.enhancement.isRunnable,
        isInstalled: (AudioProfile) -> Bool
    ) -> AudioProfile? {
        guard runnable else { return nil }
        for alias in referenceCleaningAliases {
            if let match = profiles.first(where: {
                $0.identity.alias == alias && $0.identity.task == .enhancement
                    && AudioRuntimeProvider($0) == .rust && $0.capabilities.canRun && isInstalled($0)
            }) { return match }
        }
        return nil
    }

    var referenceCleaningProfile: AudioProfile? {
        Self.referenceCleaningProfile(in: profiles) { self.isInstalled($0) }
    }

    /// Whether the reference card shows Clean reference.
    var offersReferenceCleaning: Bool { referenceCleaningProfile != nil }

    var canCleanVoiceReference: Bool {
        canEditVoiceReference && importTask == nil && speechSettings.reference?.isValid == true && offersReferenceCleaning
    }

    /// The request names the rate the region is read at (the engine refuses a
    /// job without one); the region is the reference's own trim.
    nonisolated static func referenceCleaningPlan(for reference: AudioVoiceReference) -> AudioReferenceCleaningPlan {
        var request = AudioRequest(task: .enhancement)
        request.inputSampleRate = 48_000
        return AudioReferenceCleaningPlan(request: request, clips: reference.clips, start: reference.start, duration: reference.duration)
    }

    /// Cleans the reference region. Loading the enhancement model replaces
    /// the resident one, as any other run does. The reference changes only
    /// once the cleaned take is saved, so a cancel or a failure leaves it as
    /// it was.
    func cleanVoiceReference() {
        guard page == .voiceClone, canCleanVoiceReference, let profile = referenceCleaningProfile,
              let reference = speechSettings.reference else { return }
        let limit = Double(profile.capabilities.maxInputSeconds ?? 30)
        guard reference.duration <= limit else {
            error = AppStrings.localized("This operation cannot process the full recording. Choose a shorter source.")
            return
        }
        stopReferencePreview()
        let plan = Self.referenceCleaningPlan(for: reference)
        isBusy = true; isCleaningVoiceReference = true; progress = nil; error = nil
        status = AppStrings.localized("Cleaning reference")
        let token = epoch
        let runner = referenceCleaningRunner
        jobTask = Task { [weak self] in
            guard let self else { return }
            let admitted = await ImageJobCoordinator.shared.acquire()
            defer { if admitted { Task { await ImageJobCoordinator.shared.release() } } }
            defer {
                if self.valid(token) {
                    self.isBusy = false; self.isCleaningVoiceReference = false; self.progress = nil
                    self.scheduleUnload()
                }
            }
            do {
                guard admitted else { throw CancellationError() }
                try Task.checkCancellation()
                let assets = self.assets; let lifetime = self.lifetime
                let pcm = try await Task.detached {
                    try AudioMediaIO.readMono48kTake(plan.clips, store: assets, start: plan.start, duration: plan.duration,
                                                     limit: limit, cancel: lifetime.check)
                }.value
                try Task.checkCancellation()
                guard self.valid(token) else { return }
                let report: @Sendable (AudioProgress) -> Void = { [weak self] update in
                    Task { @MainActor in
                        guard let self, self.valid(token), self.isCleaningVoiceReference,
                              let total = update.total, total > 0 else { return }
                        self.progress = min(1, Double(update.completed) / Double(total))
                    }
                }
                let output: AudioResult
                if let runner {
                    output = try await runner(profile, plan.request, pcm, report)
                } else {
                    self.status = AppStrings.localized("Loading model")
                    let session = try await self.residentSession(for: profile)
                    self.status = AppStrings.localized("Cleaning reference")
                    output = try await session.execute(plan.request, pcm: pcm, progress: report)
                }
                try Task.checkCancellation()
                guard self.valid(token) else { return }
                let audio = try Self.validateGeneratedAudio(output.pcm)
                self.status = AppStrings.localized("Saving audio"); self.progress = nil
                let stored = try await Task.detached {
                    try lifetime.check()
                    let bytes = try AudioMediaIO.wav(samples: audio.samples, sampleRate: Double(audio.format.sampleRate), channels: Int(audio.format.channels))
                    try lifetime.check()
                    return try assets.store(data: bytes, fileName: "reference.wav", mimeType: "audio/wav")
                }.value
                try Task.checkCancellation()
                guard self.valid(token) else { return }
                let clip = AudioLibraryClip(assetReference: stored.storedReference, sampleRate: Double(audio.format.sampleRate),
                                            channels: Int(audio.format.channels),
                                            frameCount: Int64(audio.samples.count / Int(audio.format.channels)))
                let take = self.cleanedReferenceTake(from: reference, clip: clip, profile: profile)
                try self.save(take)
                self.adoptCleanedReference(take, replacing: reference)
                self.status = AppStrings.localized("Reference cleaned. Confirm you have the right to use this voice.")
            } catch {
                guard self.valid(token) else { return }
                // A cancel reaches the native job, which reports it as its
                // own error rather than CancellationError.
                let cancelled = Task.isCancelled || error is CancellationError
                self.status = cancelled ? AppStrings.localized("Interrupted") : AppStrings.localized("Failed")
                self.error = cancelled ? nil : error.localizedDescription
            }
        }
    }

    /// A new library take holding the cleaned region. It points back at the
    /// source take through `sourceID`; the source itself is not touched.
    func cleanedReferenceTake(from reference: AudioVoiceReference, clip: AudioLibraryClip, profile: AudioProfile) -> AudioLibraryItem {
        var take = AudioLibraryItem(title: reference.title + " - " + AppStrings.localized("Cleaned"), kind: .enhancement)
        take.sourceID = reference.sourceID
        take.clips = [clip]
        take.recipe.task = AudioTask.enhancement.rawValue
        take.recipe.modelID = profile.identity.alias
        take.recipe.durationSeconds = clip.duration
        take.modelFamily = profile.family
        take.localModelBookmark = preferences.localModels[profile.identity.alias]
        if take.localModelBookmark == nil { take.modelIdentity = try? JSONEncoder().encode(profile.identity) }
        return take
    }

    /// Makes the cleaned take the reference, unless the user left the page
    /// or the recipe no longer holds the region that was cleaned.
    func adoptCleanedReference(_ take: AudioLibraryItem, replacing original: AudioVoiceReference) {
        guard page == .voiceClone, speechSettings.mode == .clone, speechSettings.reference == original,
              take.hasPlayableAudio else { return }
        var settings = speechSettings
        settings.reference = AudioVoiceReference(sourceID: take.id, title: take.title, clips: take.clips, duration: min(30, take.duration))
        // The transcript stays: the cleaned take is that exact region, so it
        // speaks the same words. Consent does not: it covers one exact
        // region of one source, and this is a new reference the user has not
        // confirmed (swift/AGENTS.md, voice cloning consent).
        settings.cloneOptions?.consentedReference = nil
        speechSettings = settings
        flushPreferences()
    }
}
