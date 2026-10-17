import Foundation
import TurboSpark

extension AudioWorkspaceController {
    /// The music engine refuses empty lyrics and accepts this exact token as
    /// the explicit instrumental opt-in. The lyrics editor shows it as the
    /// placeholder, and an untouched field maps to it at request time.
    nonisolated static let instrumentalLyrics = "[instrumental]"

    /// Whitespace-only lyrics mean the writer left the instrumental in place.
    static func effectiveLyrics(_ lyrics: String) -> String {
        lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? instrumentalLyrics : lyrics
    }

    static func request(from recipe: AudioRecipe, task: AudioTask) -> AudioRequest {
        var request = AudioRequest(task: task, language: recipe.language == "auto" ? nil : recipe.language)
        request.text = recipe.text; request.voice = recipe.voice; request.speed = recipe.speed
        request.caption = recipe.caption
        request.lyrics = task == .music ? effectiveLyrics(recipe.lyrics) : recipe.lyrics
        request.durationSeconds = recipe.durationSeconds; request.steps = recipe.steps; request.seed = recipe.seed
        return request
    }
    /// The request a run sends. A native cloner's run either clones, and
    /// passes its reference as PCM (`runClonesNatively`), or speaks with no
    /// reference.
    static func runRequest(_ recipe: AudioRecipe, task: AudioTask, profile: AudioProfile) -> AudioRequest {
        let base = request(from: recipe, task: task)
        guard task == .textToSpeech, clonesNatively(profile) else { return base }
        return runClonesNatively(profile, speech: recipe.speech)
            ? nativeCloneRequest(base, family: profile.family, speech: recipe.speech)
            : referenceFreeSpeechRequest(base, family: profile.family)
    }
    /// A natively cloning run (MegaTTS3, OmniVoice, Drama 3): its reference
    /// arrives as the session PCM at 24 kHz, and consent holds only for that
    /// exact region.
    nonisolated static func nativeCloneRequest(_ base: AudioRequest, family: String, speech: AudioSpeechSettings?) -> AudioRequest {
        var request = base
        let options = speech?.cloneOptions ?? AudioCloneOptions()
        request.steps = options.steps
        request.referenceSampleRate = 24_000
        request.consent = speech?.reference != nil && options.consentedReference == speech?.reference
        if family == "omnivoice" {
            // OmniVoice refuses MegaTTS3's guidance weights and reads the
            // reference transcript and delivery instruction directly.
            request.pW = nil
            request.tW = nil
            return request
        }
        if family == "fish_drama3" {
            // Drama 3 refuses speed other than 1.0, an instruction (cues go
            // inline in the script) and the guidance weights, and ignores
            // steps. A clone needs the reference's exact words.
            request.steps = nil
            request.speed = 1
            request.instruction = nil
            request.pW = nil
            request.tW = nil
            let transcript = speech?.referenceText.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            request.referenceText = transcript.isEmpty ? nil : transcript
            return request
        }
        request.pW = options.intelligibility
        request.tW = options.similarity
        // The engine aligns the reference itself and takes no style prompt.
        request.instruction = nil
        request.referenceText = nil
        return request
    }
    /// A native cloner speaking on Voiceover with no reference (Drama 3,
    /// OmniVoice): no PCM, consent, transcript or guidance weights, and the
    /// model picks the voice.
    nonisolated static func referenceFreeSpeechRequest(
        _ base: AudioRequest, family: String,
        freshSeed: () -> UInt64 = { UInt64.random(in: 1...UInt64.max) }
    ) -> AudioRequest {
        var request = base
        // Seed 0 is a new take each run, as Qwen3 does on this page. The
        // runtime takes 0 literally, so every run of a script repeated one
        // take, and Drama 3 goes silent after the first sentence at some
        // seeds (0 among them). The take saves the seed it used.
        if (request.seed ?? 0) == 0 { request.seed = freshSeed() }
        request.consent = false
        request.referenceSampleRate = nil
        request.referenceText = nil
        request.instruction = nil
        request.pW = nil
        request.tW = nil
        // The page has no steps control, and the recipe's count belongs to
        // the music page, so the engine's own default applies. Drama 3 has
        // no steps at all.
        request.steps = nil
        // Drama 3 refuses any speed but 1.0, and the page hides its slider.
        if family == "fish_drama3" { request.speed = 1 }
        return request
    }
    /// The catalog declares the CPU as this profile's runtime, so running on
    /// it is the qualified path, not a silent fallback from Metal.
    nonisolated static func runsOnDeclaredCPU(_ profile: AudioProfile) -> Bool {
        profile.capabilities.canRun && profile.capabilities.backend == "portable_cpu"
    }
    func runSection(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        var section = recipe; section.text = text
        beginRun(section)
    }
    func run() { beginRun(recipe) }
    var runSource: AudioLibraryItem? {
        guard let selected = selectedItem else { return nil }
        // Switching from a recording to a new script does not make that recording
        // an input to synthesis. Repeated takes of the same task retain lineage.
        if task == .textToSpeech || task == .music {
            guard selected.recipe.task == task.rawValue, selected.kind == kind else { return nil }
            if task == .textToSpeech, (selected.recipe.speech?.mode ?? .customVoice) != (recipe.speech?.mode ?? .customVoice) { return nil }
        }
        return selected
    }
    func beginRun(_ inputs: AudioRecipe) {
        guard canRun, let profile = selectedProfile else { return }
        let selected = runSource
        // Each editor keeps its draft when switching destinations. A retained
        // voiceover script must not name a new music take.
        let generatedTitle = String((task == .music ? inputs.caption : inputs.text).prefix(60))
        let title = task == .textToSpeech || task == .music ? generatedTitle : selected.map { $0.title + " - " + page.title } ?? generatedTitle
        var result = AudioLibraryItem(title: title, kind: kind)
        if result.title.isEmpty { result.title = page.title }
        result.recipe = inputs; result.recipe.task = task.rawValue
        if task == .music { result.recipe.lyrics = Self.effectiveLyrics(inputs.lyrics) }
        result.sourceID = selected?.id; result.status = .processing
        result.localModelBookmark = preferences.localModels[profile.identity.alias]
        result.modelFamily = profile.family
        if result.localModelBookmark == nil { result.modelIdentity = try? JSONEncoder().encode(profile.identity) }
        if task == .speechToText { result.clips = selected?.clips ?? [] }
        let synthesis = task == .music || task == .textToSpeech
        // Generation has no recoverable audio yet. Publish only after the
        // result is validated and encrypted; a failed job is not a saved take.
        if synthesis { activeGeneration = result }
        else { do { try save(result) } catch { fail(error); return } }
        selectedID = result.id
        isBusy = true; progress = nil; error = nil
        activeRunID = result.id
        generationProgress = task == .music ? AudioGenerationProgress() : nil
        speechRun = task == .textToSpeech ? AudioSpeechRunProgress() : nil
        status = AppStrings.localized("Waiting for background work")
        let token = epoch; let currentTask = task
        let portable = (page == .advanced && allowPortableExperiment) || Self.runsOnDeclaredCPU(profile)
        // Only a cloning run reads its reference as PCM. On Voiceover a
        // native cloner's request carries none (`runRequest`).
        let clonesNatively = currentTask == .textToSpeech && Self.runClonesNatively(profile, speech: inputs.speech)
        let experimentalMetal = page == .advanced && allowExperimentalMetal
        jobTask = Task { [weak self] in
            guard let self else { return }
            let admitted = await ImageJobCoordinator.shared.acquire()
            defer { if admitted { Task { await ImageJobCoordinator.shared.release() } } }
            defer {
                if self.valid(token) {
                    self.isBusy = false; self.progress = nil; self.generationProgress = nil; self.speechRun = nil; self.activeRunID = nil
                    self.scheduleUnload(); self.startNextDraft()
                }
            }
            do {
                guard admitted else { throw CancellationError() }
                try Task.checkCancellation()
                self.status = AppStrings.localized("Loading model")
                self.speechRun?.enter(.loadingModel)
                let session = profile.family == "qwen3_tts" ? nil : try await self.residentSession(for: profile, portable: portable, experimentalMetal: experimentalMetal, source: result)
                let request = Self.runRequest(inputs, task: currentTask, profile: profile)
                if currentTask == .speechToText {
                    guard let selected else { throw AudioWorkspaceError.message(AppStrings.localized("Import or select a recording first.")) }
                    guard let session else { throw CancellationError() }
                    try await self.transcribe(item: selected, into: result.id, session: session, request: request, source: .final, token: token)
                } else {
                    let pcm: [Float]
                    if [.enhancement, .separation, .alignment, .diarization, .speechDetection, .codec, .languageIdentification].contains(currentTask) {
                        guard let selected else { throw AudioWorkspaceError.message(AppStrings.localized("Import or select an audio source first.")) }
                        guard selected.duration <= Double(profile.capabilities.maxInputSeconds ?? 30) else {
                            throw AudioWorkspaceError.message(AppStrings.localized("This operation cannot process the full recording. Choose a shorter source."))
                        }
                        let assets = self.assets; let lifetime = self.lifetime
                        pcm = try await Task.detached { try AudioMediaIO.readMono16k(selected.clips, store: assets, start: 0, duration: selected.duration, cancel: lifetime.check) }.value
                    } else if clonesNatively {
                        guard let reference = inputs.speech?.reference, reference.isValid else {
                            throw AudioWorkspaceError.message(AppStrings.localized("Choose 1 to 30 seconds of reference speech."))
                        }
                        guard request.consent else {
                            throw AudioWorkspaceError.message(AppStrings.localized("Confirm you have the right to use this voice."))
                        }
                        self.status = AppStrings.localized("Preparing reference audio")
                        self.speechRun?.enter(.preparingReference)
                        let assets = self.assets; let lifetime = self.lifetime
                        pcm = try await Task.detached { try AudioMediaIO.readMono24k(reference.clips, store: assets, start: reference.start, duration: reference.duration, cancel: lifetime.check) }.value
                    } else { pcm = [] }
                    let report: @Sendable (AudioProgress) -> Void = { [weak self] update in
                        Task { @MainActor in self?.receiveProgress(update, runID: result.id, token: token) }
                    }
                    let output: AudioResult
                    if profile.family == "qwen3_tts" {
                        output = try await self.synthesizeQwenSpeech(inputs, profile: profile, source: result, report: report)
                    } else {
                        guard let session else { throw CancellationError() }
                        output = try await session.execute(request, pcm: pcm) { [weak self] update in
                            Task { @MainActor in
                                self?.receiveProgress(update, runID: result.id, token: token)
                            }
                        }
                    }
                    try Task.checkCancellation()
                    guard self.valid(token) else { return }
                    if synthesis { _ = try Self.validateGeneratedAudio(output.pcm) }
                    self.generationProgress?.saving()
                    self.speechRun?.enter(.saving)
                    self.status = AppStrings.localized("Saving audio")
                    self.progress = nil
                    var generatedClip: AudioLibraryClip?
                    if let audio = output.pcm {
                        let assets = self.assets; let lifetime = self.lifetime
                        let saved = try await Task.detached {
                            try lifetime.check()
                            let bytes = try AudioMediaIO.wav(samples: audio.samples, sampleRate: Double(audio.format.sampleRate), channels: Int(audio.format.channels))
                            try lifetime.check()
                            return try assets.store(data: bytes, fileName: "take.wav", mimeType: "audio/wav")
                        }.value
                        try Task.checkCancellation()
                        guard self.valid(token) else { return }
                        generatedClip = AudioLibraryClip(assetReference: saved.storedReference, sampleRate: Double(audio.format.sampleRate), channels: Int(audio.format.channels), frameCount: Int64(audio.samples.count / Int(audio.format.channels)))
                    }
                    try self.persistMutation(result.id) {
                        if let generatedClip { $0.clips = [generatedClip] }
                        $0.resolvedSeed = output.seed
                        if let transcript = output.transcript {
                            $0.addTranscript(Self.librarySegments(transcript, offset: 0), source: .final, language: transcript.language, nativeResult: try? JSONEncoder().encode(transcript))
                        }
                        $0.status = .completed; $0.updatedAt = Date()
                    }
                }
                guard self.valid(token) else { return }
                if currentTask == .speechToText { try self.persistMutation(result.id) { $0.status = .completed; $0.updatedAt = Date() } }
                self.status = AppStrings.localized("Saved in this profile")
            } catch {
                guard self.valid(token) else { return }
                let cancelled = Task.isCancelled
                if synthesis {
                    self.activeGeneration = nil
                    if self.selectedID == result.id { self.selectedID = selected?.id }
                } else if self.pendingSaves[result.id] == nil {
                    self.update(result.id) { $0.status = cancelled ? .interrupted : .failed; $0.failure = error.localizedDescription }
                }
                self.status = cancelled ? AppStrings.localized("Interrupted") : AppStrings.localized("Failed")
                self.error = cancelled ? nil : error.localizedDescription
            }
        }
    }
    static func librarySegments(_ transcript: AudioTranscript, offset: Double) -> [AudioLibrarySegment] {
        if transcript.segments.isEmpty, !transcript.text.isEmpty {
            return [AudioLibrarySegment(start: offset, end: offset, text: transcript.text, timing: "none")]
        }
        return transcript.segments.map {
            // Missing measured timing stays missing through the timing marker; NaN is
            // not JSON encodable and a zero-length extent is excluded from subtitles.
            AudioLibrarySegment(start: ($0.startSeconds ?? 0) + offset, end: ($0.endSeconds ?? $0.startSeconds ?? 0) + offset, text: $0.text, speaker: $0.speaker, timing: $0.timing)
        }
    }
    func transcribe(item: AudioLibraryItem, into id: UUID, session: AudioSession, request: AudioRequest, source: AudioTranscriptRevision.Source, token: UUID) async throws {
        var segments: [AudioLibrarySegment] = []
        var native: [AudioTranscript] = []
        let duration = item.duration
        let assets = self.assets; let lifetime = self.lifetime
        // Each window has a known recording offset. Model window extents remain
        // extents in the editor; they are never presented as word alignment.
        for offset in stride(from: 0.0, to: duration, by: 29.0) {
            try Task.checkCancellation()
            let seconds = min(30, duration - offset)
            let pcm = try await Task.detached { try AudioMediaIO.readMono16k(item.clips, store: assets, start: offset, duration: seconds, cancel: lifetime.check) }.value
            let result = try await session.execute(request, pcm: pcm)
            try Task.checkCancellation()
            guard valid(token) else { throw CancellationError() }
            if let transcript = result.transcript {
                native.append(transcript)
                var next = Self.librarySegments(transcript, offset: offset)
                Self.reconcileBoundary(previous: &segments, next: &next)
                segments.append(contentsOf: next)
            }
            progress = min(1, (offset + seconds) / max(1, duration))
            status = AppStrings.localized("Refining saved recording")
        }
        guard valid(token) else { throw CancellationError() }
        let metadata = try JSONEncoder().encode(native)
        // Reload inside the locked mutation so edits made during inference survive.
        try persistMutation(id) {
            $0.addTranscript(segments, source: source, language: native.compactMap(\.language).first, nativeResult: metadata)
        }
    }
    static func reconcileBoundary(previous: inout [AudioLibrarySegment], next: inout [AudioLibrarySegment]) {
        guard let last = previous.last, !next.isEmpty, next[0].start < last.end else { return }
        let left = last.text.split(whereSeparator: \.isWhitespace)
        let right = next[0].text.split(whereSeparator: \.isWhitespace)
        // Only remove exact overlap. Uncertain boundaries remain available to edit.
        for count in stride(from: min(20, min(left.count, right.count)), through: 2, by: -1) {
            if left.suffix(count).map({ $0.lowercased() }) == right.prefix(count).map({ $0.lowercased() }) {
                next[0].text = right.dropFirst(count).joined(separator: " ")
                if next[0].text.isEmpty { next.removeFirst() }
                break
            }
        }
    }
}

extension AudioWorkspaceController {
    func summarizeMeeting() {
        guard !isBusy, !hasRecordingActivity, canSummarizeMeeting, let item = selectedItem, let summaryHandler else { return }
        isBusy = true; error = nil
        let token = epoch
        jobTask = Task { [weak self] in
            guard let self else { return }
            defer { if self.valid(token) { self.isBusy = false } }
            do {
                let summary = try await summaryHandler(item)
                try Task.checkCancellation()
                guard self.valid(token) else { return }
                try self.persistMutation(item.id) { $0.summary = summary; $0.updatedAt = Date() }
            } catch { if self.valid(token), !Task.isCancelled { self.error = error.localizedDescription } }
        }
    }
}
