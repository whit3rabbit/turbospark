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
        request.durationSeconds = recipe.durationSeconds
        request.steps = task == .music ? recipe.effectiveMusicStepsOverride : recipe.steps
        request.seed = recipe.seed
        if task == .music {
            let rate = recipe.music?.frameRate ?? 25.0
            request.maxFrames = Int((recipe.durationSeconds * rate).rounded())
            request.contextSize = recipe.music?.contextSize ?? 10_240
        }
        return request
    }
    /// The request a run sends. A native cloner's run either clones, and
    /// passes its reference as PCM (`runClonesNatively`), or speaks with no
    /// reference.
    static func runRequest(_ recipe: AudioRecipe, task: AudioTask, profile: AudioProfile) -> AudioRequest {
        var base = request(from: recipe, task: task)
        // The Clean Up page reads the take at 48 kHz mono (`readMono48kTake`)
        // to enhance it and at 44.1 kHz stereo (`readStereo44kTake`) to
        // separate it.
        if task == .enhancement { base.inputSampleRate = 48_000 }
        if task == .separation {
            base.inputSampleRate = 44_100
            base.inputChannels = 2
        }
        guard task == .textToSpeech, clonesNatively(profile) else { return base }
        var request = runClonesNatively(profile, speech: recipe.speech)
            ? nativeCloneRequest(base, family: profile.family, speech: recipe.speech)
            : referenceFreeSpeechRequest(base, family: profile.family)
        // Breeze designs a voice from the instruction alone (Voice Design);
        // Voiceover speaks with the preset speaker and no instruction.
        if profile.family == "breeze_tts", recipe.speech?.mode == .design {
            let instruction = recipe.speech?.instruction.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            request.instruction = instruction.isEmpty ? nil : instruction
        }
        return request
    }
    /// A natively cloning run (MegaTTS3, OmniVoice, Drama 3): its reference
    /// arrives as the session PCM at 24 kHz, and consent holds only for that
    /// exact region.
    nonisolated static func nativeCloneRequest(
        _ base: AudioRequest, family: String, speech: AudioSpeechSettings?,
        freshSeed: () -> UInt64 = { UInt64.random(in: 1...UInt64.max) }
    ) -> AudioRequest {
        var request = base
        let options = speech?.cloneOptions ?? AudioCloneOptions()
        request.steps = options.steps
        request.referenceSampleRate = 24_000
        request.consent = speech?.reference != nil && options.consentedReference == speech?.reference
        if family == "omnivoice" {
            // OmniVoice refuses MegaTTS3's guidance weights. It prepends the
            // reference transcript to the script, and upstream transcribes
            // the reference itself when given none, so the user's transcript
            // is sent. Its instruction is voice design, which a clone does
            // not use, and the saved instruction is Qwen3's.
            request.pW = nil
            request.tW = nil
            request.instruction = nil
            let transcript = speech?.referenceText.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            request.referenceText = transcript.isEmpty ? nil : transcript
            return request
        }
        if family == "breeze_tts" {
            // Breeze has no steps, speed or guidance weights. A clone needs
            // the reference's exact words; an instruction directs the cloned
            // voice (voice direction), so it is kept when the user gave one.
            request.steps = nil
            request.speed = 1
            request.pW = nil
            request.tW = nil
            let transcript = speech?.referenceText.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            request.referenceText = transcript.isEmpty ? nil : transcript
            let instruction = speech?.instruction.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            request.instruction = instruction.isEmpty ? nil : instruction
            if (request.seed ?? 0) == 0 { request.seed = freshSeed() }
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
            // Seed 0 is a new take each run, as on Voiceover: the runtime
            // takes 0 literally, and a take that goes silent after its first
            // sentence would repeat on every Generate.
            if (request.seed ?? 0) == 0 { request.seed = freshSeed() }
            return request
        }
        request.pW = options.intelligibility
        request.tW = options.similarity
        // The engine aligns the reference itself and takes no style prompt.
        request.instruction = nil
        request.referenceText = nil
        return request
    }
    /// Retakes of a take the runtime reports as stopped early (Drama 3 went
    /// silent after a sentence, turn or pause and never spoke the rest).
    nonisolated static let earlyStopRetakes = 2
    /// The next request after `output`, or nil to keep it: a take that stopped
    /// speaking early is retaken with a fresh seed, at most `earlyStopRetakes`
    /// times, unless the user chose the seed (that seed gives the same take).
    nonisolated static func retakeRequest(
        after output: AudioResult, sent: AudioRequest, recipeSeed: UInt64?, retakes: Int,
        freshSeed: () -> UInt64 = { UInt64.random(in: 1...UInt64.max) }
    ) -> AudioRequest? {
        guard output.stoppedEarly, (recipeSeed ?? 0) == 0, retakes < earlyStopRetakes else { return nil }
        var next = sent
        next.seed = freshSeed()
        return next
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
        // Drama 3 and Breeze have no speed, and the page hides the slider.
        if family == "fish_drama3" || family == "breeze_tts" { request.speed = 1 }
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
        let cleanTranscript = cleanTranscripts
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
                let provider = AudioRuntimeProvider(profile)
                // FluidAudio runs speech recognition only. Rows for its other
                // tasks route as unsupported, which keeps them out of the picker.
                if case .fluidAudio(let family) = provider, currentTask == .speechToText {
                    guard let selected else { throw AudioWorkspaceError.message(AppStrings.localized("Import or select a recording first.")) }
                    let transcription = try await self.transcribeWithFluidAudio(item: selected, into: result.id, profile: profile, family: family, source: .final, token: token)
                    // Speaker labels are extra here too, as on the Rust path below.
                    try await self.completeFinalTranscription(item: selected, into: result.id, transcription: transcription, token: token)
                    return
                }
                if provider == .appleSpeech, currentTask == .speechToText {
                    guard let selected else { throw AudioWorkspaceError.message(AppStrings.localized("Import or select a recording first.")) }
                    let transcription = try await self.transcribeWithAppleSpeech(item: selected, into: result.id, source: .final, token: token)
                    try await self.completeFinalTranscription(item: selected, into: result.id, transcription: transcription, token: token)
                    return
                }
                if provider != .rust && provider != .qwenMLX && provider != .breezeMLX {
                    throw AudioWorkspaceError.message(AppStrings.localized("Runtime unavailable"))
                }
                // The offline Voice Isolator renders on MLX, about 13x faster
                // than the Rust Metal path; nil keeps the Rust session.
                let isolator = currentTask == .enhancement ? try await self.offlineVoiceIsolator(for: profile, source: result) : nil
                let session = provider == .qwenMLX || provider == .breezeMLX || isolator != nil ? nil : try await self.residentSession(for: profile, portable: portable, experimentalMetal: experimentalMetal, source: result)
                let request = Self.transcriptionRequest(Self.runRequest(inputs, task: currentTask, profile: profile),
                                                        family: profile.family, clean: cleanTranscript)
                var labelFailure: String?
                var stoppedEarly = false
                if currentTask == .speechToText {
                    guard let selected else { throw AudioWorkspaceError.message(AppStrings.localized("Import or select a recording first.")) }
                    guard let session else { throw CancellationError() }
                    let transcription = try await self.transcribe(item: selected, into: result.id, session: session, request: request, source: .final, token: token)
                    // Speaker labels are extra, as in the Record page's final pass.
                    labelFailure = try await self.labelSpeakersBestEffort(item: selected, into: result.id, revision: transcription.revision, words: transcription.words, token: token)
                } else {
                    let pcm: [Float]
                    if [.enhancement, .separation, .alignment, .diarization, .speechDetection, .codec, .languageIdentification].contains(currentTask) {
                        guard let selected else { throw AudioWorkspaceError.message(AppStrings.localized("Import or select an audio source first.")) }
                        guard selected.duration <= Double(profile.capabilities.maxInputSeconds ?? 30) else {
                            throw AudioWorkspaceError.message(AppStrings.localized("This operation cannot process the full recording. Choose a shorter source."))
                        }
                        let assets = self.assets; let lifetime = self.lifetime
                        if currentTask == .diarization {
                            let limit = Double(profile.capabilities.maxInputSeconds ?? 30)
                            pcm = try await Task.detached { try AudioMediaIO.readMono16kTake(selected.clips, store: assets, duration: selected.duration, limit: limit, cancel: lifetime.check) }.value
                        } else if currentTask == .enhancement {
                            let limit = Double(profile.capabilities.maxInputSeconds ?? 30)
                            pcm = try await Task.detached { try AudioMediaIO.readMono48kTake(selected.clips, store: assets, duration: selected.duration, limit: limit, cancel: lifetime.check) }.value
                        } else if currentTask == .separation {
                            let limit = Double(profile.capabilities.maxInputSeconds ?? 30)
                            pcm = try await Task.detached { try AudioMediaIO.readStereo44kTake(selected.clips, store: assets, duration: selected.duration, limit: limit, cancel: lifetime.check) }.value
                        } else {
                            pcm = try await Task.detached { try AudioMediaIO.readMono16k(selected.clips, store: assets, start: 0, duration: selected.duration, cancel: lifetime.check) }.value
                        }
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
                    var output: AudioResult
                    if provider == .qwenMLX {
                        output = try await self.synthesizeQwenSpeech(inputs, profile: profile, source: result, report: report)
                    } else if provider == .breezeMLX {
                        var sent = request
                        output = try await self.synthesizeBreezeSpeech(sent, reference: pcm, profile: profile, source: result, report: report)
                        var retakes = 0
                        while let retake = Self.retakeRequest(after: output, sent: sent, recipeSeed: inputs.seed, retakes: retakes) {
                            try Task.checkCancellation()
                            guard self.valid(token) else { return }
                            retakes += 1
                            sent = retake
                            output = try await self.synthesizeBreezeSpeech(sent, reference: pcm, profile: profile, source: result, report: report)
                        }
                    } else if let isolator {
                        output = try await isolator.enhance(request, pcm: pcm, progress: report)
                    } else {
                        guard let session else { throw CancellationError() }
                        var sent = request
                        output = try await session.execute(sent, pcm: pcm, progress: report)
                        var retakes = 0
                        while let retake = Self.retakeRequest(after: output, sent: sent, recipeSeed: inputs.seed, retakes: retakes) {
                            try Task.checkCancellation()
                            guard self.valid(token) else { return }
                            retakes += 1
                            sent = retake
                            output = try await session.execute(sent, pcm: pcm, progress: report)
                        }
                    }
                    try Task.checkCancellation()
                    guard self.valid(token) else { return }
                    if synthesis { _ = try Self.validateGeneratedAudio(output.pcm) }
                    stoppedEarly = output.stoppedEarly
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
                            var segments = Self.librarySegments(transcript, offset: 0)
                            if currentTask == .diarization {
                                for index in segments.indices { segments[index].speaker = segments[index].speaker.map(AudioSpeakerLabels.displayName) }
                            }
                            let vocabulary = AudioCustomVocabularyStore.load()
                            if !vocabulary.isEmpty {
                                segments = AudioCustomVocabulary.apply(vocabulary, to: segments)
                            }
                            $0.addTranscript(segments, source: .final, language: transcript.language, nativeResult: try? JSONEncoder().encode(transcript))
                        }
                        $0.status = .completed; $0.updatedAt = Date()
                    }
                }
                guard self.valid(token) else { return }
                if currentTask == .speechToText { try self.persistMutation(result.id) { $0.status = .completed; $0.updatedAt = Date() } }
                if let labelFailure { self.status = AppStrings.localized("Transcript saved. Speakers could not be labeled: \(labelFailure)") }
                else { self.status = AppStrings.localized("Saved in this profile") }
                if stoppedEarly {
                    self.error = AppStrings.localized("The model stopped speaking before the end of the text. Generate again for a new take.")
                }
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
    /// The revision a transcription added, and its word timings on the
    /// recording's timeline (empty for models without word alignment).
    struct Transcription {
        var revision: UUID
        var words: [ClosedRange<Double>]
    }
    @discardableResult func transcribe(item: AudioLibraryItem, into id: UUID, session: AudioSession, request: AudioRequest, source: AudioTranscriptRevision.Source, token: UUID) async throws -> Transcription {
        let windows = AudioTranscriptionPlan.windows(duration: item.duration, length: 30, overlap: 1)
        return try await transcribe(item: item, into: id, windows: windows, source: source, token: token) { pcm, _ in
            try await session.execute(request, pcm: pcm).transcript
        }
    }
    /// The window loop every recognizer shares: read each planned window,
    /// let `run` transcribe it, place its segments at the window's offset, and
    /// save once at the end.
    @discardableResult func transcribe(item: AudioLibraryItem, into id: UUID, windows: [AudioTranscriptionWindow], source: AudioTranscriptRevision.Source, token: UUID,
                    run: (_ pcm: [Float], _ window: AudioTranscriptionWindow) async throws -> AudioTranscript?) async throws -> Transcription {
        var segments: [AudioLibrarySegment] = []
        var native: [AudioTranscript] = []
        var words: [ClosedRange<Double>] = []
        let duration = item.duration
        let assets = self.assets; let lifetime = self.lifetime
        // Each window has a known recording offset. Model window extents remain
        // extents in the editor; they are never presented as word alignment.
        let clips = item.clips
        func read(_ window: AudioTranscriptionWindow) -> Task<[Float], Error> {
            Task.detached { try AudioMediaIO.readMono16k(clips, store: assets, start: window.start, duration: window.duration, cancel: lifetime.check) }
        }
        // Read and decrypt the next window while the recognizer runs this
        // one. Same windows, same order, same results: only the wait moves.
        var pending = windows.first.map(read)
        defer { pending?.cancel() }
        for (index, window) in windows.enumerated() {
            try Task.checkCancellation()
            let offset = window.start, seconds = window.duration
            guard let current = pending else { break }
            let pcm = try await current.value
            pending = index + 1 < windows.count ? read(windows[index + 1]) : nil
            let transcript = try await run(pcm, window)
            try Task.checkCancellation()
            guard valid(token) else { throw CancellationError() }
            if let transcript {
                native.append(transcript)
                for word in transcript.segments.flatMap({ $0.words ?? [] }) where word.endSeconds >= word.startSeconds {
                    words.append((word.startSeconds + offset)...(word.endSeconds + offset))
                }
                var next = Self.librarySegments(transcript, offset: offset)
                Self.reconcileBoundary(previous: &segments, next: &next)
                segments.append(contentsOf: next)
            }
            progress = min(1, (offset + seconds) / max(1, duration))
            status = AppStrings.localized("Refining saved recording")
        }
        guard valid(token) else { throw CancellationError() }
        let vocabulary = AudioCustomVocabularyStore.load()
        if !vocabulary.isEmpty {
            segments = AudioCustomVocabulary.apply(vocabulary, to: segments)
        }
        let metadata = try JSONEncoder().encode(native)
        var revision = UUID()
        // Reload inside the locked mutation so edits made during inference survive.
        try persistMutation(id) {
            revision = $0.addTranscript(segments, source: source, language: native.compactMap(\.language).first, nativeResult: metadata)
        }
        return Transcription(revision: revision, words: words)
    }
    static func reconcileBoundary(previous: inout [AudioLibrarySegment], next: inout [AudioLibrarySegment]) {
        guard let last = previous.last, !next.isEmpty, next[0].start < last.end else { return }
        let left = last.text.split(whereSeparator: \.isWhitespace)
        let right = next[0].text.split(whereSeparator: \.isWhitespace)
        // A window that cuts a sentence ends its copy with a period the next
        // window's copy lacks, so words compare without case or punctuation.
        let word = { (token: Substring) in token.lowercased().filter { $0.isLetter || $0.isNumber } }
        // Only remove exact overlap. Uncertain boundaries remain available to edit.
        for count in stride(from: min(20, min(left.count, right.count)), through: 2, by: -1) {
            if left.suffix(count).map(word) == right.prefix(count).map(word) {
                if count == left.count {
                    // The earlier window heard only the start of this segment
                    // before its cut. Keep the later, whole copy from the
                    // earlier start.
                    next[0].start = min(last.start, next[0].start)
                    previous.removeLast()
                } else {
                    next[0].text = right.dropFirst(count).joined(separator: " ")
                    if next[0].text.isEmpty { next.removeFirst() }
                }
                break
            }
        }
    }
}

extension AudioWorkspaceController {
    func summarizeMeeting(with targetModel: InstalledModel? = nil) {
        guard let item = selectedItem else { return }
        summarizeMeeting(item.id, with: targetModel)
    }

    /// Writes notes for one take. Only `item.summary` changes; the preview
    /// shows the stream and is cleared however the run ends.
    func summarizeMeeting(_ id: UUID, with targetModel: InstalledModel?) {
        guard !isBusy, !hasRecordingActivity, let item = items.first(where: { $0.id == id }),
              item.preferredTranscript != nil, canSummarize, let summaryHandler else { return }
        isBusy = true; error = nil
        summaryPreview = AudioSummaryPreview(itemID: id, progress: AudioSummaryProgress(stage: AppStrings.localized("Starting the model"), text: ""))
        let token = epoch
        jobTask = Task { [weak self] in
            guard let self else { return }
            defer { if self.valid(token) { self.isBusy = false; self.summaryPreview = nil } }
            do {
                let summary = try await summaryHandler(item, targetModel) { [weak self] progress in
                    guard let self, self.valid(token), self.summaryPreview?.itemID == id else { return }
                    self.summaryPreview?.progress = progress
                }
                try Task.checkCancellation()
                guard self.valid(token) else { return }
                try self.persistMutation(id) { $0.summary = summary; $0.updatedAt = Date() }
                if self.selectedID != id { self.status = AppStrings.localized("Meeting notes are ready for \(item.title).") }
            } catch { if self.valid(token), !Task.isCancelled { self.error = error.localizedDescription } }
        }
    }

    /// Whether any chat model could write notes: one loaded, or one installed.
    var canSummarize: Bool {
        summaryAvailable?() == true || selectedChatModel?() != nil || !(availableChatModels?().isEmpty ?? true)
    }

    var autoMeetingNotes: Bool {
        get { preferences.autoMeetingNotes ?? false }
        set { preferences.autoMeetingNotes = newValue; persistPreferences() }
    }

    var cleanTranscripts: Bool {
        get { preferences.cleanTranscripts ?? false }
        set { preferences.cleanTranscripts = newValue; persistPreferences() }
    }

    /// Only Fluid Speech Pro on the CPU has a clean style; every other model
    /// refuses `transcriptStyle`, so it is set for that family alone.
    nonisolated static let cleanTranscriptFamily = "fluid_speech_pro"
    nonisolated static func transcriptionRequest(_ base: AudioRequest, family: String, clean: Bool) -> AudioRequest {
        var request = base
        request.transcriptStyle = family == cleanTranscriptFamily && clean ? "clean" : nil
        return request
    }

    /// After a Meeting take's final pass, when the user turned it on.
    func autoSummarizeIfWanted(_ id: UUID) {
        guard autoMeetingNotes, let item = items.first(where: { $0.id == id }),
              item.recordingPreset == AudioRecordingPreset.meeting.rawValue, item.summary == nil else { return }
        summarizeMeeting(id, with: notesModel)
    }

    /// The chat model notes and Ask run on: the chosen one, else the one the
    /// notes advisor recommends. The runner loads it when it is not loaded.
    var notesModel: InstalledModel? {
        selectedChatModel?() ?? AudioMeetingNotesAdvisor.defaultModel(from: availableChatModels?() ?? [], currentSelected: nil)
    }
}
