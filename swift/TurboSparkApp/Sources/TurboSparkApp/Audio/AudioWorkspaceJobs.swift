import Foundation
import TurboSpark

extension AudioWorkspaceController {
    func request(from recipe: AudioRecipe, task: AudioTask) -> AudioRequest {
        var request = AudioRequest(task: task, language: recipe.language == "auto" ? nil : recipe.language)
        request.text = recipe.text; request.voice = recipe.voice; request.speed = recipe.speed
        request.caption = recipe.caption; request.lyrics = recipe.lyrics
        request.durationSeconds = recipe.durationSeconds; request.steps = recipe.steps; request.seed = recipe.seed
        return request
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
        result.sourceID = selected?.id; result.status = .processing
        result.localModelBookmark = preferences.localModels[profile.identity.alias]
        result.modelFamily = profile.family
        if result.localModelBookmark == nil { result.modelIdentity = try? JSONEncoder().encode(profile.identity) }
        if task == .speechToText { result.clips = selected?.clips ?? [] }
        do { try save(result) } catch { fail(error); return }
        selectedID = result.id
        isBusy = true; progress = nil; error = nil
        let token = epoch; let currentTask = task; let portable = page == .advanced && allowPortableExperiment
        let experimentalMetal = page == .advanced && allowExperimentalMetal
        jobTask = Task { [weak self] in
            guard let self else { return }
            let admitted = await ImageJobCoordinator.shared.acquire()
            defer { if admitted { Task { await ImageJobCoordinator.shared.release() } } }
            defer {
                if self.valid(token) { self.isBusy = false; self.progress = nil; self.scheduleUnload(); self.startNextDraft() }
            }
            do {
                guard admitted else { throw CancellationError() }
                let session = try await self.residentSession(for: profile, portable: portable, experimentalMetal: experimentalMetal, source: result)
                let request = self.request(from: inputs, task: currentTask)
                if currentTask == .speechToText {
                    guard let selected else { throw AudioWorkspaceError.message(String(localized: "Import or select a recording first.", bundle: .module)) }
                    try await self.transcribe(item: selected, into: result.id, session: session, request: request, source: .final, token: token)
                } else {
                    let pcm: [Float]
                    if [.enhancement, .separation, .alignment, .diarization, .speechDetection, .codec, .languageIdentification].contains(currentTask) {
                        guard let selected else { throw AudioWorkspaceError.message(String(localized: "Import or select an audio source first.", bundle: .module)) }
                        guard selected.duration <= Double(profile.capabilities.maxInputSeconds ?? 30) else {
                            throw AudioWorkspaceError.message(String(localized: "This operation cannot process the full recording. Choose a shorter source.", bundle: .module))
                        }
                        let assets = self.assets; let lifetime = self.lifetime
                        pcm = try await Task.detached { try AudioMediaIO.readMono16k(selected.clips, store: assets, start: 0, duration: selected.duration, cancel: lifetime.check) }.value
                    } else { pcm = [] }
                    let output = try await session.execute(request, pcm: pcm) { [weak self] update in
                        Task { @MainActor in
                            guard let self, self.valid(token) else { return }
                            self.status = update.stage
                            self.progress = update.total.flatMap { $0 > 0 ? Double(update.completed) / Double($0) : nil }
                        }
                    }
                    try Task.checkCancellation()
                    guard self.valid(token) else { return }
                    var generatedClip: AudioLibraryClip?
                    if let audio = output.pcm {
                        let assets = self.assets; let lifetime = self.lifetime
                        let saved = try await Task.detached {
                            try lifetime.check()
                            let bytes = try AudioMediaIO.wav(samples: audio.samples, sampleRate: Double(audio.format.sampleRate), channels: Int(audio.format.channels))
                            try lifetime.check()
                            return try assets.store(data: bytes, fileName: "take.wav", mimeType: "audio/wav")
                        }.value
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
            } catch {
                guard self.valid(token) else { return }
                let cancelled = Task.isCancelled
                if self.pendingSaves[result.id] == nil { self.update(result.id) { $0.status = cancelled ? .interrupted : .failed; $0.failure = error.localizedDescription } }
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
            status = String(localized: "Refining saved recording", bundle: .module)
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
