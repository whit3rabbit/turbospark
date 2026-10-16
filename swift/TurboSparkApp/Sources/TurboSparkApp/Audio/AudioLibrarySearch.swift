import Foundation
import TurboSpark

/// Splits transcript segments into chunks for embedding, each within the
/// model's token budget and timed by the segments it holds. EmbeddingGemma 2
/// truncates past 512 tokens without an error, so the budget leaves room for
/// the task prompt and the begin and end tokens.
enum AudioSearchChunker {
    static let tokenBudget = 480

    struct Chunk: Equatable { let start: Double; let end: Double; let text: String }

    static func chunks(_ segments: [AudioSearchProjection.Segment], budget: Int = tokenBudget,
                       count: (String) -> Int) -> [Chunk] {
        var out: [Chunk] = []
        var text = "", start = 0.0, end = 0.0, tokens = 0
        func close() {
            if !text.isEmpty { out.append(Chunk(start: start, end: end, text: text)) }
            text = ""; tokens = 0
        }
        for segment in segments where !segment.text.isEmpty {
            // A segment too long by itself is split at word boundaries.
            for piece in split(segment.text, budget: budget, count: count) {
                let size = count(piece)
                if tokens > 0 && tokens + size > budget { close() }
                if text.isEmpty { start = segment.start }
                text += (text.isEmpty ? "" : " ") + piece
                tokens += size
                end = segment.end
            }
        }
        close()
        return out
    }

    private static func split(_ text: String, budget: Int, count: (String) -> Int) -> [String] {
        guard count(text) > budget else { return [text] }
        var pieces: [String] = [], current = ""
        for word in text.split(separator: " ") {
            let next = current.isEmpty ? String(word) : current + " " + word
            if !current.isEmpty && count(next) > budget { pieces.append(current); current = String(word) }
            else { current = next }
        }
        if !current.isEmpty { pieces.append(current) }
        return pieces
    }
}

/// Library search for the Audio workspace: exact words from the index at
/// once, then meaning and sound when EmbeddingGemma 2 is installed, merged
/// into one ranked list. Also runs the background indexer that embeds
/// transcripts and audio for the semantic lanes.
@MainActor
final class AudioLibrarySearch: ObservableObject {
    @Published private(set) var results: [AudioSearchResult] = []
    /// Why meaning and sound are off; nil when they run.
    @Published private(set) var semanticUnavailableReason: String?
    @Published private(set) var isIndexing = false
    private var semantic: AudioSemanticSearch?
    private var queryTask: Task<Void, Never>?
    private var indexTask: Task<Void, Never>?
    private var failed = Set<String>()

    static let audioSpanSeconds = 300.0

    /// Finds the installed EmbeddingGemma 2 by what it embeds, not by the
    /// memory setting, which may name a text-only model.
    func refreshModel() async {
        let profiles = await Task.detached { (try? EmbeddingCatalog.profiles()) ?? [] }.value
        guard let profile = profiles.first(where: { $0.modalities.contains("audio") && $0.modalities.contains("text") }) else {
            semantic = nil
            semanticUnavailableReason = AppStrings.localized("Search matches exact words. Meaning and sound search need an embedding model this catalog does not list.")
            return
        }
        guard profile.canRun, let path = profile.installedPath else {
            semantic = nil
            semanticUnavailableReason = profile.unavailableReason
                ?? AppStrings.localized("Search matches exact words. Download \(profile.displayName) in Settings > Memory to also search by meaning and sound.")
            return
        }
        let key = "\(profile.alias)@\(profile.revision)"
        if semantic?.modelKey != key { semantic = AudioSemanticSearch(modelPath: path, modelKey: key) }
        semanticUnavailableReason = nil
    }

    func run(_ query: String, controller: AudioWorkspaceController) {
        queryTask?.cancel()
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { results = []; return }
        let words = controller.library.lexicalSearch(trimmed) + Self.substringHits(controller.items, trimmed)
        results = AudioSearchFusion.fuse(words: words, meaning: [], sound: [])
        guard let semantic, let index = controller.library.searchIndex() else { return }
        queryTask = Task { [weak self] in
            // Typing settles before the model runs.
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled, let found = try? await semantic.search(trimmed, index: index), !Task.isCancelled else { return }
            self?.results = AudioSearchFusion.fuse(words: words, meaning: found.meaning, sound: found.sound)
        }
    }

    /// Substring matches the index can miss: CJK text (unicode61 does not
    /// segment it) and a profile whose index schema did not open.
    nonisolated static func substringHits(_ items: [AudioLibraryItem], _ query: String) -> [AudioLexicalHit] {
        items.compactMap { item in
            guard !item.isEmptyGeneratedAttempt else { return nil }
            if item.title.localizedCaseInsensitiveContains(query) || item.collection.localizedCaseInsensitiveContains(query) {
                return AudioLexicalHit(itemID: item.id, start: nil, end: nil, snippet: item.title)
            }
            guard let segment = item.preferredTranscript?.segments.first(where: { $0.text.localizedCaseInsensitiveContains(query) }) else { return nil }
            return AudioLexicalHit(itemID: item.id, start: segment.start, end: segment.end, snippet: segment.text)
        }
    }

    /// Brings the index up to date: words for every item at once, then
    /// vectors one item at a time in the background. Pauses while a take
    /// records or a job runs, and backs off while chat or image work holds
    /// the device.
    func scheduleIndexing(_ controller: AudioWorkspaceController) {
        guard indexTask == nil else { return }
        for item in controller.items { try? controller.library.indexForSearch(item) }
        let token = controller.epoch
        indexTask = Task { [weak self, weak controller] in
            await self?.refreshModel()
            guard let self, let semantic = self.semantic else { self?.indexTask = nil; return }
            self.isIndexing = true
            defer { self.isIndexing = false; self.indexTask = nil }
            let tokenizer = try? TurboSparkTokenizer(modelPath: semantic.modelPath)
            defer { tokenizer?.close() }
            while !Task.isCancelled, let controller, controller.valid(token) {
                if controller.hasRecordingActivity || controller.isBusy {
                    try? await Task.sleep(for: .seconds(5)); continue
                }
                guard let (item, kind) = self.nextWork(controller.items, controller: controller, model: semantic.modelKey) else { return }
                let job = "\(item.id)-\(kind.rawValue)"
                do {
                    guard try await self.embed(item, kind, controller: controller, semantic: semantic, tokenizer: tokenizer) else {
                        try? await Task.sleep(for: .seconds(5)); continue
                    }
                    semantic.invalidate()
                } catch {
                    // Skipped for this pass; the next launch or edit retries it.
                    self.failed.insert(job)
                }
            }
        }
    }

    func stopIndexing() { indexTask?.cancel(); indexTask = nil; queryTask?.cancel() }

    private func nextWork(_ items: [AudioLibraryItem], controller: AudioWorkspaceController, model: String) -> (AudioLibraryItem, AudioVectorKind)? {
        guard let index = controller.library.searchIndex() else { return nil }
        for item in items {
            guard let projection = item.searchProjection else { continue }
            if !projection.segments.isEmpty, !failed.contains("\(item.id)-text"),
               (try? index.audioVectorKey(itemID: item.id, kind: .text)) != "\(model)|\(projection.hash)" {
                return (item, .text)
            }
            if item.hasPlayableAudio, !failed.contains("\(item.id)-audio"),
               (try? index.audioVectorKey(itemID: item.id, kind: .audio)) != "\(model)|\(item.audioFingerprint)" {
                return (item, .audio)
            }
        }
        return nil
    }

    /// One item's vectors of one kind. Returns false when the device is
    /// busy, so the caller waits and tries again.
    private func embed(_ item: AudioLibraryItem, _ kind: AudioVectorKind, controller: AudioWorkspaceController,
                       semantic: AudioSemanticSearch, tokenizer: TurboSparkTokenizer?) async throws -> Bool {
        guard let index = controller.library.searchIndex(), let projection = item.searchProjection else { return true }
        let admitted = await ImageJobCoordinator.shared.acquire()
        defer { if admitted { Task { await ImageJobCoordinator.shared.release() } } }
        guard admitted, let permit = try NativeHeavyWorkPermit.tryAcquire() else { return false }
        defer { permit.release() }
        var rows: [AudioVectorRow] = []
        let key: String
        switch kind {
        case .text:
            let count: (String) -> Int = { text in (try? tokenizer?.count(text)) ?? (text.count / 3 + 1) }
            let chunks = AudioSearchChunker.chunks(projection.segments, count: count)
            for batch in stride(from: 0, to: chunks.count, by: 16).map({ Array(chunks[$0..<min($0 + 16, chunks.count)]) }) {
                try Task.checkCancellation()
                let request = EmbeddingRequest(texts: batch.map(\.text), task: "document", dimensions: AudioSemanticSearch.textDimensions)
                let vectors = try await TurboSparkEmbedding.encode(request: request, modelPath: semantic.modelPath).texts
                for (chunk, vector) in zip(batch, vectors) {
                    rows.append(AudioVectorRow(itemID: item.id, index: rows.count, start: chunk.start, end: chunk.end, text: chunk.text, vector: vector))
                }
            }
            key = "\(semantic.modelKey)|\(projection.hash)"
        case .audio:
            let assets = controller.assets
            var start = 0.0
            while start < item.duration {
                try Task.checkCancellation()
                let length = min(Self.audioSpanSeconds, item.duration - start)
                let spanStart = start
                let pcm = try await Task.detached { try AudioMediaIO.readMono16k(item.clips, store: assets, start: spanStart, duration: length) }.value
                start += length
                guard pcm.count >= 1_600 else { continue }
                // Decrypted audio exists on disk only for this call.
                let url = try controller.scratchURL(extension: "wav")
                defer { try? FileManager.default.removeItem(at: url); controller.scratchURLs.remove(url) }
                try AudioMediaIO.wav(samples: pcm, sampleRate: 16_000, channels: 1).write(to: url, options: .atomic)
                let request = EmbeddingRequest(dimensions: AudioSemanticSearch.audioDimensions, audioPaths: [url.path])
                for window in try await TurboSparkEmbedding.encode(request: request, modelPath: semantic.modelPath).audio.first ?? [] {
                    let at = spanStart + window.startSeconds
                    rows.append(AudioVectorRow(itemID: item.id, index: rows.count, start: at, end: at + window.durationSeconds, text: nil, vector: window.embedding))
                }
            }
            key = "\(semantic.modelKey)|\(item.audioFingerprint)"
        }
        try index.replaceAudioVectors(itemID: item.id, kind: kind, model: semantic.modelKey, key: key, rows: rows)
        return true
    }
}
