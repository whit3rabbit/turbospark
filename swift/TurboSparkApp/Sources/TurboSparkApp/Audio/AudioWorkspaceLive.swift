import Foundation
import TurboSpark

/// Live text for the take being recorded, observed by the Record panel.
/// Held by the controller as one object so the panel redraws on live text
/// alone, not on every controller change.
@MainActor
final class AudioLiveText: ObservableObject {
    @Published private(set) var text = ""
    @Published private(set) var openWord: String?
    /// One user-facing line: preparing, or why live text stopped.
    @Published var status: String?
    @Published private(set) var isStreaming = false
    private(set) var itemID: UUID?
    var transcriber: AudioLiveTranscriber?
    /// The one draft revision this take's live text is saved into.
    private(set) var revisionID = UUID()
    var lastSaved = Date.distantPast
    private(set) var committedThrough: Double = 0

    func begin(itemID: UUID, transcriber: AudioLiveTranscriber) {
        self.itemID = itemID; self.transcriber = transcriber
        text = ""; openWord = nil; status = nil; isStreaming = true
        revisionID = UUID(); lastSaved = .distantPast; committedThrough = 0
    }
    func apply(_ update: AudioStreamUpdate) {
        text += update.appendedText
        openWord = update.isFinal ? nil : update.openWord
        committedThrough = max(committedThrough, update.committedThroughSeconds)
    }
    /// Streaming has ended for this take; its text stays on screen.
    func end() { isStreaming = false; transcriber = nil; openWord = nil }
    func clear() { end(); itemID = nil; text = ""; status = nil }
}

extension AudioWorkspaceController {
    static let liveTextAlias = "parakeet-unified-en-coreml"

    var liveTextProfile: AudioProfile? { profiles.first { $0.identity.alias == Self.liveTextAlias } }
    var liveTextModelPath: String? { installedPaths[Self.liveTextAlias] }

    /// Why the next take will not stream live text, or nil when it will.
    /// Without streaming, the draft-every-15-seconds path still runs.
    var liveStreamingUnavailableReason: String? {
        guard let profile = liveTextProfile else {
            return AppStrings.localized("Streaming live text needs the live text model, which this catalog does not list.")
        }
        if !profile.capabilities.canRun {
            return profile.capabilities.unavailableReason
                ?? AppStrings.localized("The live text model cannot run on this Mac.")
        }
        if liveTextModelPath == nil {
            return AppStrings.localized("Download the live text model to see words as they are spoken.")
        }
        if !Self.isEnglish(recipe.language) {
            return AppStrings.localized("Live text streams English only. Other languages show a draft every 15 seconds.")
        }
        return nil
    }
    var streamsLiveText: Bool { liveTranscription && liveStreamingUnavailableReason == nil }

    /// "auto" follows the first preferred system language: the streaming
    /// model writes fluent wrong English for other speech, so it is never
    /// guessed into a non-English take.
    nonisolated static func isEnglish(_ language: String, preferred: [String] = Locale.preferredLanguages) -> Bool {
        let code = language == "auto" ? (preferred.first ?? "") : language
        return code == "en" || code.hasPrefix("en-") || code.hasPrefix("en_")
    }

    func installLiveTextModel() {
        guard let profile = liveTextProfile, !profile.identity.repository.isEmpty else { return }
        if let installHandler { _ = installHandler(profile, []) }
    }

    /// Starts streaming live text for a take, when it applies. Called before
    /// capture starts so the feed sees the first packet.
    func startLiveText(itemID: UUID, capture: AudioCapture) {
        live.clear()
        guard streamsLiveText, let path = liveTextModelPath else { return }
        let feed = AudioLiveFeed()
        capture.attachLiveFeed(feed)
        let token = epoch
        let transcriber = AudioLiveTranscriber(feed: feed, open: { try AudioStreamSession(modelPath: path) }) { [weak self] event in
            Task { @MainActor [weak self] in self?.receiveLive(event, itemID: itemID, token: token) }
        }
        live.begin(itemID: itemID, transcriber: transcriber)
        transcriber.start()
    }

    func receiveLive(_ event: AudioLiveTranscriber.Event, itemID: UUID, token: UUID) {
        guard valid(token), live.itemID == itemID else { return }
        switch event {
        case .preparing:
            live.status = AppStrings.localized("Preparing live text. The first time on this Mac can take a few minutes.")
        case .update(let update):
            live.status = nil
            live.apply(update)
            if Date().timeIntervalSince(live.lastSaved) >= 5 { persistLiveDraft(itemID) }
        case .stopped(let reason):
            live.status = reason
            persistLiveDraft(itemID)
            // Drafts pick up where live text stopped, for the rest of the take.
            draftThrough = max(draftThrough, live.committedThrough)
            live.end()
            if !isBusy { startNextDraft() }
        }
    }

    /// Saves the take's live text into its one draft revision, replacing
    /// what the last save wrote rather than adding a revision each time.
    func persistLiveDraft(_ itemID: UUID, words: [AudioStreamWord]? = nil) {
        guard let words = words ?? live.transcriber?.snapshot().words else { return }
        let segments = Self.liveSegments(words)
        guard !segments.isEmpty else { return }
        live.lastSaved = Date()
        let revisionID = live.revisionID
        update(itemID) { $0.upsertTranscript(id: revisionID, segments, source: .draft, language: "en") }
    }

    /// Queues the final pass for a stopped take. With live text running, it
    /// first commits the tail, saves the draft, and closes the model, so the
    /// final pass never runs beside an open stream.
    func queueRefinement(_ id: UUID, refine: Bool) {
        guard live.itemID == id, let transcriber = live.transcriber else {
            if refine { pendingRefinementIDs.append(id) }
            return
        }
        live.transcriber = nil
        let token = epoch
        Task { [weak self] in
            let final = await transcriber.finishAndClose()
            guard let self, self.valid(token) else { return }
            self.persistLiveDraft(id, words: final.words)
            self.live.end()
            if refine { self.pendingRefinementIDs.append(id) }
            if !self.isBusy { self.startNextDraft() }
        }
    }

    /// Words grouped into segments at sentence ends, or every 15 seconds.
    nonisolated static func liveSegments(_ words: [AudioStreamWord], maxSeconds: Double = 15) -> [AudioLibrarySegment] {
        var segments: [AudioLibrarySegment] = []
        var current: [AudioStreamWord] = []
        func close() {
            guard let first = current.first, let last = current.last else { return }
            segments.append(AudioLibrarySegment(start: first.startSeconds, end: max(first.startSeconds, last.endSeconds),
                                                text: current.map(\.text).joined(separator: " ")))
            current = []
        }
        for word in words {
            if let first = current.first, word.endSeconds - first.startSeconds > maxSeconds { close() }
            current.append(word)
            if let end = word.text.last, ".?!".contains(end) { close() }
        }
        close()
        return segments
    }
}

extension AudioLibraryItem {
    /// Replaces the revision with `id`, or adds it when this is the first save.
    mutating func upsertTranscript(id: UUID, _ segments: [AudioLibrarySegment], source: AudioTranscriptRevision.Source, language: String? = nil) {
        if let index = transcripts.firstIndex(where: { $0.id == id }) {
            transcripts[index].segments = segments
            transcripts[index].language = language
        } else {
            transcripts.append(AudioTranscriptRevision(id: id, source: source, segments: segments, language: language))
        }
        updatedAt = Date()
    }
}
