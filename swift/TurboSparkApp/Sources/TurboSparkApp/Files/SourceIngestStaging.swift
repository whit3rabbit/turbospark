import Foundation

/// Per-file state for the Files section's import staging list.
///
/// READ-ONLY BY DESIGN. Every state here is derived from what
/// `AttachmentImporter.importDocuments` already knows (the file it is about
/// to read, and the per-file result it returns); nothing in this type
/// changes what gets attached or how. Retry reuses the same import call for
/// the one file.
enum SourceIngestStaging {
    /// Extractor refusals that a second attempt on the same file would
    /// repeat, so they get a reason and no Retry.
    enum SkipReason: Equatable {
        case tooLarge
        case unsupportedType
        case unreadable
        case noText
    }

    enum State: Equatable {
        case queued
        case reading
        /// Appended to the chat by the importer.
        case added
        case skipped(SkipReason)
        /// Anything else, with the importer's own message.
        case failed(String)
    }

    struct Entry: Equatable, Identifiable {
        let id: UUID
        let url: URL
        /// The chat the original batch targeted, so Retry attaches to the
        /// same chat even after the user switched to another one.
        let chatID: UUID?
        var state: State

        init(id: UUID = UUID(), url: URL, chatID: UUID?, state: State) {
            self.id = id
            self.url = url
            self.chatID = chatID
            self.state = state
        }

        var fileName: String { url.lastPathComponent }
    }

    struct Summary: Equatable {
        var queued = 0
        var reading = 0
        var added = 0
        var skipped = 0
        var failed = 0

        var total: Int { queued + reading + added + skipped + failed }
        var isInFlight: Bool { queued + reading > 0 }
    }

    // MARK: - Classification

    static func state(for result: AttachmentImporter.FileResult) -> State {
        guard let failure = result.failure else { return .added }
        switch failure.extractionError {
        case .unsupportedFormat: return .skipped(.unsupportedType)
        case .documentTooLarge: return .skipped(.tooLarge)
        case .unreadableFile: return .skipped(.unreadable)
        case .noExtractableText: return .skipped(.noText)
        case .invalidArchive, .extractionTimedOut, nil: return .failed(failure.message)
        }
    }

    static func canRetry(_ state: State) -> Bool {
        if case .failed = state { return true }
        return false
    }

    // MARK: - Transitions

    /// Queues `urls` for `chatID`. A file already listed is replaced, so
    /// importing it again does not show two rows for one file.
    static func staged(_ entries: [Entry], adding urls: [URL], chatID: UUID?) -> [Entry] {
        let incoming = Set(urls.map(\.standardizedFileURL))
        var result = entries.filter { !incoming.contains($0.url.standardizedFileURL) }
        result += urls.map { Entry(url: $0, chatID: chatID, state: .queued) }
        return result
    }

    static func markingReading(_ url: URL, in entries: [Entry]) -> [Entry] {
        entries.map { entry in
            guard entry.url.standardizedFileURL == url.standardizedFileURL else { return entry }
            switch entry.state {
            case .queued, .failed:
                var copy = entry
                copy.state = .reading
                return copy
            default:
                return entry
            }
        }
    }

    static func applying(_ results: [AttachmentImporter.FileResult], to entries: [Entry]) -> [Entry] {
        var byURL: [URL: State] = [:]
        for result in results {
            byURL[result.url.standardizedFileURL] = state(for: result)
        }
        return entries.map { entry in
            guard let state = byURL[entry.url.standardizedFileURL] else { return entry }
            var copy = entry
            copy.state = state
            return copy
        }
    }

    /// Fails any of `urls` still queued or reading once their batch has
    /// returned, so a row never spins after the import call ended.
    static func finishing(_ urls: [URL], in entries: [Entry], fallback: String) -> [Entry] {
        let batch = Set(urls.map(\.standardizedFileURL))
        return entries.map { entry in
            guard batch.contains(entry.url.standardizedFileURL) else { return entry }
            switch entry.state {
            case .queued, .reading:
                var copy = entry
                copy.state = .failed(fallback)
                return copy
            default:
                return entry
            }
        }
    }

    // MARK: - Aggregation

    /// Failures first (they need the user), then skips (they explain a
    /// missing file), then work in flight, then done. Stable within a group.
    static func ordered(_ entries: [Entry]) -> [Entry] {
        func rank(_ state: State) -> Int {
            switch state {
            case .failed: return 0
            case .skipped: return 1
            case .reading: return 2
            case .queued: return 3
            case .added: return 4
            }
        }
        return entries.enumerated()
            .sorted { lhs, rhs in
                let (l, r) = (rank(lhs.element.state), rank(rhs.element.state))
                return l != r ? l < r : lhs.offset < rhs.offset
            }
            .map(\.element)
    }

    static func summary(_ entries: [Entry]) -> Summary {
        var summary = Summary()
        for entry in entries {
            switch entry.state {
            case .queued: summary.queued += 1
            case .reading: summary.reading += 1
            case .added: summary.added += 1
            case .skipped: summary.skipped += 1
            case .failed: summary.failed += 1
            }
        }
        return summary
    }
}
