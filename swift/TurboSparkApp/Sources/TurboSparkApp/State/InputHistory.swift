import Foundation

/// App-wide recallable prompt history, the qwen-code input-history feature:
/// every submitted prompt is remembered and Up/Down in the composer walks
/// back through them.
///
/// Persisted to its own JSON file rather than into `settings.json`, for the
/// same reason steering presets got their own home: the settings write is
/// user-facing configuration, this is a transcript of what the user typed,
/// and the two have different privacy and size lives. Ghost chats NEVER
/// record: the no-trace rule that keeps their rows out of the chat archive
/// applies at least as strongly to a file whose only content is verbatim
/// prompts.
public struct InputHistoryEntry: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var text: String
    public var createdAt: Date

    public init(id: UUID = UUID(), text: String, createdAt: Date = Date()) {
        self.id = id
        self.text = text
        self.createdAt = createdAt
    }
}

/// The store behind the composer's Up/Down recall. MainActor: every reader
/// and writer is the submission path or the composer.
@MainActor
public final class InputHistoryStore: ObservableObject {
    /// Newest LAST, so "one step back" is `count - 2` from the end.
    @Published private(set) var entries: [InputHistoryEntry] = []

    static let capacity = 100
    /// Injectable so tests get an isolated file instead of whatever the
    /// shared store last persisted (the default is the app's real root).
    private let fileURL: URL

    public init(directory: URL = AppStorageRoot.directory) {
        self.fileURL = directory.appendingPathComponent("input_history.json")
        load()
    }

    /// Records one submitted prompt. Consecutive duplicates collapse (a
    /// retried prompt is one memory, not two), blanks are ignored, and the
    /// list is capped so a year of use cannot make the file grow forever.
    public func record(_ rawText: String) {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        if let last = entries.last, last.text == text { return }
        entries.append(InputHistoryEntry(text: text))
        if entries.count > Self.capacity {
            entries.removeFirst(entries.count - Self.capacity)
        }
        save()
    }

    /// The prompt `steps` back from the live draft: `steps == 1` is the most
    /// recent entry. Nil past the oldest end.
    public func entry(backSteps steps: Int) -> InputHistoryEntry? {
        guard steps >= 1 else { return nil }
        let index = entries.count - steps
        guard index >= 0 else { return nil }
        return entries[index]
    }

    var count: Int { entries.count }

    private func load() {
        guard let archive = AppJSONStore.load(
            [InputHistoryEntry].self, from: fileURL, label: "input history")
        else { return }
        entries = archive
    }

    private func save() {
        AppJSONStore.save(entries, to: fileURL, label: "Input history")
    }
}

/// The composer's walk through `InputHistoryStore`, as a small value so it
/// can be unit-tested without a view. MainActor: it reads the store, which
/// is where the store's isolation lives.
///
/// The rule that keeps this from fighting the editor: while browsing, the
/// draft the user had BEFORE the first Up is parked, and stepping past the
/// newest entry restores it. Any edit that is not a history step (the view
/// tells this object via `noteExternalEdit`) ends the walk and keeps the
/// current text.
@MainActor
public struct ComposerHistoryBrowser: Equatable {
    /// How far back the walk currently sits; 1 = newest entry.
    public private(set) var stepsBack: Int = 0
    /// The draft parked when the walk began.
    public private(set) var parkedDraft: String = ""

    public init() {}

    /// Whether an Up press should start (or continue) a walk. Starting is
    /// only offered from an empty draft or an existing walk -- hijacking Up
    /// mid-sentence in a multi-line draft would steal a caret move.
    public func canStepBack(draftText: String) -> Bool {
        stepsBack > 0 || draftText.isEmpty
    }

    /// Applies the previous entry. The draft the walk started FROM is
    /// parked on the first step, so stepping forward past the newest entry
    /// can restore it. Returns nil when there is nothing older; the caller
    /// leaves the text alone in that case.
    public mutating func stepBack(store: InputHistoryStore, currentDraft: String) -> String? {
        guard let entry = store.entry(backSteps: stepsBack + 1) else { return nil }
        if stepsBack == 0 { parkedDraft = currentDraft }
        stepsBack += 1
        return entry.text
    }

    /// Applies the next entry; past the newest, restores the parked draft
    /// and ends the walk (returns it with `isActive == false`).
    public mutating func stepForward(store: InputHistoryStore) -> String? {
        guard stepsBack > 0 else { return nil }
        if stepsBack == 1 {
            stepsBack = 0
            return parkedDraft
        }
        stepsBack -= 1
        return store.entry(backSteps: stepsBack)?.text
    }

    public var isActive: Bool { stepsBack > 0 }

    /// A keystroke that was not a history step: end the walk.
    public mutating func noteExternalEdit() {
        stepsBack = 0
        parkedDraft = ""
    }
}
