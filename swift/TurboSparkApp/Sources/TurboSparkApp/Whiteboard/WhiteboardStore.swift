import Foundation
import PencilKit

// Persistence for the whiteboard pane (ZP-10).
//
// LAYOUT (one, under `AppStorageRoot.directory`):
//
//     whiteboards/boards.json                   metadata (AppJSONStore)
//     whiteboards/<workspaceKey>/<id>.drawing   PKDrawing data, one per board
//
// **THE BLOB IS THE SOURCE OF TRUTH FOR STROKES, THE JSON IS REBUILDABLE.**
// The JSON holds only title and timestamps. A `.drawing` file with no row (the
// JSON was quarantined as corrupt, or the app quit between the two writes) is
// adopted as a new board with an auto title on the next load, so a damaged
// metadata file never loses a sketch. A row with no blob is an empty board.
// A blob PencilKit cannot parse is renamed `corrupt-<stamp>-<name>` (never
// overwritten) and the board opens empty with `lastIssue` set.

/// One board's metadata. Strokes live in the blob beside it.
struct WhiteboardBoard: Codable, Identifiable, Equatable {
    var id: UUID
    /// `WhiteboardStore.workspaceKey(projectID:)`.
    var workspaceKey: String
    var title: String
    var createdAt: Date
    var updatedAt: Date
}

/// What opening a board found.
enum WhiteboardLoad: Equatable {
    case empty
    case drawing(Data)
    /// The blob was unreadable and has been set aside; the board is empty.
    case recoveredFromCorruption
}

@MainActor
final class WhiteboardStore: ObservableObject {
    /// Workspace bucket for chats that are not inside a project.
    static let chatWorkspaceKey = "chat"
    /// A title longer than this is cut: it is a label, not a document.
    static let maxTitleLength = 60
    /// Writes coalesce for this long after the last stroke.
    nonisolated static let defaultDebounce: Duration = .milliseconds(600)

    @Published private(set) var boards: [WhiteboardBoard] = []
    /// A sentence for a toast or inline note when a read or write went wrong.
    @Published private(set) var lastIssue: String?

    private let root: URL
    private let debounce: Duration
    private var pending: [UUID: Data] = [:]
    private var debounceTask: Task<Void, Never>?

    init(
        directory: URL = AppStorageRoot.directory.appendingPathComponent("whiteboards", isDirectory: true),
        debounce: Duration = WhiteboardStore.defaultDebounce
    ) {
        self.root = directory
        self.debounce = debounce
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        load()
    }

    static func workspaceKey(projectID: UUID?) -> String {
        projectID?.uuidString ?? chatWorkspaceKey
    }

    // MARK: - Paths

    private var metadataURL: URL { root.appendingPathComponent("boards.json") }

    private func folder(for key: String) -> URL {
        // The key is a UUID string or "chat", but it came from a decoded file
        // too: keep it from naming anything outside `root`.
        let safe = key.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "..", with: "_")
        return root.appendingPathComponent(safe, isDirectory: true)
    }

    func blobURL(for board: WhiteboardBoard) -> URL {
        folder(for: board.workspaceKey).appendingPathComponent("\(board.id.uuidString).drawing")
    }

    // MARK: - Queries

    /// Newest-edited first.
    func boards(in key: String) -> [WhiteboardBoard] {
        boards.filter { $0.workspaceKey == key }.sorted { $0.updatedAt > $1.updatedAt }
    }

    func board(id: UUID) -> WhiteboardBoard? { boards.first { $0.id == id } }

    // MARK: - Naming

    /// The lowest "Board N" not already taken in this workspace, so a deleted
    /// board's number is reused and two boards never share a title.
    static func nextAutoTitle(existing: [String], make: (Int) -> String) -> String {
        let taken = Set(existing.map { $0.lowercased() })
        var n = 1
        while taken.contains(make(n).lowercased()) { n += 1 }
        return make(n)
    }

    /// `base` if free, else `base 2`, `base 3`, ... Case-insensitive.
    static func uniqueTitle(_ base: String, existing: [String]) -> String {
        let taken = Set(existing.map { $0.lowercased() })
        if !taken.contains(base.lowercased()) { return base }
        var n = 2
        while taken.contains("\(base) \(n)".lowercased()) { n += 1 }
        return "\(base) \(n)"
    }

    // MARK: - Mutations

    @discardableResult
    func createBoard(workspaceKey key: String, now: Date = Date()) -> WhiteboardBoard {
        let title = Self.nextAutoTitle(existing: boards(in: key).map(\.title)) { n in
            AppStrings.localized("Board \(n)")
        }
        let board = WhiteboardBoard(
            id: UUID(), workspaceKey: key, title: title, createdAt: now, updatedAt: now)
        boards.append(board)
        saveMetadata()
        return board
    }

    func rename(id: UUID, to proposed: String) {
        guard let index = boards.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = String(
            proposed.trimmingCharacters(in: .whitespacesAndNewlines).prefix(Self.maxTitleLength))
        guard !trimmed.isEmpty else { return }
        let others = boards(in: boards[index].workspaceKey).filter { $0.id != id }.map(\.title)
        let title = Self.uniqueTitle(trimmed, existing: others)
        guard title != boards[index].title else { return }
        boards[index].title = title
        saveMetadata()
    }

    /// Permanent: removes the blob too. The pane confirms first.
    func delete(id: UUID) {
        guard let board = board(id: id) else { return }
        pending[id] = nil
        try? FileManager.default.removeItem(at: blobURL(for: board))
        boards.removeAll { $0.id == id }
        saveMetadata()
    }

    // MARK: - Drawing data

    func loadDrawing(id: UUID) -> WhiteboardLoad {
        guard let board = board(id: id) else { return .empty }
        // A change still waiting on the debounce is newer than the disk.
        if let queued = pending[id] { return .drawing(queued) }
        let url = blobURL(for: board)
        guard FileManager.default.fileExists(atPath: url.path) else { return .empty }
        do {
            let data = try Data(contentsOf: url)
            _ = try PKDrawing(data: data)
            return .drawing(data)
        } catch {
            quarantine(url)
            lastIssue = AppStrings.localized(
                "The saved drawing for this board could not be read. It was set aside and the board opened empty.")
            return .recoveredFromCorruption
        }
    }

    /// Queues `data` and writes it after the debounce. Many strokes in a row
    /// cost one write.
    func scheduleSave(id: UUID, data: Data) {
        guard board(id: id) != nil else { return }
        pending[id] = data
        debounceTask?.cancel()
        let wait = debounce
        debounceTask = Task { [weak self] in
            try? await Task.sleep(for: wait)
            guard !Task.isCancelled else { return }
            self?.flush()
        }
    }

    /// Writes everything queued now. Called when the pane closes and at quit,
    /// so a stroke drawn inside the debounce window is not lost.
    func flush() {
        debounceTask?.cancel()
        debounceTask = nil
        guard !pending.isEmpty else { return }
        let queued = pending
        pending = [:]
        var wroteAny = false
        for (id, data) in queued {
            guard let index = boards.firstIndex(where: { $0.id == id }) else { continue }
            let url = blobURL(for: boards[index])
            do {
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: url, options: .atomic)
                boards[index].updatedAt = Date()
                wroteAny = true
            } catch {
                // Keep it queued so the next flush retries instead of
                // dropping the stroke.
                pending[id] = pending[id] ?? data
                lastIssue = AppStrings.localized(
                    "A whiteboard could not be saved: \(error.localizedDescription)")
            }
        }
        if wroteAny { saveMetadata() }
    }

    var hasPendingWrites: Bool { !pending.isEmpty }

    func clearIssue() { lastIssue = nil }

    // MARK: - Metadata

    private func saveMetadata() {
        AppJSONStore.save(boards, to: metadataURL, label: "Whiteboards")
    }

    private func load() {
        boards = AppJSONStore.load([WhiteboardBoard].self, from: metadataURL, label: "Whiteboards") ?? []
        adoptOrphanBlobs()
    }

    /// Blobs with no metadata row become boards again (see the file header).
    private func adoptOrphanBlobs() {
        let fm = FileManager.default
        guard let folders = try? fm.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey])
        else { return }
        var adopted = false
        let known = Set(boards.map(\.id))
        for folder in folders where (try? folder.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
            let key = folder.lastPathComponent
            let files = (try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
            for file in files where file.pathExtension == "drawing" {
                guard let id = UUID(uuidString: file.deletingPathExtension().lastPathComponent),
                    !known.contains(id)
                else { continue }
                let modified = (try? file.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? Date()
                let title = Self.nextAutoTitle(existing: boards(in: key).map(\.title)) { n in
                    AppStrings.localized("Board \(n)")
                }
                boards.append(WhiteboardBoard(
                    id: id, workspaceKey: key, title: title, createdAt: modified, updatedAt: modified))
                adopted = true
            }
        }
        if adopted { saveMetadata() }
    }

    private func quarantine(_ url: URL) {
        let stamp = Int(Date().timeIntervalSince1970)
        let target = url.deletingLastPathComponent()
            .appendingPathComponent("corrupt-\(stamp)-\(url.lastPathComponent)")
        try? FileManager.default.moveItem(at: url, to: target)
    }
}
