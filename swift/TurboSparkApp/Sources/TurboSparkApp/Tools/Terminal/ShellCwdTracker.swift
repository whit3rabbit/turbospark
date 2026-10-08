import Foundation

/// Per-project working-directory persistence for the shell tool.
///
/// Claude Code persists cwd across Bash calls by appending a `pwd -P` capture
/// to every command (`bashProvider.ts` builds the same suffix shape) and
/// resetting when the shell leaves the allowed directories
/// (`BashTool/utils.ts` `resetCwdIfOutsideProject`). Without it, `cd build &&
/// ninja` in one call is invisible to the next and the model must re-derive
/// absolute paths every turn.
///
/// The tracker is process-lifetime state keyed by (chat, project root), so a
/// second chat on the same project does not inherit the first one's `cd`
/// (a relative `rm -r build` would run in the wrong directory); it is
/// deliberately NOT persisted into the chat archive, since a saved cwd path
/// is meaningless after the project moves.
final class ShellCwdTracker: @unchecked Sendable {
    static let shared = ShellCwdTracker()

    private let lock = NSLock()
    private var cwdByRoot: [String: String] = [:]

    /// The directory the next command in `root` starts in: the remembered
    /// cwd when it still exists on disk, the root otherwise. A deleted
    /// directory must not poison every later call, so the check is a real
    /// stat and not trust in the remembered value.
    func startDirectory(for root: URL, chatID: UUID? = nil) -> URL {
        lock.lock(); defer { lock.unlock() }
        guard let remembered = cwdByRoot[Self.key(root: root, chatID: chatID)] else { return root }
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: remembered, isDirectory: &isDir)
        guard exists, isDir.boolValue else { return root }
        return URL(fileURLWithPath: remembered)
    }

    /// Records where a command finished. Returns a note for the tool result
    /// when the final directory lies OUTSIDE the project root: the tracker
    /// resets to the root in that case and the model is told so, matching
    /// Claude Code's "Shell cwd was reset" behavior. A `cd /tmp` that were
    /// allowed to persist would silently widen every later call's reach
    /// outside the directory the user attached.
    func record(finalDirectory rawPath: String, root: URL, chatID: UUID? = nil) -> String? {
        let finalPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !finalPath.isEmpty else { return nil }
        lock.lock(); defer { lock.unlock() }
        let key = Self.key(root: root, chatID: chatID)
        if Self.isInside(finalPath, root: root.path) {
            cwdByRoot[key] = finalPath
            return nil
        }
        cwdByRoot[key] = root.path
        return "Shell cwd was reset to \(root.path): a command may not leave the "
            + "project directory's working state."
    }

    private static func key(root: URL, chatID: UUID?) -> String {
        "\(chatID?.uuidString ?? "-")|\(root.path)"
    }

    /// `pwd -P` yields physical paths, so both sides resolve symlinks before
    /// comparing: /tmp vs /private/tmp on macOS would otherwise read as an
    /// escape when it is the same directory.
    static func isInside(_ path: String, root: String) -> Bool {
        let resolvedPath = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
        let resolvedRoot = URL(fileURLWithPath: root).resolvingSymlinksInPath().path
        if resolvedPath == resolvedRoot { return true }
        return resolvedPath.hasPrefix(resolvedRoot + "/")
    }

    /// Test seam: clears all remembered directories.
    func resetForTests() {
        lock.lock(); defer { lock.unlock() }
        cwdByRoot.removeAll()
    }
}
