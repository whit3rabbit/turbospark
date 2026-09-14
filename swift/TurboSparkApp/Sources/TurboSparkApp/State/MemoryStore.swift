import CryptoKit
import Foundation

/// The four memory topic types, Claude Code's memdir taxonomy
/// (`src/memdir/memoryTypes.ts` in the reference tree).
public enum MemoryEntryType: String, CaseIterable, Codable {
    case user
    case feedback
    case project
    case reference
}

/// One parsed row of a project's `MEMORY.md` index.
public struct MemoryIndexEntry: Equatable {
    /// File name inside the memory directory, e.g. `router-quirks.md`.
    public var fileName: String
    /// The memory's slug, echoed in the link text.
    public var title: String
    /// The one-line hook that decides relevance.
    public var hook: String
}

/// Claude Code-style auto-memory for one project.
///
/// The layout under the profile-aware memory base (`userScopeSubdirectory`
/// seam, so the Default profile shares `~/.turbospark/memory` and any other
/// profile keeps its own):
///
///     <base>/projects/<key>/memory/MEMORY.md          the index, injected
///     <base>/projects/<key>/memory/<slug>.md          topic files
///
/// `<key>` is the project root's symlink-resolved path with every character
/// outside `[A-Za-z0-9._-]` replaced by `-`, plus 8 hex digits of SHA-256 of
/// the resolved path. The readable prefix is what a user browses; the hash
/// is what keeps `/Users/x/a-b` and `/Users/x/a.b` -- which sanitize to the
/// same prefix -- from silently sharing one memory directory.
///
/// The INDEX is the source of truth the prompt sees; a topic file is only
/// read on demand (`read`, or the model's own file tools reaching the path).
/// Every write rewrites the index line for its slug, so the injected text
/// and the directory never disagree about what exists.
///
/// All reads and writes are serialized by one lock: the tool executor runs
/// off the main actor while settings and the composer touch the same store.
public final class MemoryStore {
    public static let shared = MemoryStore()

    /// Whether the model sees the memory prompt section and the `memory`
    /// tool at all. Mirrors `AppModel.memoryEnabled` (the settings pane
    /// writes the published var, whose `didSet` re-points this), but lives
    /// here because `AppToolCatalog.tools(for:)` and
    /// `SubagentRunner.buildSystemPrompt` are static surfaces with no
    /// `AppModel` in hand -- the same reason `CommandGate.vetoEnabled` is a
    /// static. Tests may set it directly and restore it.
    public var isModelEnabled: Bool = false

    /// Injected base directory (the `memory` user-scope root). Nil resolves
    /// through `UserProfileStore` per call, which keeps production lazy and
    /// lets tests point the whole store at a scratch directory.
    private let injectedBase: URL?

    private let lock = NSLock()
    /// Memoized index text keyed by the index file's URL, valid while the
    /// file's modification date is unchanged. Writes go through this class,
    /// so they update the cache; hand edits on disk are picked up by mtime.
    private var indexCache: [URL: (modified: Date?, content: String)] = [:]

    public init(base: URL? = nil) {
        self.injectedBase = base
    }

    // MARK: - Paths

    /// The user-scope memory root for this run's profile.
    ///
    /// Under a test runner this redirects into the scratch tree instead:
    /// prompt ASSEMBLY creates the directory as a side effect, so the
    /// automatic redirect every other store relies on has to cover the
    /// memory base too, or a test building a prompt for a project would
    /// mkdir inside the user's real `~/.turbospark` (`AppStorageRoot`'s
    /// header is the full history of that failure mode).
    public static func defaultBase() -> URL {
        if AppStorageRoot.isRunningTests {
            return AppStorageRoot.machineRoot.appendingPathComponent("memory", isDirectory: true)
        }
        return UserProfileStore.userScopeSubdirectory("memory")
    }

    private func base() -> URL {
        injectedBase ?? MemoryStore.defaultBase()
    }

    /// The sanitized, collision-proofed directory key for a project root.
    /// Pure, and the reason tests can pin the layout without touching disk.
    public static func projectKey(forProjectRoot root: URL) -> String {
        let resolved = root.resolvingSymlinksInPath().path
        let sanitized = String(resolved.map { char in
            let scalars = String(char).unicodeScalars
            if scalars.count == 1,
               let scalar = scalars.first,
               (scalar.value >= UInt32(UnicodeScalar("a").value) && scalar.value <= UInt32(UnicodeScalar("z").value))
                || (scalar.value >= UInt32(UnicodeScalar("A").value) && scalar.value <= UInt32(UnicodeScalar("Z").value))
                || (scalar.value >= UInt32(UnicodeScalar("0").value) && scalar.value <= UInt32(UnicodeScalar("9").value))
                || char == "." || char == "_" || char == "-" {
                return char
            }
            return "-"
        })
        let digest = SHA256.hash(data: Data(resolved.utf8))
        let hash = digest.prefix(4).map { String(format: "%02x", $0) }.joined()
        return sanitized + "-" + hash
    }

    /// The memory directory for a project, created on first use. Everything
    /// downstream (index, topic files, the prompt's stated path) derives
    /// from this one answer.
    public func directory(forProjectRoot root: URL) -> URL {
        let dir = base()
            .appendingPathComponent("projects", isDirectory: true)
            .appendingPathComponent(MemoryStore.projectKey(forProjectRoot: root), isDirectory: true)
            .appendingPathComponent("memory", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    public func indexURL(forProjectRoot root: URL) -> URL {
        directory(forProjectRoot: root).appendingPathComponent("MEMORY.md")
    }

    // MARK: - Index

    /// The index text, empty when nothing is remembered yet. Memoized
    /// against the file's mtime so a per-turn prompt assembly does not
    /// re-read a file that did not change.
    public func loadIndex(forProjectRoot root: URL) -> String {
        let url = indexURL(forProjectRoot: root)
        lock.lock()
        defer { lock.unlock() }
        let modified = (try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? nil
        if let cached = indexCache[url], cached.modified == modified {
            return cached.content
        }
        let content = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        indexCache[url] = (modified: modified, content: content)
        return content
    }

    /// Parses `- [title](file.md) -- hook` rows; anything else on the index
    /// is skipped rather than corrupted by a round-trip.
    public static func parseIndex(_ text: String) -> [MemoryIndexEntry] {
        var entries: [MemoryIndexEntry] = []
        for line in SkillParser.normalizedLines(text) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("- ["), let close = trimmed.firstIndex(of: "]"),
                  let openParen = trimmed[close...].firstIndex(of: "("),
                  let openRound = trimmed[openParen...].firstIndex(of: ")")
            else { continue }
            let title = String(trimmed[trimmed.index(trimmed.startIndex, offsetBy: 3)..<close])
            let fileName = String(trimmed[trimmed.index(after: openParen)..<openRound])
            let hook: String
            if let arrow = trimmed.range(of: " -- ") {
                hook = String(trimmed[arrow.upperBound...])
            } else {
                hook = ""
            }
            entries.append(MemoryIndexEntry(fileName: fileName, title: title, hook: hook))
        }
        return entries
    }

    /// The index with one row for `fileName`, replacing the existing row for
    /// that file when there is one and appending otherwise. Unparseable
    /// lines the user may have written by hand are preserved verbatim.
    public static func upsertingIndexLine(_ index: String, fileName: String, title: String, hook: String) -> String {
        let newLine = "- [\(title)](\(fileName)) -- \(hook)"
        var lines = SkillParser.normalizedLines(index)
        // A trailing newline becomes a trailing "" element; keep the text
        // body and let the join below re-add the final newline.
        while lines.last?.isEmpty == true { lines.removeLast() }
        if let existing = lines.firstIndex(where: { $0.contains("(\(fileName))") }) {
            lines[existing] = newLine
        } else {
            lines.append(newLine)
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Drops the row for `fileName`; false when no row named it.
    public static func removingIndexLine(_ index: String, fileName: String) -> String? {
        var lines = SkillParser.normalizedLines(index)
        guard let at = lines.firstIndex(where: { $0.contains("(\(fileName))") }) else { return nil }
        lines.remove(at: at)
        while lines.last?.isEmpty == true { lines.removeLast() }
        return lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")
    }

    // MARK: - Topic files

    /// The only legal topic-file stem: kebab-case, 1 to 80 characters. This
    /// is the containment boundary for the whole feature -- the tool takes a
    /// NAME, never a path, and the name cannot escape the memory directory.
    public static func isValidTopicName(_ name: String) -> Bool {
        guard (1...80).contains(name.count) else { return false }
        let parts = name.split(separator: "-", omittingEmptySubsequences: false)
        return !parts.isEmpty && parts.allSatisfy { !$0.isEmpty && $0.allSatisfy { $0.isLowercase || $0.isNumber } }
    }

    /// Turns arbitrary prose into a legal stem, for the `#` quick-save. A
    /// date prefix keeps two same-titled saves from overwriting each other.
    public static func slug(from text: String, dated: Bool) -> String {
        var base = String(text.lowercased().map { $0.isLowercase || $0.isNumber ? $0 : "-" })
        while base.contains("--") { base = base.replacingOccurrences(of: "--", with: "-") }
        base = String(base.drop(while: { $0 == "-" }).reversed().drop(while: { $0 == "-" }).reversed())
        if base.count > 48 { base = String(base.prefix(48)) }
        while base.last == "-" { base = String(base.dropLast()) }
        if base.isEmpty { base = "memory" }
        guard dated else { return base }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: Date()) + "-" + base
    }

    /// Writes (or replaces) a topic file and its index row in one step.
    /// - Returns: the file's URL; `created` is false when a file by this
    ///   name already existed and this call replaced it.
    @discardableResult
    public func saveTopic(
        projectRoot: URL, name: String, type: MemoryEntryType,
        description: String, body: String
    ) throws -> (url: URL, created: Bool) {
        guard MemoryStore.isValidTopicName(name) else {
            throw NSError(domain: "TurboSparkMemory", code: 1, userInfo: [
                NSLocalizedDescriptionKey:
                    "Invalid memory name '\(name)': use short kebab-case (lowercase letters, digits, dashes)."
            ])
        }
        let dir = directory(forProjectRoot: projectRoot)
        let fileURL = dir.appendingPathComponent(name + ".md")
        let fileName = name + ".md"
        let created = !FileManager.default.fileExists(atPath: fileURL.path)
        // One line, whatever the model sent: a description carrying newlines
        // would both break the frontmatter and smuggle index rows in.
        let oneLineDescription = description
            .replacingOccurrences(of: "\r\n", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespaces)
        let frontmatter = """
        ---
        name: \(name)
        description: \(oneLineDescription)
        type: \(type.rawValue)
        ---
        """
        let document = frontmatter + "\n\n" + body.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        try document.write(to: fileURL, atomically: true, encoding: .utf8)

        let indexText = loadIndex(forProjectRoot: projectRoot)
        let updated = MemoryStore.upsertingIndexLine(
            indexText, fileName: fileName, title: name, hook: oneLineDescription)
        try writeIndex(updated, forProjectRoot: projectRoot)
        return (url: fileURL, created: created)
    }

    /// A topic file's full text (frontmatter included), for `memory` read.
    public func readTopic(projectRoot: URL, name: String) throws -> String {
        guard MemoryStore.isValidTopicName(name) else {
            throw NSError(domain: "TurboSparkMemory", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid memory name '\(name)'."
            ])
        }
        let fileURL = directory(forProjectRoot: projectRoot).appendingPathComponent(name + ".md")
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw NSError(domain: "TurboSparkMemory", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "No memory named '\(name)' exists. Use action \"read\" with no name to see the index."
            ])
        }
        return try String(contentsOf: fileURL, encoding: .utf8)
    }

    /// Deletes a topic file and its index row. False when nothing was named
    /// `name`, which the caller reports rather than treats as an error.
    @discardableResult
    public func forgetTopic(projectRoot: URL, name: String) throws -> Bool {
        guard MemoryStore.isValidTopicName(name) else {
            throw NSError(domain: "TurboSparkMemory", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid memory name '\(name)'."
            ])
        }
        let fileURL = directory(forProjectRoot: projectRoot).appendingPathComponent(name + ".md")
        let existed = FileManager.default.fileExists(atPath: fileURL.path)
        if existed {
            try FileManager.default.removeItem(at: fileURL)
        }
        let indexText = loadIndex(forProjectRoot: projectRoot)
        if let updated = MemoryStore.removingIndexLine(indexText, fileName: name + ".md") {
            try writeIndex(updated, forProjectRoot: projectRoot)
        }
        return existed
    }

    // MARK: - Internals

    private func writeIndex(_ text: String, forProjectRoot root: URL) throws {
        let url = indexURL(forProjectRoot: root)
        try text.write(to: url, atomically: true, encoding: .utf8)
        let modified = (try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? nil
        lock.lock()
        defer { lock.unlock() }
        indexCache[url] = (modified: modified, content: text)
    }

    /// Drops every memoized index. Nothing needs this for correctness (the
    /// mtime key already catches disk edits); it exists so tests and the
    /// settings pane can force a re-read, mirroring `reloadSkills`.
    public func invalidateCache() {
        lock.lock()
        defer { lock.unlock() }
        indexCache.removeAll()
    }
}
