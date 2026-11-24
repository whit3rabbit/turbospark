import Foundation

/// The files one assistant turn touched, for the collapsed card at the end of
/// the turn (component audit, file-tree). Built only from the turn's own tool
/// calls, and only from calls that ran without error: a denied, failed or
/// unanswered call changed nothing on disk.
///
/// A state is only claimed when the call proves it. `write_file` cannot tell
/// a new file from an overwrite, so it is `written` unless the same turn
/// already read or changed that path (then the file existed: `modified`).
/// Deletes come only from `apply_patch`, whose result lists each operation;
/// a shell `rm` is not parsed. Counts are nil whenever any contributing
/// change has no line diff (whole-file writes, multi-file patches).
struct TurnFileTree: Equatable {
    enum Change: Equatable, CustomStringConvertible {
        case added, modified, written, deleted, read

        var isMutation: Bool { self != .read }

        var description: String {
            switch self {
            case .added: return "added"
            case .modified: return "modified"
            case .written: return "written"
            case .deleted: return "deleted"
            case .read: return "read"
            }
        }
    }

    struct Entry: Equatable, Identifiable {
        /// Root-relative when the path is inside the project, else as given.
        let path: String
        var change: Change
        var additions: Int?
        var deletions: Int?
        /// The latest call that changed the file (or read it, for a read-only
        /// entry). The row reveals this call's card.
        var origin: TurnSourceOrigin

        var id: String { path }

        /// Whether "View file" is offered. A delete left nothing to show.
        var canViewFile: Bool { change != .deleted }
    }

    enum Row: Equatable {
        /// A folder, single-child chains folded into one name ("a/b/c").
        case directory(name: String, depth: Int)
        case file(Entry, name: String, depth: Int)
    }

    /// A card for one file would repeat its own tool card.
    static let minimumFiles = 2

    /// Sorted by path.
    let entries: [Entry]
    let rows: [Row]

    var changedCount: Int { entries.filter { $0.change.isMutation }.count }
    var readCount: Int { entries.count - changedCount }

    /// Nil when the turn touched fewer than `minimumFiles` files or changed
    /// none (a turn that only read files has no blast radius to show).
    static func make(messages: [AppChatMessage], rootPath: String?) -> TurnFileTree? {
        var byPath: [String: Entry] = [:]
        for message in messages where message.role == .assistant {
            for call in message.toolCalls {
                guard let result = message.toolResults.first(where: { $0.callID == call.id }),
                      !result.isError
                else { continue }
                let origin = TurnSourceOrigin(messageID: message.id, callID: call.id)
                for touch in touches(of: call, output: result.output) {
                    guard let path = displayPath(touch.path, rootPath: rootPath) else { continue }
                    let incoming = Entry(
                        path: path, change: touch.change,
                        additions: touch.additions, deletions: touch.deletions, origin: origin)
                    byPath[path] = byPath[path].map { merge($0, incoming) } ?? incoming
                }
            }
        }
        let entries = byPath.values.sorted { $0.path < $1.path }
        guard entries.count >= minimumFiles,
              entries.contains(where: { $0.change.isMutation })
        else { return nil }
        return TurnFileTree(entries: entries, rows: rows(for: entries))
    }

    /// Cheap pre-check so a turn with no file-changing call never parses.
    static func hasFileChange(in messages: [AppChatMessage]) -> Bool {
        messages.contains { message in
            message.role == .assistant && message.toolCalls.contains { call in
                guard let kind = kind(of: call.name) else { return false }
                return kind != .read
            }
        }
    }

    // MARK: - Per-call touches

    private enum ToolKind { case edit, write, multiEdit, patch, document, read }

    private struct Touch {
        let path: String
        let change: Change
        let additions: Int?
        let deletions: Int?
    }

    private static func kind(of name: String) -> ToolKind? {
        let lower = name.lowercased()
        if AppToolRegistry.editFileAliases.contains(lower) { return .edit }
        if AppToolRegistry.writeFileAliases.contains(lower) { return .write }
        if AppToolRegistry.readFileAliases.contains(lower) { return .read }
        switch lower {
        case "multiedit", "multi_edit": return .multiEdit
        case "apply_patch", "applypatch": return .patch
        case "document_write": return .document
        default: return nil
        }
    }

    private static func touches(of call: AppToolCall, output: String) -> [Touch] {
        guard let kind = kind(of: call.name) else { return [] }
        switch kind {
        case .edit:
            let resolved = ToolArgumentResolver.fileArguments(call.arguments)
            guard resolved.conflict == nil, let path = resolved.path else { return [] }
            // Same counts as the card header and the peek.
            let peek = FileEditPeek.make(.edit, arguments: call.arguments)
            return [Touch(path: path, change: .modified,
                          additions: peek?.additions ?? 0, deletions: peek?.deletions ?? 0)]
        case .write:
            let resolved = ToolArgumentResolver.fileArguments(call.arguments)
            guard resolved.conflict == nil, let path = resolved.path else { return [] }
            return [Touch(path: path, change: .written, additions: nil, deletions: nil)]
        case .read:
            let resolved = ToolArgumentResolver.fileArguments(call.arguments, forRead: true)
            guard resolved.conflict == nil, let path = resolved.path else { return [] }
            return [Touch(path: path, change: .read, additions: nil, deletions: nil)]
        case .document:
            // Never overwrites (DocumentWriteExecutor), so a success is a new file.
            guard let path = call.arguments["path"] else { return [] }
            return [Touch(path: path, change: .added, additions: nil, deletions: nil)]
        case .multiEdit:
            guard let edits = try? MultiEditExecutor.parseEdits(from: call.arguments) else { return [] }
            var order: [String] = []
            var counts: [String: (add: Int, del: Int)] = [:]
            for edit in edits where !edit.filePath.isEmpty {
                if counts[edit.filePath] == nil { order.append(edit.filePath) }
                let previous = counts[edit.filePath] ?? (0, 0)
                counts[edit.filePath] = (
                    previous.add + ToolCallDiffFormatter.countLines(edit.newString),
                    previous.del + ToolCallDiffFormatter.countLines(edit.oldString))
            }
            return order.map { path in
                Touch(path: path, change: .modified,
                      additions: counts[path]?.add, deletions: counts[path]?.del)
            }
        case .patch:
            return patchTouches(call: call, output: output)
        }
    }

    /// The operations `ApplyPatchExecutor` reports as applied ("  ADD path"
    /// lines of its summary), which is what ran rather than what the patch
    /// text asked for. Per-file counts only when the patch has one file:
    /// the summary formatter counts the whole patch.
    private static func patchTouches(call: AppToolCall, output: String) -> [Touch] {
        var ops: [(path: String, change: Change)] = []
        for line in output.split(separator: "\n", omittingEmptySubsequences: true) {
            guard line.hasPrefix("  ") else { continue }
            let body = line.dropFirst(2)
            for (prefix, change) in [("ADD ", Change.added), ("UPDATE ", .modified), ("DELETE ", .deleted)]
            where body.hasPrefix(prefix) {
                ops.append((String(body.dropFirst(prefix.count)), change))
            }
        }
        guard ops.count == 1, let op = ops.first else {
            return ops.map { Touch(path: $0.path, change: $0.change, additions: nil, deletions: nil) }
        }
        let summary = ToolCallDiffFormatter.summarize(callName: call.name, arguments: call.arguments)
        return [Touch(path: op.path, change: op.change,
                      additions: summary.additions ?? 0, deletions: summary.deletions ?? 0)]
    }

    // MARK: - Merge

    /// Folds a later touch of the same path into the entry so far.
    private static func merge(_ existing: Entry, _ incoming: Entry) -> Entry {
        // A read never downgrades a change, and keeps the change's call.
        if incoming.change == .read { return existing }
        var merged = incoming
        if existing.change.isMutation {
            merged.additions = sum(existing.additions, incoming.additions)
            merged.deletions = sum(existing.deletions, incoming.deletions)
        }
        merged.change = mergedChange(existing.change, incoming.change)
        return merged
    }

    private static func sum(_ a: Int?, _ b: Int?) -> Int? {
        guard let a, let b else { return nil }
        return a + b
    }

    private static func mergedChange(_ existing: Change, _ incoming: Change) -> Change {
        switch (existing, incoming) {
        case (_, .deleted): return .deleted
        // New this turn stays new, however often it was changed after.
        case (.added, _): return .added
        // Still unknown whether the first write created it.
        case (.written, .modified), (.written, .written): return .written
        // The file existed at some point this turn, so a write replaced it.
        case (.read, .written), (.modified, .written), (.deleted, .written): return .modified
        case (.deleted, .added), (.modified, .added), (.written, .added): return .modified
        default: return incoming
        }
    }

    // MARK: - Paths and tree

    /// The path a row shows and dedupes on: inside `rootPath` it is made
    /// relative, "." and empty components are dropped, ".." is kept (it is
    /// not resolved, since that would guess at symlinks). Nil for nothing or
    /// the root itself.
    static func displayPath(_ raw: String, rootPath: String?) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let isAbsolute = trimmed.hasPrefix("/")
        let parts = trimmed.split(separator: "/").filter { $0 != "." }.map(String.init)
        if isAbsolute, let rootPath {
            let rootParts = rootPath.split(separator: "/").filter { $0 != "." }.map(String.init)
            if !rootParts.isEmpty, parts.count >= rootParts.count,
               Array(parts.prefix(rootParts.count)) == rootParts {
                let relative = parts.dropFirst(rootParts.count)
                return relative.isEmpty ? nil : relative.joined(separator: "/")
            }
        }
        guard !parts.isEmpty else { return nil }
        return (isAbsolute ? "/" : "") + parts.joined(separator: "/")
    }

    private final class Folder {
        var folders: [String: Folder] = [:]
        var files: [(name: String, entry: Entry)] = []
    }

    private static func rows(for entries: [Entry]) -> [Row] {
        let top = Folder()
        for entry in entries {
            let isAbsolute = entry.path.hasPrefix("/")
            var parts = entry.path.split(separator: "/").map(String.init)
            guard let name = parts.popLast() else { continue }
            if isAbsolute, !parts.isEmpty { parts[0] = "/" + parts[0] }
            var folder = top
            for part in parts {
                if folder.folders[part] == nil { folder.folders[part] = Folder() }
                folder = folder.folders[part]!
            }
            folder.files.append((name, entry))
        }
        var out: [Row] = []
        flatten(top, depth: 0, into: &out)
        return out
    }

    /// Folders first, then files, each alphabetical. Plain ordering, not a
    /// locale-aware one, so the order never changes with the language.
    private static func flatten(_ folder: Folder, depth: Int, into out: inout [Row]) {
        for key in folder.folders.keys.sorted() {
            var name = key
            var child = folder.folders[key]!
            while child.files.isEmpty, child.folders.count == 1, let only = child.folders.first {
                name += "/" + only.key
                child = only.value
            }
            out.append(.directory(name: name, depth: depth))
            flatten(child, depth: depth + 1, into: &out)
        }
        for file in folder.files.sorted(by: { $0.name < $1.name }) {
            out.append(.file(file.entry, name: file.name, depth: depth))
        }
    }
}
