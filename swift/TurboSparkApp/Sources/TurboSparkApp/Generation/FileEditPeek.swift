import Foundation

/// The first lines of a file edit or write, for the collapsed tool card
/// (component audit, file-edit). Read through the executor's own resolvers,
/// so the peek shows what runs. An edit lists every removed line, then every
/// added line, the way the tool replaces one block with another; counts use
/// `ToolCallDiffFormatter.countLines`, so they equal the header's +N -M.
struct FileEditPeek: Equatable {
    enum ToolKind { case edit, write, multiEdit }
    enum LineKind: Equatable { case added, removed }

    struct Line: Equatable {
        let kind: LineKind
        let text: String
    }

    static let visibleLineLimit = 6

    let lines: [Line]
    /// Lines past the peek. Never 1: a single hidden line is shown instead,
    /// since "1 more line" takes the same row as the line itself.
    let hiddenLineCount: Int
    let additions: Int
    let deletions: Int

    /// Nil when there is nothing to show or the arguments cannot be read
    /// with certainty (conflicting spellings, unparseable multiedit).
    static func make(_ kind: ToolKind, arguments: [String: String]) -> FileEditPeek? {
        let blocks: [(removed: String, added: String)]
        switch kind {
        case .edit:
            let resolved = ToolArgumentResolver.fileArguments(arguments)
            guard resolved.conflict == nil else { return nil }
            blocks = [(resolved.oldString ?? "", resolved.newString ?? "")]
        case .write:
            let resolved = ToolArgumentResolver.fileArguments(arguments)
            guard resolved.conflict == nil else { return nil }
            blocks = [("", resolved.content ?? "")]
        case .multiEdit:
            guard let edits = try? MultiEditExecutor.parseEdits(from: arguments) else { return nil }
            blocks = edits.map { ($0.oldString, $0.newString) }
        }

        // Only limit + 1 lines are ever materialized; the rest are counted.
        let keep = visibleLineLimit + 1
        var head: [Line] = []
        var additions = 0
        var deletions = 0
        for block in blocks {
            let removed = lineStats(block.removed, keep: keep - head.count)
            deletions += removed.count
            head += removed.head.map { Line(kind: .removed, text: $0) }
            let added = lineStats(block.added, keep: keep - head.count)
            additions += added.count
            head += added.head.map { Line(kind: .added, text: $0) }
        }

        let total = additions + deletions
        guard total > 0 else { return nil }
        let shown = total <= keep ? head : Array(head.prefix(visibleLineLimit))
        return FileEditPeek(
            lines: shown, hiddenLineCount: total - shown.count,
            additions: additions, deletions: deletions)
    }

    /// Line count with `countLines` semantics (empty is zero, a trailing
    /// newline adds no line) and up to `keep` leading lines, without
    /// splitting the whole text: a large write re-renders on every token.
    private static func lineStats(_ text: String, keep: Int) -> (count: Int, head: [String]) {
        guard !text.isEmpty else { return (0, []) }
        let newlines = text.utf8.reduce(0) { $1 == UInt8(ascii: "\n") ? $0 + 1 : $0 }
        let count = text.utf8.last == UInt8(ascii: "\n") ? newlines : newlines + 1
        guard keep > 0 else { return (count, []) }
        // "\r\n" is one Character in Swift, so a plain "\n" separator would
        // never split CRLF text.
        let head = text.split(
            maxSplits: keep, omittingEmptySubsequences: false,
            whereSeparator: { $0 == "\n" || $0 == "\r\n" }
        )
        .prefix(min(keep, count))
        .map(String.init)
        return (count, head)
    }
}
