import Foundation

/// OC-11: the formatter step on OC-10's post-edit seam. It registers BEFORE
/// `LSPDiagnosticsStep`, so diagnostics read the formatted file. It is
/// app-initiated (not a model tool call) and runs only after an edit tool has
/// already passed the permission ladder; the master switch, off by default, is
/// the user's trust grant, and a project may narrow it (`ProjectFormatterMode`)
/// but never widen it. Success adds a note only when the file changed.
struct FormatterPostEditStep: PostEditStep {
    let identifier = "formatter"
    var isEnabled: @Sendable () -> Bool = { FormatterSettings.isEnabled() }
    var rows: @Sendable () -> [FormatterRow] = { FormatterSettings.rows() }
    var runner = FormatterRunner()
    var searchPath: @Sendable () async -> String? = { await LoginShellPath.resolved() }

    func run(_ context: PostEditContext) async -> String? {
        guard ProjectFormatterMode.allowsPass(
            appWideEnabled: isEnabled(), project: context.formatterMode)
        else { return nil }
        let rows = rows()
        // Resolve the login-shell PATH lazily: only when a row covers a path.
        var resolvedPath: String??
        var lines: [String] = []
        for path in context.relativePaths {
            guard let row = FormatterRegistry.row(forPath: path, in: rows) else { continue }
            if resolvedPath == nil { resolvedPath = .some(await searchPath()) }
            let outcome = await runner.format(
                path: path, row: row, rootURL: context.rootURL, searchPath: resolvedPath ?? nil)
            switch outcome {
            case .skipped, .unchanged: break
            case .failed(let reason):
                lines.append("\(row.name) failed on \(path) (\(reason)). The file keeps what you wrote.")
            case .formatted(let change):
                lines.append(Self.formattedNote(path: path, formatter: row.name, change: change))
            }
        }
        guard !lines.isEmpty else { return nil }
        return "<formatter_note>\n" + lines.joined(separator: "\n") + "\n</formatter_note>"
    }

    static func formattedNote(path: String, formatter: String, change: FormatterChange) -> String {
        var text = "\(formatter) reformatted \(path). The file on disk now differs from what you wrote"
        guard change.lineDiffAvailable else {
            return text + "; read it again before your next edit."
        }
        text += " (\(change.added.count) lines added or changed, \(change.removedCount) removed)."
        let shown = change.added.prefix(FormatterRunner.maximumNoteLines)
        if !shown.isEmpty { text += " Current text of the changed lines:" }
        for item in shown {
            let line = String(item.text.prefix(200))
                .replacingOccurrences(of: "</formatter_note>", with: "<\\/formatter_note>")
            text += "\n  \(item.line): \(line)"
        }
        if change.added.count > shown.count {
            text += "\n  ... \(change.added.count - shown.count) more changed lines; read the file to see them."
        }
        return text
    }
}
