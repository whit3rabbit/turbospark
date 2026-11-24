import Foundation

/// What a finished shell card shows without expanding: the exit outcome,
/// duration, output line count, and the last few non-blank lines
/// (component audit, terminal).
///
/// Read back from the result text because `AppToolResult` carries no exit
/// code: `ShellCommandRunner` writes it into the text, and the registry
/// prefixes thrown errors with "Error: ". `make` returns nil for any shape
/// it does not recognize (a denial, a disabled tool, a background launch),
/// so the card never states an exit code it guessed.
///
/// The shell runner returns only after the command exits, so there is no
/// live tail while a command runs; that needs output streamed into the call.
struct TerminalResultSummary: Equatable {
    enum Outcome: Equatable {
        case exited(Int)
        case timedOut
    }

    let outcome: Outcome
    /// Nil when the runner compacted the middle away, so the count is unknown.
    let lineCount: Int?
    let tail: [String]
    let durationText: String?

    static let tailLineLimit = 3

    static func make(output: String, isError: Bool, durationSeconds: Double) -> TerminalResultSummary? {
        var lines = normalizedLines(output)

        if lines.first?.hasPrefix("Command running in background with ID:") == true {
            return nil
        }

        var outcome: Outcome = .exited(0)
        if isError {
            guard var first = lines.first, first.hasPrefix("Error: ") else { return nil }
            first.removeFirst("Error: ".count)
            if let code = exitCode(in: first, prefix: "Command failed (exit ", suffix: "):") {
                outcome = .exited(code)
            } else if first.hasPrefix("(Command timed out after ") {
                outcome = .timedOut
            } else {
                return nil
            }
            lines.removeFirst()
        } else {
            // Trailing "(note; note)" line: a benign exit 1 note, the cwd
            // reset note, or both. Notes are runner metadata, not output.
            if let last = lines.last, let isBenign = runnerNotes(last) {
                if isBenign { outcome = .exited(1) }
                lines.removeLast()
            }
            if lines.count == 1, lines[0] == "(Command finished: success)" {
                lines = []
            } else if lines.count == 1,
                      let code = exitCode(in: lines[0], prefix: "(Command finished: exit code ", suffix: ")") {
                outcome = .exited(code)
                lines = []
            }
        }

        // The spill trailer is the runner pointing the model at a file.
        if let last = lines.last, last.hasPrefix("[Full output (") || last.hasPrefix("[Output over the ") {
            lines.removeLast()
        }
        let compacted = lines.contains { $0.hasPrefix("... [") && $0.hasSuffix(" chars truncated] ...") }

        while let last = lines.last, isBlank(last) { lines.removeLast() }
        while let first = lines.first, isBlank(first) { lines.removeFirst() }

        let tail = Array(lines.filter { !isBlank($0) }.suffix(tailLineLimit))
        return TerminalResultSummary(
            outcome: outcome,
            lineCount: compacted ? nil : lines.count,
            tail: tail,
            durationText: durationText(seconds: durationSeconds))
    }

    /// Seconds to one decimal under a minute, then whole minutes and
    /// seconds. Nil for zero, which is the result's "not measured" default.
    static func durationText(seconds: Double) -> String? {
        guard seconds.isFinite, seconds > 0 else { return nil }
        if seconds < 60 {
            return String(format: "%.1f s", locale: Locale(identifier: "en_US_POSIX"), seconds)
        }
        let total = Int(seconds.rounded())
        return "\(total / 60) m \(total % 60) s"
    }

    // MARK: - Helpers

    /// Escapes stripped, CRLF folded, and each carriage-return redraw
    /// (progress bars) reduced to its final frame, as a terminal shows it.
    static func normalizedLines(_ output: String) -> [String] {
        let plain = ToolOutputFormatter.stripAnsi(output)
            .replacingOccurrences(of: "\r\n", with: "\n")
        return plain.components(separatedBy: "\n").map { line in
            guard line.contains("\r") else { return line }
            let frames = line.components(separatedBy: "\r")
            return frames.last(where: { !$0.isEmpty }) ?? ""
        }
    }

    private static func isBlank(_ line: String) -> Bool {
        line.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private static func exitCode(in line: String, prefix: String, suffix: String) -> Int? {
        guard line.hasPrefix(prefix), line.hasSuffix(suffix) else { return nil }
        let digits = line.dropFirst(prefix.count).dropLast(suffix.count)
        return Int(digits)
    }

    private static let benignNotes: Set<String> = [
        "No matches found", "Files differ", "Condition evaluated to false",
    ]

    /// Nil when `line` is not a runner notes line; otherwise whether it
    /// carries a benign exit 1 note.
    private static func runnerNotes(_ line: String) -> Bool? {
        guard line.hasPrefix("("), line.hasSuffix(")") else { return nil }
        let parts = line.dropFirst().dropLast().components(separatedBy: "; ")
        var benign = false
        for part in parts {
            if benignNotes.contains(part) {
                benign = true
            } else if !part.hasPrefix("Shell cwd was reset to ") {
                return nil
            }
        }
        return benign
    }
}
