import Foundation

/// The qwen-code bang-command parity: a draft starting with `!` runs as a
/// shell command in the project root and lands in the transcript as ONE user
/// message holding the command and its captured output. The output is part
/// of the message content, so the model reads it as context on the next turn
/// -- that is the feature's whole point (qwen-code's `!command` and Claude
/// Code's are the same shape).
///
/// The user typed the command, so it bypasses the permission engine the way
/// the tools' own approval flow never sees it; what it does NOT bypass is
/// the project-root requirement: with no project attached there is no
/// working directory to run in, and a home-directory fallback would run a
/// user-typed command against `~`.
extension AppModel {
    /// Whether the draft IS a bang command: a `!` followed by something that
    /// is not just more `!`. `!!` is the shell's own history spelling and
    /// the double-tap guard, not a command.
    public static func isBangCommand(_ draft: String) -> Bool {
        guard draft.hasPrefix("!") else { return false }
        let command = draft.dropFirst().trimmingCharacters(in: .whitespacesAndNewlines)
        return !command.isEmpty && !command.hasPrefix("!")
    }

    /// Runs the draft's shell command and appends the transcript row. The
    /// row is ONE message so a turn grouping never has to know about it:
    /// the renderer splits command from output at the
    /// `<shell_output>` marker, and the model reads the same text verbatim.
    public func handleBangCommand(_ draft: String) {
        guard !generating, !submitting, pendingToolCall == nil else {
            showToast("Stop the current turn before running a shell command.", style: .warning)
            return
        }
        let project = turnProject(chatID: selectedChatID) ?? selectedProject
        guard let root = project?.rootDirectoryURL, !root.path.isEmpty else {
            showToast("Attach a project first: a shell command needs a working directory.", style: .warning)
            return
        }
        let command = String(draft.dropFirst().trimmingCharacters(in: .whitespacesAndNewlines))
        let chatID = selectedChatID
        Task {
            var output = ""
            do {
                output = try await ShellCommandRunner.run(
                    command: command, rootURL: root, timeoutMs: 60_000, chatID: chatID)
            } catch {
                output = "Error: \(error.localizedDescription)"
            }
            let trimmed = Self.boundedShellOutput(output)
            let content = "!\(command)\n\n<shell_output>\n\(trimmed.isEmpty ? "(no output)" : trimmed)\n</shell_output>"
            mutateTurnMessages(for: chatID) { $0.append(AppChatMessage(role: .user, content: content)) }
            updateTokenEstimate()
            let head = command.count > 48 ? String(command.prefix(48)) + "..." : command
            showToast("Ran \(head).", style: .success)
        }
    }

    /// Head-plus-tail clamp on the captured output. A runaway `cat` on a big
    /// file must neither bloat the persisted chat nor the next prompt.
    static func boundedShellOutput(_ output: String) -> String {
        let limit = 8_000
        guard output.count > limit else { return output }
        let head = String(output.prefix(limit / 2))
        let tail = String(output.suffix(limit / 2))
        let dropped = output.count - head.count - tail.count
        return head + "\n... [\(dropped) characters truncated] ...\n" + tail
    }
}

/// The pieces of a stored bang-command message, split at the
/// `<shell_output>` marker `handleBangCommand` writes. Lives beside the
/// renderer that consumes it: the transcript row shows the command and the
/// output, and everything else (copy, history recall, prompt assembly) sees
/// the message verbatim.
struct ShellMessageContent: Equatable {
    let command: String
    let output: String?

    static func parse(_ content: String) -> ShellMessageContent? {
        guard content.hasPrefix("!") else { return nil }
        let marker = "\n\n<shell_output>\n"
        guard let markerRange = content.range(of: marker) else {
            // An interrupted run (or an old row) may hold the command alone.
            let command = content.dropFirst().trimmingCharacters(in: .whitespacesAndNewlines)
            guard !command.isEmpty, !command.hasPrefix("!") else { return nil }
            return ShellMessageContent(command: command, output: nil)
        }
        var body = String(content[markerRange.upperBound...])
        if body.hasSuffix("\n</shell_output>") {
            body = String(body.dropLast("\n</shell_output>".count))
        }
        let command = String(content[content.index(after: content.startIndex)..<markerRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ShellMessageContent(command: command, output: body)
    }
}
