import Foundation

/// Model-facing shaping of shell output, and the environment a shell child
/// runs under.
///
/// Claude Code reference: `src/tools/BashTool/utils.ts` (`formatOutput`) for
/// the truncation shape, `src/tools/BashTool/commandSemantics.ts` for the
/// benign-exit mapping, and `bashProvider.ts`'s `getEnvironmentOverrides` for
/// the hang-prevention environment.
enum ShellOutputFormatting {
    /// The cap on what the MODEL sees from one command. The pipe-level cap
    /// (1 MB per stream, `ProcessExecutor.defaultOutputCapBytes`) bounds
    /// memory; this one bounds the context window a `cargo build` would
    /// otherwise spend entirely. Claude Code defaults to 30,000 chars and
    /// keeps the head; the tail is kept too because build and test failures
    /// summarize at the END of their output, and a head-only cut hides
    /// exactly the lines that explain the failure.
    static let maxModelOutputChars = 30_000
    private static let headChars = 20_000
    private static let tailChars = 8_000

    /// Removes ANSI/VT escape sequences (colors, cursor movement, title
    /// setting) so model context is spent on text rather than on the byte
    /// soup a colored builder emits. `NO_COLOR` and `TERM=dumb` in the shell
    /// environment suppress most escapes at the source; this is the backstop
    /// for tools that color unconditionally.
    static func stripANSI(_ text: String) -> String {
        ToolOutputFormatter.stripAnsi(text)
    }

    /// Head+tail compaction to `maxModelOutputChars`. Applied AFTER
    /// `stripANSI`, so escape bytes never consume the budget.
    static func compact(_ text: String) -> String {
        guard text.count > maxModelOutputChars else { return text }
        let head = String(text.prefix(headChars))
        let tail = String(text.suffix(tailChars))
        let removed = text.count - headChars - tailChars
        return head + "\n... [\(removed) chars truncated] ...\n" + tail
    }

    /// Exit codes that carry a conventional meaning distinct from failure.
    ///
    /// `grep` exits 1 when nothing matched, `diff` 1 when files differ, and
    /// `test`/`[` 1 when the condition is false. All three are the command
    /// WORKING and the model needs the distinction drawn: reporting them as
    /// `isError` invites a retry of a command that answered its question
    /// (Claude Code reference: `commandSemantics.ts`). The head word is
    /// matched rather than the whole line, so `grep -r foo .` qualifies but
    /// `echo grep` does not.
    static func benignExitNote(command: String, exitCode: Int32) -> String? {
        guard exitCode == 1 else { return nil }
        let head = command.trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines).first?
            .lowercased() ?? ""
        let word = head.components(separatedBy: "/").last ?? head
        switch word {
        case "grep", "egrep", "fgrep", "zgrep", "rg", "ripgrep":
            return "No matches found"
        case "diff":
            return "Files differ"
        case "test", "[":
            return "Condition evaluated to false"
        default:
            return nil
        }
    }

    /// The environment a shell child runs under: the app's own environment
    /// (PATH, HOME and friends come from the parent process, matching Claude
    /// Code's `subprocessEnv()`) with the hang-prevention overrides on top.
    ///
    /// `GIT_EDITOR=true` fails a messageless `git commit` fast with an error
    /// instead of hanging forever on an editor the user cannot see; the
    /// PAGER pair keeps `git log` and `man` printing to the pipe rather than
    /// blocking on a pager reading from a terminal that is not there.
    /// `TERM=dumb` and `NO_COLOR=1` reduce escapes at the source, which the
    /// ANSI stripper only backs up.
    static func shellEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["GIT_EDITOR"] = "true"
        env["GIT_PAGER"] = "cat"
        env["PAGER"] = "cat"
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        env["TURBOSPARK_APP"] = "1"
        return env
    }
}
