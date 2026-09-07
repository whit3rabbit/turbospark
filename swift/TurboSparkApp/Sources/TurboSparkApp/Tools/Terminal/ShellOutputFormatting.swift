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

    // MARK: - Spill files

    /// How many spilled files the spill root keeps. Older ones are pruned
    /// on each write; the cap exists so a session of huge builds cannot
    /// accumulate gigabytes of logs the model will never re-read.
    static let maximumSpillFiles = 20

    /// The directory spilled output is written under, inside the app's
    /// Application Support. Created on demand.
    static var spillRootURL: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first
            ?? FileManager.default.temporaryDirectory
        return base
            .appendingPathComponent("TurboSpark", isDirectory: true)
            .appendingPathComponent("spill", isDirectory: true)
    }

    /// Whether `url` lives under the spill root. This is the predicate the
    /// `read_file` path allowlist is built on (`resolveSecurePath` refuses
    /// absolute paths everywhere else), so it resolves symlinks and
    /// standardizes both sides before the prefix compare, exactly as
    /// `PathContainment` does for project roots.
    static func isUnderSpillRoot(_ url: URL) -> Bool {
        let root = spillRootURL.standardizedFileURL.resolvingSymlinksInPath().path
        let candidate = url.standardizedFileURL.resolvingSymlinksInPath().path
        return candidate == root || candidate.hasPrefix(root + "/")
    }

    /// Compaction PLUS recovery: text over the model cap is compacted as
    /// before, and the FULL text is written to a spill file the model can
    /// go back to. Claude Code reference: `src/utils/toolResultStorage.ts`
    /// persists oversized tool results rather than dropping their middle;
    /// this is the same idea for shell output, where the dropped middle of
    /// a long build or test log is precisely where the interesting failure
    /// often is not -- but the tail is, and a `grep` over the spill file
    /// finds the rest.
    ///
    /// Under the cap this is exactly `compact(stripANSI(...))` and writes
    /// nothing. Over it, the returned string names the spill path and how
    /// to read it. The spill receives what survived the PIPE cap (1 MB per
    /// stream, `ProcessExecutor`), which is the memory bound; this layer
    /// only bounds the context window.
    ///
    /// `spillName` makes the file DETERMINISTIC: the same name is
    /// overwritten each call instead of accumulating timestamped copies.
    /// This is what the background-shell snapshot passes, because the model
    /// polls that output repeatedly and one spill file per shell is the
    /// right shape there.
    static func compactWithSpill(
        _ text: String, label: String, spillName: String? = nil
    ) -> String {
        guard text.count > maxModelOutputChars else { return text }
        let compacted = compact(text)
        guard let spilledPath = writeSpillFile(text, label: label, name: spillName) else {
            return compacted + "\n[Output over the \(maxModelOutputChars) char display cap; "
                + "spilling the full text to disk failed. Re-run with narrower scope, "
                + "e.g. tail(1) or grep(1) on the source.]"
        }
        return compacted
            + "\n[Full output (\(text.count) chars) saved to \(spilledPath.path). "
            + "Read it with read_file (absolute paths under the spill directory are allowed), "
            + "or search it from the shell, e.g. grep -n <pattern> \(spilledPath.path).]"
    }

    /// Writes one spill file and prunes old ones. Returns nil when the
    /// write failed; callers fall back to plain compaction with a note.
    /// A non-nil `name` is used verbatim (one deterministic file, each
    /// call overwriting); otherwise the name is timestamped.
    private static func writeSpillFile(_ text: String, label: String, name: String?) -> URL? {
        let fm = FileManager.default
        let root = spillRootURL
        let fileName: String
        if let name {
            let safe = name.components(separatedBy: CharacterSet.alphanumerics.inverted)
                .joined(separator: "-")
            fileName = "\(safe.isEmpty ? "output" : safe).txt"
        } else {
            let safeLabel = label
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .prefix(24)
                .joined(separator: "-")
            let stamp = String(format: "%010.0f", NSDate().timeIntervalSince1970 * 100)
            fileName = "\(stamp)-\(safeLabel.isEmpty ? "output" : safeLabel).txt"
        }
        let fileURL = root.appendingPathComponent(fileName)
        do {
            try fm.createDirectory(at: root, withIntermediateDirectories: true)
            try text.write(to: fileURL, atomically: true, encoding: .utf8)
        } catch {
            return nil
        }
        pruneSpillFiles(root: root, keeping: maximumSpillFiles)
        return fileURL
    }

    /// Keeps the `keeping` newest files under `root` by modification date.
    static func pruneSpillFiles(root: URL, keeping: Int) {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
        else { return }
        guard contents.count > keeping else { return }
        let sorted = contents.sorted { lhs, rhs in
            let lDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            let rDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            return lDate > rDate
        }
        for victim in sorted.dropFirst(keeping) {
            try? fm.removeItem(at: victim)
        }
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
