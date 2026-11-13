import Foundation

/// Analyzes shell commands to identify collapsible search/read operations vs mutating executions.
public enum TerminalCommandClassifier {
    /// Read-only search commands whose outputs are suitable for collapsible presentation.
    public static let searchCommands: Set<String> = [
        "find", "grep", "rg", "ag", "ack", "locate", "which", "whereis"
    ]

    /// Read-only inspection commands.
    public static let readCommands: Set<String> = [
        "cat", "head", "tail", "less", "more", "wc", "stat", "file", "strings",
        "jq", "awk", "cut", "sort", "uniq", "tr", "ls", "tree"
    ]

    /// Git inspection subcommands that are read-only.
    public static let gitReadSubcommands: Set<String> = [
        "status", "diff", "log", "show", "branch", "tag", "remote"
    ]

    /// Development commands the `.auto` permission mode promises to run
    /// without a prompt ("standard development commands silently ... cargo/npm
    /// builds", `AppPermissionMode.descriptionText`).
    ///
    /// **These are not SAFE in the abstract and the list does not pretend
    /// they are.** `cargo run` executes the project, `npm install` runs
    /// postinstall scripts, `make` runs whatever the Makefile says. They are
    /// auto-approvable because a user who selected `.auto` and pointed the
    /// project at a repository has already decided to trust that
    /// repository's build, which is a different claim from trusting an
    /// arbitrary string the model wrote.
    public static let buildCommands: Set<String> = [
        "cargo", "npm", "pnpm", "yarn", "bun", "make", "just", "swift", "go",
        "gradle", "mvn", "dotnet", "rustc", "tsc", "pytest", "tox", "ruff",
        "black", "mypy", "eslint", "prettier", "clippy", "rustfmt",
        "python", "python3", "node", "mkdir",
    ]

    /// Git subcommands `.auto` may run unprompted.
    ///
    /// `push`, `pull` and `checkout` are here because the DESTRUCTIVE forms
    /// of each already match `ToolRiskClassifier`'s denylist by name
    /// (`push --force`, `push -f`, `checkout -- .`, `reset --hard`,
    /// `clean -f`, `branch -D`, `stash clear`), and that is a case where a
    /// pattern list genuinely works: the dangerous variants differ from the
    /// safe ones by a flag rather than by hidden intent, and `git` is not a
    /// shell. `reset`, `clean` and `rebase` stay out -- their plain forms
    /// still rewrite a working tree or history.
    public static let gitAutoApprovableSubcommands: Set<String> =
        gitReadSubcommands.union([
            "add", "commit", "fetch", "stash", "push", "pull", "checkout", "switch",
        ])

    /// Characters that chain, redirect or substitute, anywhere in the string.
    ///
    /// Any one of these means the text is a shell program rather than a
    /// single invocation, and a head-word check on it describes the first
    /// fragment while saying nothing about the rest.
    ///
    /// Quotes and backslashes are deliberately NOT here. They hide a head
    /// word (`r""m -rf ~` and `r\m -rf ~` both execute `rm`) and are rejected
    /// for that reason by `headIsPlain` below, but in an ARGUMENT they change
    /// nothing about which program runs, and rejecting them there fails
    /// `git commit -m 'msg'`, `grep -rn 'struct' src/` and
    /// `find . -name '*.swift'` -- ordinary commands the `.auto` mode exists
    /// to run.
    private static let controlCharacters: Set<Character> = [
        "|", ";", "&", "$", "`", ">", "<", "(", ")", "{", "}", "\n", "\r",
    ]

    /// Characters that must not appear in the PROGRAM NAME, where they exist
    /// only to disguise it from a word-based check.
    private static let headObfuscationCharacters: Set<Character> = ["\"", "'", "\\", "="]

    /// Flags that hand an interpreter a program on its command line.
    ///
    /// Scoped to `interpreters` rather than checked globally, because `-e`
    /// and `-c` are ordinary flags elsewhere (`grep -e pattern`, `sort -c`).
    private static let inlineCodeFlags: Set<String> = ["-c", "-e", "--eval", "--command"]

    /// Commands that execute a program supplied as an argument.
    private static let interpreters: Set<String> = [
        "sh", "bash", "zsh", "ksh", "fish", "python", "python2", "python3",
        "perl", "ruby", "node", "deno", "php", "osascript", "env", "xargs",
    ]

    /// Whether `command` is ONE plain invocation whose program name can be
    /// read off the front, with no shell control characters anywhere.
    ///
    /// This is the precondition for every allowlist decision below it: a
    /// head-word check on a string containing `;` or `$(` classifies the
    /// first word and says nothing at all about the rest.
    public static func isSingleSimpleInvocation(_ command: String) -> Bool {
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard !trimmed.contains(where: { controlCharacters.contains($0) }) else { return false }

        // The head word must be a plain program name. Quotes and backslashes
        // in it are obfuscation (`r""m`, `r\m`), and an `=` means a
        // `VAR=value cmd` prefix, which moves the real program along by one
        // and can rewrite PATH or IFS on the way there.
        guard let head = trimmed.components(separatedBy: .whitespaces).first(where: {
            !$0.isEmpty
        }) else { return false }
        guard !head.contains(where: { headObfuscationCharacters.contains($0) }) else {
            return false
        }

        return true
    }

    /// The program name of a simple invocation, or nil when the command is
    /// not one. Path-qualified head words (`/usr/bin/grep`) resolve to their
    /// last component.
    public static func headCommand(_ command: String) -> String? {
        guard isSingleSimpleInvocation(command) else { return nil }
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let head = trimmed.components(separatedBy: .whitespaces).first(where: {
            !$0.isEmpty
        }) else { return nil }
        return (head as NSString).lastPathComponent.lowercased()
    }

    /// Whether `command` may run under `.auto` with no confirmation.
    ///
    /// **Positive test, and that direction is the whole point.** The pattern
    /// denylist in `ToolRiskClassifier` still runs first and still names a
    /// reason when it fires, but it can only ever describe attacks somebody
    /// wrote down: `rm -rf ~` matches it and `r""m -rf ~` does not, while zsh
    /// runs both. Deciding what MAY run, rather than enumerating what may
    /// not, is what makes an unrecognised string ask instead of execute.
    public static func isAutoApprovable(_ command: String) -> Bool {
        guard let head = headCommand(command) else { return false }
        let parts = command.trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }

        // **AN INTERPRETER HANDED CODE ON ITS COMMAND LINE IS A SHELL AGAIN.**
        //
        // `python3` is on the build list because `.auto` promises to run test
        // and build commands, and `python3 -m pytest` is one. But
        // `python3 -c 'import os; os.system(...)'` is the same arbitrary
        // execution this whole gate exists to stop, wearing an allowlisted
        // program name. The head word alone cannot tell those apart; the
        // inline-code flag can, and it is the ONLY thing making the
        // interpreter entries above safe to have.
        if interpreters.contains(head), hasInlineCodeFlag(head: head, args: parts.dropFirst()) {
            return false
        }

        // **A READ-ONLY HEAD WORD DOES NOT MAKE ITS FLAGS READ-ONLY.**
        // `find -delete`, `awk -f`, `rg --pre`, `tree -o` and friends write or
        // execute while wearing a head word from the read list. The
        // `ToolRiskClassifier` denylist names some of them, but this function
        // is also the only gate the subagent path calls, and a denylist only
        // knows what somebody wrote down.
        if hasDangerousFlag(head: head, args: Array(parts.dropFirst())) {
            return false
        }

        if searchCommands.contains(head) || readCommands.contains(head)
            || buildCommands.contains(head)
        {
            return true
        }

        if head == "git" {
            guard parts.count > 1 else { return false }
            let sub = parts[1].lowercased()
            guard gitAutoApprovableSubcommands.contains(sub) else { return false }
            return !hasDangerousGitForm(sub: sub, args: Array(parts.dropFirst(2)))
        }

        return false
    }

    // MARK: - Flag-level refusals

    /// Whether an interpreter is being handed code on its command line.
    ///
    /// Exact-token matching missed attached forms (`python3 -cprint`,
    /// `node --eval=x`) and clusters (`python3 -ec x`, `node -p ...`).
    /// Over-refusing is the safe direction here: a refused command asks.
    private static func hasInlineCodeFlag(head: String, args: ArraySlice<String>) -> Bool {
        for raw in args {
            let token = raw.lowercased()
            if inlineCodeFlags.contains(token) { return true }
            if token.hasPrefix("--eval=") || token.hasPrefix("--command=")
                || token == "--print" || token.hasPrefix("--print=")
            {
                return true
            }
            // Single-dash cluster of lowercase letters: `-c`, `-cCODE`, `-ec`.
            // `node -p` / `-pe` print the value of an expression, which
            // evaluates it. Case-sensitive on the raw token so `-Werror` and
            // `-E` (not code flags) are left alone.
            if raw.hasPrefix("-"), !raw.hasPrefix("--"), raw.count > 1,
                raw.dropFirst().allSatisfy({ $0.isLowercase && $0.isLetter })
            {
                if raw.contains("c") || raw.contains("e") { return true }
                if head == "node", raw.contains("p") { return true }
            }
        }
        return false
    }

    /// Flags that make an otherwise read-only command write a file or run
    /// another program.
    private static func hasDangerousFlag(head: String, args: [String]) -> Bool {
        let lowered = args.map { $0.lowercased() }
        func hasFlag(_ names: Set<String>) -> Bool {
            lowered.contains { token in
                names.contains(token) || names.contains(where: { $0.hasPrefix("--") && token.hasPrefix($0 + "=") })
            }
        }
        switch head {
        case "find", "locate":
            return hasFlag([
                "-delete", "-exec", "-execdir", "-ok", "-okdir",
                "-fprint", "-fprint0", "-fprintf", "-fls",
            ])
        case "awk":
            // `-f` runs a program file that can call `system`; `-i` is gawk's
            // in-place edit. `system` anywhere is refused because a program
            // text without braces or parens can still reach it.
            return hasFlag(["-f", "--file", "-i", "--include"])
                || lowered.contains { $0.contains("system") }
        case "sort":
            return hasFlag(["-o", "--output"])
        case "tree":
            return hasFlag(["-o"])
        case "rg", "ag", "ack":
            // ripgrep runs `--pre` on every file; `--hostname-bin` runs a program too.
            return hasFlag(["--pre", "--hostname-bin"])
        default:
            return false
        }
    }

    /// Git subcommand argument forms that write, force, delete, rewrite
    /// history or run a program, within the subcommands `.auto` admits.
    ///
    /// The destructive-by-flag cases differ from the safe ones by a flag, so a
    /// pattern list works here. Anything this cannot positively read as
    /// benign for `branch`, `tag`, `remote` and `stash` is refused, since
    /// those subcommands are reads in one spelling and writes in another.
    /// Residual: `git checkout <path>` restores a file from the index, which
    /// cannot be told from `git checkout <branch>` without reading the repo.
    private static func hasDangerousGitForm(sub: String, args: [String]) -> Bool {
        let lowered = args.map { $0.lowercased() }
        let flags = lowered.filter { $0.hasPrefix("-") }
        let positionals = lowered.filter { !$0.hasPrefix("-") }
        func hasFlag(_ names: Set<String>) -> Bool {
            flags.contains { token in
                names.contains(token) || names.contains(where: { $0.hasPrefix("--") && token.hasPrefix($0 + "=") })
            }
        }
        // `-c`-style program overrides on the transport commands.
        let runsProgram: Set<String> = ["--upload-pack", "--receive-pack", "--exec"]
        // Short flag clusters containing `f` or `d`, on the raw case-sensitive
        // arguments so `-D` and `-d` are told apart from `-u`.
        func hasShortFlag(_ letters: Set<Character>) -> Bool {
            args.contains { arg in
                arg.hasPrefix("-") && !arg.hasPrefix("--") && arg.dropFirst().contains(where: letters.contains)
            }
        }

        switch sub {
        case "diff", "log", "show":
            return hasFlag(["--output", "--ext-diff"])
        case "branch":
            if hasFlag(["--delete", "--move", "--copy", "--force", "--set-upstream-to",
                        "--unset-upstream", "--edit-description", "--track", "--no-track"]) {
                return true
            }
            if hasShortFlag(["d", "D", "m", "M", "c", "C", "f", "u"]) { return true }
            // A bare positional creates a branch unless a list flag is present.
            let listing = hasFlag(["--list", "--show-current", "--contains", "--no-contains",
                                   "--merged", "--no-merged", "--points-at"])
                || hasShortFlag(["l"])
            return !positionals.isEmpty && !listing
        case "tag":
            if hasFlag(["--delete", "--force", "--sign", "--annotate", "--message", "--file", "--edit"]) {
                return true
            }
            if hasShortFlag(["d", "f", "s", "a", "m", "F", "e", "u"]) { return true }
            let listing = hasFlag(["--list", "--contains", "--no-contains", "--merged",
                                   "--no-merged", "--points-at"]) || hasShortFlag(["l"])
            return !positionals.isEmpty && !listing
        case "remote":
            // Only the bare listing and `-v` forms. `show` and `get-url`
            // read, but `add`, `set-url`, `remove`, `rename`, `prune` write.
            let readVerbs: Set<String> = ["show", "get-url"]
            if let first = positionals.first { return !readVerbs.contains(first) }
            return false
        case "stash":
            let readOrSave: Set<String> = ["list", "show", "push", "save", "apply", "pop", "create"]
            if let first = positionals.first, !readOrSave.contains(first) { return true }
            return false
        case "fetch", "pull":
            return hasFlag(runsProgram)
        case "push":
            if hasFlag(runsProgram.union([
                "--force", "--force-with-lease", "--force-if-includes", "--delete",
                "--mirror", "--prune",
            ])) { return true }
            if hasShortFlag(["f", "d"]) { return true }
            // `+ref` forces and `:ref` deletes, whatever the flags say.
            return positionals.contains { $0.hasPrefix("+") || $0.hasPrefix(":") }
        case "checkout", "switch":
            if hasFlag(["--force", "--discard-changes", "--ours", "--theirs", "--merge", "--patch"]) {
                return true
            }
            // `-f`, `-B` and `-C` reset or overwrite; `--` and `.` restore paths.
            if hasShortFlag(["f", "B", "C", "m"]) { return true }
            if args.contains("--") || args.contains(".") { return true }
            return false
        default:
            return false
        }
    }

    /// Determines if a shell command string is purely a search or read operation.
    ///
    /// **PRESENTATION ONLY. This is not a security check and must never be
    /// used as one.** It reads the FIRST WORD and nothing else, so
    /// `cat README && python3 -c '...'` is "collapsible" on the strength of
    /// `cat`. That was fine for deciding whether to fold a card in the
    /// transcript and was catastrophic when `ToolRiskClassifier` also used it
    /// to return `.safe`, which under `.auto` means "run with no prompt".
    /// The gate is `isAutoApprovable` below; the two are kept apart so a
    /// display tweak here can never widen it again.
    public static func isCollapsible(_ command: String) -> Bool {
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let firstWord = trimmed.components(separatedBy: .whitespaces).first?.lowercased() else {
            return false
        }
        let baseCmd = (firstWord as NSString).lastPathComponent

        if searchCommands.contains(baseCmd) || readCommands.contains(baseCmd) {
            return true
        }

        if baseCmd == "git" {
            let parts = trimmed.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            if parts.count > 1 {
                let sub = parts[1].lowercased()
                return gitReadSubcommands.contains(sub)
            }
        }

        return false
    }
}
