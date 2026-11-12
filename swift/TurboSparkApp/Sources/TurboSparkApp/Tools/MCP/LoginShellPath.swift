import Foundation

/// The `PATH` the user's login shell would give a terminal, read once per app
/// run so a stdio MCP server named `npx` or `uvx` resolves from a Finder or
/// Dock launch.
///
/// A GUI launch inherits only `/usr/bin:/bin:/usr/sbin:/sbin`, so a node from
/// nvm, volta or asdf is invisible to `McpClientEngine.resolveExecutablePath`
/// unless its directory happens to be in the fixed list. ZCode and OpenCode
/// solve the same problem the same way (an `-il` shell printing its env).
///
/// **ONLY `PATH` LEAVES THE SHELL.** OpenCode merges the whole captured env
/// into every MCP child, which hands a third-party binary every token the
/// user exports in `.zshrc`. The child environment here stays the allowlist
/// in `McpClientEngine.childEnvironment`; this type only widens where a
/// command is looked up and what the child's own `PATH` holds.
///
/// **NOT RUN AT LAUNCH.** The first stdio spawn pays for it, so a user who
/// never configures an MCP server never runs their shell rc from this app.
enum LoginShellPath {
    static let beginMarker = "__TURBOSPARK_PATH_BEGIN__"
    static let endMarker = "__TURBOSPARK_PATH_END__"
    static let defaultTimeoutSeconds: TimeInterval = 4
    static let outputCapBytes = 64 * 1024

    /// What the probe shell starts with: enough for an rc file to find its
    /// own tools, nothing from this app's environment beyond identity and
    /// locale.
    static let bootstrapPath = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    static let forwardedKeys = ["HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR"]

    /// The text between the LAST marker pair, absolute entries only, or nil.
    /// rc files print banners, `fortune` output and version-manager notices
    /// before and after the command, so the markers are what separate PATH
    /// from noise.
    ///
    /// **RELATIVE ENTRIES ARE DROPPED.** A terminal PATH may hold `.` or
    /// `node_modules/.bin`, and an MCP child runs with the project root as its
    /// cwd, so such an entry would let a cloned repository's own `node` run
    /// when any stdio server starts in it.
    static func parse(_ output: String) -> String? {
        guard let end = output.range(of: endMarker, options: .backwards),
              let begin = output.range(
                of: beginMarker, options: .backwards, range: output.startIndex..<end.lowerBound)
        else { return nil }
        let entries = output[begin.upperBound..<end.lowerBound]
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: ":")
            .filter { $0.hasPrefix("/") }
        return entries.isEmpty ? nil : entries.joined(separator: ":")
    }

    /// The account's shell from the passwd entry, then `$SHELL`, each only if
    /// `/etc/shells` lists it, then zsh (the macOS default).
    ///
    /// **`$SHELL` IS NOT TRUSTED ON ITS OWN.** Any process running as the user
    /// can `launchctl setenv SHELL /tmp/x`, and this app would then run that
    /// binary as its own child, under its own privacy grants. `/etc/shells`
    /// is root-owned, so it decides what may be run.
    static func defaultShell(
        parent: [String: String] = ProcessInfo.processInfo.environment,
        passwdShell: String? = currentPasswdShell(),
        allowedShells: Set<String> = systemShells()
    ) -> String {
        for candidate in [passwdShell, parent["SHELL"]] {
            if let candidate, allowedShells.contains(candidate) { return candidate }
        }
        return "/bin/zsh"
    }

    static func currentPasswdShell() -> String? {
        guard let entry = getpwuid(getuid()), let raw = entry.pointee.pw_shell else { return nil }
        let shell = String(cString: raw)
        return shell.isEmpty ? nil : shell
    }

    /// The absolute paths `/etc/shells` lists, comments and blanks skipped.
    static func systemShells(contents: String? = try? String(contentsOfFile: "/etc/shells", encoding: .utf8)) -> Set<String> {
        Set((contents ?? "").components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { $0.hasPrefix("/") })
    }

    /// The executable and arguments for one probe, or nil for a shell that
    /// cannot run it. `-i` is what loads `.zshrc` / `.bashrc`, where nvm and
    /// most version managers put themselves; other shells get `-l` only.
    /// `printenv` rather than `$PATH` so fish, whose PATH is a list, still
    /// prints the colon form.
    static func shellInvocation(shell: String) -> (URL, [String])? {
        let name = (shell as NSString).lastPathComponent.lowercased()
        // nushell does not accept POSIX-style -c scripts; OpenCode skips it too.
        if name == "nu" { return nil }
        let script = "printf '%s' \(beginMarker); /usr/bin/printenv PATH; printf '%s' \(endMarker)"
        let flags = (name == "zsh" || name == "bash") ? "-ilc" : "-lc"
        return (URL(fileURLWithPath: shell), [flags, script])
    }

    static func probeEnvironment(parent: [String: String]) -> [String: String] {
        var environment: [String: String] = [:]
        for key in forwardedKeys {
            if let value = parent[key] { environment[key] = value }
        }
        environment["PATH"] = bootstrapPath
        // A dumb terminal keeps prompt themes and rc-time TUI setup quiet.
        environment["TERM"] = "dumb"
        return environment
    }

    /// One probe. Nil on timeout, truncation, a non-zero exit, or no markers:
    /// a half-read PATH is worse than the fixed list.
    static func capture(
        shell: String,
        parent: [String: String] = ProcessInfo.processInfo.environment,
        timeoutSeconds: TimeInterval = defaultTimeoutSeconds
    ) async -> String? {
        guard let (executable, arguments) = shellInvocation(shell: shell),
              FileManager.default.isExecutableFile(atPath: executable.path)
        else { return nil }
        // **HOME, NEVER THE PROJECT ROOT.** A direnv hook or a project-aware
        // rc in a cloned repository must not run because a server started.
        let home = URL(fileURLWithPath: parent["HOME"] ?? NSHomeDirectory(), isDirectory: true)
        guard let output = try? await ProcessExecutor.run(
            executableURL: executable,
            arguments: arguments,
            currentDirectoryURL: home,
            environment: probeEnvironment(parent: parent),
            timeoutSeconds: timeoutSeconds,
            outputCapBytes: outputCapBytes),
              !output.timedOut, !output.outputTruncated, output.exitCode == 0
        else { return nil }
        return parse(output.stdout)
    }

    /// The cached PATH, capturing it on the first call. A failed capture is
    /// cached as nil too, so a broken rc costs one timeout per run, not one
    /// per spawn.
    static func resolved() async -> String? {
        await Cache.shared.value()
    }

    /// The cached PATH without ever capturing, for synchronous callers such
    /// as the marketplace pre-check. Nil until the first stdio spawn.
    static var cached: String? {
        Cache.shared.peek()
    }

    private final class Cache: @unchecked Sendable {
        static let shared = Cache()
        private let lock = NSLock()
        private var task: Task<String?, Never>?
        private var result: String??

        func value() async -> String? {
            let pending = lock.withLock {
                if let task { return task }
                // Unstructured on purpose: a caller cancelled mid-capture
                // must not kill the probe and cache its nil for everyone.
                let started = Task { await LoginShellPath.capture(shell: LoginShellPath.defaultShell()) }
                task = started
                return started
            }
            let value = await pending.value
            lock.withLock { result = .some(value) }
            return value
        }

        func peek() -> String? {
            lock.withLock { result ?? nil }
        }
    }
}
