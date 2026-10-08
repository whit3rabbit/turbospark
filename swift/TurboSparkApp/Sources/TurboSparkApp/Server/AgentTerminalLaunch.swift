import AppKit
import Foundation
import TurboSpark

/// Launches a coding agent from the Server pane by handing the
/// `turbospark start <agent>` command to Terminal.app.
///
/// **A TERMINAL, NOT A CHILD PROCESS.** Claude Code and Codex are
/// interactive TUIs that need a TTY, which this GUI process does not have
/// and cannot fake; and `do script` runs the user's login shell, so the
/// agent binaries (`claude`, `codex`, npm-global or version-manager
/// installs) resolve through the user's own PATH. The `turbospark` half of
/// the command is spelled with the ABSOLUTE path found at button time so
/// it alone does not depend on that shell profile.
enum AgentTerminalLaunch {
    /// Locates the `turbospark` front end, mirroring the FFI daemon
    /// spawner's lookup (`crates/ffi/src/api/daemon.rs`
    /// `find_server_binary`) plus `~/.local/bin`, where `make install`
    /// puts the CLI.
    static func findCLI() -> URL? {
        var candidates: [URL] = []
        if let bundleDir = Bundle.main.executableURL?.deletingLastPathComponent() {
            candidates.append(bundleDir.appendingPathComponent("turbospark"))
        }
        if let path = ProcessInfo.processInfo.environment["PATH"] {
            candidates += pathDirectories(path)
                .map { URL(fileURLWithPath: $0).appendingPathComponent("turbospark") }
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        candidates.append(URL(fileURLWithPath: home)
            .appendingPathComponent(".cargo/bin/turbospark"))
        candidates.append(URL(fileURLWithPath: home)
            .appendingPathComponent(".local/bin/turbospark"))
        // A GUI app's PATH is launchd's minimal one, which omits Homebrew.
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/turbospark"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/turbospark"))
        return candidates.first {
            $0.isFileURL && FileManager.default.isExecutableFile(atPath: $0.path)
        }
    }

    /// Opens a new Terminal window running `command`. Throws osascript's
    /// own message on refusal -- the automation permission denied is
    /// error -1743, and the caller's fallback is the clipboard.
    static func openInTerminal(command: String) throws {
        let script = TurboSparkAgent.terminalDoScript(command: command)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script]
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let message = String(
                decoding: stderr.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw NSError(
                domain: "AgentTerminalLaunch", code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: message])
        }
    }

    /// Writes `key` to a 0600 file in the per-user temp directory and returns
    /// its path. The key travels to the launched shell through this file
    /// rather than the command text, so it never lands in Terminal's
    /// scrollback, shell history, or `ps` output.
    static func writeAPIKeyFile(_ key: String) throws -> String {
        let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("turbospark-launch-\(UUID().uuidString).key")
        // Created with the final mode: write-then-chmod would leave a window
        // where the secret is world-readable under a permissive umask.
        guard FileManager.default.createFile(
            atPath: url.path, contents: Data(key.utf8), attributes: [.posixPermissions: 0o600])
        else {
            throw NSError(
                domain: "AgentTerminalLaunch", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Could not write the API key file."])
        }
        return url.path
    }

    /// Prefixes `command` so the launched shell exports `TURBOSPARK_API_KEY`
    /// from `keyFilePath` and deletes the file before anything else runs.
    /// Without it `turbospark start` spawns an unauthenticated daemon (or the
    /// agent is refused with 401 by a keyed one).
    static func commandExportingAPIKey(_ command: String, keyFilePath: String) -> String {
        let file = ShellQuote.single(keyFilePath)
        return "export TURBOSPARK_API_KEY=\"$(cat \(file))\" && rm -f \(file) && \(command)"
    }

    /// The degrade path when Terminal cannot be scripted (automation
    /// denied, Terminal missing): the command lands on the clipboard, where
    /// pasting it reproduces the launch exactly.
    static func copyToClipboard(_ command: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(command, forType: .string)
    }

    /// Splits a PATH-style string into its directories, dropping empties.
    /// A helper rather than an inline `split` so the shape can be asserted
    /// without touching the real environment.
    static func pathDirectories(_ path: String) -> [String] {
        path.split(separator: ":", omittingEmptySubsequences: true).map(String.init)
    }
}
