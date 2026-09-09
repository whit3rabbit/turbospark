import Foundation
import TurboSpark

/// The arguments a background daemon started from this app runs with.
///
/// **A DAEMON IS A SEPARATE PROCESS AND INHERITS NOTHING.** The menu bar's
/// "Start Background Daemon" spawns `turbospark-server`, which without
/// arguments binds its own default port and starts with NO API key -- and
/// the key is the server's only access control. A start that dropped the
/// user's settings produced exactly the unauthenticated server the
/// Advanced pane warns about, on a port the pane's address row does not
/// describe, because that row reads the in-app server.
///
/// Pure, so the list can be asserted without spawning anything. The key
/// rides as `--api-key` and the daemon's spawner moves it into the child's
/// `TURBOSPARK_API_KEY` -- the fallback `crates/server/src/main.rs`
/// resolves when the flag is absent -- so it sits in neither `ps` nor the
/// run directory's meta file, the two surfaces that fallback exists to
/// keep a key off.
enum ServerDaemonLaunch {
    static func args(
        port: UInt16,
        apiKey: String?,
        guardrails: ServerOptions.Guardrails
    ) -> [String] {
        var args: [String] = []
        // 0 is "automatic" for the IN-APP server and is not passed here: the
        // daemon records what it was handed in its meta file and its status
        // reads the port back from there, so an OS-assigned port would
        // surface as 0. An unpinned start takes the daemon's own default.
        if port != 0 {
            args += ["--port", String(port)]
        }
        args += ["--guardrails", guardrails.rawValue]
        if let apiKey {
            args += ["--api-key", apiKey]
        }
        return args
    }
}
