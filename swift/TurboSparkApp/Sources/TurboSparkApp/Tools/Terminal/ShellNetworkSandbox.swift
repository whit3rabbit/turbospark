import Foundation

/// OS-level network confinement for model-run shell commands under the
/// network policy's Block all mode (Settings > Network).
///
/// The URL tools are gated per host in-process, but a shell command opens
/// its own sockets, so `curl` ignored every web setting. Under Block all the
/// command runs inside `sandbox-exec` with a Seatbelt profile that denies
/// the network, the same mechanism Codex uses for its macOS shell sandbox.
/// The other modes leave the shell unconfined: Seatbelt cannot filter by
/// domain, and the Settings pane says so.
///
/// **NEVER `(allow network* (local ip "localhost:*"))`.** It reads as "let
/// local servers listen" and is an outbound hole: an unbound outbound socket
/// has a wildcard local endpoint that the filter matches, so
/// `curl http://1.1.1.1` connected straight through it (measured, and pinned
/// by `ShellNetworkSandboxTests`). Outbound is allowed to a remote loopback
/// address only; bind and inbound are allowed on a local loopback address.
///
/// What the profile does not stop: a process asking a system daemon over
/// Mach IPC to fetch for it. The whole profile is `allow default` minus the
/// network, like `CodemodeWorkerSandbox`, so commands keep working.
enum ShellNetworkSandbox {
    static let sandboxExecPath = "/usr/bin/sandbox-exec"

    static func profile(allowLocalhost: Bool) -> String {
        var rules = ["(version 1)", "(allow default)", "(deny network*)"]
        if allowLocalhost {
            rules.append(#"(allow network-outbound (remote ip "localhost:*"))"#)
            rules.append(#"(allow network-bind network-inbound (local ip "localhost:*"))"#)
        }
        return rules.joined(separator: "\n")
    }

    /// The executable and arguments to launch `executable arguments` with,
    /// wrapped in `sandbox-exec` when the policy blocks the network.
    ///
    /// Fails closed: if `sandbox-exec` is missing, the command is refused
    /// rather than run unconfined under a policy that promised Block all.
    static func launch(
        executable: URL,
        arguments: [String],
        policy: AppNetworkPolicy = AppNetworkPolicyStore.current
    ) throws -> (executable: URL, arguments: [String]) {
        guard policy.mode == .blockAll else { return (executable, arguments) }
        guard FileManager.default.isExecutableFile(atPath: sandboxExecPath) else {
            throw NSError(
                domain: "TurboSparkTool", code: 43,
                userInfo: [NSLocalizedDescriptionKey:
                    "Network access is set to Block all, and \(sandboxExecPath) is unavailable to "
                        + "confine this command, so it was not run."])
        }
        return (
            URL(fileURLWithPath: sandboxExecPath),
            ["-p", profile(allowLocalhost: policy.allowLocalhost), executable.path] + arguments
        )
    }
}
