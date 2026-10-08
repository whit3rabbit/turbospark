import Foundation
import JavaScriptCore

/// OS-level confinement for the codemode worker process.
///
/// The worker is a copy of this app's binary running model-written
/// JavaScript in JavaScriptCore. A script has no I/O of its own, but a memory
/// bug in the engine would give native code at the user's privileges: their
/// files, the network, other processes, the clipboard. pi gets a hard
/// boundary from QuickJS compiled to WebAssembly; this is the closest native
/// equivalent: the worker applies a Seatbelt profile to itself, once, before
/// it creates a JavaScript context, and cannot lift it.
///
/// **A DENY-LIST ON TOP OF "allow default", NOT DENY-BY-DEFAULT.** Apple's
/// named deny-by-default profile (`pure-computation`) was tried first: under
/// it JavaScriptCore traps while creating a `JSContext`. A custom profile
/// that starts from allow and removes the capabilities worth protecting
/// leaves JavaScriptCore at full JIT speed (measured the same, 0.07 s vs
/// 0.07 s on a hot loop). The cost is that anything NOT listed stays allowed,
/// so every entry below earns its place and the probe pins each one.
///
/// What is denied, and what each is for:
/// - `network*`: no connecting out or accepting in (exfiltration).
/// - `process-exec*`, `process-fork`: no launching anything.
/// - `file-write*`: no writes anywhere (inherited pipes keep working).
/// - `file-read*` under `/Users` and `/Volumes`: no reading the user's files
///   or other disks. The app's own bundle directory is allowed back in, since
///   a development build lives under `/Users`.
/// - `signal`: no signalling other processes (the host, the app).
/// - `process-info*` of others: no inspecting other processes.
/// - `mach-lookup`, `iokit-open`, `user-preference-read`: no system
///   services. This is what stops the clipboard (`NSPasteboard` is a Mach
///   service); the first draft of the profile left it readable, which the
///   probe caught.
///
/// Applied in `REPLWorkerMain.serveCodemodeRequests` and FAIL CLOSED: if the
/// profile cannot be applied the worker reports that and does not run the
/// script. A worker that is already inside an App Sandbox would refuse the
/// call, which is the safe direction to fail.
enum CodemodeWorkerSandbox {
    /// Internal diagnostic entry: `--turbospark-js-repl-worker
    /// --codemode-sandbox-probe [--unconfined] [<file to try reading>]`.
    /// Prints one `name=result` line per capability so tests can pin the
    /// confinement against the real packaged binary. `--unconfined` skips
    /// applying the profile and is the control that shows each probe really
    /// detects the capability it names.
    static let probeArgument = "--codemode-sandbox-probe"
    static let unconfinedArgument = "--unconfined"

    static func profile(allowedReadRoot: String?) -> String {
        profile(allowedReadRoots: allowedReadRoot.map { [$0] } ?? [])
    }

    /// The directories the worker may still read under `/Users`: its own
    /// bundle and its executable's directory, with symlinks RESOLVED because
    /// Seatbelt matches a `subpath` against the real path. A development
    /// build is reached through a symlink (`.build/debug`), and an allow rule
    /// naming the link would never match, leaving the worker unable to read
    /// its own files.
    static func ownReadRoots() -> [String] {
        func resolved(_ path: String) -> String {
            URL(fileURLWithPath: path).resolvingSymlinksInPath().path
        }
        var roots = [resolved(Bundle.main.bundlePath)]
        if let executable = Bundle.main.executablePath {
            roots.append(resolved((executable as NSString).deletingLastPathComponent))
        }
        var seen: Set<String> = []
        return roots.filter { seen.insert($0).inserted }
    }

    static func profile(allowedReadRoots: [String]) -> String {
        var rules = [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny process-exec*)",
            "(deny process-fork)",
            "(deny file-write*)",
            "(deny file-read* (subpath \"/Users\") (subpath \"/Volumes\"))",
            "(deny signal)",
            "(allow signal (target self))",
            "(deny process-info* (target others))",
            "(deny mach-lookup)",
            "(deny iokit-open)",
            "(deny user-preference-read)",
        ]
        // After the deny, so they win: later rules override earlier ones.
        for root in allowedReadRoots {
            if let quoted = quotedSchemeString(root) {
                rules.append("(allow file-read* (subpath \(quoted)))")
            }
        }
        return rules.joined(separator: "\n")
    }

    /// Applies the profile to this process. Returns nil on success, otherwise
    /// why it could not. Irreversible; call it only in the worker.
    static func apply(profile: String? = nil) -> String? {
        let text = profile ?? Self.profile(allowedReadRoots: ownReadRoots())
        // `sandbox_init` is not exposed to Swift, and `@_silgen_name` would tie
        // the link to a private symbol; resolve it at run time instead.
        guard let initSymbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "sandbox_init") else {
            return "sandbox_init is not available on this system."
        }
        typealias SandboxInit = @convention(c) (
            UnsafePointer<CChar>, UInt64, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
        ) -> Int32
        let sandboxInit = unsafeBitCast(initSymbol, to: SandboxInit.self)

        var errorBuffer: UnsafeMutablePointer<CChar>?
        // Flags 0: the first argument is profile source, not a named profile.
        let status = sandboxInit(text, 0, &errorBuffer)
        guard status != 0 else { return nil }
        let reason = errorBuffer.map { String(cString: $0) } ?? "status \(status)"
        if let errorBuffer, let freeSymbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "sandbox_free_error") {
            typealias SandboxFreeError = @convention(c) (UnsafeMutablePointer<CChar>?) -> Void
            unsafeBitCast(freeSymbol, to: SandboxFreeError.self)(errorBuffer)
        }
        return reason
    }

    /// An SBPL string literal for a path, or nil when the path has anything
    /// that cannot be quoted safely (a control character).
    private static func quotedSchemeString(_ value: String) -> String? {
        guard !value.unicodeScalars.contains(where: { $0.value < 0x20 || $0.value == 0x7F }) else {
            return nil
        }
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }

    // MARK: Probe

    /// Applies the profile (unless `--unconfined`), then tries each
    /// capability and prints the outcome. Exits the process when done.
    static func runProbe(arguments: [String]) -> Never {
        let unconfined = arguments.contains(unconfinedArgument)
        let readFile = arguments
            .drop(while: { $0 != probeArgument }).dropFirst()
            .first { $0 != unconfinedArgument }

        func report(_ name: String, _ value: String) {
            FileHandle.standardOutput.write(Data("\(name)=\(value)\n".utf8))
        }

        // Paths are resolved BEFORE confinement: once the profile is applied,
        // reading a symlink under /Users is denied, so resolving afterwards
        // silently returns the unresolved path.
        let startedPath = Bundle.main.executablePath ?? CommandLine.arguments[0]
        let resolvedPath = URL(fileURLWithPath: startedPath).resolvingSymlinksInPath().path
        let roots = ownReadRoots()

        if unconfined {
            report("applied", "skipped")
        } else if let failure = apply() {
            report("applied", "FAILED: \(failure)")
            exit(EXIT_FAILURE)
        } else {
            report("applied", "ok")
        }

        // JavaScriptCore must still work, with its JIT, under the profile.
        let context = JSContext()
        let started = Date()
        let computed = context?.evaluateScript(
            "function f(n){return n<2?n:f(n-1)+f(n-2)} let s=0; "
                + "for(let i=0;i<2000000;i++){s=(s+i*7)%1000003} f(25)+':'+s")?.toString()
        report("javascript", computed == "75025:" + String(expectedLoopValue) ? "ok" : "WRONG \(computed ?? "nil")")
        report("javascript_seconds", String(format: "%.2f", Date().timeIntervalSince(started)))

        report("read_user_file", readFile.map { path in
            (try? Data(contentsOf: URL(fileURLWithPath: path))) != nil ? "allowed" : "denied"
        } ?? "not_tested")
        // The worker reads its own files by their real paths. (The path it was
        // STARTED by may go through a symlink, which the profile cannot let it
        // traverse; reported separately for diagnosis, not asserted.)
        report("read_own_bundle", FileHandle(forReadingAtPath: resolvedPath) != nil ? "allowed" : "denied")
        report("read_own_bundle_via_start_path", FileHandle(forReadingAtPath: startedPath) != nil ? "allowed" : "denied")
        report("own_roots", roots.joined(separator: ","))
        report("executable_path", startedPath)

        let scratch = NSTemporaryDirectory() + "codemode-probe-\(getpid())"
        let wrote = FileManager.default.createFile(atPath: scratch, contents: Data("x".utf8))
        if wrote { try? FileManager.default.removeItem(atPath: scratch) }
        report("write_file", wrote ? "allowed" : "denied")

        let spawn = Process()
        spawn.executableURL = URL(fileURLWithPath: "/usr/bin/true")
        report("exec", (try? spawn.run()) != nil ? "allowed" : "denied")
        // Polled, not `waitUntilExit`: that call has been seen to hang here.
        let reapBy = Date().addingTimeInterval(2)
        while spawn.isRunning && Date() < reapBy { usleep(10_000) }

        // `posix_spawn` starts a program without `fork`, so it is governed by
        // `process-exec*` alone. `Process.run` above forks first and is also
        // stopped by `process-fork`, which would hide a missing exec rule.
        report("spawn", spawnProbe())
        report("connect", connectProbe())
        report("signal_parent", kill(getppid(), 0) == 0 ? "allowed" : "denied")
        report("mach_lookup", machLookupProbe())

        // LAST, because success replaces this process: an in-place `execve`
        // involves no fork, so only `process-exec*` stops it (the spawn paths
        // above are also stopped by `process-fork`, which would hide a missing
        // exec rule). If it works the output simply ends at "attempting".
        report("exec_inplace", "attempting")
        let program = strdup("/usr/bin/true")
        defer { free(program) }
        let arguments: [UnsafeMutablePointer<CChar>?] = [program, nil]
        let environment: [UnsafeMutablePointer<CChar>?] = [nil]
        _ = execve("/usr/bin/true", arguments, environment)
        report("exec_inplace", "denied")
        exit(EXIT_SUCCESS)
    }

    /// What the probe's loop evaluates to; computed natively so the check
    /// does not trust the engine under test to grade itself.
    private static let expectedLoopValue: Int = {
        var s = 0
        for i in 0..<2_000_000 { s = (s + i * 7) % 1_000_003 }
        return s
    }()

    private static func spawnProbe() -> String {
        var pid: pid_t = 0
        let program = strdup("true")
        defer { free(program) }
        let arguments: [UnsafeMutablePointer<CChar>?] = [program, nil]
        let environment: [UnsafeMutablePointer<CChar>?] = [nil]
        let status = posix_spawn(&pid, "/usr/bin/true", nil, nil, arguments, environment)
        guard status == 0 else { return "denied" }
        var exitStatus: Int32 = 0
        waitpid(pid, &exitStatus, 0)
        return "allowed"
    }

    /// `denied` when the sandbox refuses (EPERM); `reached_network` when the
    /// connection got as far as the network stack and was refused there.
    private static func connectProbe() -> String {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return errno == EPERM ? "denied" : "no_socket" }
        defer { close(fd) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = UInt16(9).bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if result == 0 { return "connected" }
        return errno == EPERM ? "denied" : "reached_network"
    }

    /// Looks up the pasteboard service by name: the same Mach lookup
    /// `NSPasteboard` makes to read the clipboard.
    private static func machLookupProbe() -> String {
        guard let lookup = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "bootstrap_look_up"),
              let portSymbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "bootstrap_port")
        else { return "unavailable" }
        typealias Lookup = @convention(c) (mach_port_t, UnsafePointer<CChar>, UnsafeMutablePointer<mach_port_t>) -> kern_return_t
        let bootstrapPort = portSymbol.assumingMemoryBound(to: mach_port_t.self).pointee
        var service: mach_port_t = 0
        let status = unsafeBitCast(lookup, to: Lookup.self)(bootstrapPort, "com.apple.pasteboard.1", &service)
        return status == KERN_SUCCESS ? "allowed" : "denied"
    }
}
