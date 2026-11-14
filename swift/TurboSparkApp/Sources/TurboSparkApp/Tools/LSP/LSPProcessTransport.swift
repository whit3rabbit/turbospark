import Foundation

enum LSPLaunchError: Error, Equatable, LocalizedError {
    case notConfigured
    case emptyCommand
    case relativeCommand(String)
    case notFound(String)
    case insideWorkspace(String)
    case badArgument

    var errorDescription: String? {
        switch self {
        case .notConfigured: return "This language server is not installed on this Mac."
        case .emptyCommand: return "This language server has no command."
        case .relativeCommand(let command):
            return "'\(command)' is a path relative to the project. A server must be an absolute path or a command on your PATH."
        case .notFound(let command): return "'\(command)' was not found on your PATH."
        case .insideWorkspace(let path):
            return "Refusing to run '\(path)': it is inside the project. A server must be installed outside the workspace."
        case .badArgument: return "A server argument is not valid."
        }
    }
}

/// OC-10: what may be launched as a language server. The command is the
/// USER's configuration; this policy only refuses what would let a cloned
/// repository supply the binary.
enum LSPLaunchPolicy {
    struct Launch: Equatable {
        var executable: String
        var arguments: [String]
    }

    static func resolve(
        config: LSPServerConfig, rootURL: URL, searchPath: String?,
        discoveredSourcekit: String? = LSPDiscovery.sourcekitLSPPath()
    ) throws -> Launch {
        let command: String
        if config.isDiscovered {
            guard let path = discoveredSourcekit else { throw LSPLaunchError.notConfigured }
            command = path
        } else {
            command = config.command.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !command.isEmpty else { throw LSPLaunchError.emptyCommand }
        guard !command.contains("\0"), !command.contains("\n") else { throw LSPLaunchError.badArgument }
        let args = config.isDiscovered ? [] : config.args
        guard args.allSatisfy({ !$0.contains("\0") }) else { throw LSPLaunchError.badArgument }

        // "./tools/server" and "node_modules/.bin/server" are workspace
        // binaries by construction. Bare names go to the PATH, absolute and
        // "~" paths are taken as written.
        if command.contains("/"), !command.hasPrefix("/"), !command.hasPrefix("~") {
            throw LSPLaunchError.relativeCommand(command)
        }
        guard let resolved = McpClientEngine.resolveExecutablePath(command, relativeTo: nil, searchPath: searchPath),
              FileManager.default.isExecutableFile(atPath: resolved)
        else { throw LSPLaunchError.notFound(command) }
        // Symlinks resolved on both sides: a PATH entry or a symlink that
        // lands inside the project is still a workspace binary.
        if PathContainment.isContained(URL(fileURLWithPath: resolved), in: rootURL) {
            throw LSPLaunchError.insideWorkspace(resolved)
        }
        return Launch(executable: resolved, arguments: args)
    }

    /// The server's whole environment: PATH, HOME, LANG and TMPDIR from the
    /// allowlist in `McpClientEngine.childEnvironment`, nothing else. No
    /// credentials, no proxy settings, no passthrough.
    static func environment(
        parent: [String: String] = ProcessInfo.processInfo.environment,
        executable: String, searchPath: String?
    ) -> [String: String] {
        McpClientEngine.childEnvironment(
            parent: parent, passthrough: [], declared: [:],
            executableDirectory: (executable as NSString).deletingLastPathComponent,
            searchPath: searchPath)
    }
}

/// Live language server pids, outside the actors so the synchronous quit path
/// can kill them without awaiting.
enum LSPProcessRegistry {
    private static let lock = NSLock()
    private static var pids = Set<pid_t>()

    static func add(_ pid: pid_t) { lock.lock(); pids.insert(pid); lock.unlock() }
    static func remove(_ pid: pid_t) { lock.lock(); pids.remove(pid); lock.unlock() }
    static var count: Int { lock.lock(); defer { lock.unlock() }; return pids.count }

    static func killAllNow() {
        lock.lock()
        let snapshot = Array(pids)
        pids.removeAll()
        lock.unlock()
        for pid in snapshot { ProcessExecutor.killTreeNow(pid) }
    }
}

/// A language server child process. Spawned by absolute path with a scrubbed
/// environment and the workspace root as its working directory. stdout feeds
/// the client's decoder (which enforces the frame bounds), stderr is drained
/// into a 4 KiB tail so a chatty server cannot block on a full pipe.
final class LSPProcessTransport: LSPTransport, @unchecked Sendable {
    private let launch: LSPLaunchPolicy.Launch
    private let rootURL: URL
    private let environment: [String: String]
    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let stderrPipe = Pipe()
    private let tail = StderrTail()
    private let writeQueue = DispatchQueue(label: "turbospark.lsp.write")
    private let lock = NSLock()
    private var closeNotified = false
    private var killed = false

    init(launch: LSPLaunchPolicy.Launch, rootURL: URL, environment: [String: String]) {
        self.launch = launch
        self.rootURL = rootURL
        self.environment = environment
    }

    var stderrText: String { tail.text }

    func start(onData: @escaping @Sendable (Data) -> Void, onClose: @escaping @Sendable () -> Void) throws {
        process.executableURL = URL(fileURLWithPath: launch.executable)
        process.arguments = launch.arguments
        process.environment = environment
        process.currentDirectoryURL = rootURL
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let notifyClose: @Sendable () -> Void = { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let first = !self.closeNotified
            self.closeNotified = true
            self.lock.unlock()
            if first { onClose() }
        }
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                notifyClose()
            } else {
                onData(chunk)
            }
        }
        let tail = self.tail
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty { handle.readabilityHandler = nil } else { tail.append(chunk) }
        }
        process.terminationHandler = { _ in notifyClose() }
        do {
            try process.run()
        } catch {
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil
            throw error
        }
        LSPProcessRegistry.add(process.processIdentifier)
    }

    func write(_ data: Data) throws {
        // Pipe writes can block while the server is busy; they run on a
        // private queue so an actor never waits on the pipe. A failed write
        // closes the client through the next EOF.
        lock.lock(); let dead = killed; lock.unlock()
        if dead { throw LSPClientError.notRunning }
        let handle = stdinPipe.fileHandleForWriting
        writeQueue.async {
            // `write(contentsOf:)` throws on EPIPE; the legacy overload would
            // raise an uncatchable Objective-C exception and crash the app.
            try? handle.write(contentsOf: data)
        }
    }

    func kill() {
        lock.lock()
        let already = killed
        killed = true
        lock.unlock()
        guard !already else { return }
        let process = self.process
        let pid = process.processIdentifier
        LSPProcessRegistry.remove(pid)
        try? stdinPipe.fileHandleForWriting.close()
        guard process.isRunning else { return }
        // The ladder sleeps (SIGTERM, then SIGKILL after 2 s), so it stays off
        // the caller's thread.
        DispatchQueue.global(qos: .utility).async { ProcessExecutor.terminateAndReap(process) }
    }
}
