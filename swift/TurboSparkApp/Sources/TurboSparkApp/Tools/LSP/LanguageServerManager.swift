import Foundation

/// What one checked file turned out to be.
enum LSPFileStatus: Equatable, Sendable {
    case diagnostics([LSPDiagnostic])
    case timedOut
    case unavailable(String)
    case skipped(String)
}

struct LSPFileReport: Equatable, Sendable {
    var path: String
    var server: String
    var status: LSPFileStatus
}

struct LSPServerStatus: Equatable, Sendable {
    var name: String
    var state: String
}

struct LSPSymbolSearchResult: Equatable, Sendable {
    var symbols: [(server: String, symbol: LSPWorkspaceSymbol)]
    var notes: [String]

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.notes == rhs.notes && lhs.symbols.count == rhs.symbols.count
            && zip(lhs.symbols, rhs.symbols).allSatisfy { $0.server == $1.server && $0.symbol == $1.symbol }
    }
}

enum LSPManagerError: Error, LocalizedError, Equatable {
    case noServer(String)
    case unavailable(server: String, reason: String)
    case unreadable(String)

    var errorDescription: String? {
        switch self {
        case .noServer(let path): return "No enabled language server covers '\(path)'."
        case .unavailable(let server, let reason): return "The \(server) language server is unavailable: \(reason)"
        case .unreadable(let reason): return reason
        }
    }
}

/// OC-10: the language servers of one project root. One server per row, started
/// lazily on first use, stopped on project close, settings change, idle
/// timeout or app quit. A server that dies or fails to start gets one more
/// try; the second strike marks it unavailable until its row changes.
actor LanguageServerManager {
    typealias TransportFactory = @Sendable (LSPServerConfig, URL) async throws -> any LSPTransport

    static let maximumFileBytes = 1_024 * 1_024
    static let maximumTouchedFiles = 200

    private struct Entry {
        var config: LSPServerConfig
        var client: LSPClient?
        var starting: Task<LSPClient, Error>?
        var strikes = 0
        var lastFailure: String?
        var unavailable: Bool { strikes >= 2 }
    }

    let rootURL: URL
    private let settings: @Sendable () -> [LSPServerConfig]
    private let transportFactory: TransportFactory
    private let timeouts: LSPTimeouts
    private let idleTimeout: TimeInterval
    private var entries: [String: Entry] = [:]
    /// uri to project-relative path, for every document this manager opened.
    private var touched: [String: String] = [:]
    private var generation = 0
    private var idleTask: Task<Void, Never>?

    init(
        rootURL: URL,
        settings: @escaping @Sendable () -> [LSPServerConfig] = { LSPSettings.servers() },
        timeouts: LSPTimeouts = LSPTimeouts(),
        idleTimeout: TimeInterval = 600,
        transportFactory: @escaping TransportFactory
    ) {
        self.rootURL = PathContainment.canonical(rootURL)
        self.settings = settings
        self.timeouts = timeouts
        self.idleTimeout = idleTimeout
        self.transportFactory = transportFactory
    }

    // MARK: Selection and files

    /// The enabled server covering `path`, in settings order. Nil is the
    /// "no server for this language" passthrough.
    func serverConfig(forPath path: String) -> LSPServerConfig? {
        LSPSettings.server(forPath: path, in: settings())
    }

    private func languageID(for config: LSPServerConfig) -> String { config.languageID }

    struct FileContent: Sendable {
        var relativePath: String
        var uri: String
        var text: String
    }

    /// Reads one file for a server. The path must resolve inside the project,
    /// must not be a sensitive credential path, and must be a bounded UTF-8
    /// regular file: a language server is third-party code and gets nothing
    /// the file tools would refuse to read.
    func readFile(_ relativePath: String) throws -> FileContent {
        let cleaned = relativePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw LSPManagerError.unreadable("The file path is empty.") }
        let url: URL
        do { url = try AppToolRegistry.resolveSecurePath(relPath: cleaned, rootURL: rootURL) }
        catch { throw LSPManagerError.unreadable(error.localizedDescription) }
        if ToolRiskClassifier.isSensitivePath(cleaned) || ToolRiskClassifier.isSensitivePath(url.path) {
            throw LSPManagerError.unreadable("'\(cleaned)' is a sensitive path and is never sent to a language server.")
        }
        var info = stat()
        guard lstat(url.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
            throw LSPManagerError.unreadable("'\(cleaned)' is not a regular file in the project.")
        }
        guard Int(info.st_size) <= Self.maximumFileBytes else {
            throw LSPManagerError.unreadable("'\(cleaned)' is over \(Self.maximumFileBytes / 1_024) KiB.")
        }
        guard let data = try? Data(contentsOf: url), let text = String(data: data, encoding: .utf8) else {
            throw LSPManagerError.unreadable("'\(cleaned)' is not readable UTF-8 text.")
        }
        let relative = String(url.path.dropFirst(rootURL.path.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return FileContent(relativePath: relative, uri: url.absoluteString, text: text)
    }

    /// Whether a server-reported URI points inside this project.
    func isInsideProject(uri: String) -> Bool {
        guard let url = URL(string: uri), url.isFileURL else { return false }
        return PathContainment.isContained(url, in: rootURL)
    }

    func relativePath(forURI uri: String) -> String? {
        guard let url = URL(string: uri), url.isFileURL, PathContainment.isContained(url, in: rootURL) else { return nil }
        let path = PathContainment.canonical(url).path
        return String(path.dropFirst(rootURL.path.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    // MARK: Server lifecycle

    private func entry(for config: LSPServerConfig) async -> Entry {
        if let existing = entries[config.id] {
            if existing.config == config { return existing }
            // The row changed: the old process answers to a configuration
            // that no longer exists.
            await stop(entryID: config.id)
        }
        let fresh = Entry(config: config)
        entries[config.id] = fresh
        return fresh
    }

    private func stop(entryID: String) async {
        guard let old = entries.removeValue(forKey: entryID) else { return }
        old.starting?.cancel()
        if let client = old.client { await client.shutdown() }
    }

    /// The live client for `config`, starting it if needed. Concurrent callers
    /// share one start.
    func client(for config: LSPServerConfig) async throws -> LSPClient {
        var current = await entry(for: config)
        if current.unavailable {
            throw LSPManagerError.unavailable(server: config.name, reason: current.lastFailure ?? "it failed twice")
        }
        if let client = current.client {
            if await client.isAlive { return client }
            // It died since the last use: one strike.
            let reason = await client.failure ?? "the server exited"
            current.client = nil
            current.strikes += 1
            current.lastFailure = reason
            entries[config.id] = current
            if current.unavailable {
                throw LSPManagerError.unavailable(server: config.name, reason: reason)
            }
        }
        if let starting = entries[config.id]?.starting {
            return try await starting.value
        }
        let startGeneration = generation
        let task = Task { try await self.performStart(config, generation: startGeneration) }
        entries[config.id]?.starting = task
        return try await task.value
    }

    private func performStart(_ config: LSPServerConfig, generation startedAt: Int) async throws -> LSPClient {
        do {
            let transport = try await transportFactory(config, rootURL)
            let client = LSPClient(transport: transport, rootURL: rootURL, timeouts: timeouts)
            try await client.start()
            guard startedAt == generation, entries[config.id]?.config == config else {
                // Stopped (project closed, setting changed) while starting.
                await client.shutdown()
                throw CancellationError()
            }
            entries[config.id]?.client = client
            entries[config.id]?.starting = nil
            return client
        } catch {
            if entries[config.id]?.config == config {
                entries[config.id]?.starting = nil
                entries[config.id]?.strikes += 1
                entries[config.id]?.lastFailure = error.localizedDescription
            }
            throw error
        }
    }

    func stopAll() async {
        generation += 1
        idleTask?.cancel()
        idleTask = nil
        let ids = Array(entries.keys)
        await withTaskGroup(of: Void.self) { group in
            for id in ids { group.addTask { await self.stop(entryID: id) } }
        }
        touched.removeAll()
    }

    func statuses() async -> [LSPServerStatus] {
        var result: [LSPServerStatus] = []
        for config in settings() where config.enabled {
            guard let entry = entries[config.id] else {
                result.append(LSPServerStatus(name: config.name, state: "not started")); continue
            }
            if entry.unavailable {
                result.append(LSPServerStatus(name: config.name, state: "unavailable: \(entry.lastFailure ?? "failed twice")"))
            } else if let client = entry.client, await client.isAlive {
                result.append(LSPServerStatus(name: config.name, state: "running"))
            } else if entry.starting != nil {
                result.append(LSPServerStatus(name: config.name, state: "starting"))
            } else {
                result.append(LSPServerStatus(name: config.name, state: "not running"))
            }
        }
        return result
    }

    private func noteUse() {
        guard idleTimeout > 0 else { return }
        idleTask?.cancel()
        let timeout = idleTimeout
        idleTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            guard !Task.isCancelled else { return }
            await self?.stopAll()
        }
    }

    // MARK: Diagnostics

    /// The post-edit check: sync every covered file, then wait for each one's
    /// diagnostics within one shared budget. Nil means no enabled server
    /// covers any of `paths`, so the caller adds nothing (CUJ 3).
    func postEdit(paths: [String], budget: TimeInterval) async -> [LSPFileReport]? {
        let covered = paths.compactMap { path -> (String, LSPServerConfig)? in
            serverConfig(forPath: path).map { (path, $0) }
        }
        guard !covered.isEmpty else { return nil }
        noteUse()
        let deadline = ContinuousClock.now + .milliseconds(Int(budget * 1_000))
        let reports = await LSPBudget.run(seconds: budget) {
            await self.checkFiles(covered, deadline: deadline)
        }
        return reports ?? covered.map {
            LSPFileReport(path: $0.0, server: $0.1.name, status: .timedOut)
        }
    }

    /// `project_diagnostics`. With a file: sync it from disk and wait up to
    /// `budget`. Without: everything this manager opened, from the stores.
    func diagnostics(file: String?, budget: TimeInterval) async throws -> (reports: [LSPFileReport], statuses: [LSPServerStatus]) {
        noteUse()
        if let file {
            guard let config = serverConfig(forPath: file) else { throw LSPManagerError.noServer(file) }
            let deadline = ContinuousClock.now + .milliseconds(Int(budget * 1_000))
            let reports = await LSPBudget.run(seconds: budget) {
                await self.checkFiles([(file, config)], deadline: deadline)
            }
            return (reports ?? [LSPFileReport(path: file, server: config.name, status: .timedOut)], await statuses())
        }
        var reports: [LSPFileReport] = []
        for config in settings() where config.enabled {
            guard let live = entries[config.id]?.client, await live.isAlive else { continue }
            for uri in await live.openDocumentURIs {
                guard let path = touched[uri] else { continue }
                if let known = await live.knownDiagnostics(uri: uri) {
                    reports.append(LSPFileReport(path: path, server: config.name, status: .diagnostics(known)))
                }
            }
        }
        return (reports.sorted { $0.path < $1.path }, await statuses())
    }

    private func checkFiles(_ files: [(String, LSPServerConfig)], deadline: ContinuousClock.Instant) async -> [LSPFileReport] {
        struct Pending { var path: String; var config: LSPServerConfig; var client: LSPClient; var uri: String; var version: Int; var baseline: Int }
        var reports: [LSPFileReport] = []
        var waiting: [Pending] = []
        for (path, config) in files.prefix(5) {
            do {
                let content = try readFile(path)
                let client = try await client(for: config)
                if touched.count < Self.maximumTouchedFiles || touched[content.uri] != nil {
                    touched[content.uri] = content.relativePath
                }
                let sync = try await client.sync(uri: content.uri, languageID: languageID(for: config), text: content.text)
                waiting.append(Pending(path: content.relativePath, config: config, client: client,
                                       uri: content.uri, version: sync.version, baseline: sync.baseline))
            } catch let error as LSPManagerError {
                reports.append(LSPFileReport(path: path, server: config.name, status: statusFor(error)))
            } catch {
                reports.append(LSPFileReport(path: path, server: config.name, status: .unavailable(error.localizedDescription)))
            }
        }
        for item in waiting {
            let remaining = max(0, seconds(until: deadline))
            let outcome = await item.client.awaitDiagnostics(
                uri: item.uri, version: item.version, baseline: item.baseline, budget: remaining)
            switch outcome {
            case .fresh(let list): reports.append(LSPFileReport(path: item.path, server: item.config.name, status: .diagnostics(list)))
            case .timedOut: reports.append(LSPFileReport(path: item.path, server: item.config.name, status: .timedOut))
            case .serverGone: reports.append(LSPFileReport(path: item.path, server: item.config.name, status: .unavailable("the server stopped")))
            }
        }
        return reports
    }

    private func statusFor(_ error: LSPManagerError) -> LSPFileStatus {
        switch error {
        case .unreadable(let reason): return .skipped(reason)
        case .unavailable(_, let reason): return .unavailable(reason)
        case .noServer(let path): return .skipped("no server for \(path)")
        }
    }

    private func seconds(until deadline: ContinuousClock.Instant) -> Double {
        let delta = (deadline - ContinuousClock.now).components
        return Double(delta.seconds) + Double(delta.attoseconds) / 1e18
    }

    // MARK: Symbols

    func documentSymbols(file: String) async throws -> (server: String, relativePath: String, symbols: [LSPDocumentSymbol]) {
        guard let config = serverConfig(forPath: file) else { throw LSPManagerError.noServer(file) }
        noteUse()
        let content = try readFile(file)
        let client = try await client(for: config)
        if touched.count < Self.maximumTouchedFiles || touched[content.uri] != nil { touched[content.uri] = content.relativePath }
        _ = try await client.sync(uri: content.uri, languageID: config.languageID, text: content.text)
        return (config.name, content.relativePath, try await client.documentSymbols(uri: content.uri))
    }

    func workspaceSymbols(query: String) async -> LSPSymbolSearchResult {
        noteUse()
        var found: [(server: String, symbol: LSPWorkspaceSymbol)] = []
        var notes: [String] = []
        let configs = settings().filter(\.enabled)
        if configs.isEmpty { notes.append("No language server is enabled.") }
        for config in configs {
            do {
                let client = try await client(for: config)
                for symbol in try await client.workspaceSymbols(query: query) {
                    guard isInsideProject(uri: symbol.uri) else { continue }
                    found.append((config.name, symbol))
                }
            } catch {
                notes.append("\(config.name): \(error.localizedDescription)")
            }
        }
        return LSPSymbolSearchResult(symbols: found, notes: notes)
    }
}

/// One manager per project root, process-wide. Tools and the post-edit step
/// reach their manager here; project close, a settings change and app quit
/// stop them.
actor LanguageServerRegistry {
    static let shared = LanguageServerRegistry()

    private var managers: [String: LanguageServerManager] = [:]

    func manager(for root: URL) -> LanguageServerManager {
        let key = PathContainment.canonical(root).path
        if let existing = managers[key] { return existing }
        let created = LanguageServerManager(rootURL: root, transportFactory: LSPProductionTransport.make)
        managers[key] = created
        return created
    }

    /// Test seam: a manager over a scripted transport.
    func install(_ manager: LanguageServerManager, for root: URL) {
        managers[PathContainment.canonical(root).path] = manager
    }

    func stop(root: URL) async {
        guard let manager = managers.removeValue(forKey: PathContainment.canonical(root).path) else { return }
        await manager.stopAll()
    }

    func stopAll() async {
        let all = Array(managers.values)
        managers.removeAll()
        for manager in all { await manager.stopAll() }
    }

    nonisolated static func killAllProcessesNow() {
        LSPProcessRegistry.killAllNow()
    }
}

/// The real transport: policy-checked launch with a scrubbed environment.
enum LSPProductionTransport {
    static func make(config: LSPServerConfig, root: URL) async throws -> any LSPTransport {
        try make(config: config, root: root, searchPath: await LoginShellPath.resolved())
    }

    static func make(config: LSPServerConfig, root: URL, searchPath: String?) throws -> any LSPTransport {
        let launch = try LSPLaunchPolicy.resolve(config: config, rootURL: root, searchPath: searchPath)
        let environment = LSPLaunchPolicy.environment(executable: launch.executable, searchPath: searchPath)
        return LSPProcessTransport(launch: launch, rootURL: root, environment: environment)
    }
}
