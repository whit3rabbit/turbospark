import Foundation

/// Thread-safe newline-delimited buffer fed from a `readabilityHandler`
/// callback, which fires on an arbitrary GCD thread rather than the actor
/// that owns the transport.
private final class LineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()
    private var completeLines: [Data] = []

    /// Appends a chunk and splits out every complete (`\n`-terminated) line
    /// currently available, not just the first one -- a single `availableData`
    /// read can contain more than one JSON-RPC message when the server
    /// batches its writes.
    func append(_ chunk: Data) {
        lock.lock(); defer { lock.unlock() }
        pending.append(chunk)
        while let idx = pending.firstIndex(of: UInt8(ascii: "\n")) {
            completeLines.append(pending.subdata(in: 0..<idx))
            pending.removeSubrange(0...idx)
        }
    }

    func popLine() -> Data? {
        lock.lock(); defer { lock.unlock() }
        guard !completeLines.isEmpty else { return nil }
        return completeLines.removeFirst()
    }
}

/// JSON-RPC 2.0 Client and Process Coordinator for Model Context Protocol servers.
public actor McpClientEngine {
    public static let shared = McpClientEngine()

    public init() {}

    /// Discovers available tools by spawning the server, completing the MCP handshake, and querying `tools/list`.
    public func discoverTools(for config: McpServerConfig, workingDirectory: URL? = nil, timeoutSeconds: TimeInterval = 10.0) async throws -> [McpDiscoveredTool] {
        switch config.transport {
        case .stdio(let command, let args, let env):
            return try await discoverToolsViaStdio(
                serverName: config.name,
                command: command,
                args: args,
                env: env,
                workingDirectory: workingDirectory,
                timeoutSeconds: timeoutSeconds
            )
        case .sse(let url, let headers):
            return try await discoverToolsViaSSE(
                serverName: config.name,
                url: url,
                headers: headers,
                timeoutSeconds: timeoutSeconds
            )
        }
    }

    /// Calls a specific tool on the given MCP server.
    public func callTool(
        config: McpServerConfig,
        toolName: String,
        arguments: [String: Any],
        workingDirectory: URL? = nil,
        timeoutSeconds: TimeInterval = 30.0
    ) async throws -> String {
        switch config.transport {
        case .stdio(let command, let args, let env):
            return try await callToolViaStdio(
                serverName: config.name,
                command: command,
                args: args,
                env: env,
                toolName: toolName,
                arguments: arguments,
                workingDirectory: workingDirectory,
                timeoutSeconds: timeoutSeconds
            )
        case .sse(let url, let headers):
            return try await callToolViaSSE(
                serverName: config.name,
                url: url,
                headers: headers,
                toolName: toolName,
                arguments: arguments,
                timeoutSeconds: timeoutSeconds
            )
        }
    }

    // MARK: - Stdio Transport

    /// Spawns an MCP stdio server with stdout captured into a line buffer and
    /// stderr drained and discarded, both fed by a `readabilityHandler`
    /// running on a background thread. Draining stderr matters as much as
    /// stdout: a server that writes more than one pipe buffer (~64KB) of
    /// diagnostics with nobody reading it blocks on its own `write()` call,
    /// which hangs every subsequent call to this server, not just discovery.
    private func spawnStdioServer(
        serverName: String,
        command: String,
        args: [String],
        env: [String: String],
        workingDirectory: URL?,
        errorCode: Int
    ) throws -> (process: Process, stdinPipe: Pipe, stdoutBuffer: LineBuffer) {
        let process = Process()
        // An unresolvable command is refused by name here rather than handed
        // to `/usr/bin/env` to look up at spawn time. The old fallback made
        // "this server's command does not exist" indistinguishable from
        // "this server started and then failed", and put the lookup somewhere
        // this process could neither observe nor report.
        guard let resolvedExecutable = resolveExecutablePath(command) else {
            throw NSError(
                domain: "McpClientEngine", code: errorCode,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "MCP server '\(serverName)': command '\(command)' was not found on "
                        + "PATH or in any standard tool directory."
                ])
        }
        process.executableURL = URL(fileURLWithPath: resolvedExecutable)
        process.arguments = args

        // **A MINIMAL ENVIRONMENT, NOT THE PARENT'S.**
        //
        // This used to seed from `ProcessInfo.processInfo.environment`, which
        // hands a third-party server binary every variable this app was
        // launched with: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN`,
        // `AWS_*`, whatever else is in the launching shell. An MCP server
        // needs none of that to speak JSON-RPC on a pipe, and a config file
        // that wants a credential passes it explicitly through `env` below.
        //
        // The four kept are what a normal program needs to run at all: a
        // `PATH` for the subprocesses it shells out to, a `HOME` for its own
        // config, a locale so its output is not mojibake, and a scratch
        // directory.
        let parentEnvironment = ProcessInfo.processInfo.environment
        var environment: [String: String] = [:]
        for key in ["PATH", "HOME", "LANG", "TMPDIR"] {
            if let value = parentEnvironment[key] { environment[key] = value }
        }
        // The server's own declared variables win, including over the four
        // above: a config naming its own PATH means it.
        for (k, v) in env {
            environment[k] = v
        }
        process.environment = environment

        if let workingDirectory {
            process.currentDirectoryURL = workingDirectory
        }

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let stdoutBuffer = LineBuffer()
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
            } else {
                stdoutBuffer.append(chunk)
            }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
            }
        }

        do {
            try process.run()
        } catch {
            throw NSError(domain: "McpClientEngine", code: errorCode, userInfo: [NSLocalizedDescriptionKey: "Failed to spawn MCP server '\(serverName)': \(error.localizedDescription)"])
        }

        return (process, stdinPipe, stdoutBuffer)
    }

    private func discoverToolsViaStdio(
        serverName: String,
        command: String,
        args: [String],
        env: [String: String],
        workingDirectory: URL?,
        timeoutSeconds: TimeInterval
    ) async throws -> [McpDiscoveredTool] {
        let (process, stdinPipe, stdoutBuffer) = try spawnStdioServer(
            serverName: serverName, command: command, args: args, env: env,
            workingDirectory: workingDirectory, errorCode: 1
        )

        defer {
            // **SIGTERM IS A REQUEST** (state#61). A third-party server
            // binary that traps or ignores it simply kept running, one orphan
            // per tool call, holding its own child processes and sockets for
            // the life of the app. `ProcessExecutor.terminateAndReap`
            // escalates to SIGKILL after 2 s and is what every other spawn
            // site here already uses.
            if process.isRunning {
                ProcessExecutor.terminateAndReap(process)
            }
        }

        // Step 1: Send `initialize`
        let initRequest: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [
                    "tools": ["listChanged": true]
                ],
                "clientInfo": [
                    "name": "TurboSpark",
                    "version": "1.0.0"
                ]
            ]
        ]
        try sendJsonRpc(initRequest, to: stdinPipe)

        // Read response
        _ = try await readJsonRpcResponse(from: stdoutBuffer, expectedId: 1, timeoutSeconds: timeoutSeconds)

        // Step 2: Send `notifications/initialized`
        let initializedNotification: [String: Any] = [
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": [:] as [String: Any]
        ]
        try sendJsonRpc(initializedNotification, to: stdinPipe)

        // Step 3: Send `tools/list`
        let toolsRequest: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": [:] as [String: Any]
        ]
        try sendJsonRpc(toolsRequest, to: stdinPipe)

        let toolsResponse = try await readJsonRpcResponse(from: stdoutBuffer, expectedId: 2, timeoutSeconds: timeoutSeconds)

        guard let result = toolsResponse["result"] as? [String: Any],
              let toolsArray = result["tools"] as? [[String: Any]] else {
            return []
        }

        var discovered: [McpDiscoveredTool] = []
        for rawTool in toolsArray {
            guard let name = rawTool["name"] as? String else { continue }
            let desc = (rawTool["description"] as? String) ?? "Tool provided by MCP server '\(serverName)'"
            var schemaJSON = "{}"
            if let schemaDict = rawTool["inputSchema"] as? [String: Any],
               let schemaData = try? JSONSerialization.data(withJSONObject: schemaDict, options: []),
               let schemaStr = String(data: schemaData, encoding: .utf8) {
                schemaJSON = schemaStr
            }
            discovered.append(McpDiscoveredTool(name: name, description: desc, inputSchemaJSON: schemaJSON, serverName: serverName))
        }

        return discovered
    }

    private func callToolViaStdio(
        serverName: String,
        command: String,
        args: [String],
        env: [String: String],
        toolName: String,
        arguments: [String: Any],
        workingDirectory: URL?,
        timeoutSeconds: TimeInterval
    ) async throws -> String {
        let (process, stdinPipe, stdoutBuffer) = try spawnStdioServer(
            serverName: serverName, command: command, args: args, env: env,
            workingDirectory: workingDirectory, errorCode: 2
        )

        defer {
            // **SIGTERM IS A REQUEST** (state#61). A third-party server
            // binary that traps or ignores it simply kept running, one orphan
            // per tool call, holding its own child processes and sockets for
            // the life of the app. `ProcessExecutor.terminateAndReap`
            // escalates to SIGKILL after 2 s and is what every other spawn
            // site here already uses.
            if process.isRunning {
                ProcessExecutor.terminateAndReap(process)
            }
        }

        // Initialize handshake
        let initRequest: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": ["tools": [:] as [String: Any]],
                "clientInfo": ["name": "TurboSpark", "version": "1.0.0"]
            ]
        ]
        try sendJsonRpc(initRequest, to: stdinPipe)
        _ = try await readJsonRpcResponse(from: stdoutBuffer, expectedId: 1, timeoutSeconds: timeoutSeconds)

        let initializedNotification: [String: Any] = [
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": [:] as [String: Any]
        ]
        try sendJsonRpc(initializedNotification, to: stdinPipe)

        // Call tool
        let callRequest: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": toolName,
                "arguments": arguments
            ]
        ]
        try sendJsonRpc(callRequest, to: stdinPipe)

        let callResponse = try await readJsonRpcResponse(from: stdoutBuffer, expectedId: 2, timeoutSeconds: timeoutSeconds)

        if let errorDict = callResponse["error"] as? [String: Any],
           let errorMsg = errorDict["message"] as? String {
            throw NSError(domain: "McpClientEngine", code: 3, userInfo: [NSLocalizedDescriptionKey: "MCP Tool error: \(errorMsg)"])
        }

        guard let result = callResponse["result"] as? [String: Any] else {
            return "Tool executed successfully."
        }

        if let isError = result["isError"] as? Bool, isError {
            if let contentList = result["content"] as? [[String: Any]] {
                let text = contentList.compactMap { $0["text"] as? String }.joined(separator: "\n")
                throw NSError(domain: "McpClientEngine", code: 4, userInfo: [NSLocalizedDescriptionKey: text.isEmpty ? "MCP Tool reported error." : text])
            }
            throw NSError(domain: "McpClientEngine", code: 4, userInfo: [NSLocalizedDescriptionKey: "MCP Tool reported failure."])
        }

        if let contentList = result["content"] as? [[String: Any]] {
            let textPieces = contentList.compactMap { $0["text"] as? String }
            if !textPieces.isEmpty {
                return textPieces.joined(separator: "\n")
            }
        }

        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted]),
           let str = String(data: data, encoding: .utf8) {
            return str
        }

        return "Executed successfully."
    }

    // MARK: - SSE Transport (Remote)

    // MARK: - SSE transport: NOT IMPLEMENTED, AND IT SAYS SO
    //
    // Both arms below used to fabricate a result. `discoverToolsViaSSE` built
    // a `URLRequest`, never sent it, and returned an empty list that reads as
    // "this server publishes no tools". `callToolViaSSE` returned the literal
    // string "SSE remote tool execution completed." for every call, so an SSE
    // server config reported every tool as having succeeded: the model was
    // told an external action happened, the transcript showed a green result,
    // and nothing had left the process.
    //
    // That is the failure `AppToolRegistry.execute`'s default case was fixed
    // for (T5) and it was still live here. Throwing is not a smaller feature
    // than lying, it is the only honest state until the transport is written.

    private func discoverToolsViaSSE(
        serverName: String,
        url: URL,
        headers: [String: String],
        timeoutSeconds: TimeInterval
    ) async throws -> [McpDiscoveredTool] {
        throw NSError(
            domain: "McpClientEngine", code: 6,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "MCP server '\(serverName)' uses the SSE transport, which this client "
                    + "does not implement. Configure it as a stdio server, or remove it."
            ])
    }

    private func callToolViaSSE(
        serverName: String,
        url: URL,
        headers: [String: String],
        toolName: String,
        arguments: [String: Any],
        timeoutSeconds: TimeInterval
    ) async throws -> String {
        throw NSError(
            domain: "McpClientEngine", code: 6,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Cannot call '\(toolName)': MCP server '\(serverName)' uses the SSE "
                    + "transport, which this client does not implement. The call was NOT "
                    + "executed."
            ])
    }

    // MARK: - JSON-RPC Utilities

    private func sendJsonRpc(_ dict: [String: Any], to pipe: Pipe) throws {
        let data = try JSONSerialization.data(withJSONObject: dict, options: [])
        // The legacy `FileHandle.write(Data)` overload raises an uncatchable
        // Objective-C exception (not a Swift error) on a write failure such
        // as EPIPE when the child has already died -- there is no `try`
        // available to catch it, so it crashes the app. `write(contentsOf:)`
        // is the throwing equivalent and lets a dead server surface as a
        // normal Swift error instead.
        try pipe.fileHandleForWriting.write(contentsOf: data)
        try pipe.fileHandleForWriting.write(contentsOf: Data("\n".utf8))
    }

    /// Polls a `LineBuffer` fed by a background `readabilityHandler` rather
    /// than reading the pipe directly. Reading the pipe inline (the previous
    /// `pipe.fileHandleForReading.availableData`) blocks until the child
    /// writes, so the deadline check above it can only fire BETWEEN reads --
    /// if the child never writes again, the deadline never gets re-evaluated
    /// and the timeout never fires. Polling a buffer that something else is
    /// filling concurrently keeps this loop's own tick honoring the deadline
    /// regardless of the child's I/O timing.
    private func readJsonRpcResponse(from buffer: LineBuffer, expectedId: Int, timeoutSeconds: TimeInterval) async throws -> [String: Any] {
        let deadline = Date().addingTimeInterval(timeoutSeconds)

        while Date() < deadline {
            // Drain every line already buffered before waiting again: a
            // single chunk can contain more than one JSON-RPC message when
            // the server batches its writes, and looking at only the first
            // one strands the rest until the next chunk arrives, which may
            // never come.
            while let lineData = buffer.popLine() {
                if let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any],
                   let id = json["id"] as? Int, id == expectedId {
                    return json
                }
            }
            try await Task.sleep(nanoseconds: 30_000_000) // 30ms
        }

        throw NSError(domain: "McpClientEngine", code: 5, userInfo: [NSLocalizedDescriptionKey: "Timed out waiting for MCP response (id: \(expectedId))."])
    }

    /// The absolute path of `name`, or nil when it resolves nowhere.
    private func resolveExecutablePath(_ name: String) -> String? {
        if name.hasPrefix("/") || name.hasPrefix("./") || name.hasPrefix("../") {
            return name
        }
        let commonPaths = [
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/usr/bin",
            "/bin",
            "\(FileManager.default.homeDirectoryForCurrentUser.path)/.cargo/bin",
            "\(FileManager.default.homeDirectoryForCurrentUser.path)/.nvm/versions/node/current/bin"
        ]
        for path in commonPaths {
            let candidate = "\(path)/\(name)"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }

        // Then the real PATH, which is what the `/usr/bin/env` fallback was
        // reaching for and the reason that fallback must not simply be
        // deleted: an nvm, asdf or pyenv install puts `node` and `python`
        // somewhere no hard-coded list can predict.
        //
        // Searching it HERE rather than delegating to `env` is the change.
        // The old code returned `/usr/bin/env` for anything it could not
        // find, so a missing or misspelled command silently became a PATH
        // lookup at spawn time whose result nothing in this process could
        // see or report. Resolving it ourselves means an unresolvable
        // command is an error naming itself, and a resolvable one is spawned
        // by absolute path.
        let searchPath = ProcessInfo.processInfo.environment["PATH"] ?? ""
        for path in searchPath.components(separatedBy: ":") where !path.isEmpty {
            let candidate = "\(path)/\(name)"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }

        return nil
    }
}
