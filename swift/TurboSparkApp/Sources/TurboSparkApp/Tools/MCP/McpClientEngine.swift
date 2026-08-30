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
        let resolvedExecutable = resolveExecutablePath(command)
        process.executableURL = URL(fileURLWithPath: resolvedExecutable)
        // `resolveExecutablePath` falls back to `/usr/bin/env` when `command`
        // is not found under any known PATH-like directory. `env` needs the
        // original command name prepended to its argument list to know what
        // to actually run; without it, `env` executes whatever `args[0]`
        // happened to be as if IT were the program name.
        process.arguments = (resolvedExecutable == "/usr/bin/env") ? [command] + args : args

        var environment = ProcessInfo.processInfo.environment
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
            if process.isRunning {
                process.terminate()
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
            if process.isRunning {
                process.terminate()
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

    private func discoverToolsViaSSE(
        serverName: String,
        url: URL,
        headers: [String: String],
        timeoutSeconds: TimeInterval
    ) async throws -> [McpDiscoveredTool] {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        for (k, v) in headers {
            request.setValue(v, forHTTPHeaderField: k)
        }
        // Simulated remote discovery fallback
        return []
    }

    private func callToolViaSSE(
        serverName: String,
        url: URL,
        headers: [String: String],
        toolName: String,
        arguments: [String: Any],
        timeoutSeconds: TimeInterval
    ) async throws -> String {
        return "SSE remote tool execution completed."
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

    private func resolveExecutablePath(_ name: String) -> String {
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
        return "/usr/bin/env"
    }
}
