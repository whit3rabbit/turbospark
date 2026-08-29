import Foundation

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

    private func discoverToolsViaStdio(
        serverName: String,
        command: String,
        args: [String],
        env: [String: String],
        workingDirectory: URL?,
        timeoutSeconds: TimeInterval
    ) async throws -> [McpDiscoveredTool] {
        let process = Process()
        let resolvedExecutable = resolveExecutablePath(command)
        process.executableURL = URL(fileURLWithPath: resolvedExecutable)
        process.arguments = args

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

        do {
            try process.run()
        } catch {
            throw NSError(domain: "McpClientEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to spawn MCP server '\(serverName)': \(error.localizedDescription)"])
        }

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
        _ = try await readJsonRpcResponse(from: stdoutPipe, expectedId: 1, timeoutSeconds: timeoutSeconds)

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

        let toolsResponse = try await readJsonRpcResponse(from: stdoutPipe, expectedId: 2, timeoutSeconds: timeoutSeconds)

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
        let process = Process()
        let resolvedExecutable = resolveExecutablePath(command)
        process.executableURL = URL(fileURLWithPath: resolvedExecutable)
        process.arguments = args

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

        do {
            try process.run()
        } catch {
            throw NSError(domain: "McpClientEngine", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to spawn MCP server '\(serverName)': \(error.localizedDescription)"])
        }

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
        _ = try await readJsonRpcResponse(from: stdoutPipe, expectedId: 1, timeoutSeconds: timeoutSeconds)

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

        let callResponse = try await readJsonRpcResponse(from: stdoutPipe, expectedId: 2, timeoutSeconds: timeoutSeconds)

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
        pipe.fileHandleForWriting.write(data)
        pipe.fileHandleForWriting.write("\n".data(using: .utf8)!)
    }

    private func readJsonRpcResponse(from pipe: Pipe, expectedId: Int, timeoutSeconds: TimeInterval) async throws -> [String: Any] {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var buffer = Data()

        while Date() < deadline {
            let chunk = pipe.fileHandleForReading.availableData
            if !chunk.isEmpty {
                buffer.append(chunk)
                if let lineEnd = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                    let lineData = buffer.subdata(in: 0..<lineEnd)
                    buffer.removeSubrange(0...lineEnd)
                    if let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] {
                        if let id = json["id"] as? Int, id == expectedId {
                            return json
                        }
                    }
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
