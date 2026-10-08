import Foundation

/// Executor for parallel execution of independent tool calls.
public enum BatchToolExecutor {
    public static let maxBatchSize: Int = 25

    public struct BatchItem: Codable, Sendable {
        public var tool: String
        public var parameters: [String: String]

        enum CodingKeys: String, CodingKey {
            case tool
            case parameters
        }

        public init(tool: String, parameters: [String: String] = [:]) {
            self.tool = tool
            self.parameters = parameters
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.tool = try container.decode(String.self, forKey: .tool)
            if let dict = try? container.decode([String: String].self, forKey: .parameters) {
                self.parameters = dict
            } else if let anyDict = try? container.decode([String: AnyCodableValue].self, forKey: .parameters) {
                var stringDict: [String: String] = [:]
                for (k, v) in anyDict {
                    stringDict[k] = v.asString
                }
                self.parameters = stringDict
            } else {
                self.parameters = [:]
            }
        }
    }

    private enum AnyCodableValue: Codable, Sendable {
        case string(String)
        case int(Int)
        case double(Double)
        case bool(Bool)
        /// Nested containers are decoded structurally (not flattened to ""),
        /// then re-encoded as compact JSON text exactly as the non-batch path
        /// flattens container arguments.
        indirect case array([AnyCodableValue])
        indirect case object([String: AnyCodableValue])

        var asString: String {
            switch self {
            case .string(let s): return s
            case .int(let i): return String(i)
            case .double(let d): return String(d)
            case .bool(let b): return String(b)
            case .array, .object:
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
                guard let data = try? encoder.encode(self) else { return "" }
                return String(decoding: data, as: UTF8.self)
            }
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .string(str)
            } else if let intVal = try? container.decode(Int.self) {
                self = .int(intVal)
            } else if let boolVal = try? container.decode(Bool.self) {
                self = .bool(boolVal)
            } else if let doubleVal = try? container.decode(Double.self) {
                self = .double(doubleVal)
            } else if let array = try? container.decode([AnyCodableValue].self) {
                self = .array(array)
            } else if let object = try? container.decode([String: AnyCodableValue].self) {
                self = .object(object)
            } else {
                self = .string("")
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .string(let s): try container.encode(s)
            case .int(let i): try container.encode(i)
            case .double(let d): try container.encode(d)
            case .bool(let b): try container.encode(b)
            case .array(let a): try container.encode(a)
            case .object(let o): try container.encode(o)
            }
        }
    }

    public static func parseItems(from arguments: [String: String]) throws -> [BatchItem] {
        let decoder = JSONDecoder()
        if let raw = arguments["tool_calls"], let data = raw.data(using: .utf8) {
            if let items = try? decoder.decode([BatchItem].self, from: data) {
                return items
            }
        }
        if let raw = arguments["tool_calls_json"], let data = raw.data(using: .utf8) {
            if let items = try? decoder.decode([BatchItem].self, from: data) {
                return items
            }
        }
        throw NSError(
            domain: "TurboSparkTool",
            code: 35,
            userInfo: [NSLocalizedDescriptionKey: "Missing or invalid 'tool_calls' parameter for batch execution."]
        )
    }

    public static func execute(
        arguments: [String: String],
        project: AppProject?,
        chatID: UUID? = nil,
        subagentDepth: Int = 0,
        webToolsEnabled: Bool = true,
        fallbackMode: AppPermissionMode? = nil
    ) async throws -> String {
        let items = try parseItems(from: arguments)
        guard !items.isEmpty else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 36,
                userInfo: [NSLocalizedDescriptionKey: "Empty 'tool_calls' array provided to batch."]
            )
        }
        guard items.count <= maxBatchSize else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 37,
                userInfo: [NSLocalizedDescriptionKey: "Batch size exceeds maximum limit of \(maxBatchSize) calls (received \(items.count))."]
            )
        }

        // Anti-recursion check
        for item in items {
            let lower = item.tool.lowercased()
            if lower == "batch" {
                throw NSError(
                    domain: "TurboSparkTool",
                    code: 38,
                    userInfo: [NSLocalizedDescriptionKey: "Recursive batch execution is not allowed."]
                )
            }
        }

        // Execute items concurrently preserving original ordering
        var results = [(Int, String, AppToolResult)]()
        results.reserveCapacity(items.count)

        let gate = MutationGate()
        await withTaskGroup(of: (Int, String, AppToolResult).self) { group in
            for (idx, item) in items.enumerated() {
                group.addTask {
                    var call = AppToolCall(
                        name: item.tool,
                        arguments: item.parameters,
                        category: AppToolCatalog.category(for: item.tool, projectURL: project?.rootDirectoryURL)
                    )
                    let sessionID = chatID?.uuidString ?? "batch"
                    let projectDirectory = project?.rootDirectoryPath
                    let hookDecision = await AppHookExecutionEngine.shared.evaluatePreToolUse(
                        sessionID: sessionID,
                        toolName: call.name,
                        toolArguments: call.arguments,
                        workingDirectory: projectDirectory,
                        projectBoundHookDirectory: projectDirectory)
                    if let updated = hookDecision.updatedInput {
                        for (key, value) in updated { call.arguments[key] = value }
                    }
                    if hookDecision.preventContinuation || hookDecision.behavior != .allow {
                        let reason = hookDecision.continuationStopReason ?? hookDecision.reason
                            ?? "Nested batch call was not allowed by a PreToolUse hook."
                        return (idx, item.tool, refused(call, reason: reason))
                    }

                    let sessionApproved = await SessionApprovalStore.shared.isApproved(
                        sessionID: sessionID, toolName: call.name, command: call.shellCommand)
                    switch AppToolPermissionEngine.evaluate(
                        call: call, project: project, sessionApproved: sessionApproved,
                        fallbackMode: fallbackMode)
                    {
                    case .deny(let reason):
                        return (idx, item.tool, refused(call, reason: reason))
                    case .ask(_, let reason):
                        let permissionResults = await AppHookExecutionEngine.shared.dispatch(
                            event: .permissionRequest,
                            sessionID: sessionID,
                            toolName: call.name,
                            toolArguments: call.arguments,
                            workingDirectory: projectDirectory,
                            projectBoundHookDirectory: projectDirectory)
                        let verdict = AppHookDecisionAggregator.aggregate(
                            permissionResults, event: .permissionRequest)
                        guard !verdict.preventContinuation,
                              verdict.permissionDecision == .allow else {
                            // A batch has no nested approval surface. Never treat approval
                            // of its outer wrapper as approval of this child call.
                            let refusal = verdict.continuationStopReason
                                ?? verdict.permissionReason
                                ?? "Nested call requires separate approval: \(reason)"
                            return (idx, item.tool, refused(call, reason: refusal))
                        }
                    case .allow:
                        break
                    }

                    // Mutating children run one at a time. Two edit_file calls on
                    // one path otherwise both read the same original and the
                    // second write drops the first edit while both report
                    // SUCCESS. Read-only children stay parallel.
                    let mutates = call.category == .fileWrite || call.category == .terminal
                    if mutates { await gate.acquire() }
                    let res = await AppToolRegistry.execute(
                        call: call,
                        in: project,
                        chatID: chatID,
                        subagentDepth: subagentDepth,
                        webToolsEnabled: webToolsEnabled
                    )
                    if mutates { await gate.release() }
                    var hookResults = await AppHookExecutionEngine.shared.dispatch(
                        event: .postToolUse,
                        sessionID: sessionID,
                        toolName: call.name,
                        toolArguments: call.arguments,
                        toolOutput: res.output,
                        toolDurationSeconds: res.durationSeconds,
                        isError: res.isError,
                        workingDirectory: projectDirectory,
                        projectBoundHookDirectory: projectDirectory)
                    if res.isError {
                        hookResults += await AppHookExecutionEngine.shared.dispatch(
                            event: .postToolUseFailure,
                            sessionID: sessionID,
                            toolName: call.name,
                            toolArguments: call.arguments,
                            toolOutput: res.output,
                            toolDurationSeconds: res.durationSeconds,
                            isError: true,
                            workingDirectory: projectDirectory,
                            projectBoundHookDirectory: projectDirectory)
                    }
                    let verdict = AppHookDecisionAggregator.aggregate(
                        hookResults, event: .postToolUse)
                    var output = res.output
                    if let note = verdict.blockReason ?? verdict.feedbackMessage, !note.isEmpty {
                        output += "\n\n<hook_feedback>\n\(note)\n</hook_feedback>"
                    }
                    if let context = verdict.additionalContext, !context.isEmpty {
                        output += "\n\n<hook_context>\n\(context)\n</hook_context>"
                    }
                    return (idx, item.tool, AppToolResult(
                        callID: res.callID,
                        output: output,
                        isError: res.isError,
                        durationSeconds: res.durationSeconds))
                }
            }

            for await result in group {
                results.append(result)
            }
        }

        results.sort { $0.0 < $1.0 }

        var successCount = 0
        var failCount = 0
        var formattedParts: [String] = []

        for (idx, toolName, res) in results {
            if res.isError {
                failCount += 1
                formattedParts.append("### Call \(idx + 1): `\(toolName)` (FAILED)\n\(res.output)")
            } else {
                successCount += 1
                formattedParts.append("### Call \(idx + 1): `\(toolName)` (SUCCESS)\n\(res.output)")
            }
        }

        let summary = "Batch execution completed: \(successCount) succeeded, \(failCount) failed (Total: \(items.count)).\n\n"
        return summary + formattedParts.joined(separator: "\n\n")
    }

    private static func refused(_ call: AppToolCall, reason: String) -> AppToolResult {
        AppToolResult(
            callID: call.id,
            output: "Error: Nested batch call '\(call.name)' was refused. \(reason)",
            isError: true,
            durationSeconds: 0)
    }
}

/// FIFO mutual exclusion for the mutating children of one batch.
actor MutationGate {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
    }

    func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}
