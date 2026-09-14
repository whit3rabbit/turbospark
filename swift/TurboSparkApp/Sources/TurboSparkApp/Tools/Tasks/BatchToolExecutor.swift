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

        var asString: String {
            switch self {
            case .string(let s): return s
            case .int(let i): return String(i)
            case .double(let d): return String(d)
            case .bool(let b): return String(b)
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
        webToolsEnabled: Bool = true
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

        await withTaskGroup(of: (Int, String, AppToolResult).self) { group in
            for (idx, item) in items.enumerated() {
                group.addTask {
                    let call = AppToolCall(
                        name: item.tool,
                        arguments: item.parameters,
                        category: AppToolCatalog.category(for: item.tool, projectURL: project?.rootDirectoryURL)
                    )
                    let res = await AppToolRegistry.execute(
                        call: call,
                        in: project,
                        chatID: chatID,
                        subagentDepth: subagentDepth,
                        webToolsEnabled: webToolsEnabled
                    )
                    return (idx, item.tool, res)
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
}
