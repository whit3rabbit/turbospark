import Foundation

/// The progressive-disclosure tools used to find and inspect dynamic MCP tools.
public enum ToolSearchToolDefinitions {
    public static let toolSearch = OpenAITool.function(
        name: "tool_search",
        description: "Search the currently granted deferred MCP tools by capability, name, server, or parameter. Pass queries as a JSON string array, or one query per line.",
        parameters: .object(
            properties: [
                "queries": .string(description: "A JSON string array such as [\"create issue\", \"send message\"], or newline-separated queries."),
                "limit": .integer(description: "Maximum matches per query, from 1 to 25.", defaultVal: "5")
            ],
            required: ["queries"]
        )
    )

    public static let toolDescribe = OpenAITool.function(
        name: "tool_describe",
        description: "Load the full schemas for named deferred MCP tools. Pass names as a JSON string array, or one name per line.",
        parameters: .object(
            properties: [
                "names": .string(description: "A JSON string array of exact tool names, or newline-separated names.")
            ],
            required: ["names"]
        )
    )

    public static let toolCall = OpenAITool.function(
        name: "tool_call",
        description: "Call one named deferred MCP tool with a JSON object of arguments. The underlying MCP permission and hook gates still apply.",
        parameters: .object(
            properties: [
                "name": .string(description: "The exact deferred MCP tool name returned by tool_search or tool_describe."),
                "arguments": .string(description: "A JSON object containing the MCP tool arguments.", defaultVal: "{}")
            ],
            required: ["name"]
        )
    )

    public static let all: [OpenAITool] = [toolSearch, toolDescribe, toolCall]
}

/// A dynamic tool entry that is visible to the current turn but whose full
/// schema is loaded only after discovery.
public struct DeferredToolDescriptor: Sendable, Equatable {
    public let name: String
    public let serverName: String
    public let toolName: String
    public let description: String
    public let inputSchemaJSON: String

    public init(
        name: String,
        serverName: String,
        toolName: String,
        description: String,
        inputSchemaJSON: String
    ) {
        self.name = name
        self.serverName = serverName
        self.toolName = toolName
        self.description = description
        self.inputSchemaJSON = inputSchemaJSON
    }
}

public enum ToolSearchCatalog {
    public static let defaultLimit = 5
    public static let maxLimit = 25
    public static let listingCharacterLimit = 8_000
    private static let descriptionCharacterLimit = 160

    /// Rebuilds from the live server list and cache on every request. A
    /// session-local catalog would become stale when a server is refreshed or
    /// disabled while a chat is still open.
    public static func descriptors(
        servers: [McpServerConfig], permissions: AppProjectPermissions?
    ) -> [DeferredToolDescriptor] {
        var result: [DeferredToolDescriptor] = []
        var seen = Set<String>()
        for server in servers where server.isEnabled {
            guard let tools = McpToolCatalogCache.shared.tools(forServerName: server.name) else { continue }
            for tool in tools {
                let advertised = AppToolCatalogMcp.advertisedName(
                    server: server.name, tool: tool.name)
                if let permissions,
                   permissions.mcpDenyMatches(serverName: server.name, toolName: tool.name) {
                    continue
                }
                guard seen.insert(advertised.lowercased()).inserted else { continue }
                result.append(DeferredToolDescriptor(
                    name: advertised,
                    serverName: server.name,
                    toolName: tool.name,
                    description: tool.description,
                    inputSchemaJSON: tool.inputSchemaJSON))
            }
        }
        return result.sorted { $0.name < $1.name }
    }

    public static func promptListing(
        descriptors: [DeferredToolDescriptor], contextTokens: Int? = nil
    ) -> String {
        let contextLimit = contextTokens.map { max(600, $0 / 20) } ?? listingCharacterLimit
        let limit = min(listingCharacterLimit, contextLimit)
        var lines = [
            "",
            "## Deferred MCP Tools",
            "Dynamic MCP tools are discoverable through `tool_search`. Use `tool_describe` for the full schema, then `tool_call` with the exact name and a JSON arguments object. Calls that require interactive approval must be made as the direct MCP tool in the main conversation.",
        ]
        for descriptor in descriptors {
            let description = shortDescription(descriptor.description)
            let line = "- `" + descriptor.name + "`: " + description
            let candidate = (lines + [line]).joined(separator: "\n")
            guard candidate.count <= limit else { break }
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }

    public static func search(
        queries: [String], descriptors: [DeferredToolDescriptor], limit: Int = defaultLimit
    ) -> String {
        let boundedLimit = min(max(limit, 1), maxLimit)
        var groups: [[String: Any]] = []
        var summaries: [String: [String: Any]] = [:]

        for rawQuery in queries {
            let query = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
            let queryTerms = tokens(query)
            let answerable = queryTerms.filter { term in
                descriptors.contains { documentTokens(for: $0).contains(term) }
            }
            let rarestTerms: Set<String> = {
                guard let rarestCount = answerable.map({
                    documentCount(for: $0, descriptors: descriptors)
                }).min() else { return [] }
                return Set(answerable.filter {
                    documentCount(for: $0, descriptors: descriptors) == rarestCount
                })
            }()
            let matches = descriptors.filter { descriptor in
                let document = documentTokens(for: descriptor)
                guard !rarestTerms.isDisjoint(with: document) else { return false }
                if answerable.count >= 4 {
                    let matched = Set(answerable).intersection(document).count
                    guard matched * 2 >= answerable.count else { return false }
                }
                return !answerable.isEmpty && (queryTerms.isEmpty || !Set(queryTerms).isDisjoint(with: document))
            }
            .sorted { lhs, rhs in
                let leftScore = score(lhs, queryTerms: queryTerms)
                let rightScore = score(rhs, queryTerms: queryTerms)
                return leftScore == rightScore ? lhs.name < rhs.name : leftScore > rightScore
            }
            .prefix(boundedLimit)

            let names = matches.map(\.name)
            var group: [String: Any] = ["query": query, "matches": names]
            if names.isEmpty {
                group["available_sources"] = Array(Set(descriptors.map(\.serverName))).sorted()
            } else {
                for descriptor in matches {
                    summaries[descriptor.name] = summary(for: descriptor)
                }
            }
            groups.append(group)
        }
        return encode(["results": groups, "tools": summaries])
    }

    public static func describe(
        names: [String], descriptors: [DeferredToolDescriptor]
    ) -> String {
        let byName = Dictionary(uniqueKeysWithValues: descriptors.map { ($0.name.lowercased(), $0) })
        var tools: [String: [String: Any]] = [:]
        var notFound: [String] = []
        for name in names.map({ $0.trimmingCharacters(in: .whitespacesAndNewlines) }).filter({ !$0.isEmpty }) {
            guard let descriptor = byName[name.lowercased()] else {
                notFound.append(name)
                continue
            }
            tools[descriptor.name] = summary(for: descriptor, includeSchema: true)
        }
        return encode(["tools": tools, "not_found": notFound])
    }

    public static func parseList(_ raw: String) -> [String] {
        if let data = raw.data(using: .utf8),
           let values = try? JSONSerialization.jsonObject(with: data) as? [String] {
            return values
        }
        return raw.split(whereSeparator: { $0 == "\n" || $0 == "," })
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    public static func parseJSONObject(_ raw: String) throws -> [String: Any] {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = text.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw NSError(domain: "TurboSparkToolSearch", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Expected 'arguments' to be a JSON object."
            ])
        }
        return object
    }

    public static func stringArguments(from object: [String: Any]) -> [String: String] {
        object.reduce(into: [:]) { result, item in
            if let value = item.value as? String {
                result[item.key] = value
            } else if let data = try? JSONSerialization.data(withJSONObject: item.value, options: [.sortedKeys]),
                      let value = String(data: data, encoding: .utf8) {
                result[item.key] = value
            } else {
                result[item.key] = String(describing: item.value)
            }
        }
    }

    private static func summary(
        for descriptor: DeferredToolDescriptor, includeSchema: Bool = false
    ) -> [String: Any] {
        var value: [String: Any] = [
            "description": descriptor.description,
            "server": descriptor.serverName,
            "tool": descriptor.toolName,
            "required": requiredNames(from: descriptor.inputSchemaJSON)
        ]
        if includeSchema {
            value["parameters"] = (try? JSONSerialization.jsonObject(
                with: Data(descriptor.inputSchemaJSON.utf8))) ?? ["type": "object"]
        }
        return value
    }

    private static func requiredNames(from schema: String) -> [String] {
        guard let data = schema.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [] }
        return (object["required"] as? [String])?.sorted() ?? []
    }

    private static func shortDescription(_ raw: String) -> String {
        let clean = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard clean.count > descriptionCharacterLimit else { return clean.isEmpty ? "No description." : clean }
        return String(clean.prefix(descriptionCharacterLimit)) + "..."
    }

    private static func documentTokens(for descriptor: DeferredToolDescriptor) -> Set<String> {
        Set(tokens([descriptor.name, descriptor.serverName, descriptor.toolName, descriptor.description, descriptor.inputSchemaJSON].joined(separator: " ")))
    }

    private static func tokens(_ value: String) -> [String] {
        value.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map { stem(String($0)) }
            .filter { $0.count > 1 }
    }

    private static func stem(_ value: String) -> String {
        if value.hasSuffix("ies") && value.count > 4 { return String(value.dropLast(3)) + "y" }
        if value.hasSuffix("ing") && value.count > 5 { return String(value.dropLast(3)) }
        if value.hasSuffix("ed") && value.count > 4 { return String(value.dropLast(2)) }
        if value.hasSuffix("s") && value.count > 3 { return String(value.dropLast()) }
        return value
    }

    private static func documentCount(
        for term: String, descriptors: [DeferredToolDescriptor]
    ) -> Int {
        descriptors.reduce(0) { $0 + (documentTokens(for: $1).contains(term) ? 1 : 0) }
    }

    private static func score(
        _ descriptor: DeferredToolDescriptor, queryTerms: [String]
    ) -> Int {
        let nameTokens = Set(tokens("\(descriptor.serverName) \(descriptor.toolName) \(descriptor.name)"))
        let descriptionTokens = Set(tokens(descriptor.description))
        let argumentTokens = Set(tokens(descriptor.inputSchemaJSON))
        return queryTerms.reduce(0) { score, term in
            score + (nameTokens.contains(term) ? 8 : 0)
                + (descriptionTokens.contains(term) ? 3 : 0)
                + (argumentTokens.contains(term) ? 1 : 0)
        }
    }

    private static func encode(_ value: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
              let result = String(data: data, encoding: .utf8) else { return "{}" }
        return result
    }
}

public enum ToolSearchExecutor {
    public static func execute(
        call: AppToolCall, project: AppProject?, chatID: UUID?
    ) async throws -> String {
        let servers = AppToolCatalogMcp.visibleServers(
            global: GlobalMcpFileStore.load().servers, project: project)
        let descriptors = ToolSearchCatalog.descriptors(
            servers: servers, permissions: project?.permissions)
        switch call.name.lowercased() {
        case "tool_search":
            let queries = ToolSearchCatalog.parseList(call.arguments["queries"] ?? "")
            guard !queries.isEmpty else { throw missing("queries") }
            let limit = Int(call.arguments["limit"] ?? "") ?? ToolSearchCatalog.defaultLimit
            return ToolSearchCatalog.search(queries: queries, descriptors: descriptors, limit: limit)
        case "tool_describe":
            let names = ToolSearchCatalog.parseList(call.arguments["names"] ?? "")
            guard !names.isEmpty else { throw missing("names") }
            return ToolSearchCatalog.describe(names: names, descriptors: descriptors)
        case "tool_call":
            guard let name = call.arguments["name"], !name.isEmpty else { throw missing("name") }
            guard descriptors.contains(where: { $0.name.caseInsensitiveCompare(name) == .orderedSame }) else {
                throw NSError(domain: "TurboSparkToolSearch", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "Deferred tool '\(name)' was not found in the current granted MCP catalog."
                ])
            }
            let rawArguments = call.arguments["arguments"] ?? "{}"
            let arguments = try ToolSearchCatalog.parseJSONObject(rawArguments)
            return try await AppToolRegistry.executeDeferredMcpCall(
                name: name, arguments: arguments, project: project, chatID: chatID)
        default:
            throw NSError(domain: "TurboSparkToolSearch", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported Tool Search operation '\(call.name)'."
            ])
        }
    }

    private static func missing(_ name: String) -> NSError {
        NSError(domain: "TurboSparkToolSearch", code: 4, userInfo: [
            NSLocalizedDescriptionKey: "Missing '\(name)' argument."
        ])
    }
}
