import Foundation

/// Dynamic MCP tool advertisement: one entry per discovered tool of every
/// visible enabled server, named `mcp__<server>__<tool>`.
///
/// The turn path has no `tools` array -- the model reads its vocabulary from
/// the system prompt's tool lines and emits `<tool_call>` blocks -- so the
/// `OpenAITool` values built here exist to carry the advertised NAME and a
/// DESCRIPTION that includes the argument names and types. The parameter
/// schema itself is summarized rather than passed through verbatim, because
/// the typed `JSONSchema` model cannot round-trip every keyword a server may
/// send and prose is the only channel the model actually reads.
///
/// Tools come from `McpToolCatalogCache`, which is refreshed in the
/// background on project selection and server edits; a turn never waits on
/// discovery. A tool advertised but since removed from the server fails the
/// call with the server's own "unknown tool" error, which is honest.
public enum AppToolCatalogMcp {
    /// Server description text cap in the advertised line, mirroring the
    /// reference implementation's truncation of MCP tool descriptions. The
    /// argument summary is appended AFTER the cap, because it is the part
    /// the model cannot guess.
    public static let descriptionLimit = 160

    /// The servers whose tools are visible to a turn in `project`: enabled
    /// global and project servers, plus the plugin-contributed ones. The
    /// same composition the "Connected MCP Servers" prompt section uses.
    public static func visibleServers(global: [McpServerConfig], project: AppProject?) -> [McpServerConfig] {
        (global + (project?.mcpServers ?? [])).filter { $0.isEnabled }
            + PluginManager.shared.pluginMcpServers(projectURL: project?.rootDirectoryURL)
    }

    /// One `OpenAITool` per discovered tool across `servers`, with denied
    /// rules stripped (C3): a tool matching a project DENY rule, at server
    /// or tool level, never reaches the model at all -- the same eager
    /// removal the reference implementation applies to its tool pool.
    /// Disabled servers are refused here too, not only in `visibleServers`,
    /// so a caller that forgets the filter still cannot advertise a server
    /// the user switched off. Output is sorted by advertised name so the
    /// prompt is stable across turns when discovery order varies.
    public static func toolDefinitions(
        servers: [McpServerConfig],
        permissions: AppProjectPermissions?
    ) -> [OpenAITool] {
        var byName: [String: OpenAITool] = [:]
        for server in servers where server.isEnabled {
            guard let tools = McpToolCatalogCache.shared.tools(forServerName: server.name) else { continue }
            for tool in tools {
                let advertised = advertisedName(server: server.name, tool: tool.name)
                guard byName[advertised] == nil else { continue }
                if let permissions,
                   permissions.mcpDenyMatches(serverName: server.name, toolName: tool.name) {
                    continue
                }
                byName[advertised] = OpenAITool.function(
                    name: advertised,
                    description: advertisedDescription(for: tool),
                    parameters: .emptyObject()
                )
            }
        }
        return byName.keys.sorted().compactMap { byName[$0] }
    }

    /// `mcp__<server>__<tool>`, the spelling both the executor and the
    /// permission engine resolve.
    public static func advertisedName(server: String, tool: String) -> String {
        "mcp__\(server)__\(tool)"
    }

    /// The line the model reads: the server's own description (truncated),
    /// then the argument summary from the tool's input schema.
    public static func advertisedDescription(for tool: McpDiscoveredTool) -> String {
        var description = tool.description.trimmingCharacters(in: .whitespacesAndNewlines)
        if description.count > descriptionLimit {
            description = String(description.prefix(descriptionLimit)) + "..."
        }
        if let arguments = argumentsSummary(for: tool.inputSchemaJSON) {
            description += (description.isEmpty ? "" : " ") + "Arguments: \(arguments)."
        }
        return description
    }

    /// `name (type, required), name (type, optional), ...` from a JSON
    /// Schema object, alphabetically ordered for stability. Nil when the
    /// schema parses to nothing usable -- the line then carries the
    /// description alone.
    public static func argumentsSummary(for schemaJSON: String) -> String? {
        guard let data = schemaJSON.data(using: .utf8),
              let schema = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let properties = schema["properties"] as? [String: Any],
              !properties.isEmpty else {
            return nil
        }
        let required = Set((schema["required"] as? [String]) ?? [])
        let pieces = properties.keys.sorted().map { name -> String in
            let type = typeSummary(properties[name])
            let requiredPart = required.contains(name) ? "" : ", optional"
            if type.isEmpty {
                return requiredPart.isEmpty ? name : "\(name)\(requiredPart)"
            }
            return "\(name): \(type)\(requiredPart)"
        }
        return pieces.isEmpty ? nil : pieces.joined(separator: ", ")
    }

    private static func typeSummary(_ value: Any?) -> String {
        guard let dict = value as? [String: Any] else { return "" }
        if let type = dict["type"] as? String { return type }
        // A property typed only by union keywords (anyOf / oneOf / allOf)
        // still deserves a name in the summary; "any" says a type exists.
        if dict["anyOf"] != nil || dict["oneOf"] != nil || dict["allOf"] != nil {
            return "any"
        }
        return ""
    }
}
