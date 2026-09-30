import Foundation
import CoreFoundation

public struct AppMcpToolCatalogSnapshot: Sendable, Equatable {
    public let definitions: [OpenAITool]
    public let deferredDescriptors: [DeferredToolDescriptor]

    public init(definitions: [OpenAITool], deferredDescriptors: [DeferredToolDescriptor]) {
        self.definitions = definitions
        self.deferredDescriptors = deferredDescriptors
    }
}

/// Dynamic MCP tool advertisement: one entry per discovered tool of every
/// visible enabled server, named `mcp__<server>__<tool>`.
///
/// The turn path has no `tools` array -- the model reads its vocabulary from
/// the system prompt's tool lines and emits `<tool_call>` blocks. The captured
/// `OpenAITool` values retain the supported input-schema subset, while the
/// deferred descriptor retains the server's raw schema. Unsupported schema
/// keywords are marked so validation refuses them rather than weakening the
/// server's declared constraints.
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
        resolvedServers(global: global, project: project).filter { $0.isEnabled }
    }

    public static func resolvedServers(global: [McpServerConfig], project: AppProject?) -> [McpServerConfig] {
        let inherited = global.map { server in
            var result = server
            result.isEnabled = project?.enabledMcpServers[server.id.uuidString] ?? server.isEnabled
            return result
        }
        var seen = Set<String>()
        return (inherited + (project?.mcpServers ?? [])
            + PluginManager.shared.pluginMcpServers(projectURL: project?.rootDirectoryURL))
            .filter { seen.insert($0.name.lowercased()).inserted }
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
        catalogSnapshot(servers: servers, permissions: permissions).definitions
    }

    /// Captures direct definitions and deferred descriptors from one cache
    /// read, after enabled-server and project deny filters.
    public static func catalogSnapshot(
        servers: [McpServerConfig],
        permissions: AppProjectPermissions?
    ) -> AppMcpToolCatalogSnapshot {
        var byName: [String: OpenAITool] = [:]
        var descriptorsByName: [String: DeferredToolDescriptor] = [:]
        var seenNames = Set<String>()
        for server in servers where server.isEnabled {
            guard let tools = McpToolCatalogCache.shared.tools(forServerName: server.name) else { continue }
            for tool in tools {
                let advertised = advertisedName(server: server.name, tool: tool.name)
                guard seenNames.insert(advertised.lowercased()).inserted else { continue }
                if let permissions,
                   permissions.mcpDenyMatches(serverName: server.name, toolName: tool.name) {
                    continue
                }
                byName[advertised] = OpenAITool.function(
                    name: advertised,
                    description: advertisedDescription(for: tool),
                    parameters: parameterSchema(from: tool.inputSchemaJSON)
                )
                descriptorsByName[advertised] = DeferredToolDescriptor(
                    name: advertised,
                    serverName: server.name,
                    toolName: tool.name,
                    description: tool.description,
                    inputSchemaJSON: tool.inputSchemaJSON
                )
            }
        }
        let names = byName.keys.sorted()
        return AppMcpToolCatalogSnapshot(
            definitions: names.compactMap { byName[$0] },
            deferredDescriptors: names.compactMap { descriptorsByName[$0] })
    }

    /// Converts the JSON Schema subset represented by `JSONSchema`. Any
    /// validation keyword outside that subset is retained as an internal
    /// marker so the call validator refuses the schema instead of weakening it.
    static func parameterSchema(from schemaJSON: String) -> JSONSchema {
        guard let data = schemaJSON.data(using: .utf8),
              let raw = try? JSONSerialization.jsonObject(with: data),
              let schema = raw as? [String: Any] else {
            return JSONSchema(type: "object", validationIssue: "invalid MCP input schema JSON")
        }
        var issue = validationIssue(in: schema)
        let type: String
        if let rawType = schema["type"] {
            guard let stringType = rawType as? String else {
                return JSONSchema(type: "object", validationIssue: issue ?? "union parameter types are unsupported")
            }
            type = stringType
        } else {
            type = "object"
        }
        guard type == "object" else {
            return JSONSchema(
                type: type,
                validationIssue: issue ?? "MCP parameter schema must be an object")
        }
        if schema["items"] != nil || schema["enum"] != nil {
            issue = issue ?? "items and enum are unsupported on object parameters"
        }

        var properties: [String: JSONSchemaProperty] = [:]
        if let rawProperties = schema["properties"] {
            guard let objectProperties = rawProperties as? [String: Any] else {
                return JSONSchema(type: "object", validationIssue: "properties must be an object")
            }
            for name in objectProperties.keys.sorted() {
                guard let rawProperty = objectProperties[name] as? [String: Any] else {
                    properties[name] = JSONSchemaProperty(
                        type: "unsupported", validationIssue: "property schema must be an object")
                    continue
                }
                properties[name] = propertySchema(from: rawProperty)
            }
        }

        let required: [String]?
        if let rawRequired = schema["required"] {
            if let names = rawRequired as? [String] {
                required = names
            } else {
                return JSONSchema(type: "object", validationIssue: "required must be a string array")
            }
        } else {
            required = nil
        }

        let additionalProperties: Bool?
        if let rawAdditional = schema["additionalProperties"] {
            if let value = jsonBoolean(rawAdditional) {
                additionalProperties = value
            } else {
                return JSONSchema(
                    type: "object",
                    properties: properties,
                    required: required,
                    validationIssue: "schema-valued additionalProperties is unsupported")
            }
        } else {
            additionalProperties = nil
        }

        return JSONSchema(
            type: "object",
            description: schema["description"] as? String,
            properties: properties,
            required: required,
            additionalProperties: additionalProperties,
            validationIssue: issue)
    }

    private static let supportedKeywords: Set<String> = [
        "type", "description", "title", "properties", "required", "items", "enum",
        "additionalProperties", "default", "examples", "example", "$schema", "$comment",
        "deprecated", "readOnly", "writeOnly"
    ]

    private static func validationIssue(in schema: [String: Any]) -> String? {
        let unsupported = schema.keys.filter { !supportedKeywords.contains($0) }.sorted()
        guard let first = unsupported.first else { return nil }
        return "unsupported JSON Schema keyword \(first)"
    }

    private static func propertySchema(from schema: [String: Any]) -> JSONSchemaProperty {
        var issue = validationIssue(in: schema)
        guard let type = schema["type"] as? String else {
            return JSONSchemaProperty(
                type: "unsupported", validationIssue: issue ?? "property schema has no supported type")
        }

        var properties: [String: JSONSchemaProperty]?
        if let rawProperties = schema["properties"] {
            if let objectProperties = rawProperties as? [String: Any] {
                properties = [:]
                for name in objectProperties.keys.sorted() {
                    if let rawProperty = objectProperties[name] as? [String: Any] {
                        properties?[name] = propertySchema(from: rawProperty)
                    } else {
                        properties?[name] = JSONSchemaProperty(
                            type: "unsupported", validationIssue: "property schema must be an object")
                    }
                }
            } else {
                issue = issue ?? "properties must be an object"
            }
        }

        var items: JSONSchemaProperty?
        if let rawItems = schema["items"] {
            if let itemSchema = rawItems as? [String: Any] {
                items = propertySchema(from: itemSchema)
            } else {
                issue = issue ?? "items must be an object schema"
            }
        }

        let required: [String]?
        if let rawRequired = schema["required"] {
            if let names = rawRequired as? [String] {
                required = names
            } else {
                required = nil
                issue = issue ?? "required must be a string array"
            }
        } else {
            required = nil
        }

        let enumValues: [String]?
        if let rawEnum = schema["enum"] {
            if let strings = rawEnum as? [String] {
                enumValues = strings
            } else {
                enumValues = nil
                issue = issue ?? "only string enums are represented"
            }
        } else {
            enumValues = nil
        }

        let additionalProperties: Bool?
        if let rawAdditional = schema["additionalProperties"] {
            if let value = jsonBoolean(rawAdditional) {
                additionalProperties = value
            } else {
                additionalProperties = nil
                issue = issue ?? "schema-valued additionalProperties is unsupported"
            }
        } else {
            additionalProperties = nil
        }

        return JSONSchemaProperty(
            type: type,
            description: schema["description"] as? String,
            enumValues: enumValues,
            items: items,
            properties: properties,
            required: required,
            defaultVal: schema["default"] as? String,
            additionalProperties: additionalProperties,
            validationIssue: issue)
    }

    private static func jsonBoolean(_ value: Any) -> Bool? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) == CFBooleanGetTypeID() else {
            return nil
        }
        return number.boolValue
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
