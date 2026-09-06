import Foundation

/// Parsing and matching for persisted MCP permission rules, stored on
/// `AppProjectPermissions.mcpAllowRules` / `mcpDenyRules`.
///
/// Rule syntax follows the reference implementation: `mcp__server` matches
/// every tool on that server, `mcp__server__tool` matches one tool exactly,
/// and `mcp__server__*` is accepted as an explicit wildcard spelling of the
/// server-level form. Tool names may THEMSELVES contain `__`, so everything
/// after the second separator is the tool name -- the same parse
/// `AppToolRegistry.executeMcpCall` applies to an incoming call, which is
/// what keeps a rule written for `server__a__b` matching the tool the
/// executor would actually dial.
public enum McpPermissionRule {
    /// Splits `rule` into its target, or nil when it is not an MCP rule at
    /// all (no `mcp__` prefix) or names no server. Comparison is
    /// case-insensitive end to end, matching both resolvers.
    public static func target(of rule: String) -> (server: String, tool: String?)? {
        let trimmed = rule.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.lowercased().hasPrefix("mcp__") else { return nil }
        let body = String(trimmed.dropFirst("mcp__".count))
        let parts = body.components(separatedBy: "__")
        guard let server = parts.first, !server.isEmpty else { return nil }
        guard parts.count > 1 else { return (normalized(server), nil) }
        let tool = parts[1...].joined(separator: "__")
        if tool.isEmpty || tool == "*" {
            return (normalized(server), nil)
        }
        return (normalized(server), normalized(tool))
    }

    /// Whether `rule` covers a call to `toolName` on `serverName`. A
    /// server-level rule covers every tool; a tool-level rule is exact.
    public static func matches(_ rule: String, serverName: String, toolName: String?) -> Bool {
        guard let target = target(of: rule) else { return false }
        guard McpServerConfig.normalizedName(serverName) == target.server else { return false }
        guard let expected = target.tool else { return true }
        guard let actual = toolName else { return false }
        return McpServerConfig.normalizedName(actual) == expected
    }

    /// The server/tool pair a tool CALL addresses, in whichever spelling
    /// the model used: a dynamic `mcp__server__tool` name (the tool name
    /// may itself contain `__`) or the static `call_mcp_tool` form with its
    /// `server` / `toolName` arguments. Nil when the call addresses
    /// nothing MCP-shaped.
    ///
    /// Shared by the permission engine, the approval card, and rule
    /// writing, so a rule is always written against the same parse the
    /// engine will match it with.
    public static func targetOfCall(name: String, arguments: [String: String]) -> (server: String, tool: String?)? {
        if name.lowercased().hasPrefix("mcp__") {
            let parts = name.components(separatedBy: "__")
            guard parts.count >= 3 else { return nil }
            return (parts[1], parts[2...].joined(separator: "__"))
        }
        guard let server = arguments["server"] ?? arguments["server_name"] else { return nil }
        let tool = arguments["toolName"] ?? arguments["tool"] ?? arguments["name"]
        return (server, tool)
    }

    private static func normalized(_ value: String) -> String {
        McpServerConfig.normalizedName(value)
    }
}

extension AppProjectPermissions {
    /// Whether any persisted DENY rule covers this call.
    public func mcpDenyMatches(serverName: String, toolName: String?) -> Bool {
        mcpDenyRules.contains { McpPermissionRule.matches($0, serverName: serverName, toolName: toolName) }
    }

    /// Whether any persisted ALLOW rule covers this call.
    public func mcpAllowMatches(serverName: String, toolName: String?) -> Bool {
        mcpAllowRules.contains { McpPermissionRule.matches($0, serverName: serverName, toolName: toolName) }
    }

    /// The canonical spelling stored when a user grants a rule: the full
    /// `mcp__server__tool` or `mcp__server` form. Deduplicated by target --
    /// a server-level rule makes any tool-level rule for the same server
    /// redundant.
    public func addingMcpRule(serverName: String, toolName: String?, allow: Bool) -> AppProjectPermissions {
        var copy = self
        let rule = "mcp__\(serverName)" + (toolName.map { "__\($0)" } ?? "")
        var list = allow ? copy.mcpAllowRules : copy.mcpDenyRules
        guard !list.contains(where: { McpPermissionRule.matches($0, serverName: serverName, toolName: toolName) }) else {
            return copy
        }
        list.append(rule)
        if allow {
            copy.mcpAllowRules = list
        } else {
            copy.mcpDenyRules = list
        }
        return copy
    }
}
