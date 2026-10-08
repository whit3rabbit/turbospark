import Foundation

/// Two spellings of one argument that disagree.
///
/// A model (or an attacker steering it) can send `path` and `TargetFile` in
/// the same call. When the approval card, the risk classifier and the
/// executor each pick a different winner, the user approves one file and a
/// different one is written. Rejecting the call is the only answer all of
/// them can agree on without having to agree on a precedence order.
struct ToolArgumentConflict: Error, LocalizedError, Equatable {
    let field: String
    let keys: [String]

    var errorDescription: String? {
        "Conflicting values were supplied for '\(field)' (\(keys.joined(separator: " vs "))). "
            + "Send exactly one spelling of each argument."
    }
}

/// THE single place that decides which argument key means what for the
/// file tools and the MCP bridge. The executor, the approval card, the diff
/// formatter and the risk classifier all read through here, so the value
/// that is shown, assessed, and executed is the same value by construction.
enum ToolArgumentResolver {
    static let filePathKeys = [
        "path", "file_path", "filePath", "TargetFile", "AbsolutePath", "file",
    ]
    static let readPathKeys = filePathKeys + ["resource"]
    static let contentKeys = ["content", "CodeContent", "text", "code"]
    static let oldStringKeys = [
        "old_string", "oldString", "target", "oldStr", "TargetContent",
    ]
    static let newStringKeys = [
        "new_string", "newString", "replacement", "newStr", "ReplacementContent", "file_text",
    ]

    /// The value for one logical argument, or a conflict when two present
    /// spellings differ. Identical duplicates are harmless and accepted.
    static func resolve(
        _ arguments: [String: String], keys: [String], field: String
    ) throws -> String? {
        var first: (key: String, value: String)?
        for key in keys {
            guard let value = arguments[key] else { continue }
            if let first {
                if first.value != value {
                    throw ToolArgumentConflict(field: field, keys: [first.key, key])
                }
            } else {
                first = (key, value)
            }
        }
        return first?.value
    }

    /// Every spelling of the file path that is present, for checks that must
    /// consider all of them (sensitive-path gates).
    static func allPathValues(_ arguments: [String: String]) -> [String] {
        readPathKeys.compactMap { arguments[$0] }
    }

    /// Non-throwing resolution of everything the file tools read, so a
    /// SwiftUI body can use it. `conflict` is set when any argument is
    /// ambiguous; the affected field is then nil.
    struct FileArguments {
        var path: String?
        var content: String?
        var oldString: String?
        var newString: String?
        var conflict: ToolArgumentConflict?
    }

    static func fileArguments(_ arguments: [String: String], forRead: Bool = false) -> FileArguments {
        var result = FileArguments()
        func attempt(_ keys: [String], _ field: String) -> String? {
            do {
                return try resolve(arguments, keys: keys, field: field)
            } catch let error as ToolArgumentConflict {
                if result.conflict == nil { result.conflict = error }
                return nil
            } catch {
                return nil
            }
        }
        result.path = attempt(forRead ? readPathKeys : filePathKeys, "path")
        result.content = attempt(contentKeys, "content")
        result.oldString = attempt(oldStringKeys, "old_string")
        result.newString = attempt(newStringKeys, "new_string")
        return result
    }

    /// Text a card or summary shows for the path: the resolved path, a
    /// visible conflict marker (never one of the competing values), or the
    /// caller's placeholder when none was supplied.
    static func displayPath(
        _ arguments: [String: String], forRead: Bool = false, placeholder: String = "file"
    ) -> String {
        let resolved = fileArguments(arguments, forRead: forRead)
        if resolved.conflict != nil { return "(conflicting path arguments, call will be refused)" }
        return resolved.path ?? placeholder
    }

    // MARK: - MCP bridge target

    static let mcpServerKeys = ["server", "server_name", "serverName", "ServerName"]
    /// `name` is deliberately NOT here: it is also a very common argument OF
    /// the target tool (`create_repo {name: ...}`), so it only names the
    /// target when no explicit tool key is present.
    static let mcpToolKeys = ["toolName", "tool_name", "ToolName", "tool"]

    /// The server and tool a `call_mcp_tool` call addresses, shared by the
    /// permission rules, the risk classifier and the executor.
    static func mcpTarget(
        _ arguments: [String: String]
    ) throws -> (server: String?, tool: String?) {
        let server = try resolve(arguments, keys: mcpServerKeys, field: "server")
        let tool = try resolve(arguments, keys: mcpToolKeys, field: "toolName")
            ?? arguments["name"]
        return (server, tool)
    }

    /// Tool names that reach the `call_mcp_tool` handler in `execute`.
    static let mcpBridgeToolNames: Set<String> = ["call_mcp_tool", "callmcptool", "mcp_tool"]
}

/// Builds the `arguments` object of an MCP `tools/call` request.
///
/// The executor works on `[String: String]`, which flattens numbers,
/// booleans, arrays and objects to text. A server that validates against its
/// input schema (zod, ajv) rejects `"5"` where an integer is declared. The
/// flat string stays authoritative: a typed value is used only when its own
/// string projection still equals the string the gate, a hook or a card
/// saw, so an argument edited after validation is sent as the edited text.
enum McpWireArguments {
    static func build(
        strings: [String: String], typed: [String: ToolCallJSONValue]?
    ) -> [String: Any] {
        var wire: [String: Any] = strings
        guard let typed else { return wire }
        for (key, text) in strings {
            guard let original = typed[key],
                  ToolCallDispatchGate.executorArguments(from: [key: original])[key] == text
            else { continue }
            wire[key] = jsonObject(original)
        }
        return wire
    }

    /// Same merge for the deferred `tool_call` bridge, whose arguments were
    /// parsed into Foundation objects rather than `ToolCallJSONValue`.
    static func build(strings: [String: String], original: [String: Any]) -> [String: Any] {
        var wire: [String: Any] = strings
        for (key, text) in strings {
            guard let value = original[key],
                  ToolSearchCatalog.stringArguments(from: [key: value])[key] == text
            else { continue }
            wire[key] = value
        }
        return wire
    }

    static func jsonObject(_ value: ToolCallJSONValue) -> Any {
        switch value {
        case .string(let string): return string
        case .number(let number): return NSDecimalNumber(decimal: number)
        case .boolean(let flag): return flag
        case .null: return NSNull()
        case .array(let items): return items.map(jsonObject)
        case .object(let fields): return fields.mapValues(jsonObject)
        }
    }
}
