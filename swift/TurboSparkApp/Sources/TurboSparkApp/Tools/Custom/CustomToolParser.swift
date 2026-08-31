import Foundation

/// Errors that can occur during custom tool parsing.
public enum CustomToolParseError: Error, LocalizedError, Sendable {
    case fileNotFound(String)
    case invalidEncoding(String)
    case invalidFormat(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Tool file not found at: \(path)"
        case .invalidEncoding(let path):
            return "Could not read tool file with UTF-8 encoding from: \(path)"
        case .invalidFormat(let msg):
            return "Invalid custom tool format: \(msg)"
        }
    }
}

/// Parser for loading CustomToolDefinition models from JSON and YAML disk files.
public enum CustomToolParser {
    public static func parse(fileURL: URL, scope: SkillScope) throws -> CustomToolDefinition {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw CustomToolParseError.fileNotFound(fileURL.path)
        }

        guard let data = try? Data(contentsOf: fileURL) else {
            throw CustomToolParseError.invalidEncoding(fileURL.path)
        }

        let ext = fileURL.pathExtension.lowercased()
        if ext == "json" {
            let decoder = JSONDecoder()
            do {
                var tool = try decoder.decode(CustomToolDefinition.self, from: data)
                tool.sourcePath = fileURL.path
                tool.scope = scope
                return tool
            } catch {
                // Fallback to tolerant dictionary parsing if strict decode fails
                if let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    return try parseDictionary(dict, sourceURL: fileURL, scope: scope)
                }
                throw CustomToolParseError.invalidFormat(error.localizedDescription)
            }
        } else {
            guard let str = String(data: data, encoding: .utf8) else {
                throw CustomToolParseError.invalidEncoding(fileURL.path)
            }
            return try parseYAMLContent(str, sourceURL: fileURL, scope: scope)
        }
    }

    /// Tolerant parser for dictionary-based representations.
    public static func parseDictionary(_ dict: [String: Any], sourceURL: URL, scope: SkillScope) throws -> CustomToolDefinition {
        var fallbackName = sourceURL.deletingPathExtension().lastPathComponent
        if fallbackName.uppercased() == "TOOL" {
            fallbackName = sourceURL.deletingLastPathComponent().lastPathComponent
        }

        let name = (dict["name"] as? String) ?? fallbackName
        let displayName = (dict["displayName"] as? String) ?? (dict["display_name"] as? String) ?? name.capitalized
        let description = (dict["description"] as? String) ?? (dict["toolDescription"] as? String) ?? "Custom tool: \(name)"
        let categoryStr = (dict["category"] as? String)?.lowercased() ?? "terminal"
        let category: AppToolCategory
        switch categoryStr {
        case "fileread", "read": category = .fileRead
        case "filewrite", "write": category = .fileWrite
        case "web": category = .web
        case "mcp": category = .mcp
        case "automation": category = .automation
        default: category = .terminal
        }

        var execution = CustomToolExecution()
        if let execDict = dict["execution"] as? [String: Any] {
            let typeStr = (execDict["type"] as? String)?.lowercased() ?? "command"
            let type = CustomToolExecutionType(rawValue: typeStr) ?? .command
            let command = execDict["command"] as? String
            let script = (execDict["scriptContent"] as? String) ?? (execDict["script"] as? String)
            let interpreter = (execDict["scriptInterpreter"] as? String) ?? (execDict["interpreter"] as? String) ?? "/bin/zsh"
            let httpURL = (execDict["httpURL"] as? String) ?? (execDict["url"] as? String)
            let httpMethod = (execDict["httpMethod"] as? String) ?? (execDict["method"] as? String) ?? "POST"
            let httpHeaders = execDict["httpHeaders"] as? [String: String]
            let arguments = execDict["arguments"] as? [String]
            let env = execDict["environment"] as? [String: String]
            let timeout = (execDict["timeoutSeconds"] as? Double) ?? (execDict["timeout"] as? Double) ?? 30.0

            execution = CustomToolExecution(
                type: type,
                command: command,
                scriptContent: script,
                scriptInterpreter: interpreter,
                httpURL: httpURL,
                httpMethod: httpMethod,
                httpHeaders: httpHeaders,
                arguments: arguments,
                environment: env,
                timeoutSeconds: timeout
            )
        } else if let directCmd = dict["command"] as? String {
            execution = CustomToolExecution(type: .command, command: directCmd)
        }

        var schema = JSONSchema.emptyObject()
        if let paramsDict = dict["parameters"] as? [String: Any],
           let paramsData = try? JSONSerialization.data(withJSONObject: paramsDict, options: []),
           let decodedSchema = try? JSONDecoder().decode(JSONSchema.self, from: paramsData) {
            schema = decodedSchema
        }

        let isEnabled = (dict["isEnabled"] as? Bool) ?? (dict["enabled"] as? Bool) ?? true

        return CustomToolDefinition(
            name: name,
            displayName: displayName,
            toolDescription: description,
            category: category,
            scope: scope,
            parameters: schema,
            execution: execution,
            isEnabled: isEnabled,
            sourcePath: sourceURL.path
        )
    }

    private static func parseYAMLContent(_ rawText: String, sourceURL: URL, scope: SkillScope) throws -> CustomToolDefinition {
        var fallbackName = sourceURL.deletingPathExtension().lastPathComponent
        if fallbackName.uppercased() == "TOOL" {
            fallbackName = sourceURL.deletingLastPathComponent().lastPathComponent
        }

        var name = fallbackName
        var description = "Custom tool: \(name)"
        var command: String?

        let lines = rawText.components(separatedBy: "\n")
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("name:") {
                name = trimmed.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
            } else if trimmed.hasPrefix("description:") {
                description = trimmed.dropFirst(12).trimmingCharacters(in: .whitespacesAndNewlines)
            } else if trimmed.hasPrefix("command:") {
                command = trimmed.dropFirst(8).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        return CustomToolDefinition(
            name: name,
            toolDescription: description,
            category: .terminal,
            scope: scope,
            execution: CustomToolExecution(type: .command, command: command ?? "echo 'Ran custom tool \(name)'"),
            sourcePath: sourceURL.path
        )
    }
}
