import Foundation

/// Parsing errors for agent definition files.
public enum AgentParseError: Error, LocalizedError, Sendable {
    case fileNotFound(String)
    case invalidEncoding(String)
    case invalidFormat(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Agent file not found at: \(path)"
        case .invalidEncoding(let path):
            return "Could not read agent file with UTF-8 encoding from: \(path)"
        case .invalidFormat(let msg):
            return "Invalid agent format: \(msg)"
        }
    }
}

/// Lightweight parser for Markdown (with YAML frontmatter) and JSON agent definition files.
public enum AgentParser {
    /// Parses an agent definition file (.md or .json).
    public static func parseFile(
        at fileURL: URL,
        scope: AppAgentScope = .userGlobal,
        sourceAgent: AgentSourceAgent = .turboSpark
    ) throws -> AppAgentDefinition {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw AgentParseError.fileNotFound(fileURL.path)
        }

        guard let rawString = try? String(contentsOf: fileURL, encoding: .utf8) else {
            throw AgentParseError.invalidEncoding(fileURL.path)
        }

        let ext = fileURL.pathExtension.lowercased()
        if ext == "json" {
            return try parseJSONContent(
                rawString,
                sourceURL: fileURL,
                scope: scope,
                sourceAgent: sourceAgent
            )
        } else {
            return parseMarkdownContent(
                rawString,
                sourceURL: fileURL,
                scope: scope,
                sourceAgent: sourceAgent
            )
        }
    }

    /// Parses Markdown text containing optional YAML frontmatter.
    public static func parseMarkdownContent(
        _ rawText: String,
        sourceURL: URL,
        scope: AppAgentScope = .userGlobal,
        sourceAgent: AgentSourceAgent = .turboSpark
    ) -> AppAgentDefinition {
        let (frontmatterText, bodyText) = extractFrontmatterAndBody(from: rawText)
        let parsedDict = frontmatterText.map(parseYAMLKeyValue) ?? [:]

        var fallbackName = sourceURL.deletingPathExtension().lastPathComponent
        if fallbackName.uppercased() == "AGENT" {
            fallbackName = sourceURL.deletingLastPathComponent().lastPathComponent
        }

        let name = parsedDict["name"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? fallbackName
        let displayName = parsedDict["display_name"] ?? parsedDict["displayName"] ?? parsedDict["title"] ?? name.capitalized
        let description = parsedDict["description"] ?? parsedDict["when_to_use"] ?? parsedDict["whenToUse"] ?? "Autonomous agent: \(name)"
        let model = parsedDict["model"]
        let maxTurns = Int(parsedDict["max_turns"] ?? parsedDict["maxTurns"] ?? "") ?? 5

        let tools = parseListField(parsedDict["tools"])
        let disallowedTools = parseListField(parsedDict["disallowed_tools"] ?? parsedDict["disallowedTools"])

        let prompt = parsedDict["prompt"] ?? bodyText.trimmingCharacters(in: .whitespacesAndNewlines)

        return AppAgentDefinition(
            name: name,
            displayName: displayName,
            agentDescription: description,
            systemPrompt: prompt,
            tools: tools,
            disallowedTools: disallowedTools,
            model: model,
            maxTurns: maxTurns,
            sourceAgent: sourceAgent,
            scope: scope,
            filePath: sourceURL.path,
            isEnabled: true
        )
    }

    /// Parses JSON text into an AppAgentDefinition.
    public static func parseJSONContent(
        _ jsonString: String,
        sourceURL: URL,
        scope: AppAgentScope = .userGlobal,
        sourceAgent: AgentSourceAgent = .turboSpark
    ) throws -> AppAgentDefinition {
        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AgentParseError.invalidFormat("Malformed JSON in agent definition.")
        }

        var fallbackName = sourceURL.deletingPathExtension().lastPathComponent
        if fallbackName.uppercased() == "AGENT" {
            fallbackName = sourceURL.deletingLastPathComponent().lastPathComponent
        }

        let name = (json["name"] as? String) ?? fallbackName
        let displayName = (json["displayName"] as? String) ?? (json["display_name"] as? String) ?? (json["title"] as? String) ?? name.capitalized
        let description = (json["description"] as? String) ?? (json["whenToUse"] as? String) ?? (json["when_to_use"] as? String) ?? "Autonomous agent: \(name)"
        let systemPrompt = (json["prompt"] as? String) ?? (json["systemPrompt"] as? String) ?? (json["system_prompt"] as? String) ?? ""
        let model = json["model"] as? String
        let maxTurns = (json["maxTurns"] as? Int) ?? (json["max_turns"] as? Int) ?? 5

        let tools = (json["tools"] as? [String]) ?? parseListField(json["tools"] as? String)
        let disallowedTools = (json["disallowedTools"] as? [String]) ?? (json["disallowed_tools"] as? [String]) ?? parseListField(json["disallowedTools"] as? String)

        return AppAgentDefinition(
            name: name,
            displayName: displayName,
            agentDescription: description,
            systemPrompt: systemPrompt,
            tools: tools,
            disallowedTools: disallowedTools,
            model: model,
            maxTurns: maxTurns,
            sourceAgent: sourceAgent,
            scope: scope,
            filePath: sourceURL.path,
            isEnabled: true
        )
    }

    // MARK: - Frontmatter Extraction Helpers

    private static func extractFrontmatterAndBody(from text: String) -> (frontmatter: String?, body: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("---") else {
            return (nil, text)
        }

        let lines = text.components(separatedBy: "\n")
        guard lines.first?.trimmingCharacters(in: .whitespacesAndNewlines) == "---" else {
            return (nil, text)
        }

        var frontmatterLines: [String] = []
        var bodyLines: [String] = []
        var inFrontmatter = true
        var foundEnd = false

        for line in lines.dropFirst() {
            if inFrontmatter {
                if line.trimmingCharacters(in: .whitespacesAndNewlines) == "---" {
                    inFrontmatter = false
                    foundEnd = true
                } else {
                    frontmatterLines.append(line)
                }
            } else {
                bodyLines.append(line)
            }
        }

        if !foundEnd {
            return (nil, text)
        }

        return (frontmatterLines.joined(separator: "\n"), bodyLines.joined(separator: "\n"))
    }

    private static func parseYAMLKeyValue(_ yaml: String) -> [String: String] {
        var result: [String: String] = [:]
        let lines = yaml.components(separatedBy: "\n")
        var currentKey: String?
        var currentArrayValues: [String] = []

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }

            if trimmed.hasPrefix("- ") && currentKey != nil {
                let val = String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespacesAndNewlines)
                currentArrayValues.append(stripQuotes(val))
                continue
            }

            if let key = currentKey, !currentArrayValues.isEmpty {
                result[key] = currentArrayValues.joined(separator: ",")
                currentArrayValues = []
                currentKey = nil
            }

            if let colonIndex = line.firstIndex(of: ":") {
                let key = String(line[..<colonIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
                let val = String(line[line.index(after: colonIndex)...]).trimmingCharacters(in: .whitespacesAndNewlines)
                if val.isEmpty {
                    currentKey = key
                    currentArrayValues = []
                } else {
                    result[key] = stripQuotes(val)
                    currentKey = nil
                }
            }
        }

        if let key = currentKey, !currentArrayValues.isEmpty {
            result[key] = currentArrayValues.joined(separator: ",")
        }

        return result
    }

    private static func parseListField(_ raw: String?) -> [String]? {
        guard let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        var cleaned = raw
        if cleaned.hasPrefix("[") && cleaned.hasSuffix("]") {
            cleaned = String(cleaned.dropFirst().dropLast())
        }
        let items = cleaned.components(separatedBy: ",")
            .map { stripQuotes($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            .filter { !$0.isEmpty }
        return items.isEmpty ? nil : items
    }

    private static func stripQuotes(_ s: String) -> String {
        var str = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if (str.hasPrefix("\"") && str.hasSuffix("\"")) || (str.hasPrefix("'") && str.hasSuffix("'")) {
            str = String(str.dropFirst().dropLast())
        }
        return str
    }
}
