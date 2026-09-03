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

        // **BLANK IS ABSENT.** The fallback covered a MISSING key only, so
        // `name: ""` produced an agent keyed on the empty string -- which
        // takes `seenNames` and the effective map's `""` slot, so two such
        // files silently collapse into one and neither is reachable by name.
        // `AppSkill.name` already treats blank as absent; this is the same
        // rule one file over.
        let declaredName = parsedDict["name"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = (declaredName?.isEmpty == false) ? declaredName! : fallbackName
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

    /// Parses the frontmatter's `key: value` pairs, lists and block scalars.
    ///
    /// **BLOCK SCALARS WERE STORED AS THE LITERAL `"|"`.** A value was treated
    /// as a nested one only when it was EMPTY, so `description: |` stored the
    /// pipe as the description and then read every indented continuation line
    /// as its own `key: value` -- including any prose line containing a colon,
    /// and including a `name:` written inside the block, which then replaced
    /// the agent's real name. `SkillParser` next door already handled `|` and
    /// `>`; this is that branch ported over.
    ///
    /// Index-based rather than `for line in lines` because consuming a block
    /// means skipping ahead, which the old loop had no way to express -- which
    /// is why it reparsed the body instead.
    private static func parseYAMLKeyValue(_ yaml: String) -> [String: String] {
        var result: [String: String] = [:]
        let lines = yaml.components(separatedBy: "\n")

        func isIndented(_ line: String) -> Bool {
            line.hasPrefix(" ") || line.hasPrefix("\t")
        }

        var i = 0
        while i < lines.count {
            let raw = lines[i]
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                i += 1
                continue
            }
            guard let colonIndex = trimmed.firstIndex(of: ":") else {
                i += 1
                continue
            }

            let key = String(trimmed[..<colonIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            let val = String(trimmed[trimmed.index(after: colonIndex)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            i += 1

            if !val.isEmpty && val != "|" && val != ">" && val != "|-" && val != ">-" {
                result[key] = stripQuotes(val)
                continue
            }

            // A block scalar (`|`, `>`) or a bare `key:`. Both are followed by
            // indented lines; whether those are a LIST or prose is decided by
            // the first one, and either way they are consumed here rather
            // than falling back into the key loop.
            let isExplicitBlock = val == "|" || val == ">" || val == "|-" || val == ">-"
            var listValues: [String] = []
            var blockLines: [String] = []
            while i < lines.count {
                let sub = lines[i]
                let subTrimmed = sub.trimmingCharacters(in: .whitespaces)
                if subTrimmed.isEmpty {
                    // A blank line inside a block is part of it; after a list
                    // it ends the entry.
                    if isExplicitBlock || !blockLines.isEmpty {
                        blockLines.append("")
                        i += 1
                        continue
                    }
                    break
                }
                if !isExplicitBlock && subTrimmed.hasPrefix("- ") {
                    listValues.append(
                        stripQuotes(
                            String(subTrimmed.dropFirst(2))
                                .trimmingCharacters(in: .whitespacesAndNewlines)))
                    i += 1
                    continue
                }
                guard isIndented(sub) else { break }
                blockLines.append(subTrimmed)
                i += 1
            }

            if !listValues.isEmpty {
                result[key] = listValues.joined(separator: ",")
            } else if !blockLines.isEmpty {
                // Folded (`>`) joins with spaces, literal (`|`) with newlines.
                let separator = (val == ">" || val == ">-") ? " " : "\n"
                result[key] = blockLines.joined(separator: separator)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            } else if isExplicitBlock {
                // An empty block is an empty string, never the marker itself.
                result[key] = ""
            }
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
