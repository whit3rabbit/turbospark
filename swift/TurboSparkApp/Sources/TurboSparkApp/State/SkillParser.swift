import Foundation

/// Parsing errors for skill files.
public enum SkillParseError: Error, LocalizedError, Sendable {
    case fileNotFound(String)
    case invalidEncoding(String)
    case invalidFrontmatter(String)
    case unreadableContent

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Skill file not found at: \(path)"
        case .invalidEncoding(let path):
            return "Could not read text with UTF-8 encoding from: \(path)"
        case .invalidFrontmatter(let msg):
            return "Invalid skill frontmatter: \(msg)"
        case .unreadableContent:
            return "Skill content could not be read."
        }
    }
}

/// Lightweight, resilient parser for YAML frontmatter in Markdown skill files.
public enum SkillParser {
    /// Parses a raw Markdown skill file from disk.
    public static func parseFile(
        at fileURL: URL,
        scope: SkillScope = .userGlobal,
        agentOrigin: SkillSourceAgent = .turboSpark
    ) throws -> AppSkill {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw SkillParseError.fileNotFound(fileURL.path)
        }

        guard let rawString = try? String(contentsOf: fileURL, encoding: .utf8) else {
            throw SkillParseError.invalidEncoding(fileURL.path)
        }

        let skillDir = fileURL.deletingLastPathComponent()
        var refFiles: [String] = []
        if fileURL.lastPathComponent.uppercased() == "SKILL.MD" {
            if let dirContents = try? FileManager.default.contentsOfDirectory(atPath: skillDir.path) {
                refFiles = dirContents.filter { $0 != "SKILL.md" && $0 != "SKILL.MD" && !$0.hasPrefix(".") }
            }
        }

        return parseContent(
            rawText: rawString,
            sourceURL: fileURL,
            skillDirectoryURL: fileURL.lastPathComponent.uppercased() == "SKILL.MD" ? skillDir : nil,
            scope: scope,
            agentOrigin: agentOrigin,
            referenceFiles: refFiles
        )
    }

    /// Parses Markdown text with optional YAML frontmatter into an AppSkill.
    public static func parseContent(
        rawText: String,
        sourceURL: URL = URL(fileURLWithPath: "/skills/skill.md"),
        skillDirectoryURL: URL? = nil,
        scope: SkillScope = .userGlobal,
        agentOrigin: SkillSourceAgent = .turboSpark,
        referenceFiles: [String] = []
    ) -> AppSkill {
        let (frontmatterText, bodyText) = extractFrontmatterAndBody(from: rawText)
        let manifest: SkillManifest
        if let frontmatterText {
            manifest = parseFrontmatterYAML(frontmatterText)
        } else {
            manifest = SkillManifest()
        }

        return AppSkill(
            manifest: manifest,
            content: bodyText.trimmingCharacters(in: .whitespacesAndNewlines),
            sourceURL: sourceURL,
            skillDirectoryURL: skillDirectoryURL,
            scope: scope,
            agentOrigin: agentOrigin,
            isEnabled: true,
            referenceFiles: referenceFiles
        )
    }

    /// Splits `---` delimited frontmatter from the markdown body.
    public static func extractFrontmatterAndBody(from text: String) -> (frontmatter: String?, body: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("---") else {
            return (nil, text)
        }

        let lines = text.components(separatedBy: .newlines)
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else {
            return (nil, text)
        }

        var closingIndex: Int?
        for i in 1..<lines.count {
            if lines[i].trimmingCharacters(in: .whitespaces) == "---" {
                closingIndex = i
                break
            }
        }

        guard let closing = closingIndex else {
            return (nil, text)
        }

        let frontmatterLines = lines[1..<closing]
        let frontmatter = frontmatterLines.joined(separator: "\n")

        let bodyLines = lines[(closing + 1)...]
        let body = bodyLines.joined(separator: "\n")

        return (frontmatter, body)
    }

    /// Parses YAML frontmatter key-value pairs into a `SkillManifest`.
    public static func parseFrontmatterYAML(_ yamlText: String) -> SkillManifest {
        var name: String?
        var description: String?
        var allowedTools: [String] = []
        var argumentHint: String?
        var arguments: [SkillArgument] = []
        var userInvocable: Bool = true
        var disableModelInvocation: Bool?
        var model: String?
        var context: SkillExecutionContext = .inline
        var agent: String?
        var paths: [String] = []
        var shell: SkillShellType = .bash

        let lines = yamlText.components(separatedBy: .newlines)
        var i = 0

        while i < lines.count {
            let rawLine = lines[i]
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") {
                i += 1
                continue
            }

            guard let colonPos = line.firstIndex(of: ":") else {
                i += 1
                continue
            }

            let key = String(line[..<colonPos]).trimmingCharacters(in: .whitespaces).lowercased()
            let valueRest = String(line[line.index(after: colonPos)...]).trimmingCharacters(in: .whitespaces)

            switch key {
            case "name":
                name = unquote(valueRest)
            case "description":
                if valueRest.isEmpty || valueRest == "|" || valueRest == ">" {
                    // Multiline string
                    var descLines: [String] = []
                    i += 1
                    while i < lines.count {
                        let subLine = lines[i]
                        if subLine.hasPrefix("  ") || subLine.hasPrefix("\t") {
                            descLines.append(subLine.trimmingCharacters(in: .whitespaces))
                            i += 1
                        } else {
                            i -= 1
                            break
                        }
                    }
                    description = descLines.joined(separator: " ")
                } else {
                    description = unquote(valueRest)
                }
            case "allowed-tools", "allowed_tools":
                if valueRest.hasPrefix("[") && valueRest.hasSuffix("]") {
                    allowedTools = parseInlineArray(valueRest)
                } else if valueRest.isEmpty {
                    // Multiline list items starting with -
                    i += 1
                    while i < lines.count {
                        let subLine = lines[i].trimmingCharacters(in: .whitespaces)
                        if subLine.hasPrefix("- ") {
                            let item = String(subLine.dropFirst(2)).trimmingCharacters(in: .whitespaces)
                            allowedTools.append(unquote(item))
                            i += 1
                        } else if subLine.isEmpty {
                            i += 1
                        } else {
                            i -= 1
                            break
                        }
                    }
                } else {
                    allowedTools = [unquote(valueRest)]
                }
            case "argument-hint", "argument_hint":
                argumentHint = unquote(valueRest)
            case "arguments":
                if valueRest.hasPrefix("[") && valueRest.hasSuffix("]") {
                    let items = parseInlineArray(valueRest)
                    arguments = items.map { SkillArgument(name: $0) }
                } else if valueRest.isEmpty {
                    // Multiline list of arguments
                    i += 1
                    var currentArg: SkillArgument?
                    while i < lines.count {
                        let subLine = lines[i].trimmingCharacters(in: .whitespaces)
                        if subLine.hasPrefix("- name:") {
                            if let cur = currentArg { arguments.append(cur) }
                            let argName = unquote(String(subLine.dropFirst(7)).trimmingCharacters(in: .whitespaces))
                            currentArg = SkillArgument(name: argName)
                        } else if subLine.hasPrefix("- ") && !subLine.contains(":") {
                            if let cur = currentArg { arguments.append(cur) }
                            let argName = unquote(String(subLine.dropFirst(2)).trimmingCharacters(in: .whitespaces))
                            currentArg = SkillArgument(name: argName)
                        } else if subLine.hasPrefix("placeholder:") {
                            currentArg?.placeholder = unquote(String(subLine.dropFirst(12)).trimmingCharacters(in: .whitespaces))
                        } else if subLine.hasPrefix("default:") {
                            currentArg?.defaultValue = unquote(String(subLine.dropFirst(8)).trimmingCharacters(in: .whitespaces))
                        } else if subLine.hasPrefix("description:") {
                            currentArg?.description = unquote(String(subLine.dropFirst(12)).trimmingCharacters(in: .whitespaces))
                        } else if subLine.isEmpty {
                            // ignore blank
                        } else if !subLine.hasPrefix("  ") && !subLine.hasPrefix("\t") {
                            i -= 1
                            break
                        }
                        i += 1
                    }
                    if let cur = currentArg { arguments.append(cur) }
                }
            case "user-invocable", "user_invocable":
                userInvocable = (valueRest.lowercased() == "true" || valueRest == "1" || valueRest.lowercased() == "yes")
            case "disable-model-invocation", "disable_model_invocation":
                disableModelInvocation = (valueRest.lowercased() == "true" || valueRest == "1" || valueRest.lowercased() == "yes")
            case "model":
                model = unquote(valueRest)
            case "context":
                let ctxStr = unquote(valueRest).lowercased()
                context = (ctxStr == "fork") ? .fork : .inline
            case "agent":
                agent = unquote(valueRest)
            case "paths":
                if valueRest.hasPrefix("[") && valueRest.hasSuffix("]") {
                    paths = parseInlineArray(valueRest)
                } else if valueRest.isEmpty {
                    i += 1
                    while i < lines.count {
                        let subLine = lines[i].trimmingCharacters(in: .whitespaces)
                        if subLine.hasPrefix("- ") {
                            let item = String(subLine.dropFirst(2)).trimmingCharacters(in: .whitespaces)
                            paths.append(unquote(item))
                            i += 1
                        } else if subLine.isEmpty {
                            i += 1
                        } else {
                            i -= 1
                            break
                        }
                    }
                } else {
                    paths = [unquote(valueRest)]
                }
            case "shell":
                let shStr = unquote(valueRest).lowercased()
                shell = (shStr == "powershell" || shStr == "ps") ? .powershell : .bash
            default:
                break
            }

            i += 1
        }

        return SkillManifest(
            name: name,
            description: description,
            allowedTools: allowedTools,
            argumentHint: argumentHint,
            arguments: arguments,
            userInvocable: userInvocable,
            disableModelInvocation: disableModelInvocation,
            model: model,
            context: context,
            agent: agent,
            paths: paths,
            shell: shell
        )
    }

    /// Strips leading and trailing quotes from YAML string scalar.
    public static func unquote(_ s: String) -> String {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\"")) || (trimmed.hasPrefix("'") && trimmed.hasSuffix("'")) {
            if trimmed.count >= 2 {
                return String(trimmed.dropFirst().dropLast())
            }
        }
        return trimmed
    }

    /// Parses JSON-like or YAML inline bracket array: `["a", "b", "c"]`.
    public static func parseInlineArray(_ raw: String) -> [String] {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("[") && trimmed.hasSuffix("]") else { return [] }
        let inside = String(trimmed.dropFirst().dropLast()).trimmingCharacters(in: .whitespaces)
        if inside.isEmpty { return [] }

        return inside
            .split(separator: ",")
            .map { unquote(String($0).trimmingCharacters(in: .whitespacesAndNewlines)) }
            .filter { !$0.isEmpty }
    }

    /// Generates standard SKILL.md formatted text with YAML frontmatter from an `AppSkill`.
    public static func serializeSkill(_ skill: AppSkill) -> String {
        var lines: [String] = ["---"]
        if let name = skill.manifest.name, !name.isEmpty {
            lines.append("name: \(name)")
        }
        if let desc = skill.manifest.description, !desc.isEmpty {
            lines.append("description: \(desc)")
        }
        if !skill.manifest.allowedTools.isEmpty {
            lines.append("allowed-tools:")
            for tool in skill.manifest.allowedTools {
                lines.append("  - \(tool)")
            }
        }
        if let hint = skill.manifest.argumentHint, !hint.isEmpty {
            lines.append("argument-hint: \(hint)")
        }
        if !skill.manifest.arguments.isEmpty {
            lines.append("arguments:")
            for arg in skill.manifest.arguments {
                lines.append("  - name: \(arg.name)")
                if let p = arg.placeholder { lines.append("    placeholder: \(p)") }
                if let d = arg.defaultValue { lines.append("    default: \(d)") }
                if let desc = arg.description { lines.append("    description: \(desc)") }
            }
        }
        if !skill.manifest.userInvocable {
            lines.append("user-invocable: false")
        }
        if let dmi = skill.manifest.disableModelInvocation {
            lines.append("disable-model-invocation: \(dmi)")
        }
        if let model = skill.manifest.model {
            lines.append("model: \(model)")
        }
        if skill.manifest.context == .fork {
            lines.append("context: fork")
        }
        if let agent = skill.manifest.agent {
            lines.append("agent: \(agent)")
        }
        if !skill.manifest.paths.isEmpty {
            lines.append("paths:")
            for path in skill.manifest.paths {
                lines.append("  - \(path)")
            }
        }
        if skill.manifest.shell == .powershell {
            lines.append("shell: powershell")
        }
        lines.append("---")
        lines.append("")
        lines.append(skill.content)
        lines.append("")
        return lines.joined(separator: "\n")
    }
}
