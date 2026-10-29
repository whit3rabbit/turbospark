import Foundation

/// One tool as the profile editor and the composer popover list it.
public struct ToolProfileEntry: Identifiable, Equatable, Sendable {
    /// The canonical key a profile stores (`AppToolCatalog.profileKey(for:)`).
    public let id: String
    public let name: String
    public let summary: String
}

/// A titled run of tools. `title` is a localization key, not display text.
public struct ToolProfileGroup: Identifiable, Equatable, Sendable {
    public let title: String
    public let entries: [ToolProfileEntry]
    public var id: String { title }
}

extension AppToolCatalog {
    /// Tools the harness needs whatever the profile says. A large result is
    /// replaced by a reference the model resolves with `recall_tool_output`, so
    /// removing it would strand every elided result.
    static let profileAlwaysOnKeys: Set<String> = ["recall_tool_output"]

    /// The progressive-disclosure front for MCP servers. They are named like
    /// built-ins but exist only to reach MCP tools, so they follow that switch.
    private static let profileMcpFrontKeys: Set<String> = [
        "tool_search", "tool_describe", "tool_call",
    ]

    /// The key a profile stores for a tool name.
    ///
    /// The file handlers collapse to `read`, `write` and `edit` through the same
    /// alias sets `execute` dispatches on, so a stored profile survives the
    /// advertised names changing (`FileRead` to `Read`) and a built-in profile
    /// can say `read` and mean whichever spelling is offered.
    public static func profileKey(for toolName: String) -> String {
        let name = toolName.lowercased()
        if AppToolRegistry.readFileAliases.contains(name) { return "read" }
        if AppToolRegistry.writeFileAliases.contains(name) { return "write" }
        if AppToolRegistry.editFileAliases.contains(name) { return "edit" }
        return name
    }

    /// Whether a definition belongs to the MCP-and-custom switch rather than
    /// to the per-name allowlist. A custom tool cannot take a built-in name
    /// (`CustomToolManager` rejects collisions), so "not built in" is exactly
    /// "custom".
    static func isMcpOrCustomTool(named toolName: String) -> Bool {
        let name = toolName.lowercased()
        if profileMcpFrontKeys.contains(name) { return true }
        if builtInCategory(for: name) == .mcp { return true }
        return !isBuiltInToolName(name)
    }

    /// `definitions` narrowed to what `profile` allows.
    static func applying(_ profile: AppToolProfile, to definitions: [OpenAITool]) -> [OpenAITool] {
        if profile.allowsEverything { return definitions }
        let allowed = profile.allowedTools.map(Set.init)
        return definitions.filter { tool in
            let name = tool.function.name
            let key = profileKey(for: name)
            if profileAlwaysOnKeys.contains(key) { return true }
            if isMcpOrCustomTool(named: name) { return profile.includesMcpAndCustom }
            guard let allowed else { return true }
            return allowed.contains(key)
        }
    }

    /// A captured turn narrowed to `profile`, deferred MCP descriptors included.
    static func applying(_ profile: AppToolProfile, to captured: TurnAvailableTools) -> TurnAvailableTools {
        if profile.allowsEverything { return captured }
        return TurnAvailableTools(
            definitions: applying(profile, to: captured.definitions),
            promptDefinitions: applying(profile, to: captured.promptDefinitions),
            deferredMcpTools: profile.includesMcpAndCustom ? captured.deferredMcpTools : [])
    }

    /// Every built-in tool, grouped for the profile editor and the popover.
    ///
    /// Built from the same static lists `tools(for:)` assembles, so a tool
    /// added to a list shows up here with no second table to keep in step. The
    /// MCP-and-custom switch is not listed because it has no fixed members.
    ///
    /// `rootless` drops the tools that need a workspace root. Chat has none, so
    /// listing `read` or `bash` there would offer a switch that changes nothing.
    ///
    /// `agentType` keeps only tools that agent profile is ever offered (a Chat
    /// turn or a researcher project has no shell, and no automation tools), so
    /// the composer's "6/38" counts what the turn can use. Nil lists every
    /// built-in, which is what the Settings editor wants: a profile is global.
    public static func profileGroups(
        agentType: AppAgentType? = nil, rootless: Bool = false
    ) -> [ToolProfileGroup] {
        let offeredKeys = agentType.map(offeredProfileKeys(for:))
        let sources: [(title: String, tools: [OpenAITool])] = [
            ("Files", fileTools),
            ("Terminal", terminalTools),
            ("Web", webTools),
            ("Tasks and agents", taskAgentTools),
            ("Planning and skills", planningInteractiveTools),
            ("Projects and artifacts", projectArtifactTools),
            ("Memory", memoryTools),
            ("Automation", automationTools),
            ("Browser", BrowserToolDefinitions.all),
        ]
        var seen = profileAlwaysOnKeys
        var groups: [ToolProfileGroup] = []
        for source in sources {
            var entries: [ToolProfileEntry] = []
            for tool in source.tools where AppToolRegistry.isImplemented(tool.function.name) {
                let key = profileKey(for: tool.function.name)
                guard offeredKeys?.contains(key) ?? true,
                      !isMcpOrCustomTool(named: tool.function.name),
                      !(rootless && AppToolRegistry.workspaceRootedToolNames.contains(tool.function.name.lowercased())),
                      seen.insert(key).inserted
                else { continue }
                entries.append(ToolProfileEntry(
                    id: key, name: tool.function.name,
                    summary: firstSentence(of: tool.function.description)))
            }
            if !entries.isEmpty { groups.append(ToolProfileGroup(title: source.title, entries: entries)) }
        }
        return groups
    }

    /// How many tools the editor can toggle, for the "6/38" count.
    public static func profileToolCount(
        agentType: AppAgentType? = nil, rootless: Bool = false
    ) -> Int {
        profileGroups(agentType: agentType, rootless: rootless).reduce(0) { $0 + $1.entries.count }
    }

    private static let offeredKeysLock = NSLock()
    private static nonisolated(unsafe) var offeredKeysCache: [String: Set<String>] = [:]

    /// The profile keys `agentType` is offered, read once from the same
    /// `tools(for:)` a turn uses. Cached because the composer reads it on every
    /// redraw and building the list reads the skill index. The memory flag is
    /// in the key because it changes the list mid-session.
    static func offeredProfileKeys(for agentType: AppAgentType) -> Set<String> {
        let cacheKey = "\(agentType)|\(memoryTools.isEmpty)"
        offeredKeysLock.lock()
        if let cached = offeredKeysCache[cacheKey] {
            offeredKeysLock.unlock()
            return cached
        }
        offeredKeysLock.unlock()
        let keys = Set(tools(for: agentType, webToolsEnabled: true).map {
            profileKey(for: $0.function.name)
        })
        offeredKeysLock.lock()
        offeredKeysCache[cacheKey] = keys
        offeredKeysLock.unlock()
        return keys
    }

    private static func firstSentence(of text: String) -> String {
        let flat = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        let trimmed = flat.trimmingCharacters(in: .whitespaces)
        if let end = trimmed.range(of: ". ") { return String(trimmed[..<end.lowerBound]) + "." }
        return trimmed.count > 120 ? String(trimmed.prefix(117)) + "..." : trimmed
    }
}
