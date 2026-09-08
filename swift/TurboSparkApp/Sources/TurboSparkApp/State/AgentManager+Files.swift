import Foundation

/// Writing agents to disk: creating one, saving edits, removing one.
///
/// Separate from discovery and precedence, which only ever READ. The
/// distinction is not cosmetic here -- these are the calls that take a
/// name the user just typed, so they are where a name that resolves
/// outside the agents directory has to be refused (the `..` footgun
/// `SkillManager.createSkill` already learned). An agent file becomes a
/// subagent's system prompt, so everything written here is user-authored
/// content in the user's own directories; a PROJECT-scoped create is the
/// one case worth a second thought, and it writes under
/// `<root>/.turbospark/agents/` like any other TurboSpark-scoped project
/// file rather than into a foreign tool's directory.
extension AgentManager {
    // MARK: - Agent Creation & File Management

    /// Errors surfaced by the write path, with UI-presentable text.
    public enum AgentFileError: Error, LocalizedError {
        case invalidName(String)
        case scopeNotAllowed(AppAgentScope)
        case projectRootRequired
        case destinationExists(String)
        case missingFilePath(String)

        public var errorDescription: String? {
            switch self {
            case .invalidName(let name):
                return "'\(name)' is not a usable agent name: it resolves outside the agents directory."
            case .scopeNotAllowed(let scope):
                return "A \(scope.label)-scoped agent is not a file this editor can write: built-ins are code and plugin agents are owned by their plugin."
            case .projectRootRequired:
                return "A project-scoped agent needs an open project to write into."
            case .destinationExists(let name):
                return "An agent file named '\(name)' already exists. Rename the new agent or delete the old file first."
            case .missingFilePath(let name):
                return "Agent '\(name)' has no file on disk to save to."
            }
        }
    }

    /// Creates and writes a new agent file in the given scope.
    ///
    /// - Parameter directoryOverride: redirects a USER-scope create to an
    ///   explicit directory. `defaultUserAgentsDirectory` is the real
    ///   `~/.turbospark/agents` (the storage page's documented gap), so a
    ///   test cannot redirect it after the fact -- the override keeps the
    ///   write path exercisable without touching the user's own tree.
    @discardableResult
    public func createAgent(
        name: String,
        displayName: String? = nil,
        description: String,
        systemPrompt: String,
        tools: [String]? = nil,
        disallowedTools: [String]? = nil,
        maxTurns: Int = 5,
        scope: AppAgentScope,
        projectRootURL: URL? = nil,
        directoryOverride: URL? = nil
    ) throws -> AppAgentDefinition {
        // Same footgun as `SkillManager.createSkill`: `..` survives the
        // slash replacement and resolves to the directory's PARENT. Cheaper
        // to refuse than to reason about.
        let sanitizedName = name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: "\\", with: "-")
            .lowercased()
        // A dot-LEADING name is refused as well as `.` and `..` exactly:
        // `../evil` sanitizes to `..-evil`, a legal file name that
        // `scanDirectory`'s `skipsHiddenFiles` would then never discover --
        // a created agent that silently does not exist.
        guard !sanitizedName.isEmpty, sanitizedName != ".", sanitizedName != "..",
              !sanitizedName.hasPrefix(".") else {
            throw AgentFileError.invalidName(name)
        }

        let targetDir: URL
        switch scope {
        case .userGlobal:
            targetDir = directoryOverride ?? defaultUserAgentsDirectory
        case .project:
            guard let projectRootURL else {
                throw AgentFileError.projectRootRequired
            }
            targetDir = projectRootURL
                .appendingPathComponent(".turbospark", isDirectory: true)
                .appendingPathComponent("agents", isDirectory: true)
        case .builtIn, .plugin:
            throw AgentFileError.scopeNotAllowed(scope)
        }

        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        let fileURL = targetDir.appendingPathComponent("\(sanitizedName).md")
        // **AN OVERWRITE IS A DELETE OF USER CONTENT.** A name the user
        // typed can collide with a file they wrote by hand; the editor
        // decides, by surfacing the error, not by replacing it.
        guard !FileManager.default.fileExists(atPath: fileURL.path) else {
            throw AgentFileError.destinationExists(sanitizedName)
        }

        let agent = AppAgentDefinition(
            name: sanitizedName,
            displayName: displayName,
            agentDescription: description,
            systemPrompt: systemPrompt,
            tools: tools,
            disallowedTools: disallowedTools,
            maxTurns: maxTurns,
            sourceAgent: .turboSpark,
            scope: scope,
            filePath: fileURL.path,
            isEnabled: true
        )
        let serialized = AgentParser.serializeAgent(agent)
        try serialized.write(to: fileURL, atomically: true, encoding: .utf8)
        return agent
    }

    /// Saves modifications to an existing user- or project-scoped agent.
    public func saveAgent(_ agent: AppAgentDefinition) throws {
        switch agent.scope {
        case .builtIn, .plugin:
            throw AgentFileError.scopeNotAllowed(agent.scope)
        case .userGlobal, .project:
            break
        }
        guard let path = agent.filePath, !path.isEmpty else {
            throw AgentFileError.missingFilePath(agent.name)
        }
        let fileURL = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let serialized = AgentParser.serializeAgent(agent)
        try serialized.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    /// Deletes an agent's file. Built-ins are code (refuse), plugin agents
    /// are owned by their plugin (refuse).
    public func deleteAgent(_ agent: AppAgentDefinition) throws {
        switch agent.scope {
        case .builtIn, .plugin:
            throw AgentFileError.scopeNotAllowed(agent.scope)
        case .userGlobal, .project:
            break
        }
        guard let path = agent.filePath, !path.isEmpty else {
            throw AgentFileError.missingFilePath(agent.name)
        }
        try FileManager.default.removeItem(at: URL(fileURLWithPath: path))
    }
}
