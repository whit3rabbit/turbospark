import Foundation

/// Central registry and file manager for discovering, loading, creating, and executing skills.
public final class SkillManager: @unchecked Sendable {
    public static let shared = SkillManager()

    private let fileManager = FileManager.default

    public init() {}

    // MARK: - Enabled/Disabled Persistence
    //
    // `SkillParser` hardcodes `isEnabled: true` when reading a skill file --
    // there is no such field in the SKILL.md frontmatter format, on purpose:
    // it is a per-USER preference, not something that belongs in a file
    // meant to be shared or checked into a repo. `AppModel.toggleSkillEnabled`
    // used to flip `isEnabled` only on the in-memory `AppSkill` copy, which
    // `reloadSkills()` (switching projects, importing a skill, anything that
    // re-scans disk) silently discarded back to enabled (state#12). Persisted
    // here, keyed by lowercased name, and applied uniformly in
    // `scanDirectory` so every discovery path -- `AppModel.reloadSkills()`
    // AND `resolveEffectiveSkills` (what the `skill` tool itself resolves
    // through) -- sees the same state.

    private var disabledSkillNames: Set<String> {
        // Under `AppStorageRoot`, not `UserDefaults.standard` (state#57):
        // the suite was mutating real preferences and the bundle-identity
        // change re-enabled everything on install. See `DisabledItemStore`.
        get { DisabledItemStore.names(for: .skills) }
        set { DisabledItemStore.setNames(newValue, for: .skills) }
    }

    /// The persisted key for one skill.
    ///
    /// **SCOPE PLUS NAME, NOT NAME ALONE.** A project skill may deliberately
    /// share a user skill's name -- that is what project precedence IS -- and
    /// keying on the name alone disabled both together, so turning off a
    /// project's `deploy` also turned off the user's own. The old
    /// name-only keys are still honoured on read so nobody's existing
    /// preference is silently forgotten.
    private static func disabledKey(scope: SkillScope, name: String) -> String {
        "\(scope.isProjectScope ? "project" : "user"):\(name.lowercased())"
    }

    /// Whether the given skill is persisted as user-disabled.
    public func isSkillDisabled(scope: SkillScope, name: String) -> Bool {
        let names = disabledSkillNames
        return names.contains(Self.disabledKey(scope: scope, name: name))
            || names.contains(name.lowercased())
    }

    /// Persists the enabled/disabled state for one skill.
    public func setSkillEnabled(_ enabled: Bool, scope: SkillScope, name: String) {
        var names = disabledSkillNames
        let key = Self.disabledKey(scope: scope, name: name)
        if enabled {
            names.remove(key)
            // Also clears a pre-scope key, or re-enabling would appear to do
            // nothing for anyone upgrading.
            names.remove(name.lowercased())
        } else {
            names.insert(key)
        }
        disabledSkillNames = names
        invalidateResolutionCache()
    }

    // MARK: - Standard Directories

    /// Standard user skills directory (~/.turbospark/skills).
    public var defaultUserSkillsDirectory: URL {
        let home = fileManager.homeDirectoryForCurrentUser
        let dir = home.appendingPathComponent(".turbospark", isDirectory: true)
            .appendingPathComponent("skills", isDirectory: true)
        try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Known agent user-scope skill roots as documented in agent-config support matrix.
    public var knownUserAgentSkillRoots: [(agent: SkillSourceAgent, relativePath: String)] {
        [
            (.turboSpark, ".turbospark/skills"),
            (.claude, ".claude/skills"),
            (.antigravity, ".gemini/antigravity/skills"),
            (.gemini, ".gemini/skills"),
            (.antigravity, ".agents/skills"),
            (.openCode, ".config/opencode/skills"),
            (.pi, ".pi/agent/skills"),
            (.cursor, ".cursor/skills"),
            (.windsurf, ".codeium/windsurf/skills"),
            (.kilo, ".kilo/skills"),
            (.crush, ".crush/skills"),
            (.cline, ".cline/skills"),
            (.forge, ".forge/skills"),
            (.qwen, ".qwen/skills"),
            (.copilot, ".copilot/skills")
        ]
    }

    /// Known project-scope relative directories to search within a codebase.
    public var knownProjectSkillSubdirectories: [(agent: SkillSourceAgent, relativePath: String)] {
        [
            (.turboSpark, ".turbospark/skills"),
            (.claude, ".claude/skills"),
            (.antigravity, ".agents/skills"),
            (.openCode, ".opencode/skills"),
            (.pi, ".pi/skills"),
            (.cursor, ".cursor/skills"),
            (.copilot, ".github/skills"),
            (.windsurf, ".windsurf/skills"),
            (.kilo, ".kilo/skills"),
            (.crush, ".crush/skills"),
            (.cline, ".cline/skills"),
            (.forge, ".forge/skills"),
            (.qwen, ".qwen/skills")
        ]
    }

    // MARK: - Discovery

    /// Discovers all user-global skills from ~/.turbospark/skills and other agent roots.
    public func discoverUserSkills(includeExternalAgents: Bool = true) -> [AppSkill] {
        var skills: [AppSkill] = []
        var seenNames = Set<String>()

        // 1. Primary TurboSpark user directory
        let primaryURL = defaultUserSkillsDirectory
        let primarySkills = scanDirectory(primaryURL, scope: .userGlobal, defaultAgent: .turboSpark)
        for skill in primarySkills {
            let key = skill.name.lowercased()
            if !seenNames.contains(key) {
                seenNames.insert(key)
                skills.append(skill)
            }
        }

        // 2. Scan standard external user agent roots if enabled
        if includeExternalAgents {
            let home = fileManager.homeDirectoryForCurrentUser
            for (agent, relPath) in knownUserAgentSkillRoots where agent != .turboSpark {
                let agentURL = home.appendingPathComponent(relPath, isDirectory: true)
                guard fileManager.fileExists(atPath: agentURL.path) else { continue }
                let agentSkills = scanDirectory(agentURL, scope: .userGlobal, defaultAgent: agent)
                for skill in agentSkills {
                    let key = skill.name.lowercased()
                    if !seenNames.contains(key) {
                        seenNames.insert(key)
                        skills.append(skill)
                    }
                }
            }
        }

        return skills.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    /// Discovers all project-local skills inside a given project workspace directory.
    public func discoverProjectSkills(projectRootURL: URL) -> [AppSkill] {
        guard fileManager.fileExists(atPath: projectRootURL.path) else { return [] }

        var skills: [AppSkill] = []
        var seenNames = Set<String>()
        let projectPath = projectRootURL.path

        for (agent, relPath) in knownProjectSkillSubdirectories {
            let subURL = projectRootURL.appendingPathComponent(relPath, isDirectory: true)
            guard fileManager.fileExists(atPath: subURL.path) else { continue }

            let scanned = scanDirectory(
                subURL, scope: .projectLocal(projectPath: projectPath), defaultAgent: agent,
                containedIn: projectRootURL)
            for skill in scanned {
                let key = skill.name.lowercased()
                if !seenNames.contains(key) {
                    seenNames.insert(key)
                    skills.append(skill)
                }
            }
        }

        return skills.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    /// Overrides a freshly-parsed skill's `isEnabled` (always `true` out of
    /// `SkillParser`) with the persisted per-user preference, if any.
    private func applyingPersistedEnabledState(to skill: AppSkill) -> AppSkill {
        guard isSkillDisabled(scope: skill.scope, name: skill.name) else { return skill }
        var updated = skill
        updated.isEnabled = false
        return updated
    }

    /// Scans a specific skills directory for both folder-based (SKILL.md) and single-file (.md) skills.
    /// - Parameter containedIn: the project root a PROJECT-scoped scan must
    ///   keep its files inside (state#39). Nil for a user scope, where the
    ///   files are the user's own.
    public func scanDirectory(
        _ dirURL: URL, scope: SkillScope, defaultAgent: SkillSourceAgent,
        containedIn root: URL? = nil
    ) -> [AppSkill] {
        guard let entries = try? fileManager.contentsOfDirectory(atPath: dirURL.path) else { return [] }

        var results: [AppSkill] = []
        for entry in entries {
            if entry.hasPrefix(".") || entry == "node_modules" { continue }

            let itemURL = dirURL.appendingPathComponent(entry)
            var isDir: ObjCBool = false
            guard fileManager.fileExists(atPath: itemURL.path, isDirectory: &isDir) else { continue }

            if isDir.boolValue {
                // Directory-based skill format: <name>/SKILL.md or <name>/skill.md
                let skillMdURL = itemURL.appendingPathComponent("SKILL.md")
                let skillMdLowerURL = itemURL.appendingPathComponent("skill.md")

                let targetURL: URL?
                if fileManager.fileExists(atPath: skillMdURL.path) {
                    targetURL = skillMdURL
                } else if fileManager.fileExists(atPath: skillMdLowerURL.path) {
                    targetURL = skillMdLowerURL
                } else {
                    targetURL = nil
                }

                if let targetURL,
                    let skill = try? SkillParser.parseFile(
                        at: targetURL, scope: scope, agentOrigin: defaultAgent, containedIn: root) {
                    results.append(applyingPersistedEnabledState(to: skill))
                }
            } else if itemURL.pathExtension.lowercased() == "md" {
                // Single-file skill format: <name>.md
                if let skill = try? SkillParser.parseFile(
                    at: itemURL, scope: scope, agentOrigin: defaultAgent, containedIn: root) {
                    results.append(applyingPersistedEnabledState(to: skill))
                }
            }
        }

        return results
    }

    // MARK: - Combined Precedence Resolution

    /// Merges project and user skills with project precedence over user skills for matching names.
    /// Drops the memoized resolution. Call after anything that changes what
    /// is on disk or which skills are enabled.
    public func invalidateResolutionCache() {
        resolutionCache = nil
    }

    private var resolutionCache: (key: String, skills: [AppSkill])?

    /// Resolves the skills in effect, memoized per project root.
    ///
    /// Every `skill` tool call and every system-prompt build walked the user
    /// and project skill directories from scratch, synchronously on the main
    /// actor. Same reasoning as `AgentManager`'s cache next door.
    public func resolveEffectiveSkills(projectURL: URL?) -> [AppSkill] {
        let key = projectURL?.standardizedFileURL.path ?? ""
        if let cached = resolutionCache, cached.key == key {
            return cached.skills
        }
        let skills = computeEffectiveSkills(projectURL: projectURL)
        resolutionCache = (key, skills)
        return skills
    }

    /// The names of USER skills a project skill is currently overriding
    /// (state#57).
    ///
    /// **PRECEDENCE WITH NO DISCLOSURE IS INDISTINGUISHABLE FROM A BROKEN
    /// SKILL.** A project skill deliberately shadows a same-named user one;
    /// that is what precedence IS. But nothing anywhere said so, so a user
    /// whose own `deploy` stopped behaving reads it as their skill being
    /// broken rather than replaced -- and a cloned repository can shadow any
    /// skill by name, silently. Agents already publish
    /// `constrainedProjectAgentNames` for the same reason; this is that list
    /// on the skill side.
    public func shadowedUserSkillNames(projectURL: URL?) -> [String] {
        guard let projectURL else { return [] }
        let userNames = Set(discoverUserSkills().map { $0.name.lowercased() })
        return discoverProjectSkills(projectRootURL: projectURL)
            .map { $0.name }
            .filter { userNames.contains($0.lowercased()) }
            .sorted()
    }

    private func computeEffectiveSkills(projectURL: URL?) -> [AppSkill] {
        let userSkills = discoverUserSkills()
        guard let projectURL else { return userSkills }

        let projectSkills = discoverProjectSkills(projectRootURL: projectURL)
        var mergedMap: [String: AppSkill] = [:]

        // 1. Insert user skills first
        for skill in userSkills {
            mergedMap[skill.name.lowercased()] = skill
        }

        // 2. Override with project skills (higher precedence)
        for skill in projectSkills {
            mergedMap[skill.name.lowercased()] = skill
        }

        return Array(mergedMap.values).sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    // MARK: - Skill Creation & File Management

    /// Creates and writes a new skill to disk in the given scope.
    @discardableResult
    public func createSkill(
        name: String,
        description: String,
        content: String,
        allowedTools: [String] = [],
        paths: [String] = [],
        scope: SkillScope,
        projectRootURL: URL? = nil
    ) throws -> AppSkill {
        // `..` survived the slash replacement, so `createSkill(name: "..")`
        // resolved to the skills directory's PARENT and wrote a SKILL.md
        // there. User-typed rather than model-controlled, so this is a
        // footgun rather than an escalation -- and cheaper to refuse than to
        // reason about.
        let sanitizedName = name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: "\\", with: "-")
            .lowercased()
        guard !sanitizedName.isEmpty, sanitizedName != ".", sanitizedName != ".." else {
            throw NSError(
                domain: "TurboSparkSkill", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "'\(name)' is not a usable skill name: it resolves outside the skills directory."
                ])
        }

        let targetDir: URL
        switch scope {
        case .userGlobal, .bundled:
            targetDir = defaultUserSkillsDirectory.appendingPathComponent(sanitizedName, isDirectory: true)
        case .projectLocal:
            guard let projectRootURL else {
                throw NSError(domain: "TurboSparkSkill", code: 1, userInfo: [NSLocalizedDescriptionKey: "Project root URL required for project-scoped skill."])
            }
            targetDir = projectRootURL
                .appendingPathComponent(".turbospark", isDirectory: true)
                .appendingPathComponent("skills", isDirectory: true)
                .appendingPathComponent(sanitizedName, isDirectory: true)
        }

        try fileManager.createDirectory(at: targetDir, withIntermediateDirectories: true)
        let skillMdURL = targetDir.appendingPathComponent("SKILL.md")

        let manifest = SkillManifest(
            name: name,
            description: description,
            allowedTools: allowedTools,
            paths: paths
        )

        let appSkill = AppSkill(
            manifest: manifest,
            content: content,
            sourceURL: skillMdURL,
            skillDirectoryURL: targetDir,
            scope: scope,
            agentOrigin: .turboSpark,
            isEnabled: true
        )

        let serialized = SkillParser.serializeSkill(appSkill)
        try serialized.write(to: skillMdURL, atomically: true, encoding: .utf8)

        return appSkill
    }

    /// Saves modifications to an existing skill on disk.
    public func saveSkill(_ skill: AppSkill) throws {
        let serialized = SkillParser.serializeSkill(skill)
        let targetURL = skill.sourceURL
        let targetDir = targetURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: targetDir, withIntermediateDirectories: true)
        try serialized.write(to: targetURL, atomically: true, encoding: .utf8)
    }

    /// Deletes a skill and its containing folder if it is directory-based.
    public func deleteSkill(_ skill: AppSkill) throws {
        if skill.isDirectoryBased, let dirURL = skill.skillDirectoryURL {
            try fileManager.removeItem(at: dirURL)
        } else {
            try fileManager.removeItem(at: skill.sourceURL)
        }
    }

    /// Imports a skill directory or file into TurboSpark user or project scope.
    @discardableResult
    /// - Parameter overwrite: whether to replace an existing skill of the
    ///   same name. Defaults to false so a collision is REPORTED; the import
    ///   used to delete the existing directory outright, which is
    ///   unrecoverable user content.
    public func importSkill(
        from sourceURL: URL,
        targetScope: SkillScope,
        projectRootURL: URL? = nil,
        overwrite: Bool = false
    ) throws -> AppSkill {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: sourceURL.path, isDirectory: &isDir) else {
            throw SkillParseError.fileNotFound(sourceURL.path)
        }

        let destinationBaseDir: URL
        switch targetScope {
        case .userGlobal, .bundled:
            destinationBaseDir = defaultUserSkillsDirectory
        case .projectLocal:
            guard let projectRootURL else {
                throw NSError(domain: "TurboSparkSkill", code: 2, userInfo: [NSLocalizedDescriptionKey: "Project root required for project-scoped import."])
            }
            destinationBaseDir = projectRootURL
                .appendingPathComponent(".turbospark", isDirectory: true)
                .appendingPathComponent("skills", isDirectory: true)
        }

        try fileManager.createDirectory(at: destinationBaseDir, withIntermediateDirectories: true)

        if isDir.boolValue {
            let folderName = sourceURL.lastPathComponent
            let destFolderURL = destinationBaseDir.appendingPathComponent(folderName, isDirectory: true)
            // **AN IMPORT USED TO DELETE WHATEVER WAS ALREADY THERE.** A skill
            // directory is the user's own edited content; replacing it
            // silently on a name collision is unrecoverable. The caller
            // decides, by passing `overwrite`.
            if fileManager.fileExists(atPath: destFolderURL.path) {
                guard overwrite else {
                    throw SkillImportError.destinationExists(name: folderName)
                }
                try fileManager.removeItem(at: destFolderURL)
            }
            try fileManager.copyItem(at: sourceURL, to: destFolderURL)

            let skillMdURL = destFolderURL.appendingPathComponent("SKILL.md")
            let fallbackMdURL = destFolderURL.appendingPathComponent("skill.md")
            let readURL = fileManager.fileExists(atPath: skillMdURL.path) ? skillMdURL : fallbackMdURL

            return try SkillParser.parseFile(at: readURL, scope: targetScope, agentOrigin: .custom)
        } else {
            let fileName = sourceURL.lastPathComponent
            let destFileURL = destinationBaseDir.appendingPathComponent(fileName)
            if fileManager.fileExists(atPath: destFileURL.path) {
                guard overwrite else {
                    throw SkillImportError.destinationExists(name: fileName)
                }
                try fileManager.removeItem(at: destFileURL)
            }
            try fileManager.copyItem(at: sourceURL, to: destFileURL)
            return try SkillParser.parseFile(at: destFileURL, scope: targetScope, agentOrigin: .custom)
        }
    }

    // MARK: - Argument Substitution & Execution

    /// Replaces `${arg_name}`, `${SKILL_DIR}`, `${CLAUDE_SKILL_DIR}`, and `${SESSION_ID}` placeholders.
    public func substituteArguments(
        content: String,
        arguments: [String: String] = [:],
        skillDirectoryURL: URL? = nil,
        sessionID: String? = nil
    ) -> String {
        var result = content

        for (key, val) in arguments {
            let placeholder = "${\(key)}"
            result = result.replacingOccurrences(of: placeholder, with: val)
        }

        if let skillDir = skillDirectoryURL {
            result = result.replacingOccurrences(of: "${SKILL_DIR}", with: skillDir.path)
            result = result.replacingOccurrences(of: "${CLAUDE_SKILL_DIR}", with: skillDir.path)
        }

        if let sessionID {
            result = result.replacingOccurrences(of: "${SESSION_ID}", with: sessionID)
            result = result.replacingOccurrences(of: "${CLAUDE_SESSION_ID}", with: sessionID)
        }

        return result
    }

    /// Evaluates glob path patterns to check if a skill matches a given file path.
    public func matchesPath(skill: AppSkill, filePath: String) -> Bool {
        if skill.manifest.paths.isEmpty { return true }
        let cleanPath = filePath.trimmingCharacters(in: .whitespacesAndNewlines)

        for pattern in skill.manifest.paths {
            if globMatch(pattern: pattern, path: cleanPath) {
                return true
            }
        }
        return false
    }

    private func globMatch(pattern: String, path: String) -> Bool {
        var regexPattern = "^"
        let chars = Array(pattern)
        var i = 0

        while i < chars.count {
            let c = chars[i]
            if c == "*" {
                if i + 1 < chars.count && chars[i + 1] == "*" {
                    // Double star **
                    i += 2
                    if i < chars.count && chars[i] == "/" {
                        regexPattern.append("(.*/)?")
                        i += 1
                    } else {
                        regexPattern.append(".*")
                    }
                } else {
                    regexPattern.append("[^/]*")
                    i += 1
                }
            } else if c == "?" {
                regexPattern.append("[^/]")
                i += 1
            } else if [".", "(", ")", "[", "]", "{", "}", "^", "$", "+", "|", "\\"].contains(c) {
                regexPattern.append("\\\(c)")
                i += 1
            } else {
                regexPattern.append(c)
                i += 1
            }
        }
        regexPattern.append("$")

        guard let regex = try? NSRegularExpression(pattern: regexPattern, options: [.caseInsensitive]) else {
            return path.lowercased().contains(pattern.lowercased())
        }

        let range = NSRange(location: 0, length: (path as NSString).length)
        return regex.firstMatch(in: path, options: [], range: range) != nil
    }
}
