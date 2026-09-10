import Foundation

/// Central registry and file manager for discovering, loading, creating, and executing skills.
public final class SkillManager: @unchecked Sendable {
    public static let shared = SkillManager()

    // `internal` rather than `private` only because `SkillManager+Files` is
    // in the file next door; nothing outside this module can see it.
    let fileManager = FileManager.default

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
        if case .projectLocal(let path) = scope {
            let root = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
            return "project:\(root):\(name.lowercased())"
        }
        return "user:\(name.lowercased())"
    }

    /// Whether the given skill is persisted as user-disabled.
    public func isSkillDisabled(scope: SkillScope, name: String) -> Bool {
        let names = disabledSkillNames
        let key = Self.disabledKey(scope: scope, name: name)
        if names.contains("enabled:" + key) { return false }
        return names.contains(key)
            || (scope.isProjectScope && names.contains("project:" + name.lowercased()))
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
            names.insert("enabled:" + key)
        } else {
            names.remove("enabled:" + key)
            names.insert(key)
        }
        disabledSkillNames = names
        invalidateResolutionCache()
    }

    // MARK: - Standard Directories

    /// Standard user skills directory: `~/.turbospark/skills` for the Default
    /// profile (where other agent harnesses read the same files), inside that
    /// profile's own folder for anyone else, who shares nothing.
    public var defaultUserSkillsDirectory: URL {
        UserProfileStore.userScopeSubdirectory("skills")
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

        // 2. Scan standard external user agent roots if enabled. The
        // cross-agent roots are SHARED home-directory trees, so a non-default
        // profile -- whose whole point is owning its own content -- skips
        // them entirely.
        if includeExternalAgents, UserProfileStore.isDefault {
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
        cacheLock.lock()
        resolutionCache = nil
        cacheLock.unlock()
    }

    /// **GUARDED, BECAUSE THIS TYPE IS `@unchecked Sendable` AND THE CACHE IS
    /// READ OFF THE MAIN ACTOR** (state#69). The writers are main-actor
    /// (`reloadSkills`, `invalidateResolutionCache`, the settings pane
    /// bodies) and the reader is not: `AppToolRegistry.execute` is a
    /// `nonisolated async` function running on the cooperative pool, and the
    /// `skill` tool resolves through here. A tuple of an optional and an
    /// array is several words wide, so an unsynchronized read racing a write
    /// is a torn read rather than a stale one. `@MainActor` on the type would
    /// force every tool call to hop, which is why this is a lock.
    ///
    /// The compute deliberately runs OUTSIDE the lock: two concurrent misses
    /// duplicate a directory walk, where holding it across `computeEffectiveSkills`
    /// would serialize a filesystem scan behind a lock this type also takes
    /// from its own discovery path.
    private let cacheLock = NSLock()
    private var resolutionCache: (key: String, skills: [AppSkill])?

    /// Resolves the skills in effect, memoized per project root.
    ///
    /// Every `skill` tool call and every system-prompt build walked the user
    /// and project skill directories from scratch, synchronously on the main
    /// actor. Same reasoning as `AgentManager`'s cache next door.
    public func resolveEffectiveSkills(projectURL: URL?) -> [AppSkill] {
        let key = projectURL?.standardizedFileURL.path ?? ""
        cacheLock.lock()
        let cached = resolutionCache
        cacheLock.unlock()
        if let cached, cached.key == key {
            return cached.skills
        }
        let skills = computeEffectiveSkills(projectURL: projectURL)
        cacheLock.lock()
        resolutionCache = (key, skills)
        cacheLock.unlock()
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
        // Plugin skills are namespaced (`plugin:skill`), so they cannot
        // collide with a bare user or project name and need no precedence
        // rule -- the colon IS the namespace, as in Claude Code.
        let pluginSkills = PluginManager.shared.pluginSkills(projectURL: projectURL)
        guard let projectURL else {
            return (userSkills + pluginSkills)
                .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        }

        let projectSkills = discoverProjectSkills(projectRootURL: projectURL)
        var mergedMap: [String: AppSkill] = [:]

        // 1. Insert user skills first
        for skill in userSkills {
            mergedMap[skill.name.lowercased()] = skill
        }

        // 2. Plugin skills beside them (disjoint by namespace)
        for skill in pluginSkills {
            mergedMap[skill.name.lowercased()] = skill
        }

        // 3. Override with project skills (higher precedence)
        for skill in projectSkills {
            mergedMap[skill.name.lowercased()] = skill
        }

        let project = AppProjectFileStore.load().projects.first {
            $0.rootDirectoryURL?.standardizedFileURL.resolvingSymlinksInPath()
                == projectURL.standardizedFileURL.resolvingSymlinksInPath()
        }
        return mergedMap.values.map { skill in
            var result = skill
            if let enabled = project?.enabledSkills[skill.name.lowercased()] {
                // Disabling a plugin still removes all its contributions upstream.
                result.isEnabled = enabled
            }
            return result
        }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    // MARK: - Argument Substitution & Execution

    /// Replaces `${arg_name}`, `$ARGUMENTS`, `$ARGUMENTS[n]`, positional
    /// `$0`-`$9`, `${SKILL_DIR}`, `${CLAUDE_SKILL_DIR}`, `${SESSION_ID}`, and
    /// `${CLAUDE_SESSION_ID}` placeholders.
    ///
    /// `$ARGUMENTS` and the positional forms are Claude Code's spellings, and
    /// a skill discovered from `~/.claude/skills` is expected to use them:
    /// without them the literal placeholder ships to the model inside an
    /// otherwise-substituted body. When the body carries NO placeholder of
    /// any supported kind and raw arguments arrived, the raw string is
    /// appended the way Claude Code appends it, so the arguments still reach
    /// the model.
    ///
    /// A braceless `$foo` for a DECLARED argument is deliberately not
    /// substituted: a skill body is full of shell variables (`$HOME`, `$?`,
    /// `$0` inside a command example), and without shell-quoting semantics
    /// the braceless named form is indistinguishable from prose. `${foo}` is
    /// the unambiguous spelling here.
    public func substituteArguments(
        content: String,
        arguments: [String: String] = [:],
        skillDirectoryURL: URL? = nil,
        sessionID: String? = nil
    ) -> String {
        var result = content

        let rawArgs = arguments["arguments"] ?? arguments["args"] ?? ""
        let positionals = SkillManager.splitPositionalArguments(rawArgs)

        // Indexed forms before the bare one: `$ARGUMENTS[0]` contains bare
        // `$ARGUMENTS` as a substring, so the bare replacement must come
        // last. Positionals run HIGH to LOW so `$1` cannot eat the prefix of
        // a literal `$10`.
        for (index, value) in positionals.prefix(10).enumerated().reversed() {
            result = result.replacingOccurrences(of: "$ARGUMENTS[\(index)]", with: value)
            result = result.replacingOccurrences(of: "$\(index)", with: value)
        }
        result = result.replacingOccurrences(of: "$ARGUMENTS", with: rawArgs)

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

        // The fallback arm reads the ORIGINAL body: a body that asked for
        // positionals that never arrived did ask, and appending the raw
        // string on top of an unfilled `$0` would double the arguments in.
        let hadPlaceholder = content.contains("${")
            || content.contains("$ARGUMENTS")
            || SkillManager.containsPositionalPlaceholder(content)
        if !hadPlaceholder && !rawArgs.isEmpty {
            result += "\n\nARGUMENTS: \(rawArgs)"
        }

        return result
    }

    /// Splits a raw argument string on whitespace, keeping quoted spans whole
    /// and dropping the quote characters (Claude Code parses arguments with
    /// shell-quote; this is the same shape without a shell). `$1` indexes the
    /// first element.
    static func splitPositionalArguments(_ raw: String) -> [String] {
        var parts: [String] = []
        var current = ""
        var quote: Character? = nil
        for character in raw {
            if let open = quote {
                if character == open { quote = nil } else { current.append(character) }
            } else if character == "\"" || character == "'" {
                quote = character
            } else if character.isWhitespace {
                if !current.isEmpty {
                    parts.append(current)
                    current = ""
                }
            } else {
                current.append(character)
            }
        }
        if !current.isEmpty { parts.append(current) }
        return parts
    }

    /// Whether any `$<digit>` appears in the text (the positional form).
    static func containsPositionalPlaceholder(_ text: String) -> Bool {
        var previous: Character? = nil
        for character in text {
            if character.isNumber, previous == "$" { return true }
            previous = character
        }
        return false
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
