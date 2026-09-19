import Foundation

extension AppModel {
    /// User skills the selected project is currently shadowing by name
    /// (state#57). The skill-side twin of `constrainedProjectAgentNames`.
    public var shadowedUserSkillNames: [String] {
        SkillManager.shared.shadowedUserSkillNames(
            projectURL: selectedProject?.rootDirectoryURL)
    }

    /// Combined list of active skills, with project-level skills taking precedence over user-level skills with matching names.
    ///
    /// This routes through `SkillManager.resolveEffectiveSkills` rather than
    /// re-merging the published arrays, because only the manager's path also
    /// merges PLUGIN skills -- a second merge here would show them in the
    /// prompt while the `skill` tool resolved a different set.
    public var effectiveSkills: [AppSkill] {
        SkillManager.shared.resolveEffectiveSkills(
            projectURL: selectedProject?.rootDirectoryURL)
    }

    /// All skills (both user and project scope) for management in settings.
    public var allManagedSkills: [AppSkill] {
        var list = userSkills
        list.append(contentsOf: projectSkills)
        return list.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    /// Reloads both user-scoped and project-scoped skills from disk.
    public func reloadSkills() {
        // The explicit re-scan, so it drops the memoized resolution first --
        // otherwise a skill the user just created or imported would not
        // appear until the project changed.
        SkillManager.shared.invalidateResolutionCache()
        userSkills = SkillManager.shared.discoverUserSkills()
        if let projectURL = selectedProject?.rootDirectoryURL {
            projectSkills = SkillManager.shared.discoverProjectSkills(projectRootURL: projectURL)
        } else {
            projectSkills = []
        }
    }

    /// Creates and saves a new skill to disk in user or project scope.
    @discardableResult
    public func createNewSkill(
        name: String,
        description: String,
        content: String,
        allowedTools: [String] = [],
        paths: [String] = [],
        scope: SkillScope
    ) -> Result<AppSkill, Error> {
        do {
            let projectURL = scope.projectRootURL
            let skill = try SkillManager.shared.createSkill(
                name: name,
                description: description,
                content: content,
                allowedTools: allowedTools,
                paths: paths,
                scope: scope,
                projectRootURL: projectURL
            )
            reloadSkills()
            showToast("Skill '\(skill.name)' created in \(scope.label).", style: .info)
            return .success(skill)
        } catch {
            showToast("Failed to create skill: \(error.localizedDescription)", style: .error)
            return .failure(error)
        }
    }

    /// Updates and persists an existing skill on disk.
    public func updateSkill(_ skill: AppSkill) {
        do {
            try SkillManager.shared.saveSkill(skill)
            reloadSkills()
            showToast("Saved changes to '\(skill.name)'.", style: .info)
        } catch {
            showToast("Failed to save skill: \(error.localizedDescription)", style: .error)
        }
    }

    /// Deletes a skill from disk.
    public func removeSkill(_ skill: AppSkill) {
        defer { reloadSkills() }
        do {
            try SkillManager.shared.deleteSkill(skill)
            showToast("Removed skill '\(skill.name)'.", style: .info)
        } catch {
            showToast("Failed to delete skill: \(error.localizedDescription)", style: .error)
        }
    }

    /// Toggles and PERSISTS the enabled state of a skill.
    ///
    /// A skill file has no `enabled` field (deliberately -- it is a
    /// per-user preference, not something that belongs in a file meant to
    /// be shared or checked into a repo), so the state lives in
    /// `SkillManager`'s own store instead. Toggling only the in-memory
    /// `AppSkill` copy here used to be silently discarded by the next
    /// `reloadSkills()` -- switching projects, importing a skill, anything
    /// that re-scans disk -- which always came back `isEnabled: true`
    /// (state#12).
    ///
    /// Keyed on SCOPE plus name: a project skill may deliberately share a
    /// user skill's name -- that is what project precedence is -- and the
    /// name-only key disabled both at once.
    public func toggleSkillEnabled(_ skill: AppSkill) {
        SkillManager.shared.setSkillEnabled(
            !skill.isEnabled, scope: skill.scope, name: skill.name)
        reloadSkills()
    }

    /// Imports a skill from an external directory into user or project scope.
    @discardableResult
    public func importSkill(from sourceURL: URL, targetScope: SkillScope) -> Result<AppSkill, Error> {
        do {
            let projectURL = targetScope.projectRootURL
            let skill = try SkillManager.shared.importSkill(
                from: sourceURL,
                targetScope: targetScope,
                projectRootURL: projectURL
            )
            reloadSkills()
            showToast("Imported '\(skill.name)' into \(targetScope.label).", style: .info)
            return .success(skill)
        } catch {
            showToast("Import failed: \(error.localizedDescription)", style: .error)
            return .failure(error)
        }
    }

    /// Finds and resolves a skill by name from effective skills.
    public func findSkill(named name: String) -> AppSkill? {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return effectiveSkills.first { $0.name.lowercased() == clean }
    }

    /// Executes a skill by substituting arguments and returning the formatted instructions payload.
    ///
    /// **THE SAME TWO GATES THE `skill` TOOL APPLIES** (state#64). This had
    /// neither, so it ran a skill the user had switched off (state#12's other
    /// half, on the path nothing currently calls) and would run one declaring
    /// `disable-model-invocation` if a model ever reached it. The user slash
    /// path (`handleSkillSlashCommand`) is its caller, which is exactly why
    /// the gates live here rather than at a call site: the next caller would
    /// otherwise inherit the hole. (`disable-model-invocation` stays a
    /// MODEL-side gate; a user may always invoke their own skill.)
    public func executeSkill(named name: String, arguments: [String: String] = [:]) -> String? {
        guard let skill = findSkill(named: name), skill.isEnabled else {
            return nil
        }
        let sessionID = selectedChatID.uuidString
        let expanded = SkillManager.shared.substituteArguments(
            content: skill.content,
            arguments: arguments,
            skillDirectoryURL: skill.skillDirectoryURL,
            sessionID: sessionID
        )

        var output = "### Skill: \(skill.name)\n\(expanded)"
        if !skill.referenceFiles.isEmpty {
            output += "\n\n*Reference Files available in skill directory:* \(skill.referenceFiles.joined(separator: ", "))"
        }
        return output
    }

    /// Handles user-typed slash commands for skills (e.g. `/my-skill [args]` or `/skill <name> [args]`).
    /// Returns true if recognized and handled.
    ///
    /// Acts on the CURRENTLY SELECTED chat: it rewrites `promptText` and
    /// calls `run()`, both of which resolve the selection themselves, so a
    /// chat-id parameter would be an argument nothing reads.
    public func handleSkillSlashCommand(_ input: String) -> Bool {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("/") else { return false }

        let parts = trimmed.dropFirst().components(separatedBy: " ")
        guard let command = parts.first?.lowercased(), !command.isEmpty else { return false }
        let rest = parts.dropFirst().joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)

        var targetSkillName: String?
        var rawArgs: String = ""

        if command == "skill" {
            let subParts = rest.components(separatedBy: " ")
            if let first = subParts.first, !first.isEmpty {
                targetSkillName = first
                rawArgs = subParts.dropFirst().joined(separator: " ")
            }
        } else if let skill = findSkill(named: command) {
            targetSkillName = skill.name
            rawArgs = rest
        } else {
            let all = allManagedSkills
            if let disabled = all.first(where: { $0.name.lowercased() == command }) {
                showToast("Skill '\(disabled.name)' is currently disabled. Enable it in Settings > Skills to use it.", style: .warning)
                return true
            }
        }

        guard let skillName = targetSkillName, let skill = findSkill(named: skillName) else {
            return false
        }

        guard skill.isEnabled else {
            showToast("Skill '\(skill.name)' is disabled.", style: .warning)
            return true
        }

        guard skill.manifest.userInvocable else {
            showToast("Skill '\(skill.name)' is not user-invocable.", style: .warning)
            return true
        }

        var argsDict: [String: String] = [:]
        if !rawArgs.isEmpty {
            if let firstArg = skill.manifest.arguments.first?.name {
                argsDict[firstArg] = rawArgs
            }
            argsDict["arguments"] = rawArgs
            argsDict["args"] = rawArgs
        }

        if skill.manifest.context == .fork {
            // The SUBSTITUTED BODY rides along, not just the name: an agent
            // handed only "execute skill [name]" would have to find and load
            // the skill itself, and `disable-model-invocation` skills would
            // be unreachable from here at all. `agent:` names the runner,
            // general-purpose the fallback (Claude Code's
            // `command.agent ?? 'general-purpose'`).
            guard let payload = executeSkill(named: skill.name, arguments: argsDict) else {
                return false
            }
            let requested = skill.manifest.agent.flatMap { $0.isEmpty ? nil : $0 }
            let agent = requested.flatMap { findAgent(named: $0) }
                ?? findAgent(named: "general-purpose")
                ?? AgentManager.shared.builtInAgents.first
            if let agent {
                var prompt = "Execute the following skill instructions:\n\n\(payload)"
                if let requested, agent.name.lowercased() != requested.lowercased() {
                    prompt += "\n\n(Requested agent '\(requested)' was not found; running as '\(agent.name)'.)"
                }
                runAgentTaskDirectly(agent: agent, prompt: prompt)
            }
            return true
        } else {
            guard let payload = executeSkill(named: skill.name, arguments: argsDict) else {
                return false
            }
            // Programmatic expansion of a skill payload, which is often
            // larger than the paste threshold; never a paste.
            writePromptTextDirectly(
                rawArgs.isEmpty
                    ? "Execute the following skill instructions:\n\n\(payload)"
                    : "Execute the following skill instructions with arguments: \(rawArgs)\n\n\(payload)")
            run()
            return true
        }
    }
}
