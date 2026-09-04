import Foundation

extension AppModel {
    /// User skills the selected project is currently shadowing by name
    /// (state#57). The skill-side twin of `constrainedProjectAgentNames`.
    public var shadowedUserSkillNames: [String] {
        SkillManager.shared.shadowedUserSkillNames(
            projectURL: selectedProject?.rootDirectoryURL)
    }

    /// Combined list of active skills, with project-level skills taking precedence over user-level skills with matching names.
    public var effectiveSkills: [AppSkill] {
        var merged: [String: AppSkill] = [:]
        for skill in userSkills where skill.isEnabled {
            merged[skill.name.lowercased()] = skill
        }
        for skill in projectSkills where skill.isEnabled {
            merged[skill.name.lowercased()] = skill
        }
        return Array(merged.values).sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
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
            let projectURL = selectedProject?.rootDirectoryURL
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
        do {
            try SkillManager.shared.deleteSkill(skill)
            reloadSkills()
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
            let projectURL = selectedProject?.rootDirectoryURL
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
    /// `disable-model-invocation` if a model ever reached it. It has no
    /// caller today, which is exactly why the gates have to be here rather
    /// than at a call site: the first one added would otherwise inherit the
    /// hole.
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
}
