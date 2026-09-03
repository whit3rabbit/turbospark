import Foundation

extension AppModel {
    /// Filtered list of chats according to the active project selection.
    public var filteredChats: [AppChat] {
        guard let projectID = selectedProjectID else {
            return chats
        }
        return chats.filter { $0.projectID == projectID }
    }

    /// The project a chat belongs to, or nil.
    ///
    /// Resolved from `AppChat.projectID` rather than from `selectedProjectID`
    /// so it cannot move under a turn.
    public func project(forChat chatID: UUID) -> AppProject? {
        guard let index = chats.firstIndex(where: { $0.id == chatID }),
            let projectID = chats[index].projectID
        else { return nil }
        return projects.first { $0.id == projectID }
    }

    /// The project a GENERATION TURN runs under (state#30).
    ///
    /// **NOT `selectedProject`.** state#19 pinned the project an approved
    /// call executes against; the turn that FOLLOWS it took its system
    /// prompt, workspace root, agent type, step cap, skill-state toggle and
    /// the next call's permission evaluation from whatever was selected at
    /// that moment. `selectProject` guards only on `!generating`, which is
    /// false for the whole time a call sits at an approval card, so the
    /// switch is not merely possible -- it is legal precisely when a turn is
    /// mid-flight.
    ///
    /// The `interactionMode` gate is preserved from the call site this
    /// replaced: conversational Chat mode sends no project, hence no system
    /// prompt and no tool definitions, and `extractToolCalls` refuses to
    /// parse a call that was never offered.
    public func turnProject(chatID: UUID) -> AppProject? {
        guard interactionMode == .projects else { return nil }
        return project(forChat: chatID)
    }

    /// Creates and persists a new codebase project.
    @discardableResult
    public func createProject(
        name: String,
        rootDirectoryPath: String? = nil,
        agentType: AppAgentType = .coder,
        rulePreference: AppRulePreference = .agentsFirst,
        customInstructions: String = "",
        permissions: AppProjectPermissions = .newProjectDefault,
        maxAutonomousSteps: Int = 5,
        skillStateEnabled: Bool = false,
        forgeGuardrailsEnabled: Bool? = nil
    ) -> AppProject {
        var instructions = customInstructions
        if instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let path = rootDirectoryPath,
           let autoRules = detectProjectRules(directoryPath: path, preference: rulePreference) {
            instructions = autoRules
        }

        let project = AppProject(
            name: name,
            rootDirectoryPath: rootDirectoryPath,
            agentType: agentType,
            rulePreference: rulePreference,
            customInstructions: instructions,
            permissions: permissions,
            maxAutonomousSteps: maxAutonomousSteps,
            forgeGuardrailsEnabled: forgeGuardrailsEnabled,
            skillStateEnabled: skillStateEnabled
        )

        projects.insert(project, at: 0)
        selectedProjectID = project.id
        if let path = project.rootDirectoryPath, !path.isEmpty {
            worktree = WorktreeModel(rootDirectoryPath: path)
        } else {
            worktree = nil
        }
        persistProjects()
        reloadSkills()
        reloadAgents()
        AppHookStore.shared.refresh(projectDirectory: project.rootDirectoryPath)

        // Create initial chat for this project
        createChat(projectID: project.id)
        return project
    }

    /// Selects an active project or clears the project filter.
    public func selectProject(id: UUID?) {
        guard !generating else { return }
        selectedProjectID = id
        // Stale-write hashes are per workspace. Carrying them across a
        // project switch means an `edit_file` in the new project can be
        // refused (or, worse, allowed) on the strength of a hash recorded
        // against a same-named file in the old one. `FileSnapshotStore`
        // documented this clearing since it was written and nothing ever
        // called it.
        Task { await FileSnapshotStore.shared.reset() }
        if let id, let proj = projects.first(where: { $0.id == id }), let path = proj.rootDirectoryPath, !path.isEmpty {
            if let existing = worktree {
                existing.updateRoot(path: path)
            } else {
                worktree = WorktreeModel(rootDirectoryPath: path)
            }
        } else {
            worktree = nil
        }
        persistProjects()
        reloadSkills()
        reloadAgents()
        AppHookStore.shared.refresh(projectDirectory: selectedProject?.rootDirectoryPath)

        // If the currently selected chat doesn't belong to the newly selected project, switch selection
        if let id {
            let matching = chats.filter { $0.projectID == id }
            if let first = matching.first {
                selectChat(id: first.id)
            } else {
                createChat(projectID: id)
            }
        }
    }

    /// Updates an existing project and persists changes.
    public func updateProject(_ project: AppProject) {
        guard let index = projects.firstIndex(where: { $0.id == project.id }) else { return }
        var updated = project
        updated.updatedAt = Date()
        projects[index] = updated
        persistProjects()
        reloadSkills()
        reloadAgents()
        if selectedProjectID == project.id {
            AppHookStore.shared.refresh(projectDirectory: updated.rootDirectoryPath)
        }
    }

    /// Deletes a project and optionally clears project references from its chats.
    ///
    /// **DELETING MUST UNDO EVERYTHING SELECTING DID** (state#26). `createProject`,
    /// `selectProject` and `updateProject` all rebind `AppHookStore` to the
    /// project directory, and `selectProject` additionally clears `worktree`
    /// and resets `FileSnapshotStore`. This did none of it, and
    /// `AppHookStore` only rebuilds on refresh -- so after deleting a project
    /// the git pane kept rendering the deleted repository and its
    /// `PreToolUse` hooks kept running for every later tool call, in a
    /// workspace the user had just removed.
    public func deleteProject(id: UUID) {
        guard !generating else { return }
        let wasSelected = selectedProjectID == id
        projects.removeAll { $0.id == id }
        if wasSelected {
            selectedProjectID = nil
        }
        for index in chats.indices where chats[index].projectID == id {
            chats[index].projectID = nil
        }
        if wasSelected {
            worktree = nil
            // Stale-write hashes are per workspace; see `selectProject`.
            Task { await FileSnapshotStore.shared.reset() }
        }
        persistProjects()
        persistChats()
        reloadSkills()
        reloadAgents()
        if wasSelected {
            AppHookStore.shared.refresh(projectDirectory: selectedProject?.rootDirectoryPath)
        }
    }

    /// Scans a local codebase directory for AGENTS.md, CLAUDE.md, or rules files according to preference.
    public func detectProjectRules(
        directoryPath: String,
        preference: AppRulePreference = .agentsFirst
    ) -> String? {
        ProjectRuleDetector.detectRules(in: directoryPath, preference: preference)?.content
    }

    /// Scans a local codebase directory and returns full detection details (conflicts, symlinks, status).
    public func detectProjectRulesDetails(
        directoryPath: String,
        preference: AppRulePreference = .agentsFirst
    ) -> ProjectRulesDetectionResult? {
        ProjectRuleDetector.detectRules(in: directoryPath, preference: preference)
    }
}
