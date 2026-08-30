import Foundation

extension AppModel {
    /// Filtered list of chats according to the active project selection.
    public var filteredChats: [AppChat] {
        guard let projectID = selectedProjectID else {
            return chats
        }
        return chats.filter { $0.projectID == projectID }
    }

    /// Creates and persists a new codebase project.
    @discardableResult
    public func createProject(
        name: String,
        rootDirectoryPath: String? = nil,
        agentType: AppAgentType = .coder,
        rulePreference: AppRulePreference = .agentsFirst,
        customInstructions: String = "",
        permissions: AppProjectPermissions = .standard,
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
        persistProjects()
        reloadSkills()

        // Create initial chat for this project
        createChat(projectID: project.id)
        return project
    }

    /// Selects an active project or clears the project filter.
    public func selectProject(id: UUID?) {
        guard !generating else { return }
        selectedProjectID = id
        persistProjects()
        reloadSkills()

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
    }

    /// Deletes a project and optionally clears project references from its chats.
    public func deleteProject(id: UUID) {
        guard !generating else { return }
        projects.removeAll { $0.id == id }
        if selectedProjectID == id {
            selectedProjectID = nil
        }
        for index in chats.indices where chats[index].projectID == id {
            chats[index].projectID = nil
        }
        persistProjects()
        persistChats()
        reloadSkills()
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
