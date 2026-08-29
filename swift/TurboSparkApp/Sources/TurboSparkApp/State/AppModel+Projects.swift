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
        customInstructions: String = "",
        permissions: AppProjectPermissions = .standard,
        maxAutonomousSteps: Int = 5
    ) -> AppProject {
        var instructions = customInstructions
        if instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let path = rootDirectoryPath,
           let autoRules = detectProjectRules(directoryPath: path) {
            instructions = autoRules
        }

        let project = AppProject(
            name: name,
            rootDirectoryPath: rootDirectoryPath,
            agentType: agentType,
            customInstructions: instructions,
            permissions: permissions,
            maxAutonomousSteps: maxAutonomousSteps
        )

        projects.insert(project, at: 0)
        selectedProjectID = project.id
        persistProjects()

        // Create initial chat for this project
        createChat(projectID: project.id)
        return project
    }

    /// Selects an active project or clears the project filter.
    public func selectProject(id: UUID?) {
        guard !generating else { return }
        selectedProjectID = id
        persistProjects()

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
    }

    /// Scans a local codebase directory for AGENTS.md, CLAUDE.md, or rules files.
    public func detectProjectRules(directoryPath: String) -> String? {
        let candidates = ["AGENTS.md", "CLAUDE.md", ".rules", "RULES.md"]
        let rootURL = URL(fileURLWithPath: directoryPath, isDirectory: true)

        for candidate in candidates {
            let fileURL = rootURL.appendingPathComponent(candidate)
            if FileManager.default.fileExists(atPath: fileURL.path),
               let content = try? String(contentsOf: fileURL, encoding: .utf8) {
                let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return String(trimmed.prefix(6000))
                }
            }
        }
        return nil
    }
}
