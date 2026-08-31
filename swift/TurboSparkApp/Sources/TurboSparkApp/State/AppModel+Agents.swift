import Foundation
import TurboSpark

extension AppModel {
    /// Combined list of active agents, with project-level agents taking precedence over user and built-in agents with matching names.
    public var effectiveAgents: [AppAgentDefinition] {
        AgentManager.shared.resolveEffectiveAgents(projectURL: selectedProject?.rootDirectoryURL)
            .filter { $0.isEnabled }
    }

    /// All discovered agents (built-in, user, project) for inspection and configuration.
    public var allManagedAgents: [AppAgentDefinition] {
        AgentManager.shared.resolveEffectiveAgents(projectURL: selectedProject?.rootDirectoryURL)
    }

    /// Reloads agents from built-in and on-disk definitions.
    public func reloadAgents() {
        discoveredAgents = allManagedAgents
    }

    /// Toggles and persists the enabled state of an agent.
    public func toggleAgentEnabled(_ agent: AppAgentDefinition) {
        AgentManager.shared.setAgentEnabled(!agent.isEnabled, name: agent.name)
        reloadAgents()
    }

    /// Finds an agent by name.
    public func findAgent(named name: String) -> AppAgentDefinition? {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allManagedAgents.first { $0.name.lowercased() == clean }
    }

    /// Handles REPL slash commands for direct agent invocations (e.g. `/explore <query>`, `/plan <task>`, `/agent <name> <task>`).
    /// Returns true if a command was recognized and handled.
    public func handleAgentSlashCommand(_ input: String) -> Bool {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("/") else { return false }

        let parts = trimmed.dropFirst().components(separatedBy: " ")
        guard let command = parts.first?.lowercased(), !command.isEmpty else { return false }
        let rest = parts.dropFirst().joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)

        var targetAgentName: String?
        var taskPrompt: String = ""

        switch command {
        case "explore":
            targetAgentName = "explore"
            taskPrompt = rest
        case "plan":
            targetAgentName = "plan"
            taskPrompt = rest
        case "review", "reviewer":
            targetAgentName = "reviewer"
            taskPrompt = rest
        case "agent":
            let subParts = rest.components(separatedBy: " ")
            if let first = subParts.first, !first.isEmpty {
                targetAgentName = first
                taskPrompt = subParts.dropFirst().joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
            }
        default:
            if let agent = findAgent(named: command) {
                targetAgentName = agent.name
                taskPrompt = rest
            }
        }

        guard let agentName = targetAgentName, let agent = findAgent(named: agentName) else {
            return false
        }

        guard !taskPrompt.isEmpty else {
            showToast("Please provide a prompt for agent '\(agent.displayName)'.", style: .warning)
            return true
        }

        runAgentTaskDirectly(agent: agent, prompt: taskPrompt)
        return true
    }

    /// Executes an isolated subagent task directly from user REPL command.
    public func runAgentTaskDirectly(agent: AppAgentDefinition, prompt: String) {
        guard session != nil else {
            showToast("No active model session. Please load a model first.", style: .error)
            return
        }

        let chatIndex: Int
        if let existing = selectedChatIndex {
            chatIndex = existing
        } else {
            let newChat = AppChat(id: selectedChatID, projectID: selectedProjectID)
            chats.insert(newChat, at: 0)
            chatIndex = 0
            selectedChatID = newChat.id
        }

        // Clear composer draft
        chats[chatIndex].draft = ""
        chats[chatIndex].draftAttachments = []
        chats[chatIndex].updatedAt = Date()

        // Append user turn reflecting the agent command
        let userTurn = AppChatMessage(
            role: .user,
            content: "[Agent: \(agent.displayName)] \(prompt)"
        )
        chats[chatIndex].messages.append(userTurn)
        persistChats()

        generating = true
        phase = .prefill
        outputText = "Running isolated subagent [\(agent.displayName)]...\n"

        Task { @MainActor in
            let result = await SubagentRunner.run(
                agent: agent,
                taskPrompt: prompt,
                session: self.session,
                project: self.selectedProject
            )

            self.generating = false
            self.phase = .idle
            self.outputText = ""

            let assistantContent = """
            ### Subagent: \(agent.displayName)
            \(result.finalResponse)

            *(Completed in \(result.totalTurns) turn(s), \(result.totalToolCalls) tool call(s), \(String(format: "%.2f", result.durationSeconds))s)*
            """

            let assistantTurn = AppChatMessage(
                role: .assistant,
                content: assistantContent,
                stopReason: result.status
            )

            if let idx = self.chats.firstIndex(where: { $0.id == self.selectedChatID }) {
                self.chats[idx].messages.append(assistantTurn)
                self.chats[idx].updatedAt = Date()
                self.persistChats()
            }
        }
    }
}
