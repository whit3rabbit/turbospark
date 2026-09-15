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
    ///
    /// This is the explicit re-scan, so it drops the cache first: resolution
    /// is now memoized per project root (13 recursive directory walks on the
    /// main actor otherwise, once per lookup), and a reload that read the
    /// cache back would never see a file the user just added.
    public func reloadAgents() {
        AgentManager.shared.invalidateResolutionCache()
        discoveredAgents = allManagedAgents
    }

    /// Project agent files held to a built-in's tool ceiling for taking its
    /// name. They still override the prompt; they cannot widen what it may do.
    public var constrainedProjectAgentNames: [String] {
        AgentManager.shared.constrainedProjectAgentNames(
            projectURL: selectedProject?.rootDirectoryURL)
    }

    /// User agents a PROJECT agent is currently overriding (state#105). The
    /// agent-side twin of `shadowedUserSkillNames`.
    public var shadowedUserAgentNames: [String] {
        AgentManager.shared.shadowedUserAgentNames(
            projectURL: selectedProject?.rootDirectoryURL)
    }

    /// Toggles and persists the enabled state of an agent.
    public func toggleAgentEnabled(_ agent: AppAgentDefinition) {
        // Scoped, so toggling a project agent does not also toggle the
        // built-in of that name (state#57).
        AgentManager.shared.setAgentEnabled(
            !agent.isEnabled, name: agent.name, scope: agent.scope)
        reloadAgents()
    }

    // MARK: - Agent File Management (Settings editor)

    /// Creates and writes a new agent file in user or project scope.
    /// A project scope without an open project is refused by the manager.
    @discardableResult
    public func createAgent(
        name: String,
        displayName: String?,
        description: String,
        systemPrompt: String,
        tools: [String]?,
        disallowedTools: [String]?,
        maxTurns: Int,
        scope: AppAgentScope
    ) -> Result<AppAgentDefinition, Error> {
        do {
            let agent = try AgentManager.shared.createAgent(
                name: name,
                displayName: displayName,
                description: description,
                systemPrompt: systemPrompt,
                tools: tools,
                disallowedTools: disallowedTools,
                maxTurns: maxTurns,
                scope: scope,
                projectRootURL: selectedProject?.rootDirectoryURL)
            reloadAgents()
            showToast("Agent '\(agent.name)' created in \(scope.label) scope.", style: .info)
            return .success(agent)
        } catch {
            showToast("Failed to create agent: \(error.localizedDescription)", style: .error)
            return .failure(error)
        }
    }

    /// Updates and persists an existing agent on disk.
    public func updateAgent(_ agent: AppAgentDefinition) {
        do {
            try AgentManager.shared.saveAgent(agent)
            reloadAgents()
            showToast("Saved changes to '\(agent.name)'.", style: .info)
        } catch {
            showToast("Failed to save agent: \(error.localizedDescription)", style: .error)
        }
    }

    /// Deletes an agent's file from disk.
    public func deleteAgent(_ agent: AppAgentDefinition) {
        do {
            try AgentManager.shared.deleteAgent(agent)
            reloadAgents()
            showToast("Removed agent '\(agent.name)'.", style: .info)
        } catch {
            showToast("Failed to delete agent: \(error.localizedDescription)", style: .error)
        }
    }

    /// Finds an ENABLED agent by name.
    ///
    /// **DISABLED MEANS DISABLED ON BOTH PATHS** (state#93). The `agent` TOOL
    /// checks `agentDef.isEnabled` and refuses; this resolved against
    /// `allManagedAgents`, which is the inspection list and deliberately
    /// includes the switched-off ones -- so `/explore` ran an agent the user
    /// had turned off, with its full prompt and its full tool set. Same shape
    /// as state#12 on the skill side, where the flag was persisted and one
    /// caller ignored it.
    public func findAgent(named name: String) -> AppAgentDefinition? {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allManagedAgents.first { $0.name.lowercased() == clean && $0.isEnabled }
    }

    /// The same lookup WITHOUT the enabled filter, for telling "no such
    /// agent" apart from "that one is switched off" (state#93). A slash
    /// command that silently does nothing reads as a broken command.
    func findAnyAgent(named name: String) -> AppAgentDefinition? {
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

        // The command names and their fixed agents live in the shared
        // `BuiltInSlashCommand` table, which the menu and the popup read too.
        // A table row the parser lost (or a case this dispatch never
        // recorded) reddens the registry drift test rather than offering a
        // command that falls through as prose.
        if let builtin = BuiltInSlashCommand.matching(command) {
            if let fixed = builtin.fixedAgentName {
                targetAgentName = fixed
                taskPrompt = rest
            } else {
                // `/agent <name> <task>`: the target is the command's own
                // first argument.
                let subParts = rest.components(separatedBy: " ")
                if let first = subParts.first, !first.isEmpty {
                    targetAgentName = first
                    taskPrompt = subParts.dropFirst().joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        } else {
            // `findAnyAgent`, so a bare `/name` naming a SWITCHED-OFF agent is
            // recognized here and refused by name below, rather than falling
            // through as "not a command" and being sent to the model as prose
            // (state#93).
            if let agent = findAnyAgent(named: command) {
                targetAgentName = agent.name
                taskPrompt = rest
            }
        }

        guard let agentName = targetAgentName else { return false }
        guard let agent = findAgent(named: agentName) else {
            // Recognized, and refused with a reason rather than falling
            // through to be sent to the model as prose (state#93).
            if let disabled = findAnyAgent(named: agentName) {
                showToast(
                    "Agent '\(disabled.displayName)' is switched off. Enable it in Settings > "
                        + "Agents to use it.", style: .warning)
                return true
            }
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
    ///
    /// **RUNS UNDER THE SAME LIFECYCLE AS AN ORDINARY TURN.** It used to set
    /// `generating` by hand with no `!generating` guard, no epoch bump, and
    /// its `Task` stored nowhere -- so it could start beside a running turn,
    /// `cancel()` cancelled a nil `runTask` while leaving
    /// `isCancellationPending` set (greying Stop out permanently), and the
    /// chat was resolved at COMPLETION rather than captured, landing the
    /// result in whatever the user had switched to.
    public func runAgentTaskDirectly(agent: AppAgentDefinition, prompt: String) {
        guard session != nil else {
            showToast("No active model session. Please load a model first.", style: .error)
            return
        }
        guard !generating else {
            showToast("A turn is already running. Wait for it to finish first.", style: .warning)
            return
        }

        // Captured BEFORE anything can move the selection, the way `run()`
        // captures `submissionChatID` (state#64). Nothing here awaits today,
        // so resolving `selectedChatIndex` twice happens to agree -- but the
        // two reads are what the next `await` inserted between them would
        // break, and this function already learned that lesson once for its
        // completion path.
        let submissionChatID = selectedChatID
        let chatIndex: Int
        if let existing = chats.firstIndex(where: { $0.id == submissionChatID }) {
            chatIndex = existing
        } else {
            let newChat = AppChat(id: submissionChatID, projectID: selectedProjectID)
            chats.insert(newChat, at: 0)
            chatIndex = 0
            selectedChatID = newChat.id
        }

        // Clear composer draft
        if chats[chatIndex].isGhost {
            mutateGhostPayload(for: submissionChatID) { $0.draft = "" }
        } else {
            chats[chatIndex].draft = ""
        }
        chats[chatIndex].draftAttachments = []
        chats[chatIndex].updatedAt = Date()

        // Append user turn reflecting the agent command
        let userTurn = AppChatMessage(
            role: .user,
            content: "[Agent: \(agent.displayName)] \(prompt)"
        )
        mutateTurnMessages(for: submissionChatID) { $0.append(userTurn) }

        // Captured before the run, like every other turn: a subagent run is
        // seconds to minutes of work and the user is free to click away.
        let turnChatID = submissionChatID
        let project = turnProject(chatID: submissionChatID) ?? selectedProject

        generationEpoch += 1
        let myEpoch = generationEpoch
        generating = true
        isCancellationPending = false
        phase = .prefill
        outputText = "Running isolated subagent [\(agent.displayName)]...\n"

        // Same live progress surface an `agent` TOOL call gets: the sink
        // routes into `applySubagentEvent`, which creates the card's state
        // on the runner's `.started` event and drops it on `.finished`.
        let runKey = UUID().uuidString
        let sink = AppToolRegistry.subagentProgressSink

        runTask = Task { @MainActor in
            defer {
                // Guarded for the same reason an ordinary turn's tail is: a
                // newer turn may already have claimed this state.
                if self.generationEpoch == myEpoch {
                    self.generating = false
                    self.phase = .idle
                    self.isCancellationPending = false
                    self.outputText = ""
                    self.runTask = nil
                }
                self.liveSubagentRuns.removeValue(forKey: runKey)
            }
            let result = await SubagentRunner.run(
                agent: agent,
                taskPrompt: prompt,
                session: self.session,
                project: project,
                chatID: turnChatID,
                // The app-wide default, global SOUL, and selected personality,
                // never a per-chat override: a subagent runs in a fresh context.
                userSystemPrompt: self.appWideSystemPrompt,
                samplingOptions: self.samplingOptions(),
                progress: { event in await sink?(runKey, event) }
            )

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

            self.mutateTurnMessages(for: turnChatID) { $0.append(assistantTurn) }
        }
    }
}
