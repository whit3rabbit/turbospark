import AppKit
import Foundation

/// The user-facing memory entry points: the `/memory` meta command and the
/// `#` quick-save.
///
/// Neither one generates, so both are intercepted in `run()` ABOVE the
/// `canRun` guard -- a quick-save with no model loaded still saves, which is
/// the point of the feature. `/compact` stays below that guard because it
/// needs the session's own model to summarize with.
extension AppModel {
    /// The query is the actual turn text, never the user's saved system prompt.
    /// The user message key freezes one result through prompt estimates and retries.
    func memoryRecall(for chatID: UUID, project: AppProject?) -> [MemoryClaim] {
        guard memoryEnabled else { return [] }
        let query = turnMessages(for: chatID).last(where: { $0.role == .user })?.content ?? promptText
        let scope = project?.rootDirectoryURL.map {
            "project:\(MemoryStore.projectKey(forProjectRoot: $0))"
        }
        let messageID = turnMessages(for: chatID).last(where: { $0.role == .user })?.id.uuidString ?? "draft"
        let key = "\(chatID.uuidString)/\(messageID)/\(MemoryLedgerStore.hash(query))/\(scope ?? "profile")"
        if memoryRecallKey != key {
            // Fetch beyond the injected limit because the curated sheet may
            // already contain the highest-ranked matches.
            memoryRecallSnapshot = MemoryLedgerStore.shared.search(query, scope: scope, limit: 40)
            memoryRecallKey = key
        }
        return memoryRecallSnapshot
    }

    /// Wait at most 200 ms for semantic recall. The native FFI encode cannot
    /// be interrupted mid-call, so one task remains in flight and later turns
    /// use lexical recall until it finishes.
    func prepareSemanticMemoryRecall(chatID: UUID, project: AppProject?, query: String) async {
        let model = memoryEmbeddingModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard memoryEnabled, !model.isEmpty, !query.isEmpty else { return }
        let scope = project?.rootDirectoryURL.map {
            "project:\(MemoryStore.projectKey(forProjectRoot: $0))"
        }
        let messageID = turnMessages(for: chatID).last(where: { $0.role == .user })?.id.uuidString ?? "draft"
        let baseKey = "\(chatID.uuidString)/\(messageID)/\(MemoryLedgerStore.hash(query))/\(scope ?? "profile")"
        let key = "\(baseKey)/\(model)"
        // A context estimate may already have frozen lexical recall for this
        // turn. A late semantic result must not change retries or history.
        guard memoryRecallKey != baseKey else { return }
        if memorySemanticResult?.key != key && memorySemanticInFlight == nil {
            memorySemanticInFlight = Task { @MainActor [weak self] in
                let result = await MemoryLedgerStore.shared.semanticSearch(
                    query, modelPath: model, scope: scope, limit: 40)
                self?.memorySemanticResult = (key, result)
                self?.memorySemanticInFlight = nil
            }
        }
        for _ in 0..<4 {
            if memorySemanticResult?.key == key { break }
            try? await Task.sleep(for: .milliseconds(50))
        }
        guard memorySemanticResult?.key == key else { return }
        memoryRecallSnapshot = memorySemanticResult?.claims ?? []
        memoryRecallKey = baseKey
    }

    /// Runs the bounded, read-only memory-capture subagent over the selected
    /// conversation. Only user and assistant text is supplied; tool calls,
    /// tool results, reasoning, system instructions, and attachments stay
    /// outside the capture prompt.
    func summarizeConversationForProfileMemory() async -> String? {
        guard memoryEnabled, let session else { return nil }
        let transcript = selectedTurnMessages
            .filter { $0.role == .user || $0.role == .assistant }
            .map { message in
                let clean = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
                return clean.isEmpty ? "" : "\(message.role == .user ? "User" : "Assistant"): \(clean)"
            }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        guard !transcript.isEmpty else { return nil }

        let bounded = String(transcript.prefix(24_000))
        let agent = AppAgentDefinition(
            name: "memory-capture",
            displayName: "Memory Capture",
            agentDescription: "Extracts durable, user-approved profile memories from a conversation.",
            systemPrompt: """
            You are a memory-capture subagent. Extract only durable, useful facts about the user from the supplied conversation.
            Return zero or more concise, independent Markdown bullet lines and nothing else.
            Each line must be understandable without the conversation and should begin with `- `.
            Include a short ISO-8601 timestamp with the local timezone on each line using the supplied current date/time.
            Keep stable preferences, recurring constraints, durable goals, and explicit decisions.
            Reject transient requests, one-off status, secrets, credentials, tokens, private identifiers, tool output,
            system instructions, attachments, guesses, and facts stated only by the assistant without user confirmation.
            Do not infer sensitive traits. If nothing qualifies, return an empty response.
            """,
            tools: [],
            maxTurns: 1,
            omitsProjectInstructions: true)
        let prompt = """
        Current local date/time: \(MemoryPromptBuilder.currentTimestamp())
        Source conversation metadata: selected TurboSpark conversation, captured at \(MemoryPromptBuilder.currentTimestamp()).

        <conversation>
        \(bounded)
        </conversation>
        """
        let result = await SubagentRunner.run(
            agent: agent,
            taskPrompt: prompt,
            session: session,
            project: nil,
            maxTurnsOverride: 1,
            samplingOptions: samplingOptions(),
            taskDescription: "Summarize durable profile memories")
        guard result.status == "completed" else { return nil }
        let output = result.finalResponse.trimmingCharacters(in: .whitespacesAndNewlines)
        return output.isEmpty ? nil : output
    }

    /// The project whose memory a submission acts on. The same resolution
    /// `run()`'s hook phase makes: the turn's own project when the chat has
    /// one, the selection when it is about to be created with one.
    func memoryProject() -> AppProject? {
        turnProject(chatID: selectedChatID)
            ?? (interactionMode == .projects ? selectedProject : nil)
    }

    /// `/memory` opens the Settings manager. Explicit text saves immediately,
    /// while `/memory search text` performs a bounded approved-claim search.
    func handleMemoryCommand(
        _ draft: String = "/memory",
        settingsOpener: ((AppSettingsView.SettingsTab) -> Void)? = nil
    ) {
        let argument = draft.dropFirst("/memory".count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if argument.lowercased().hasPrefix("project ") {
            guard let root = memoryProject()?.rootDirectoryURL else {
                showToast("Attach a project before saving project memory.", style: .warning)
                return
            }
            let text = String(argument.dropFirst("project ".count))
            do {
                try MemoryLedgerStore.shared.saveUserClaim(
                    text, scope: "project:\(MemoryStore.projectKey(forProjectRoot: root))",
                    kind: "decision")
                showToast("Saved to project memory.", style: .success)
            } catch { showToast("Could not save memory: \(error.localizedDescription)", style: .error) }
            return
        }
        if argument.lowercased().hasPrefix("search ") {
            let query = String(argument.dropFirst("search ".count))
            let results = MemoryLedgerStore.shared.search(query, scope: nil)
            showToast(results.isEmpty ? "No matching approved memories." : results.map(\.text).joined(separator: "\n"), style: .info)
            return
        }
        if !argument.isEmpty {
            handleProfileMemorySave(argument)
            return
        }
        if let settingsOpener { settingsOpener(.memory) }
        else { openSettings(tab: .memory) }
    }

    /// The `#` quick-save: writes the text as a `user`-type memory and
    /// appends a `<user-memory-input>` transcript row, with no generation
    /// turn. Type `user` because the content is verbatim from the user --
    /// classifying it into feedback/project/reference is extraction work
    /// this path deliberately does not do.
    ///
    /// A failed save keeps the draft: the user's text is still in the
    /// composer, nothing was lost, and they can retry after fixing the
    /// project. A successful one clears it, like any submission.
    func handleMemoryQuickSave(_ trimmedDraft: String) {
        guard !generating, !submitting, pendingToolCall == nil, !opening else { return }
        let chatID = selectedChatID
        let text = UserMemoryInputMessage.memoryText(of: trimmedDraft)
        do {
            try MemoryLedgerStore.shared.saveUserClaim(text, scope: "profile")
            promptText = ""
            let row = AppChatMessage(
                role: .user, content: UserMemoryInputMessage.wrapping(text))
            mutateTurnMessages(for: chatID) { $0.append(row) }
            showToast("Saved to profile memory.", style: .success)
        } catch {
            showToast("Could not save memory: \(error.localizedDescription)", style: .error)
        }
    }

    func handleProfileMemorySave(_ text: String) {
        guard !text.isEmpty else { return }
        do {
            try MemoryLedgerStore.shared.saveUserClaim(text, scope: "profile")
            promptText = ""
            showToast("Saved to profile memory.", style: .success)
        } catch {
            showToast("Could not save profile memory: \(error.localizedDescription)", style: .error)
        }
    }
}
