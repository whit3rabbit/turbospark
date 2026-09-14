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

    /// `/memory` opens the profile store. `/memory text` appends a dated
    /// profile memory, and `/memory search text` performs a bounded search.
    func handleMemoryCommand(
        _ draft: String = "/memory",
        opener: (URL) -> Void = { NSWorkspace.shared.open($0) }
    ) {
        guard MemoryStore.shared.isModelEnabled else {
            showToast("Memory is disabled in Settings.", style: .warning)
            return
        }
        let argument = draft.dropFirst("/memory".count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if argument.lowercased().hasPrefix("search ") {
            let query = String(argument.dropFirst("search ".count))
            let results = ProfileMemoryStore.shared.lexicalSearch(query)
            showToast(results.isEmpty ? "No matching profile memories." : results.joined(separator: "\n"), style: .info)
            return
        }
        if !argument.isEmpty {
            handleProfileMemorySave(argument)
            return
        }
        opener(ProfileMemoryStore.shared.directory)
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
        guard MemoryStore.shared.isModelEnabled else {
            showToast("Memory is disabled in Settings.", style: .warning)
            return
        }
        let text = UserMemoryInputMessage.memoryText(of: trimmedDraft)
        do {
            if let root = memoryProject()?.rootDirectoryURL, !root.path.isEmpty {
                let firstLine = SkillParser.normalizedLines(text).first { !$0.trimmingCharacters(in: .whitespaces).isEmpty } ?? text
                let hook = String(firstLine.trimmingCharacters(in: .whitespaces).prefix(120))
                _ = try MemoryStore.shared.saveTopic(
                    projectRoot: root,
                    name: MemoryStore.slug(from: text, dated: true),
                    type: .user,
                    description: hook,
                    body: text)
            }
            _ = try ProfileMemoryStore.shared.append(text)
            Task { try? await ProfileMemoryStore.shared.rebuildIndex(modelPath: memoryEmbeddingModel) }
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
            _ = try ProfileMemoryStore.shared.append(text)
            Task { try? await ProfileMemoryStore.shared.rebuildIndex(modelPath: memoryEmbeddingModel) }
            promptText = ""
            showToast("Saved to profile memory.", style: .success)
        } catch {
            showToast("Could not save profile memory: \(error.localizedDescription)", style: .error)
        }
    }
}
