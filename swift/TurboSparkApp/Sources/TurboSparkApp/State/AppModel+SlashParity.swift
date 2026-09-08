import AppKit
import Foundation
import TurboSpark

/// The qwen-code local-command parity layer: `/copy`, `/model`, `/theme`,
/// `/tasks`, `/btw`, `/fork`, `/rewind` and friends. Dispatched from
/// `run()`'s session-less band (`BuiltInSlashCommand.isLocalCommand`), so a
/// local command never generates, never reaches a hook, and -- unlike a
/// queued prompt -- works WHILE a turn is running, which is the whole point
/// of `/btw` and `/tasks`.
///
/// Each handler enforces its own preconditions and answers through a toast:
/// there is no transcript row for a local command (qwen-code suppresses the
/// echo for the same reason -- a local row would split the running turn).
extension AppModel {
    public func handleLocalCommand(_ draft: String) {
        let words = draft.dropFirst().split(separator: " ").map { String($0) }
        let name = words.first?.lowercased() ?? ""
        let arguments = words.dropFirst()
        switch name {
        case "copy":
            runCopyCommand(in: selectedTurnMessages.last {
                $0.role == .assistant && !$0.content.isEmpty
            }?.content ?? "")
        case "model":
            runModelCommand(arguments.map { String($0) })
        case "context":
            showContextSheet = true
        case "tasks":
            showTasksSheet = true
        case "tools":
            showToolsSheet = true
        case "status":
            showStatusSheet = true
        case "mcp":
            openSettings(tab: .mcp)
        case "agents":
            openSettings(tab: .agents)
        case "skills":
            openSettings(tab: .skills)
        case "settings":
            openSettings(tab: .general)
        case "schedule":
            openSettings(tab: .automation)
        case "theme":
            runThemeCommand(arguments.first.map { String($0) })
        case "language":
            runLanguageCommand(arguments.first.map { String($0) })
        case "approval-mode":
            runApprovalModeCommand(arguments.first.map { String($0) })
        case "new":
            createChat(projectID: selectedProjectID)
        case "clear":
            guard !generating, !submitting else {
                showToast("Stop the current turn before clearing.", style: .warning)
                return
            }
            clearOutput()
        case "rename":
            runRenameCommand(arguments.joined(separator: " "))
        case "delete":
            // Never destructive without the same confirmation the sidebar's
            // Delete action shows.
            confirmDeleteChat = true
        case "archive":
            setChatArchived(id: selectedChatID, archived: true)
        case "unarchive":
            setChatArchived(id: selectedChatID, archived: false)
        case "rewind":
            showRewindSheet = true
        case "branch":
            runBranchCommand(arguments.joined(separator: " "))
        case "btw":
            runBtwCommand(arguments.joined(separator: " "))
        case "fork":
            runForkCommand(arguments.joined(separator: " "))
        case "recap":
            runRecapCommand()
        default:
            break
        }
    }

    // MARK: - /copy

    private func runCopyCommand(in message: String) {
        let outcome = CopyCommandParser.resolve(draft: promptText, in: message)
        switch outcome {
        case .copied(let text):
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            showToast("Copied to the clipboard.", style: .success)
        case .nothingFound(let what):
            showToast("Nothing to copy: \(what).", style: .info)
        case .indexOutOfRange(let requested, let available, let noun):
            showToast(
                "No \(noun) number \(requested): the last answer has \(available).",
                style: .warning)
        case .notACopyCommand:
            break
        }
    }

    // MARK: - /model

    private func runModelCommand(_ arguments: [String]) {
        let query = arguments.joined(separator: " ").lowercased()
        guard !query.isEmpty else {
            showToast(
                "Loaded model: \(selected?.alias ?? "none"). Give a name to switch, "
                    + "e.g. /model \(installed.first?.alias ?? "gemma4").",
                style: .info)
            return
        }
        let matches = installed.filter { $0.alias.lowercased().contains(query) }
        switch matches.count {
        case 0:
            showToast("No installed model matches '\(query)'.", style: .warning)
        case 1:
            selectModel(matches[0])
            showToast("Switched to \(matches[0].alias).", style: .success)
        default:
            let names = matches.prefix(4).map(\.alias).joined(separator: ", ")
            showToast(
                "'\(query)' matches \(matches.count) models: \(names). Be more specific.",
                style: .info)
        }
    }

    // MARK: - /theme, /language, /approval-mode

    private func runThemeCommand(_ argument: String?) {
        guard let argument, !argument.isEmpty else {
            showToast(
                "Appearance is \(AppearanceManager.shared.appearance.rawValue). "
                    + "Use /theme light, /theme dark or /theme system.",
                style: .info)
            return
        }
        guard let appearance = AppAppearance(rawValue: argument.lowercased()) else {
            showToast("Unknown theme '\(argument)': use light, dark or system.", style: .warning)
            return
        }
        AppearanceManager.shared.appearance = appearance
        showToast("Appearance set to \(appearance.rawValue).", style: .success)
    }

    private func runLanguageCommand(_ argument: String?) {
        guard let argument, !argument.isEmpty else {
            let current = UserDefaults.standard.string(forKey: AppLanguage.storageKey)
                ?? AppLanguage.system.rawValue
            showToast(
                "UI language is \(current). Give a code, e.g. /language de or /language zh-Hans.",
                style: .info)
            return
        }
        let query = argument.lowercased()
        let match = AppLanguage.allCases.first {
            $0.rawValue.lowercased() == query || $0.rawValue.lowercased().hasPrefix(query)
                || $0.label.lowercased() == query
        }
        guard let match else {
            let codes = AppLanguage.allCases.map(\.rawValue).joined(separator: ", ")
            showToast("Unknown language '\(argument)'. Codes: \(codes).", style: .warning)
            return
        }
        UserDefaults.standard.set(match.rawValue, forKey: AppLanguage.storageKey)
        showToast("UI language set to \(match.label).", style: .success)
    }

    private func runApprovalModeCommand(_ argument: String?) {
        guard let argument, !argument.isEmpty else {
            showToast(
                "Permission mode is \(effectivePermissionMode.rawValue). Use /approval-mode "
                    + AppPermissionMode.allCases.map(\.rawValue).joined(separator: "|") + ".",
                style: .info)
            return
        }
        let query = argument.lowercased().replacingOccurrences(of: "_", with: "-")
        let match = AppPermissionMode.allCases.first {
            $0.rawValue.lowercased().contains(query) || $0.label.lowercased().contains(query)
        }
        guard let match else {
            showToast(
                "Unknown mode '\(argument)': use "
                    + AppPermissionMode.allCases.map(\.rawValue).joined(separator: ", ") + ".",
                style: .warning)
            return
        }
        setEffectivePermissionMode(match)
        showToast("Permission mode set to \(match.label).", style: .success)
    }

    // MARK: - /rename

    private func runRenameCommand(_ title: String) {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            showToast("Give the new title: /rename My Topic.", style: .info)
            return
        }
        renameChat(id: selectedChatID, title: trimmed)
    }

    // MARK: - Archive / duplicate (qwen-code session actions)

    /// Archives or restores a chat. Ghost chats are refused: an archive is
    /// a durable hiding place, and nothing about a ghost may outlive the
    /// process.
    public func setChatArchived(id: UUID, archived: Bool) {
        guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
        if chats[index].isGhost {
            showToast("A temporary chat cannot be archived; end Ghost Mode instead.", style: .info)
            return
        }
        guard chats[index].isArchived != archived else { return }
        chats[index].isArchived = archived
        persistChats()
        showToast(archived ? "Chat archived." : "Chat restored.", style: .success)
    }

    /// Duplicates the selected chat under fresh identities and selects the
    /// copy. Refused mid-run, like every other selection-changing action.
    @discardableResult
    public func duplicateChat(id: UUID) -> UUID? {
        guard !generating, !submitting, pendingToolCall == nil,
            let index = chats.firstIndex(where: { $0.id == id }),
            !chats[index].isGhost
        else {
            if chats.first(where: { $0.id == id })?.isGhost == true {
                showToast("A temporary chat cannot be duplicated.", style: .info)
            }
            return nil
        }
        let copy = chats[index].duplicated()
        chats.insert(copy, at: 0)
        selectedChatID = copy.id
        persistChats()
        showToast("Duplicated as '\(copy.title)'.", style: .success)
        return copy.id
    }

    /// The archived chats, what the sidebar's Archived section renders.
    public var archivedChats: [AppChat] {
        AppChat.sortedForSidebar(chats.filter { $0.isArchived && !$0.isGhost })
    }

    // MARK: - Transcript turn collapse

    /// Whether any turn in the selected chat has rows that a collapse would
    /// hide, what the context-menu command's enabled state reads.
    public var canCollapseTurns: Bool {
        let byID = Dictionary(uniqueKeysWithValues: selectedTurnMessages.map { ($0.id, $0) })
        return TurnCollapseModel.turns(in: selectedTurnMessages).contains { turn in
            TurnCollapseModel.hiddenCount(for: turn, messagesByID: byID) > 0
        }
    }

    /// Collapses every collapsible turn at once (the context menu's
    /// "Collapse finished turns").
    public func collapseAllTurns() {
        let byID = Dictionary(uniqueKeysWithValues: selectedTurnMessages.map { ($0.id, $0) })
        var anchors: Set<UUID> = []
        for turn in TurnCollapseModel.turns(in: selectedTurnMessages) {
            if turn.isCollapsible,
                TurnCollapseModel.hiddenCount(for: turn, messagesByID: byID) > 0,
                let anchor = turn.anchorID
            {
                anchors.insert(anchor)
            }
        }
        collapsedTurnAnchors = anchors
    }

    // MARK: - /rewind

    /// One rewindable prompt: the anchor row of a transcript turn, with
    /// what the rewind sheet shows for it.
    public struct RewindTarget: Identifiable, Equatable {
        public let id: UUID
        public let createdAt: Date
        public let preview: String
    }

    /// The rewindable prompts of the selected chat, oldest first. A chat
    /// with nothing after its first prompt has nothing to rewind TO.
    public var rewindTargets: [RewindTarget] {
        selectedChat.messages.enumerated().compactMap { index, message in
            guard message.role == .user else { return nil }
            // The LAST prompt is not a target: rewinding to it would drop
            // nothing (there is nothing after it), so it is not offered.
            guard index < selectedChat.messages.count - 1 else { return nil }
            return RewindTarget(
                id: message.id,
                createdAt: message.createdAt,
                preview: String(message.content.prefix(120)).replacingOccurrences(of: "\n", with: " "))
        }
    }

    /// Drops every message AFTER the anchor prompt, keeping the prompt
    /// itself: the qwen-code rewind semantics (the conversation, not any
    /// files the tools wrote). Refused mid-run; also drops the chat's
    /// queued prompts and any parked response variants, which would
    /// otherwise replay onto a conversation they predate.
    public func rewindTo(anchorMessageID: UUID) {
        guard !generating, !submitting, pendingToolCall == nil else {
            showToast("Stop the current turn before rewinding.", style: .warning)
            return
        }
        guard let index = chats.firstIndex(where: { $0.id == selectedChatID }),
            let anchorIndex = chats[index].messages.firstIndex(where: { $0.id == anchorMessageID })
        else { return }
        let removed = chats[index].messages.count - (anchorIndex + 1)
        guard removed > 0 else { return }
        chats[index].messages.removeSubrange((anchorIndex + 1)...)
        // The checklist is transcript state too: a TodoWrite list from a
        // rewound turn describes work the conversation no longer contains.
        chats[index].todos = []
        chats[index].updatedAt = Date()
        pendingUserMessages[selectedChatID] = nil
        pendingResponseVariants[selectedChatID] = nil
        collapsedTurnAnchors.removeAll()
        outputText = ""
        outputReasoningText = ""
        persistChats()
        updateTokenEstimate()
        showToast("Rewound \(removed) message\(removed == 1 ? "" : "s").", style: .success)
    }

    // MARK: - /branch

    /// Forks the whole conversation into a new chat at the latest prompt.
    /// The transcript copies (like a duplicate) but the fork is titled by
    /// the directive given and immediately archived-free; the ORIGINAL
    /// stays selected, matching a branch's "leave this line alone" intent.
    @discardableResult
    public func branchConversation(titled newTitle: String?) -> UUID? {
        guard !generating, !submitting, pendingToolCall == nil,
            !selectedChat.isGhost, !selectedChat.messages.isEmpty
        else {
            showToast(
                selectedChat.isGhost
                    ? "A temporary chat cannot be branched."
                    : "Nothing to branch yet.",
                style: .info)
            return nil
        }
        var fork = selectedChat.duplicated()
        fork.title = newTitle.flatMap {
            let trimmed = $0.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : String(trimmed.prefix(80))
        } ?? (selectedChat.title + " (branch)")
        chats.insert(fork, at: 0)
        persistChats()
        showToast("Branched into '\(fork.title)'.", style: .success)
        return fork.id
    }

    private func runBranchCommand(_ title: String) {
        _ = branchConversation(titled: title.isEmpty ? nil : title)
    }

    // MARK: - /btw, /fork (background side questions)

    /// `/btw <question>`: a side question answered by a BACKGROUND agent so
    /// the running turn is not disturbed (qwen-code's side-task semantics,
    /// on top of this app's background subagent runs).
    private func runBtwCommand(_ question: String) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            showToast("Give the question: /btw What did we decide about retries?", style: .info)
            return
        }
        launchParityAgent(
            taskPrompt: """
            A side question from the user, asked while their main conversation \
            continues. Answer it directly and briefly; do not restate the \
            conversation. Question: \(trimmed)
            """,
            taskDescription: "Side question")
    }

    /// `/fork <directive>`: continues THIS conversation in a background
    /// agent. The transcript rides along (tail-bounded) so the fork picks
    /// up mid-thread, and its completion arrives as a task notification in
    /// the originating chat.
    private func runForkCommand(_ directive: String) {
        let trimmed = directive.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            showToast("Give the directive: /fork Turn the last answer into a test plan.", style: .info)
            return
        }
        let transcript = AppChatCompaction.renderTranscript(
            Array(selectedChat.messages.suffix(40)))
        let bounded = String(transcript.suffix(12_000))
        launchParityAgent(
            taskPrompt: """
            Continue the conversation below as a background task. The final \
            user directive is what to do; the transcript before it is the \
            context. Your final response is delivered back to the \
            originating chat.

            <conversation>
            \(bounded)
            </conversation>

            <directive>
            \(trimmed)
            </directive>
            """,
            taskDescription: trimmed)
    }

    private func launchParityAgent(
        taskPrompt: String, taskDescription: String
    ) {
        guard let agent = effectiveAgents.first else {
            showToast("No enabled agent definition to run this on.", style: .warning)
            return
        }
        guard session != nil else {
            showToast("Load a model first: a side task needs one to run.", style: .warning)
            return
        }
        let chatID = selectedChatID
        let project = turnProject(chatID: chatID)
        Task {
            do {
                let id = try await launchBackgroundAgent(BackgroundAgentLaunch(
                    agent: agent, taskPrompt: taskPrompt, taskDescription: taskDescription,
                    project: project, chatID: chatID, depth: 0, userSystemPrompt: ""))
                showToast("Side task \(id) started in the background.", style: .success)
            } catch {
                showToast(error.localizedDescription, style: .error)
            }
        }
    }

    // MARK: - /recap

    /// `/recap`: one-shot summarize of the conversation, shown in a sheet.
    /// Unlike `/compact` it MUTATES NOTHING: the recap is a reading, not a
    /// replacement, so it is safe on a ghost chat and safe mid-thread.
    public func runRecapCommand() {
        guard !isRecapping else {
            showToast("A recap is already running.", style: .info)
            return
        }
        guard session != nil else {
            showToast("Load a model first: a recap needs one to run.", style: .warning)
            return
        }
        let rows = turnMessages(for: selectedChatID)
        guard !rows.isEmpty else {
            showToast("Nothing to recap yet.", style: .info)
            return
        }
        let transcript = String(
            AppChatCompaction.renderTranscript(rows).suffix(16_000))
        let messages = [
            ChatMessage(
                role: .system,
                content: "You summarize conversations faithfully and concisely."),
            ChatMessage(
                role: .user,
                content: """
                Summarize this conversation so far: what was asked, what was \
                done and decided, and what is still open. Keep it under 250 \
                words.

                <transcript>
                \(transcript)
                </transcript>
                """),
        ]
        var options = GenerateOptions()
        options.reasoning = .off
        options.temperature = 0.2
        options.maxNewTokens = 500

        recapText = ""
        isRecapping = true
        showRecapSheet = true
        Task {
            defer { isRecapping = false }
            guard let session else { return }
            do {
                for try await event in session.generate(messages, options: options) {
                    if case .content(let chunk) = event {
                        recapText += chunk
                    }
                }
            } catch is CancellationError {
                return
            } catch {
                showToast("Recap failed: \(error.localizedDescription)", style: .error)
                showRecapSheet = false
            }
            let finished = recapText.trimmingCharacters(in: .whitespacesAndNewlines)
            if finished.isEmpty {
                showToast("The model returned an empty recap.", style: .warning)
                showRecapSheet = false
            }
        }
    }

    // MARK: - Two-stage Esc cancel (qwen-code escapeIntent parity)

    /// The composer's Escape, while a turn is running: the first press
    /// ARMS the cancel (the footer says "Press Esc again to stop"), the
    /// second within the window stops the turn. Returns true when the
    /// press was consumed by the cancel flow, so the caller keeps focus.
    @discardableResult
    public func handleEscCancelIntent() -> Bool {
        guard generating, canCancel else { return false }
        escCancelArmTask?.cancel()
        escCancelArmTask = nil
        if isEscCancelArmed {
            isEscCancelArmed = false
            cancel()
            return true
        }
        isEscCancelArmed = true
        escCancelArmTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            self?.isEscCancelArmed = false
        }
        return true
    }

    /// Disarms the two-stage Esc, on turn end and on an explicit stop.
    public func disarmEscCancel() {
        escCancelArmTask?.cancel()
        escCancelArmTask = nil
        isEscCancelArmed = false
    }
}
