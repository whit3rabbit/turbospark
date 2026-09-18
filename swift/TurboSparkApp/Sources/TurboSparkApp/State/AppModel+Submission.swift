import Foundation
import TurboSpark

/// Everything between pressing Send and the first token being asked for.
///
/// **THE HOOK IS AWAITED BEFORE THE MESSAGE IS APPENDED**, or a
/// `UserPromptSubmit` hook cannot actually stop the turn: by the time this
/// app used to fire it, the user's message was already in the transcript and
/// prefill was already starting. A block now simply never appends, which
/// needs no undo step because there is nothing to undo.
///
/// Split from `AppModel+Generation`, which owns the STREAM. The two phases
/// answer different questions -- this one is about the draft, the
/// attachments and the chat, that one about the window, the options and the
/// events -- and the file had grown to 684 lines holding both.
extension AppModel {
    public func run() {
        let userDraft = promptText.trimmingCharacters(in: .whitespacesAndNewlines)

        if imageModeEnabled {
            generateImage()
            return
        }

        // The memory commands NEVER GENERATE, so they sit above the `canRun`
        // guard and work with no model loaded -- the point of a quick-save.
        // Everything below needs the session. A `#` draft is not a slash
        // command and never reaches the meta gate or a hook; `/memory` is a
        // meta command dispatched by name off the shared table.
        if UserMemoryInputMessage.isQuickSaveDraft(userDraft) {
            handleMemoryQuickSave(userDraft)
            return
        }
        if BuiltInSlashCommand.isMemoryCommand(userDraft) {
            handleMemoryCommand(userDraft)
            return
        }
        // The other never-generate commands (`/stats`, `/export`, `/help`)
        // sit in the same session-less band as `/memory`: each is pure UI
        // over the chat row, so they work with no model loaded too.
        if BuiltInSlashCommand.isLocalMetaCommand(userDraft) {
            handleLocalMetaCommand(userDraft)
            return
        }
        // `/goal` sits in this band too: status and clear never generate,
        // and the SET arm wants to be above the queue gate so that setting
        // a goal while the chat is busy parks the directive turn behind the
        // running work rather than refusing. Not recorded in prompt
        // history: it is a command, not a recalled prompt.
        if BuiltInSlashCommand.isGoalCommand(userDraft) {
            handleGoalCommand(userDraft)
            return
        }
        // The LOCAL commands (`/copy`, `/theme`, `/tasks`, `/btw`, ...)
        // share the session-less band: none of them generate, and several
        // exist to run WHILE a turn is generating, so they must precede the
        // queue gate, not ride behind it. Handlers enforce their own
        // preconditions (a `/fork` with no model toasts a refusal).
        if BuiltInSlashCommand.isLocalCommand(userDraft) {
            handleLocalCommand(userDraft)
            return
        }
        // A `!` draft runs a shell command and lands its output in the
        // transcript as context (qwen-code bang-command parity). It never
        // generates, and it is not queued: mid-turn output would land inside
        // a running turn's rows, so the handler refuses while busy instead.
        if Self.isBangCommand(userDraft) {
            promptText = ""
            handleBangCommand(userDraft)
            return
        }
        // Input history (qwen-code parity): every accepted prompt is
        // remembered for Up-arrow recall, ghost chats excepted -- history
        // persists to disk and a ghost prompt must not.
        let historyChatIsGhost =
            chats.first(where: { $0.id == selectedChatID })?.isGhost ?? false
        if !historyChatIsGhost {
            promptHistory.record(userDraft)
        }

        // **A BUSY CHAT QUEUES THE DRAFT INSTEAD OF DROPPING THE KEYPRESS**
        // (Claude Code's message queue). When the only failing `canRun`
        // terms are the busy ones -- a turn in flight, an awaited hook, an
        // approval card -- the draft is parked and sent from a turn tail.
        // Everything else (no session, an open in flight, an empty draft)
        // refuses exactly as before.
        if !canRun {
            guard canQueue, session != nil else { return }
            enqueueCurrentDraft(chatID: selectedChatID)
            return
        }
        guard session != nil else { return }

        // A meta-command is not a prompt, and it is handled before everything
        // else here for the same reason the agent slash commands are handled
        // inside the task: placement decides whether a `UserPromptSubmit`
        // hook sees it. The hook receives prompt content, and a maintenance
        // command is not content -- unlike an agent command, which IS a
        // prompt and must be hookable. The names live in the shared
        // `BuiltInSlashCommand` table; the drift test pins the meta count
        // and every dispatcher arm, so a new one edits this dispatcher.
        // (`/memory` was dispatched above: it does not need the session.)
        if BuiltInSlashCommand.isMetaCommand(userDraft) {
            handleCompactCommand(userDraft)
            return
        }

        // Captured BEFORE the hook await below. The draft this turn was built
        // from belongs to THIS chat; a switch while a slow `UserPromptSubmit`
        // hook runs must not land the message in whatever chat the user moved
        // to. Same capture `executeGenerationTurn` makes for the turn itself.
        let submissionChatID = selectedChatID
        // The project this submission's hooks run under. A chat that does not
        // exist yet has no `projectID`, so the selection is what it will be
        // created with -- the same value `run()`'s lazy create below passes
        // (state#67).
        let submissionProject =
            turnProject(chatID: submissionChatID)
            ?? (interactionMode == .projects ? selectedProject : nil)

        // Set synchronously, before the `Task`: `generating` is not raised
        // until `executeGenerationTurn`, so nothing else refuses a second
        // Return pressed while the hook is still running.
        submitting = true

        // `UserPromptSubmit` has to be awaited BEFORE the message is
        // appended to the chat, or a hook cannot actually stop the turn: by
        // the time this app used to fire it (inside `executeGenerationTurn`,
        // fire-and-forget), the user's message was already in the
        // transcript and prefill was already starting. A block now simply
        // never appends the message and leaves the draft intact, which
        // needs no "remove/annotate" step because there is nothing to undo.
        // **STORED, SO STOP CAN REACH IT** (state#33). A `UserPromptSubmit`
        // hook is awaited with a 120 s budget and `submitting` is true for
        // all of it; with the task held nowhere, `cancel()` had nothing to
        // cancel and both Send and Stop were dead for the duration.
        submissionTask = Task {
            defer {
                self.submitting = false
                self.submissionTask = nil
                // A cancel that reached only the submission has no `runTask`
                // tail to clear this, and a latched flag greys Stop out and
                // makes `continueAgentLoop` refuse every later turn.
                if !self.generating { self.isCancellationPending = false }
            }
            self.stopHookReentryCount = 0
            // A user prompt is the goal loop's external clock: it lifts a
            // stall pause and resets the idle check-in cap (CC: "up to
            // three idle check-ins per goal BETWEEN YOUR PROMPTS").
            self.resetGoalForUserPrompt(chatID: submissionChatID)
            // Retry/Edit park response variants immediately before their own
            // turn; an ordinary submission must never inherit a stale one.
            self.pendingResponseVariants[submissionChatID] = nil

            // **`@path` MENTIONS RESOLVE FIRST**, before the attachments are
            // read and before the hook: each resolvable token becomes an
            // ordinary attachment (a folder through the same bounded walk
            // the folder picker uses), so the hook sees the content it gates
            // on, not the pointer. Unresolvable tokens stay as prose in the
            // message, silently.
            if MentionResolver.draftContainsMentions(userDraft) {
                await MentionResolver.resolveMentions(
                    in: userDraft,
                    projectRoot: submissionProject?.rootDirectoryURL,
                    chatID: submissionChatID,
                    into: self)
            }

            // **A PICTURE IS NOT AN ATTACHMENT WITH EMPTY TEXT.** Images go to
            // the vision tower by path and are injected at the marker the
            // template renders; inlining them here would produce an empty
            // "--- Attachment ---" block and send the model nothing at all.
            // Only a picture whose source is still on disk can be sent, because
            // the engine reads the file itself.
            //
            // Captured BY ID rather than through `promptAttachments`: the
            // mention resolution above can take seconds, and a chat switched
            // in that window must not donate ITS attachments to this turn.
            let attachments: [AppPromptAttachment] = {
                if let row = self.chats.first(where: { $0.id == submissionChatID }) {
                    return row.draftAttachments
                }
                return self.activeDraftChat.id == submissionChatID
                    ? self.activeDraftChat.draftAttachments : []
            }()
            let (imageDocs, textDocs) = attachments.reduce(
                into: ([AppPromptAttachment](), [AppPromptAttachment]())
            ) { split, doc in
                if doc.isSendableImage {
                    split.0.append(doc)
                } else {
                    split.1.append(doc)
                }
            }
            // **A PICTURE THAT CANNOT BE SENT IS REFUSED, NOT DROPPED.** This used
            // to resolve to an empty list when `visionIsActive` was false, which
            // put the image in neither list and then cleared the draft below: an
            // image-only turn returned at the emptiness guard and Send read as
            // dead. The install already states WHY it refuses
            // (`info.vision.reason`), and that is the sentence a user needs.
            if !imageDocs.isEmpty && !visionIsActive {
                let reason = visionRefusalReason ?? "the loaded model has no active vision tower"
                showToast(
                    "Cannot send \(imageDocs.count == 1 ? "this image" : "these images"): \(reason)",
                    style: .warning)
                return
            }
            let promptImages: [ChatImage] = imageDocs.compactMap { $0.sourcePath.map(ChatImage.path) }

            var fullUserContent = userDraft
            if !textDocs.isEmpty {
                let docsText = textDocs.map { doc in
                    "--- Attachment: \(doc.fileName) (\(doc.formatLabel)) ---\n\(doc.extractedText)\n--- End of \(doc.fileName) ---"
                }.joined(separator: "\n\n")
                if fullUserContent.isEmpty {
                    fullUserContent = docsText
                } else {
                    fullUserContent = "\(fullUserContent)\n\n\(docsText)"
                }
            }

            // A turn carrying only a picture has no text and is still a turn.
            guard !fullUserContent.isEmpty || !promptImages.isEmpty else { return }

            // **GHOST MODE DISPATCHES NO `UserPromptSubmit` HOOK.** The hook
            // receives the full prompt text, and a hook script is free to
            // log it -- a user who asked for a conversation that leaves no
            // trace has not consented to that. The empty verdict is the
            // "every hook allowed it" answer, so the turn simply proceeds.
            let submissionIsGhost =
                self.chats.first(where: { $0.id == submissionChatID })?.isGhost ?? false
            let verdict = submissionIsGhost
                ? AppHookVerdict()
                : await self.evaluateUserPromptSubmit(
                    prompt: fullUserContent, chatID: submissionChatID, project: submissionProject)
            guard !Task.isCancelled else { return }
            // `continue: false` from a hook's JSON output refuses the prompt
            // outright. Like a block, nothing is appended and the draft is
            // kept; unlike a block (whose reason is an error naming the
            // hook), the stopReason is Claude Code's "shown to the user"
            // message.
            if verdict.preventContinuation {
                self.error = verdict.continuationStopReason
                    ?? "A hook declined to continue this prompt."
                return
            }
            if verdict.isBlocked {
                self.error = verdict.blockReason ?? "Prompt blocked by a UserPromptSubmit hook."
                return
            }
            // The model can be unloaded while the hook runs only if something
            // widened `canUnloadModel`; this is the backstop, and it refuses
            // BEFORE the draft is cleared rather than after.
            guard self.session != nil else {
                self.error = "The model was unloaded before this prompt could be sent."
                return
            }

            // An agent slash command (`/explore`, `/plan`, `/agent`) is a
            // prompt like any other and is dispatched HERE rather than ahead
            // of the `Task`: run before the await, it reaches
            // `runAgentTaskDirectly` -- which appends a user turn and starts
            // generating -- with no `UserPromptSubmit` hook ever consulted,
            // so a hook that blocks every prompt did not block these.
            if self.handleAgentSlashCommand(fullUserContent) {
                return
            }

            if self.handleSkillSlashCommand(fullUserContent) {
                return
            }

            let chatIndex: Int
            if let existing = self.chats.firstIndex(where: { $0.id == submissionChatID }) {
                chatIndex = existing
            } else {
                let newChat = AppChat(id: submissionChatID, projectID: self.selectedProjectID)
                self.chats.insert(newChat, at: 0)
                chatIndex = 0
                self.selectedChatID = newChat.id
                // `createChat()` dispatches `SessionStart` on its own path;
                // this is the OTHER place a chat gets created lazily
                // (the very first prompt in a fresh window), which had no
                // dispatch at all.
                Task {
                    _ = await self.dispatchLifecycleHook(
                        event: .sessionStart, chatID: submissionChatID,
                        project: submissionProject, source: "startup")
                }
            }

            // Clear draft & attachments. The ghost draft is vaulted, so the
            // clear is too; attachments stay on the row (paths only, and
            // the row is never persisted anyway).
            if self.chats[chatIndex].isGhost {
                self.mutateGhostPayload(for: submissionChatID) { $0.draft = "" }
            } else {
                self.chats[chatIndex].draft = ""
            }
            self.chats[chatIndex].draftAttachments = []
            self.chats[chatIndex].updatedAt = Date()

            // Invisible-character sanitization sits at the model boundary:
            // tag characters, bidi overrides, and friends are invisible in
            // the UI but real token content to the model, so they are
            // stripped here and the SANITIZED text is what gets stored
            // below, keeping the transcript equal to what the model read.
            // (`UnicodeSanitization`'s own doc carries the scope decision:
            // prompts only, never tool output.)
            var contentForModel = UnicodeSanitization.sanitize(fullUserContent)
            if let context = verdict.additionalContext, !context.isEmpty {
                contentForModel += "\n\n<hook_context>\n\(context)\n</hook_context>"
            }

            // Auto-title from the draft -- REAL chats only. A ghost chat
            // keeps its fixed title: the title is derived from the prompt,
            // and the row's title field is as in-memory as everything else,
            // but a stable "Temporary Chat" is what the sidebar promises.
            if !self.chats[chatIndex].isGhost,
                (self.chats[chatIndex].title == "New Chat" || self.chats[chatIndex].title.isEmpty),
                !userDraft.isEmpty {
                self.chats[chatIndex].title = String(userDraft.prefix(40)).replacingOccurrences(of: "\n", with: " ")
            }

            // Append user turn. Ghost chats seal the message into the vault
            // instead of the row; the ordinary path persists inside the
            // helper, exactly once, as before.
            let userMessage = AppChatMessage(
                role: .user,
                content: contentForModel,
                imagePaths: promptImages.compactMap {
                    if case .path(let p) = $0 { return p } else { return nil }
                })
            self.mutateTurnMessages(for: submissionChatID) { $0.append(userMessage) }

            self.executeGenerationTurn(step: 0, chatID: submissionChatID)
        }
    }
}
