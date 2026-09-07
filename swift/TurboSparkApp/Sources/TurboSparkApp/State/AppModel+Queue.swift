import Foundation
import TurboSpark

/// One prompt parked while its chat is busy.
///
/// Claude Code reference: `src/utils/messageQueueManager.ts` -- input typed
/// while a turn runs is QUEUED and sent when the turn ends, never silently
/// dropped. This app used to drop it: Return was swallowed whenever
/// `canRun` was false, so a user who kept typing into a long turn had to
/// notice, wait, and send again.
///
/// In-memory only, like `pendingTaskNotifications`: the queue describes the
/// gap between two turns of a live session, and a relaunch has nothing to
/// resume it against.
public struct QueuedUserPrompt: Identifiable, Equatable {
    public let id: UUID
    /// The draft text, trimmed at enqueue time the way `run()` trims it.
    public var text: String
    /// The draft attachments, MOVED out of the chat row at enqueue time.
    public var attachments: [AppPromptAttachment]

    public init(id: UUID = UUID(), text: String, attachments: [AppPromptAttachment] = []) {
        self.id = id
        self.text = text
        self.attachments = attachments
    }
}

extension AppModel {
    /// Moves the current draft into the queue for its chat.
    ///
    /// **MIRRORS `run()`'S CAPTURE AND CLEAR EXACTLY**, because the entry
    /// has to survive the same journey a sent prompt would: the text is
    /// trimmed here (the composer may still hold trailing whitespace), the
    /// attachments leave the row (so they cannot be mutated or re-sent by
    /// a user edit that lands before the drain), and the ghost draft is
    /// cleared through the vault payload, not the row. A ghost chat's
    /// queued text stays in memory only, the same exposure class as the
    /// composer holding it while it is typed.
    ///
    /// Only ever called for the SELECTED chat: it captures `promptText`
    /// and `promptAttachments`, both of which describe the selection.
    func enqueueCurrentDraft(chatID: UUID) {
        let text = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachments = promptAttachments
        guard !text.isEmpty || !attachments.isEmpty else { return }
        pendingUserMessages[chatID, default: []].append(
            QueuedUserPrompt(text: text, attachments: attachments))
        if let index = chats.firstIndex(where: { $0.id == chatID }) {
            if chats[index].isGhost {
                mutateGhostPayload(for: chatID) { $0.draft = "" }
            } else {
                chats[index].draft = ""
            }
            chats[index].draftAttachments = []
            chats[index].updatedAt = Date()
        }
        promptText = ""
    }

    /// The selected chat's queued prompts, for the composer pill.
    public func queuedMessages(for chatID: UUID) -> [QueuedUserPrompt] {
        pendingUserMessages[chatID] ?? []
    }

    /// Puts one queued entry back into the draft for editing (the pill's
    /// remove affordance). Attachments return to the row first, so
    /// `promptAttachments` reads them the way a fresh attach would have.
    public func restoreQueuedMessage(id: UUID) {
        let chatID = selectedChatID
        guard var queued = pendingUserMessages[chatID],
            let index = queued.firstIndex(where: { $0.id == id })
        else { return }
        let entry = queued.remove(at: index)
        if queued.isEmpty {
            pendingUserMessages[chatID] = nil
        } else {
            pendingUserMessages[chatID] = queued
        }
        if let rowIndex = chats.firstIndex(where: { $0.id == chatID }) {
            chats[rowIndex].draftAttachments += entry.attachments
        }
        promptText = promptText.isEmpty
            ? entry.text
            : "\(promptText)\n\n\(entry.text)"
    }

    /// Sends the chat's first queued prompt NOW, if the chat is idle.
    ///
    /// **GOES THROUGH `run()`, NOT AROUND IT.** The entry is restored into
    /// the composer and submitted the ordinary way, so a queued meta
    /// command dispatches as a command, `@` mentions resolve, the
    /// `UserPromptSubmit` hook runs, and the auto-title fires -- every one
    /// of those would be skipped by appending the message directly.
    ///
    /// Only the FIRST entry goes; the rest wait for the turn this starts,
    /// whose tail calls this again. Called from the generation-turn tail
    /// (before the task-notification drain, so USER intent goes before a
    /// background agent's completion note) and idempotent whenever the
    /// chat is not idle.
    func drainPendingUserMessagesIfIdle(chatID: UUID) {
        guard canInjectTaskNotification(into: chatID),
            let queued = pendingUserMessages[chatID], !queued.isEmpty
        else { return }
        let first = queued[0]
        if queued.count > 1 {
            pendingUserMessages[chatID] = Array(queued.dropFirst())
        } else {
            pendingUserMessages[chatID] = nil
        }
        if let rowIndex = chats.firstIndex(where: { $0.id == chatID }) {
            chats[rowIndex].draftAttachments = first.attachments
        }
        promptText = first.text
        run()
    }

    /// Delivers parked prompts MID-TURN at an agent-loop step boundary, and
    /// returns whether any message reached the transcript.
    ///
    /// **THE QUEUE HAS TWO DELIVERY POINTS, WITH AN ORDER BETWEEN THEM.**
    /// The tail drain above sends a parked prompt as a NEW turn once this
    /// one ends. But a user who types "stop, use Python instead" into a
    /// twenty-step agent turn does not want the answer to the first
    /// instruction finished first -- Claude Code folds mid-turn input into
    /// the running turn, and opencode v2's session contract makes it the
    /// DEFAULT delivery mode (`steer`: deliver at the next safe step
    /// boundary, vs `queue`: wait for idle). This is that boundary drain:
    /// `continueOrStop` calls it before issuing the next model call of a
    /// running loop, the transcript row is appended directly, and the next
    /// step's `buildAppendOnlyHistory` carries it to the model naturally.
    ///
    /// The assembly mirrors `run()`'s submission body on purpose -- mention
    /// resolution first, then attachments off the row, then the hook, then
    /// sanitization -- minus the turn lifecycle around it: the draft was
    /// already captured at enqueue time, the chat already exists and is
    /// titled, and `executeGenerationTurn` is already running. Ghost chats
    /// dispatch no hook, for the same no-trace rule `run()` states.
    ///
    /// A hook block (or an image the loaded model cannot send) RE-PARKS the
    /// entry at the front and stops the batch, closest to `run()`'s
    /// "nothing appended, draft kept": the user can pull it back out of the
    /// pill, and the turn's tail drain will submit it through `run()` where
    /// the same verdict becomes the ordinary error path. Later entries stay
    /// ahead of anything enqueued during the awaits, so order is preserved.
    ///
    /// Not the tail drain's replacement: entries that arrive during the
    /// FINAL step have no boundary left to catch and are still the tail
    /// drain's. Boundary first, tail second -- the same priority order the
    /// two modes have in the contract this ports.
    func deliverSteersAtBoundary(chatID: UUID, project: AppProject?) async -> Bool {
        // An approval card means the loop is parked between steps: the next
        // boundary is whichever `continueOrStop` the approved (or denied)
        // call lands on, not this one. A pending cancel means the turn is
        // already leaving; a steer must not outrun the Stop.
        guard pendingToolCall == nil, !isCancellationPending else { return false }
        var delivered = false
        // Cancel is checked per entry, and the entry in flight is re-parked:
        // the loop runs inside the turn's `runTask`, and a Stop pressed while
        // a slow hook runs must not have its remaining steers appended after
        // the turn it stopped (state#38's shape).
        while !Task.isCancelled,
            let queue = pendingUserMessages[chatID], let entry = queue.first {
            pendingUserMessages[chatID] = Array(queue.dropFirst())

            // Restore onto the row so mention resolution and the row-based
            // read below behave exactly as they do in `run()`.
            if let rowIndex = chats.firstIndex(where: { $0.id == chatID }) {
                chats[rowIndex].draftAttachments = entry.attachments
            }
            if MentionResolver.draftContainsMentions(entry.text) {
                await MentionResolver.resolveMentions(
                    in: entry.text,
                    projectRoot: project?.rootDirectoryURL,
                    chatID: chatID,
                    into: self)
            }
            let attachments = chats.first(where: { $0.id == chatID })?.draftAttachments ?? []
            if let rowIndex = chats.firstIndex(where: { $0.id == chatID }) {
                chats[rowIndex].draftAttachments = []
            }
            let (imageDocs, textDocs) = attachments.reduce(
                into: ([AppPromptAttachment](), [AppPromptAttachment]())
            ) { split, doc in
                if doc.isSendableImage {
                    split.0.append(doc)
                } else {
                    split.1.append(doc)
                }
            }
            if !imageDocs.isEmpty && !visionIsActive {
                showToast(
                    "Cannot send \(imageDocs.count == 1 ? "this image" : "these images"): "
                        + "\(visionRefusalReason ?? "the loaded model has no active vision tower")",
                    style: .warning)
                reparkSteer(entry, chatID: chatID)
                break
            }
            let promptImages: [ChatImage] = imageDocs.compactMap { $0.sourcePath.map(ChatImage.path) }
            var fullUserContent = entry.text
            if !textDocs.isEmpty {
                let docsText = textDocs.map { doc in
                    "--- Attachment: \(doc.fileName) (\(doc.formatLabel)) ---\n\(doc.extractedText)\n--- End of \(doc.fileName) ---"
                }.joined(separator: "\n\n")
                fullUserContent = fullUserContent.isEmpty
                    ? docsText
                    : "\(fullUserContent)\n\n\(docsText)"
            }
            guard !fullUserContent.isEmpty || !promptImages.isEmpty else { continue }

            let isGhost = chats.first(where: { $0.id == chatID })?.isGhost ?? false
            let verdict = isGhost
                ? AppHookVerdict()
                : await evaluateUserPromptSubmit(
                    prompt: fullUserContent, chatID: chatID, project: project)
            // Same post-hook guard `run()` carries: the hook's await is the
            // one window a Stop can land in, and a cancelled steer is not
            // delivered.
            guard !Task.isCancelled else {
                reparkSteer(entry, chatID: chatID)
                break
            }
            if verdict.preventContinuation {
                self.error = verdict.continuationStopReason
                    ?? "A hook declined to continue this prompt."
                reparkSteer(entry, chatID: chatID)
                break
            }
            if verdict.isBlocked {
                self.error = verdict.blockReason ?? "Prompt blocked by a UserPromptSubmit hook."
                reparkSteer(entry, chatID: chatID)
                break
            }

            var contentForModel = UnicodeSanitization.sanitize(fullUserContent)
            if let context = verdict.additionalContext, !context.isEmpty {
                contentForModel += "\n\n<hook_context>\n\(context)\n</hook_context>"
            }
            let userMessage = AppChatMessage(
                role: .user,
                content: contentForModel,
                imagePaths: promptImages.compactMap {
                    if case .path(let p) = $0 { return p } else { return nil }
                })
            mutateTurnMessages(for: chatID) { $0.append(userMessage) }
            delivered = true
        }
        return delivered
    }

    /// Puts an undelivered steer back at the FRONT of its queue, ahead of
    /// anything enqueued while its hook was being awaited.
    private func reparkSteer(_ entry: QueuedUserPrompt, chatID: UUID) {
        pendingUserMessages[chatID] = [entry] + (pendingUserMessages[chatID] ?? [])
    }
}
