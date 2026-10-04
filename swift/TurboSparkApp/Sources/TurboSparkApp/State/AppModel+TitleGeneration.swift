import Foundation
import TurboSpark

struct PendingAppTitleGeneration: Equatable, Sendable {
    let chatID: UUID
    let attemptID: UUID
    let firstUserMessage: String
}

extension AppModel {
    /// Claims title generation before the user row is appended. The row
    /// mutation immediately after this call persists the marker with it.
    func recordTitleGenerationAttempt(chatID: UUID, firstUserMessage: String) {
        guard let index = chats.firstIndex(where: { $0.id == chatID }),
            !chats[index].isGhost,
            !chats[index].messages.contains(where: { $0.role == .user }),
            chats[index].titleProvenance == .unclaimed,
            chats[index].title == "New Chat" || chats[index].title.isEmpty,
            !chats[index].titleGenerationAttempted
        else { return }

        let attemptID = UUID()
        chats[index].titleGenerationAttempted = true
        pendingTitleGenerationTokens[chatID] = attemptID
        pendingTitleGenerations.append(PendingAppTitleGeneration(
            chatID: chatID,
            attemptID: attemptID,
            firstUserMessage: firstUserMessage))
    }

    /// Starts title work only after the foreground turn has drained. Parked
    /// inputs are not active session work; the active-turn guards below, not
    /// queue contents, determine idleness. Foreground work interrupts an
    /// in-flight title stream.
    func drainPendingTitleGenerationIfIdle(session: TurboSparkSession? = nil) {
        guard !generating,
            !submitting,
            pendingToolCall == nil,
            titleGenerationTask == nil
        else { return }

        while let pending = pendingTitleGenerations.first {
            guard let index = chats.firstIndex(where: { $0.id == pending.chatID }),
                !chats[index].isGhost,
                chats[index].titleProvenance == .unclaimed,
                chats[index].titleGenerationAttempted,
                pendingTitleGenerationTokens[pending.chatID] == pending.attemptID
            else {
                pendingTitleGenerations.removeFirst()
                if pendingTitleGenerationTokens[pending.chatID] == pending.attemptID {
                    pendingTitleGenerationTokens[pending.chatID] = nil
                }
                continue
            }

            guard titleGenerationEnqueueOverride != nil || session != nil else { return }
            pendingTitleGenerations.removeFirst()
            let enqueueOverride = titleGenerationEnqueueOverride
            titleGenerationTask = Task { @MainActor [weak self] in
                guard let self else { return }
                let candidate: String?
                if Task.isCancelled {
                    candidate = nil
                } else if let enqueueOverride {
                    candidate = await enqueueOverride(pending.firstUserMessage)
                } else if let session {
                    candidate = await AppTitleGeneration.generateTitle(
                        session: session, firstUserMessage: pending.firstUserMessage)
                } else {
                    candidate = nil
                }

                self.applyTitleIfUngoverned(
                    chatID: pending.chatID,
                    candidate: Task.isCancelled ? nil : candidate,
                    attemptID: pending.attemptID)
                self.titleGenerationTask = nil
                self.drainPendingTitleGenerationIfIdle(session: session)
            }
            return
        }
    }

    /// Title work uses the same serial model queue. End its consumer before
    /// foreground work; the binding skips cancelled queued requests and
    /// cancels active generation without claiming that its C call has exited.
    func interruptTitleGenerationForForeground() async {
        guard let task = titleGenerationTask else { return }
        task.cancel()
        await task.value
    }

    /// Applies a sidecar result only to the still-unclaimed chat from the
    /// exact attempt that produced it.
    func applyTitleIfUngoverned(chatID: UUID, candidate: String?, attemptID: UUID) {
        guard pendingTitleGenerationTokens[chatID] == attemptID else { return }
        pendingTitleGenerationTokens[chatID] = nil

        guard let candidate, !candidate.isEmpty,
            let index = chats.firstIndex(where: { $0.id == chatID }),
            !chats[index].isGhost,
            chats[index].titleProvenance == .unclaimed,
            chats[index].titleGenerationAttempted
        else { return }

        chats[index].title = String(candidate.prefix(AppTitleGeneration.titleCharacterLimit))
        chats[index].titleProvenance = .generated
        chats[index].updatedAt = Date()
        persistChats()
    }

    /// Invalidates both queued work and any eventual result from in-flight work.
    func invalidatePendingTitleGeneration(for chatID: UUID) {
        pendingTitleGenerationTokens[chatID] = nil
        pendingTitleGenerations.removeAll { $0.chatID == chatID }
    }
}
