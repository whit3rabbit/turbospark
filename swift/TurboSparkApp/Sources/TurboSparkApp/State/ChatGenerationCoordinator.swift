import Foundation

/// Where one chat stands relative to the single turn lane (ZP-23).
///
/// There is no `.submitting` case on purpose: `submitting` is the window
/// before a turn exists, it stays a global flag, and it still blocks chat
/// switching (see `ChatGenerationCoordinator.verdict`).
public enum ChatTurnState: Equatable, Sendable {
    /// Nothing in flight and nothing parked.
    case idle
    /// Has parked input waiting for the lane to free up.
    case queued
    /// Holds the lane: its turn (or the tool it started) is running.
    case generating
    /// Parked on a tool call that needs the user. Holds no lane.
    case awaitingTool
    /// Parked on a question for the user. HOLDS the lane (see
    /// `ChatGenerationCoordinator.park`): the question blocks inside the
    /// tool's own execution task, which is the app's single turn slot. The
    /// chat still reads as `.awaitingQuestion`, not `.generating`, so the
    /// sidebar says what it is waiting for.
    case awaitingQuestion
}

/// An operation whose legality depends on what other chats are doing.
public enum ChatLaneOperation: Equatable, Sendable {
    /// Select `target` (or create a chat, which selects it).
    case select(target: UUID?)
    /// Delete one chat.
    case delete(chat: UUID)
    /// Replace or unload the resident model.
    case switchModel
}

/// The coordinator's answer to a `ChatLaneOperation`.
public enum ChatLaneVerdict: Equatable, Sendable {
    case allowed
    /// Refused. `blockingChat` names the chat in the way, or is nil when the
    /// busy work is not attributed to any chat (an audio summary, a bang
    /// command, an image job): those keep the old blanket refusal.
    case refused(blockingChat: UUID?)
}

/// What the app was doing when a guard asked, as plain values.
///
/// A snapshot rather than the live `AppModel`, so every cell of the guard
/// matrix is reachable from a test without a model or a session.
public struct ChatLaneSnapshot: Equatable, Sendable {
    public var generating: Bool
    /// The chat that owns the running turn. Nil while `generating` is true
    /// means the work is not a chat turn.
    public var generatingChatID: UUID?
    public var submitting: Bool
    /// An approval sits in the slot but names no chat (legacy writers), so it
    /// cannot be parked under a chat key and must keep pinning the user.
    public var hasUnattributedApproval: Bool

    public init(
        generating: Bool = false, generatingChatID: UUID? = nil,
        submitting: Bool = false, hasUnattributedApproval: Bool = false
    ) {
        self.generating = generating
        self.generatingChatID = generatingChatID
        self.submitting = submitting
        self.hasUnattributedApproval = hasUnattributedApproval
    }
}

/// The ZP-23 scheduler: one turn lane, a foreground-first admission order,
/// and the per-chat guard decisions that replace the blanket single-active
/// guard.
///
/// **THE LANE IS THE APP'S SINGLE TURN SLOT, NOT THE ENGINE.** The engine
/// already serializes requests (one `Engine` per session behind a mutex, and
/// `QueuedGenerationStream` on the Swift side). What cannot run twice is the
/// turn state on `AppModel`: one `runTask`, one epoch, one live stream
/// buffer, one approval slot. So concurrency here means: any chat can be
/// selected while another one runs, other chats park behind the lane, and a
/// chat that needs the user releases the lane. It does not mean two turns
/// stepping at once. That needs per-chat turn state and is deferred
/// (`docs/designs/zcode-parity/ZP-23-concurrent-chats.md`).
///
/// **CAP.** `laneCapacity` is 1 and a constant: one slot per top-level chat,
/// subagents run inside their parent chat's slot, and a request at the cap
/// queues rather than refuses. There is no settings field in v1; the engine's
/// KV is sized per session, and a second concurrent turn is not offered, so
/// there is nothing hardware-dependent to bound.
///
/// **LADDER POSITION.** Background chats never auto-run tools: a tool call
/// proposed by a chat that is not selected parks (`awaitingTool`) and is
/// evaluated by the ordinary permission path only once the chat is
/// foreground, so the permission ladder, hooks and approval cards behave
/// exactly as for a foreground chat. Nothing here widens what is allowed.
public struct ChatGenerationCoordinator: Equatable, Sendable {
    /// One lane in v1. See the type doc.
    public static let laneCapacity = 1

    public private(set) var laneHolder: UUID?
    /// Chats waiting for the lane, oldest first.
    public private(set) var admissionQueue: [UUID] = []
    /// Chats parked on the user, with why.
    public private(set) var parked: [UUID: ChatTurnState] = [:]

    public init() {}

    // MARK: - The lease

    public enum Admission: Equatable, Sendable {
        case granted
        /// Waiting; `position` is 1-based.
        case queued(position: Int)
    }

    /// Takes the lane for `chat`, or queues it. Idempotent for the holder.
    @discardableResult
    public mutating func requestLane(for chat: UUID) -> Admission {
        if laneHolder == nil || laneHolder == chat {
            laneHolder = chat
            admissionQueue.removeAll { $0 == chat }
            parked[chat] = nil
            return .granted
        }
        if !admissionQueue.contains(chat) { admissionQueue.append(chat) }
        let position = (admissionQueue.firstIndex(of: chat) ?? admissionQueue.count - 1) + 1
        return .queued(position: position)
    }

    /// Records that `chat` has parked input waiting for the lane. Never
    /// grants it: a chat with only parked input is not running anything, and
    /// handing it the lane would hold the lane for a turn that never starts.
    public mutating func enqueue(_ chat: UUID) {
        guard laneHolder != chat, parked[chat] == nil, !admissionQueue.contains(chat) else {
            return
        }
        admissionQueue.append(chat)
    }

    public mutating func dequeue(_ chat: UUID) {
        admissionQueue.removeAll { $0 == chat }
    }

    /// Frees the lane if, and only if, `chat` holds it. A release by any
    /// other chat is ignored: a stale tail from an older turn must not free
    /// the lane a newer turn owns.
    @discardableResult
    public mutating func releaseLane(for chat: UUID) -> Bool {
        guard laneHolder == chat else { return false }
        laneHolder = nil
        return true
    }

    /// The lane's holder is gone without a named owner (the app-wide flag
    /// dropped). Unconditional by design, the flag is the source of truth.
    public mutating func releaseHolder() {
        // A question parks under the holder and dies with the turn.
        if let holder = laneHolder, parked[holder] == .awaitingQuestion { parked[holder] = nil }
        laneHolder = nil
    }

    /// The next chat to offer the lane to: the foreground chat when it is
    /// waiting, else the oldest waiter. Does not grant; the caller starts
    /// the turn, whose start path calls `requestLane`.
    public func nextAdmission(foreground: UUID?) -> UUID? {
        guard laneHolder == nil else { return nil }
        if let foreground, admissionQueue.contains(foreground) { return foreground }
        return admissionQueue.first
    }

    // MARK: - Parking

    /// Marks `chat` as waiting on the user.
    ///
    /// `.awaitingTool` releases the lane: the tool proposal already ended
    /// the turn and the approval needs no engine. `.awaitingQuestion` does
    /// not: the question is answered inside the running tool's execution
    /// task, so the app's single turn slot stays occupied until it is
    /// answered. Releasing it needs the turn and the tool task decoupled
    /// (`docs/designs/zcode-parity/ZP-23-concurrent-chats.md`). The state is
    /// recorded either way, and `unpark` clears it when the answer lands.
    public mutating func park(_ chat: UUID, as state: ChatTurnState) {
        guard state == .awaitingTool || state == .awaitingQuestion else { return }
        if state == .awaitingTool, laneHolder == chat { laneHolder = nil }
        admissionQueue.removeAll { $0 == chat }
        parked[chat] = state
    }

    public mutating func unpark(_ chat: UUID) {
        parked[chat] = nil
    }

    /// Drops every trace of a chat (deleted).
    public mutating func forget(_ chat: UUID) {
        if laneHolder == chat { laneHolder = nil }
        admissionQueue.removeAll { $0 == chat }
        parked[chat] = nil
    }

    public func state(of chat: UUID) -> ChatTurnState {
        if laneHolder == chat { return parked[chat] == .awaitingQuestion ? .awaitingQuestion : .generating }
        if let state = parked[chat] { return state }
        if admissionQueue.contains(chat) { return .queued }
        return .idle
    }

    // MARK: - Guards

    /// Whether `op` may proceed given what the app is doing.
    ///
    /// Replaces the blanket "anything running blocks everything" guard. A
    /// chat blocks only operations that touch it, with these exceptions kept
    /// on purpose:
    /// - `submitting` blocks selection and deletion (the hook window has no
    ///   turn yet, and `run()` captured the chat it will append into);
    /// - work that is not a chat turn keeps the blanket refusal;
    /// - an approval with no chat of its own cannot be parked, so it still
    ///   pins the user. An attributed approval does not: `selectChat` parks
    ///   it under its chat and restores it, re-validated, on return.
    public static func verdict(_ op: ChatLaneOperation, _ snapshot: ChatLaneSnapshot)
        -> ChatLaneVerdict
    {
        switch op {
        case .switchModel:
            // The model is shared by every chat, so ANY busy chat blocks.
            if snapshot.generating { return .refused(blockingChat: snapshot.generatingChatID) }
            if snapshot.submitting { return .refused(blockingChat: nil) }
            return .allowed
        case .select:
            if snapshot.submitting { return .refused(blockingChat: nil) }
            if snapshot.generating && snapshot.generatingChatID == nil {
                return .refused(blockingChat: nil)
            }
            if snapshot.hasUnattributedApproval { return .refused(blockingChat: nil) }
            return .allowed
        case .delete(let chat):
            if snapshot.submitting { return .refused(blockingChat: nil) }
            if snapshot.generating {
                if snapshot.generatingChatID == nil || snapshot.generatingChatID == chat {
                    return .refused(blockingChat: snapshot.generatingChatID)
                }
            }
            return .allowed
        }
    }
}
