import Foundation
import Combine

/// One progress event out of a live `SubagentRunner` run.
///
/// A subagent's generation used to be consumed silently
/// (`SubagentRunner.runBody` accumulated `.content` into a local string),
/// so a run that took minutes showed the user nothing until its tool result
/// landed. These events are the observable seam: the runner emits them at
/// its loop seams and the app routes them to whichever surface is hosting
/// the run (a foreground card in the streaming row, or a background task
/// card).
public enum SubagentProgressEvent: Sendable {
    /// First event of every run, emitted by the `run` wrapper before the
    /// loop body, so every exit path (depth, no session, cancellation,
    /// overflow, error) is bracketed by `started` and `finished`. Carries
    /// the run's own chat so a foreground card binds to the conversation
    /// that proposed it, never to whatever chat is selected later.
    case started(
        agentName: String, displayName: String, taskDescription: String,
        promptHead: String, chatID: UUID?)
    case turnStarted(number: Int)
    /// A decoded content delta from the subagent's current turn.
    case content(String)
    case toolStarted(name: String, summary: String)
    /// `isError` reads the `<tool_error>` tag the observation carries, and
    /// `durationSeconds` is 0 -- the gate returns a message, not a timing,
    /// so there is no honest number to report per tool at this level.
    case toolFinished(name: String, summary: String, isError: Bool)
    /// Last event of every run, carrying the same status string the
    /// `SubagentRunResult` reports (completed, max_turns, context_overflow,
    /// cancelled, failed, error).
    case finished(status: String)
}

/// One tool call a subagent made, as shown in a progress card.
public struct SubagentToolRow: Identifiable, Equatable, Sendable {
    public var id = UUID()
    public var name: String
    public var summary: String
    public var isError: Bool

    public init(id: UUID = UUID(), name: String, summary: String, isError: Bool) {
        self.id = id
        self.name = name
        self.summary = summary
        self.isError = isError
    }
}

/// Where a run is hosted. A foreground run's card lives in the streaming
/// row and disappears when the tool result lands; a background run's card
/// persists until the user dismisses it, because nothing else in the
/// transcript represents it while it works.
public enum SubagentRunMode: String, Sendable {
    case foreground
    case background
}

/// Live, observable state of one subagent run, keyed by the string the
/// host surface chose (a tool-call UUID for foreground runs, a `bga_N` id
/// for background ones).
///
/// MainActor because every writer is the app's event sink hopping to the
/// main actor, and every reader is a SwiftUI view. The runner itself never
/// touches this type -- it emits `SubagentProgressEvent` values.
@MainActor
public final class SubagentRunState: ObservableObject, Identifiable {
    public let id: String
    public let mode: SubagentRunMode
    public let chatID: UUID?
    public let startedAt = Date()
    @Published public var agentName: String
    @Published public var displayName: String
    @Published public var taskDescription: String
    @Published public var promptHead: String
    @Published public var turns: Int = 0
    @Published public var streamedText: String = ""
    @Published public var toolRows: [SubagentToolRow] = []
    @Published public var status: String = "running"
    @Published public var result: SubagentRunResult?

    public init(
        id: String, mode: SubagentRunMode, chatID: UUID?,
        agentName: String = "", displayName: String = "",
        taskDescription: String = "", promptHead: String = ""
    ) {
        self.id = id
        self.mode = mode
        self.chatID = chatID
        self.agentName = agentName
        self.displayName = displayName
        self.taskDescription = taskDescription
        self.promptHead = promptHead
    }

    /// Applies one event. Shared by every host surface so the two dicts in
    /// `AppModel` cannot drift apart in what they record.
    public func apply(_ event: SubagentProgressEvent) {
        switch event {
        case .started(let agentName, let displayName, let taskDescription, let promptHead, _):
            self.agentName = agentName
            self.displayName = displayName
            self.taskDescription = taskDescription
            self.promptHead = promptHead
        case .turnStarted(let number):
            turns = number
        case .content(let chunk):
            streamedText += chunk
        case .toolStarted(let name, let summary):
            toolRows.append(SubagentToolRow(name: name, summary: summary, isError: false))
        case .toolFinished(let name, let summary, let isError):
            // The started row for this call is the last one with the same
            // name; mark it rather than append a second row, so a card shows
            // one row per call with its outcome.
            if let idx = toolRows.lastIndex(where: { $0.name == name && !$0.isError }) {
                toolRows[idx] = SubagentToolRow(id: toolRows[idx].id, name: name, summary: summary, isError: isError)
            } else {
                toolRows.append(SubagentToolRow(name: name, summary: summary, isError: isError))
            }
        case .finished(let status):
            self.status = status
        }
    }
}
