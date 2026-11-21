import Foundation

/// The knowledge sources the next turn's prompt draws on, as rows for the
/// composer's "What feeds this turn" popover.
///
/// READ-ONLY BY DESIGN. Every switch behind these rows is global or has
/// side effects a quick composer toggle would hide: the Memory settings
/// toggle is the only writer of `memoryEnabled` (MemorySettingsPaneView),
/// turning recall off deletes its store, project rules have no switch at
/// all, and MCP servers already have the Tools popover. So the list reports
/// state and points at where it is changed; it invents no per-turn
/// suppression.
///
/// Each row's condition restates the gate the prompt path itself applies
/// (`buildSystemPromptSections`, `conversationRecallEligible`), so the list
/// cannot claim a source the turn will not send. Token figures are passed in
/// from the context ring's breakdown or the attachment estimate; a source
/// with no figure the app can stand behind gets none.
enum TurnKnowledgeSources {
    /// Display order is declaration order.
    enum Source: String, CaseIterable, Equatable {
        case projectRules
        case memory
        case conversationRecall
        case pinnedInstructions
        case attachments
        case mcpServers
    }

    enum Status: Equatable {
        /// Sent with the next turn.
        case included
        /// The switch for it is off.
        case off
        /// It applies here but holds nothing (project with no rules).
        case notSet
        /// Added only when older turns no longer fit the window.
        case whenNeeded
        /// Cannot apply in this chat, whatever the switch says.
        case unavailable(Reason)
    }

    enum Reason: Equatable {
        case temporaryChat
        case protectedProfile
        case skillState
    }

    /// Who the source applies to, so a row can say whether changing it
    /// affects only this chat or every chat.
    enum Scope: Equatable {
        case allChats
        case thisProject
        case thisChat
        case nextMessage
    }

    struct Row: Equatable, Identifiable {
        let source: Source
        let status: Status
        let scope: Scope
        /// Raw-string or characters-per-token estimate. Nil when unknown.
        let estimatedTokens: Int?
        /// Item count where one is meaningful (attachments, servers).
        let count: Int?

        var id: Source { source }
    }

    struct Inputs: Equatable {
        var hasProject: Bool
        var projectRulesSet: Bool
        var memoryEnabled: Bool
        var recallEnabled: Bool
        var isTemporaryChat: Bool
        var isProtectedProfile: Bool
        var skillStateEnabled: Bool
        var memoryTokens: Int?
        var pinnedInstructionTokens: Int?
        var attachmentCount: Int
        var attachmentTokens: Int?
        var mcpServerCount: Int
        var mcpServerTokens: Int?
    }

    static func rows(_ inputs: Inputs) -> [Row] {
        // A zero or negative count is no estimate; showing "0 tokens" would
        // read as free rather than unknown.
        func positive(_ value: Int?) -> Int? {
            guard let value, value > 0 else { return nil }
            return value
        }

        var rows: [Row] = []

        // Project-derived sections exist only on the project branch of the
        // prompt builder.
        if inputs.hasProject {
            rows.append(Row(
                source: .projectRules,
                status: inputs.projectRulesSet ? .included : .notSet,
                scope: .thisProject, estimatedTokens: nil, count: nil))
        }

        rows.append(Row(
            source: .memory,
            status: inputs.memoryEnabled ? .included : .off,
            scope: .allChats,
            // A stale count from before the switch went off must not show.
            estimatedTokens: inputs.memoryEnabled ? positive(inputs.memoryTokens) : nil,
            count: nil))

        rows.append(Row(
            source: .conversationRecall, status: recallStatus(inputs),
            // Evidence is fitted against the room left at send time, so no
            // figure exists before the turn runs.
            scope: .allChats, estimatedTokens: nil, count: nil))

        if let tokens = positive(inputs.pinnedInstructionTokens) {
            rows.append(Row(
                source: .pinnedInstructions, status: .included, scope: .thisChat,
                estimatedTokens: tokens, count: nil))
        }

        if inputs.attachmentCount > 0 {
            rows.append(Row(
                source: .attachments, status: .included, scope: .nextMessage,
                estimatedTokens: positive(inputs.attachmentTokens), count: inputs.attachmentCount))
        }

        if inputs.hasProject, inputs.mcpServerCount > 0 {
            rows.append(Row(
                source: .mcpServers, status: .included, scope: .thisProject,
                estimatedTokens: positive(inputs.mcpServerTokens), count: inputs.mcpServerCount))
        }

        return rows
    }

    /// Mirrors `conversationRecallEligible`. A protected profile comes first
    /// because it forces the settings toggle off; the switch itself comes
    /// next because it is the one thing the user can act on.
    private static func recallStatus(_ inputs: Inputs) -> Status {
        if inputs.isProtectedProfile { return .unavailable(.protectedProfile) }
        if !inputs.recallEnabled { return .off }
        if inputs.isTemporaryChat { return .unavailable(.temporaryChat) }
        if inputs.hasProject, inputs.skillStateEnabled { return .unavailable(.skillState) }
        return .whenNeeded
    }
}
