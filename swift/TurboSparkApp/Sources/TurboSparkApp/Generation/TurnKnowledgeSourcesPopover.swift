import SwiftUI

extension AppModel {
    /// The popover's inputs, read from the same state the prompt path reads.
    ///
    /// `projectRulesSet` is passed in because resolving it reads the rule
    /// files from disk; the popover resolves it once when it opens rather
    /// than on every model publish.
    func turnKnowledgeSourceInputs(projectRulesSet: Bool) -> TurnKnowledgeSources.Inputs {
        let project = turnProject(chatID: selectedChatID)
        let components = contextUsageSummary?.components ?? []
        // The labels are the literals `buildEstimateParts` assigns; summed
        // because the memory slice can arrive as more than one row kind.
        func tokens(where match: (ContextUsageComponent) -> Bool) -> Int? {
            let matched = components.filter(match)
            return matched.isEmpty ? nil : matched.reduce(0) { $0 + $1.tokens }
        }
        let mcpCount = project.map {
            AppToolCatalogMcp.visibleServers(global: globalMcpServers, project: $0).count
        } ?? 0
        return TurnKnowledgeSources.Inputs(
            hasProject: project != nil,
            projectRulesSet: projectRulesSet,
            memoryEnabled: memoryEnabled,
            recallEnabled: conversationRecallEnabled,
            isTemporaryChat: selectedChat.isGhost,
            isProtectedProfile: ProfileVaultCoordinator.shared.isProtected,
            skillStateEnabled: project?.skillStateEnabled ?? false,
            memoryTokens: tokens { $0.kind == .memory },
            pinnedInstructionTokens: tokens { $0.label == "Pinned instructions" },
            attachmentCount: promptAttachments.count,
            attachmentTokens: AttachmentContextBudget.totalTokens(promptAttachments),
            mcpServerCount: mcpCount,
            mcpServerTokens: tokens { $0.label == "MCP servers" })
    }
}

/// "What feeds this turn": a read-only list of the knowledge sources the
/// next message draws on, opened from the project pill.
///
/// Read-only because no per-message switch exists for any of them (see
/// `TurnKnowledgeSources`). Rows whose switch lives in Settings link there.
struct TurnKnowledgeSourcesPopover: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    /// Resolved on appear: reading the rule files is disk work.
    @State private var projectRulesSet = false

    var body: some View {
        let rows = TurnKnowledgeSources.rows(
            model.turnKnowledgeSourceInputs(projectRulesSet: projectRulesSet))
        VStack(alignment: .leading, spacing: 12) {
            Text("What feeds this turn", bundle: .module)
                .font(theme.ui(.callout, weight: .semibold))
                .accessibilityAddTraits(.isHeader)

            VStack(alignment: .leading, spacing: 10) {
                ForEach(rows) { row in
                    rowView(row)
                }
            }

            Divider()

            Text("Shown for reference. These are changed in Settings or the project, not per message. Token counts are estimates.", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(theme.ui(.small))
        // Width scales with Text Size, like the Tools popover beside it.
        .frame(width: (320 * AppMetrics.textScale(theme)).rounded(), alignment: .leading)
        .padding(16)
        .onAppear {
            let project = model.turnProject(chatID: model.selectedChatID)
            projectRulesSet = !(project?.escapedProjectInstructionsForPrompt().isEmpty ?? true)
        }
    }

    private func rowView(_ row: TurnKnowledgeSources.Row) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: Self.sourceImage(row.source))
                .foregroundStyle(.appSecondary)
                .frame(width: 16)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    sourceName(row.source)
                        .font(theme.ui(.small, weight: .medium))
                    if let count = row.count {
                        Text(verbatim: "(\(count))")
                            .foregroundStyle(.appSecondary)
                            .monospacedDigit()
                    }
                }
                HStack(spacing: 8) {
                    statusLabel(row.status)
                    if let tokens = row.estimatedTokens {
                        // "~" marks the figure as an estimate, as in the Tools popover.
                        (Text(verbatim: "~") + Text("\(tokens) tokens", bundle: .module))
                            .foregroundStyle(.appSecondary)
                            .monospacedDigit()
                    }
                }
                .font(theme.ui(.tiny))
                scopeLabel(row.scope)
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 4)
            if let tab = Self.settingsTab(row.source) {
                Button {
                    model.openSettings(tab: tab)
                } label: {
                    Text("Open Settings", bundle: .module)
                }
                .buttonStyle(.link)
                .font(theme.ui(.tiny))
                .accessibilityLabel(Text("Open Settings", bundle: .module) + Text(verbatim: ", ") + sourceName(row.source))
            }
        }
        // Contain, not combine: combining would swallow the Settings link's
        // own action.
        .accessibilityElement(children: .contain)
    }

    private func sourceName(_ source: TurnKnowledgeSources.Source) -> Text {
        switch source {
        case .projectRules: return Text("Project Rules & Instructions", bundle: .module)
        case .memory: return Text("Memory", bundle: .module)
        case .conversationRecall: return Text("Recall earlier messages in long chats", bundle: .module)
        case .pinnedInstructions: return Text("Pinned instructions", bundle: .module)
        case .attachments: return Text("Attachments", bundle: .module)
        case .mcpServers: return Text("MCP servers", bundle: .module)
        }
    }

    /// Glyph plus word, so the state never rests on color alone.
    private func statusLabel(_ status: TurnKnowledgeSources.Status) -> some View {
        HStack(spacing: 3) {
            Image(systemName: Self.statusImage(status))
                .accessibilityHidden(true)
            switch status {
            case .included: Text("On", bundle: .module)
            case .off: Text("Off", bundle: .module)
            case .notSet: Text("Not set", bundle: .module)
            case .whenNeeded: Text("On, used when older messages no longer fit", bundle: .module)
            case .unavailable(.temporaryChat): Text("Not in temporary chats", bundle: .module)
            case .unavailable(.protectedProfile): Text("Off in a protected profile", bundle: .module)
            case .unavailable(.skillState): Text("Off while bounded execution state is on", bundle: .module)
            }
        }
        .foregroundStyle(status == .included ? AnyShapeStyle(.appSuccess) : AnyShapeStyle(.appSecondary))
    }

    private func scopeLabel(_ scope: TurnKnowledgeSources.Scope) -> Text {
        switch scope {
        case .allChats: return Text("Applies to all chats", bundle: .module)
        case .thisProject: return Text("Applies to this project", bundle: .module)
        case .thisChat: return Text("Applies to this chat only", bundle: .module)
        case .nextMessage: return Text("Applies to the next turn", bundle: .module)
        }
    }

    private static func sourceImage(_ source: TurnKnowledgeSources.Source) -> String {
        switch source {
        case .projectRules: return "doc.text"
        case .memory: return "brain"
        case .conversationRecall: return "text.magnifyingglass"
        case .pinnedInstructions: return "pin"
        case .attachments: return "paperclip"
        case .mcpServers: return "server.rack"
        }
    }

    private static func statusImage(_ status: TurnKnowledgeSources.Status) -> String {
        switch status {
        case .included: return "checkmark.circle"
        case .off: return "minus.circle"
        case .notSet: return "circle.dashed"
        case .whenNeeded: return "clock"
        case .unavailable: return "nosign"
        }
    }

    /// Where the row's switch lives. Project rules are edited in the project
    /// sheet, pins in the transcript and attachments in the tray, so those
    /// rows have no Settings link.
    private static func settingsTab(_ source: TurnKnowledgeSources.Source) -> AppSettingsView.SettingsTab? {
        switch source {
        case .memory: return .memory
        case .conversationRecall: return .general
        case .mcpServers: return .mcp
        case .projectRules, .pinnedInstructions, .attachments: return nil
        }
    }
}
