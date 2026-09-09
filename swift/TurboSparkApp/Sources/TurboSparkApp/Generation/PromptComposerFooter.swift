import SwiftUI

/// The composer's control row: mode toggle and attachments on the leading
/// edge, compact model/reasoning/guardrails status and the send action on
/// the trailing edge.
///
/// One `HStack` with a real `Spacer` between a leading `FlowLayout` and a
/// trailing cluster, exactly the shape `FlowLayout` documents itself for
/// ("so this can sit beside a Spacer without claiming the row"). The
/// leading side keeps the wrap-on-overflow safety `FlowLayout` exists for
/// (`FlowLayout.swift`'s own comment: an `HStack` compresses a capsule pill
/// with no floor); the trailing side is now flat text rather than boxed
/// pills, so it reads as one slim row in the common case, matching the
/// Claude / Claude Code composer footer.
struct PromptComposerFooter: View {
    @ObservedObject var model: AppModel
    @Binding var showingPromptTips: Bool
    let isExtractingDocuments: Bool
    let onAttachFiles: () -> Void
    let onAttachFolder: () -> Void
    let onNewProject: () -> Void
    let onAddMcpServer: () -> Void
    let onCreateSkill: () -> Void
    let onInsertPromptText: (String) -> Void
    var promptFocused: FocusState<Bool>.Binding

    @ScaledMetric private var iconButtonSize: CGFloat = 28
    @Environment(\.appTheme) private var theme

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            leading
            Spacer(minLength: 8)
            trailing
        }
    }

    private var leading: some View {
        HStack(spacing: 6) {
            PromptComposerPlusMenu(
                model: model,
                iconButtonSize: iconButtonSize,
                isRunning: model.isRunning,
                isExtracting: isExtractingDocuments,
                onAttachFiles: onAttachFiles,
                onAttachFolder: onAttachFolder,
                onNewProject: onNewProject,
                onAddMcpServer: onAddMcpServer,
                onCreateSkill: onCreateSkill,
                onInsertPromptText: onInsertPromptText
            )

            ToolApprovalDropdown(model: model)

            SearchToggleButton(model: model)

            if model.selectedProject != nil {
                PromptProjectContextPill(model: model)
            }
        }
    }

    private var trailing: some View {
        HStack(spacing: 6) {
            // First, not last: `clearAction` below is conditional on the
            // draft, and the ring should not change position with it.
            ContextUsageRingView(model: model)

            clearAction

            PromptAudioInputButton(promptFocused: promptFocused)

            GenerateControl(model: model)
        }
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused.wrappedValue = true
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .themedFont(.callout)
                    .foregroundStyle(.tertiary)
                    .frame(width: iconButtonSize, height: iconButtonSize)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Clear prompt")
            .accessibilityLabel("Clear prompt")
            .accessibilityHint("Empties the prompt editor")
        }
    }
}
