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
    let onAttach: () -> Void
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
        FlowLayout(spacing: 6, lineSpacing: 6) {
            PromptAttachDocumentButton(
                iconButtonSize: iconButtonSize,
                isRunning: model.isRunning,
                isExtracting: isExtractingDocuments,
                onAttach: onAttach
            )
            PromptTipsButton(
                iconButtonSize: iconButtonSize,
                showingTips: $showingPromptTips
            )
            // Guarded here as well as inside the pill: a view that renders
            // nothing is still a subview, and FlowLayout would leave its
            // spacing behind as a phantom gap.
            if model.selectedProject != nil {
                PromptProjectContextPill(model: model)
            }
        }
    }

    private var trailing: some View {
        HStack(spacing: 4) {
            PromptModelSelectorPill(model: model)
            if model.reasoningPickerEnabled {
                statusDot
                PromptReasoningPillControl(model: model)
            }
            statusDot
            ForgeGuardrailsPillControl(model: model)

            if model.estimatedPromptTokens > 0 {
                Text("\(model.estimatedPromptTokens) tokens")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .fixedSize()
                    .padding(.leading, 4)
                    .help("Estimated prompt length: \(model.estimatedPromptTokens) tokens")
                    .accessibilityLabel("Estimated prompt length: \(model.estimatedPromptTokens) tokens")
            }

            clearAction
                .padding(.leading, 2)

            GenerateControl(model: model)
                .padding(.leading, 4)
        }
    }

    private var statusDot: some View {
        Text("\u{00B7}")
            .font(theme.ui(points: 11))
            .foregroundStyle(.tertiary)
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused.wrappedValue = true
            } label: {
                Label("Clear prompt", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: iconButtonSize, height: iconButtonSize)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear prompt")
            .accessibilityLabel("Clear prompt")
            .accessibilityHint("Empties the prompt editor")
        } else if !model.isRunning && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label("Clear chat history", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: iconButtonSize, height: iconButtonSize)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear chat history")
            .accessibilityLabel("Clear chat history")
            .accessibilityHint("Removes the current chat transcript")
        }
    }
}
