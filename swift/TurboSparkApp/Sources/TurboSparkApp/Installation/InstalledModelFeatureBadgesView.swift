import SwiftUI
import TurboSpark

/// Feature badges strip displaying capabilities, architecture highlights, and quant formats.
struct InstalledModelFeatureBadgesView: View {
    let descriptor: ModelFeatureDescriptor

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model Highlights & Features", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)

            FlowLayout(spacing: 6, lineSpacing: 6) {
                // MoE vs Dense
                if descriptor.routingType == .moe {
                    ModelFeatureBadgeView.moe(details: descriptor.routingDetails, style: .regular)
                } else {
                    ModelFeatureBadgeView.dense(details: "Dense Transformer", style: .regular)
                }

                // Quantization format
                ModelFeatureBadgeView.quant(descriptor.quantFormat, style: .regular)

                // Drafter
                if descriptor.speculativeDrafter == .dynamicSloth {
                    ModelFeatureBadgeView.dynamicSloth(style: .regular)
                } else if descriptor.speculativeDrafter == .mtp {
                    ModelFeatureBadgeView.mtp(style: .regular)
                }

                // Weight container
                ModelFeatureBadgeView.format(descriptor.format, style: .regular)

                // Live Steering
                if descriptor.isSteeringReady {
                    ModelFeatureBadgeView.steeringReady(style: .regular)
                }

                // Linear Attention
                if descriptor.hasLinearAttention {
                    ModelFeatureBadgeView.linearAttention(style: .regular)
                }

                // Chunked Prefill
                if descriptor.supportsChunkedPrefill {
                    ModelFeatureBadgeView(
                        title: "Chunked Prefill",
                        iconSystemName: "rectangle.split.3x1.fill",
                        tintColor: .teal,
                        tooltip: "Chunked GPU prefill for long prompts without watchdog timeouts",
                        style: .regular
                    )
                }

                // Reasoning
                if descriptor.supportsReasoning {
                    ModelFeatureBadgeView(
                        title: "Reasoning Channel",
                        iconSystemName: "brain.head.profile",
                        tintColor: .indigo,
                        tooltip: "Supports chain-of-thought and internal reasoning extraction",
                        style: .regular
                    )
                }

                // Tool calls. THREE states, not two: the fact is a property
                // of the checkpoint's dialect, which is resolved from the
                // tokenizer at load and appears in no file this pane can read
                // -- so before the model is opened the honest answer is that
                // it is not known. This badge was an unconditional `true`,
                // i.e. one that could not fail (swift/CLAUDE.md Gotcha 22).
                //
                // Note what "Prompted" does NOT mean: guardrails still works
                // there, and the rescue that recovers a call from raw prose
                // is worth MORE on a checkpoint whose framing hands none
                // over. That is why there is no arm that hides the badge.
                switch descriptor.supportsToolCalls {
                case .some(true):
                    ModelFeatureBadgeView(
                        title: "Native Tool Calls",
                        iconSystemName: "hammer.fill",
                        tintColor: .orange,
                        tooltip:
                            "This checkpoint's own markup frames tool calls, so a call arrives "
                            + "already parsed. Forge Guardrails validates its arguments.",
                        style: .regular
                    )
                case .some(false):
                    ModelFeatureBadgeView(
                        title: "Prompted Tool Calls",
                        iconSystemName: "hammer",
                        tintColor: .secondary,
                        tooltip:
                            "This checkpoint's framing hands no tool call over, so one has to be "
                            + "prompted for and recovered from the reply. That is exactly what "
                            + "Forge Guardrails' rescue does, so leave it on here.",
                        style: .regular
                    )
                case .none:
                    ModelFeatureBadgeView(
                        title: "Tool Calls: unknown",
                        iconSystemName: "questionmark.circle",
                        tintColor: .secondary,
                        tooltip:
                            "Load this model to find out. Whether its own markup frames tool "
                            + "calls is a property of its chat dialect, which is only resolved "
                            + "when the tokenizer loads.",
                        style: .regular
                    )
                }

                // Vision
                if descriptor.supportsVision {
                    ModelFeatureBadgeView(
                        title: "Vision Pipeline",
                        iconSystemName: "eye.fill",
                        tintColor: .pink,
                        tooltip: "Multi-modal vision pipeline with mRoPE positional dispatch",
                        style: .regular
                    )
                }

                // Storage source
                ModelFeatureBadgeView.source(descriptor.storageSource, style: .regular)
            }
        }
        .padding(14)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }
}
