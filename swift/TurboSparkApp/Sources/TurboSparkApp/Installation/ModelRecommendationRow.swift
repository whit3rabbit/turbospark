import SwiftUI
import TurboSpark

/// Row view displaying a model recommendation, hardware fit specs, and install action.
struct ModelRecommendationRow: View {
    let recommendation: ModelRecommendation
    @ObservedObject var model: AppModel

    private var isMoE: Bool {
        ModelRecommendationRow.isMoEModel(recommendation)
    }

    var body: some View {
        let visuals = ModelFamilyVisuals.resolve(
            alias: recommendation.alias,
            family: recommendation.family ?? "",
            name: recommendation.name
        )

        HStack(alignment: .center, spacing: 12) {
            // Icon
            ModelLogoView(visuals: visuals, size: 40, cornerRadius: 10)

            // Info
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(recommendation.alias)
                        .themedFont(.base, weight: .bold)

                    if isMoE {
                        HStack(spacing: 3) {
                            Image(systemName: "bolt.fill")
                                .themedFont(.tiny)
                                .accessibilityHidden(true)
                            Text("MoE Streaming", bundle: .module)
                        }
                        .themedFont(.tiny, weight: .bold)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1.5)
                        .background(Color.purple.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(Color.purple)
                        .help("Streams weights dynamically with minimal RAM footprint")
                    }

                    RecommendationVerdictBadge(verdict: recommendation.verdict)
                }

                Text(recommendation.name)
                    .themedFont(.small)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                HStack(spacing: 8) {
                    HStack(spacing: 3) {
                        Image(systemName: "memorychip")
                            .themedFont(.tiny)
                        Text(isMoE ? "RAM: ~\(MetricFormat.storage(recommendation.countedBytes)) (Cache)" : "RAM: ~\(MetricFormat.storage(recommendation.countedBytes))")
                    }
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.secondary)
                    .help("Estimated unified memory footprint")

                    Text("-", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.tertiary)

                    HStack(spacing: 3) {
                        Image(systemName: "internaldrive")
                            .themedFont(.tiny)
                        Text("Disk: \(MetricFormat.storage(recommendation.installBytes))", bundle: .module)
                    }
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.secondary)
                    .help("On-disk installation size")

                    if let minRate = recommendation.toksPerSecondMin, let maxRate = recommendation.toksPerSecondMax {
                        Text("-", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)
                        Text("\(Int(minRate))-\(Int(maxRate)) tok/s", bundle: .module)
                            .themedFont(.tiny).monospacedDigit()
                            .foregroundStyle(TurboSparkTheme.accentColor)
                    }
                }
            }

            Spacer(minLength: 8)

            // Install Button
            Button {
                model.installModel(alias: recommendation.alias)
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.down.circle.fill")
                        .accessibilityHidden(true)
                    Text("Install & Start", bundle: .module)
                }
                .themedFont(.base, weight: .semibold)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .disabled(model.isInstallingModel)
            .help("Install and start \(recommendation.alias)")
            .accessibilityLabel("Install and start \(recommendation.alias)")
            .accessibilityHint("Downloads the model and starts a new chat with it")
        }
        .padding(12)
        .background(
            Color(nsColor: .controlBackgroundColor).opacity(0.6),
            in: RoundedRectangle(cornerRadius: 12)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(
                    isMoE ? Color.purple.opacity(0.3) : Color(nsColor: .separatorColor).opacity(0.5),
                    lineWidth: isMoE ? 1 : 0.5
                )
        )
    }

    /// Determines whether a recommendation refers to a Mixture-of-Experts architecture.
    static func isMoEModel(_ rec: ModelRecommendation) -> Bool {
        let alias = rec.alias.lowercased()
        let family = rec.family?.lowercased() ?? ""
        return rec.slotCacheSlots > 0
            || rec.verdict == .streams
            || alias.contains("gemma")
            || alias.contains("qwen3moe")
            || alias.contains("gptoss")
            || alias.contains("ornith")
            || alias.contains("ternary")
            || alias.contains("mixtral")
            || family.contains("moe")
    }
}

/// Visual badge reflecting the hardware fit verdict of a model recommendation.
struct RecommendationVerdictBadge: View {
    let verdict: ModelRecommendation.FitVerdict

    var body: some View {
        switch verdict {
        case .resident:
            HStack(spacing: 3) {
                Image(systemName: "checkmark.circle.fill")
                    .themedFont(.micro, weight: .bold)
                    .accessibilityHidden(true)
                Text("Fits Memory", bundle: .module)
            }
            .themedFont(.tiny, weight: .semibold)
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(Color.green.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.green)
            .help("Fits entirely in unified memory")
        case .streams:
            HStack(spacing: 3) {
                Image(systemName: "bolt.fill")
                    .themedFont(.micro, weight: .bold)
                    .accessibilityHidden(true)
                Text("Streams Fast", bundle: .module)
            }
            .themedFont(.tiny, weight: .semibold)
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(Color.blue.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.blue)
            .help("Streams experts dynamically from fast NVMe storage")
        case .tight:
            HStack(spacing: 3) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .themedFont(.micro, weight: .bold)
                Text("Tight Fit", bundle: .module)
            }
            .themedFont(.tiny, weight: .semibold)
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(Color.orange.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.orange)
            .help("Fits with tight memory headroom")
        case .refused:
            HStack(spacing: 3) {
                Image(systemName: "xmark.octagon.fill")
                    .themedFont(.micro, weight: .bold)
                Text("Too Large", bundle: .module)
            }
            .themedFont(.tiny, weight: .semibold)
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(Color.red.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.red)
            .help("Exceeds available system memory")
        case .unknown:
            EmptyView()
        }
    }
}

