import SwiftUI
import TurboSpark

/// Row view displaying a model recommendation, hardware fit specs, and install action.
struct ModelRecommendationRow: View {
    let recommendation: ModelRecommendation
    @ObservedObject var model: AppModel
    /// The library the install goes into (nil is This Mac). The list above the
    /// rows owns the menu that chose it.
    var library: String? = nil

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
                    Text(verbatim: recommendation.alias)
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
                        .help(Text("Streams weights dynamically with minimal RAM footprint", bundle: .module))
                    }

                    RecommendationVerdictBadge(verdict: recommendation.verdict)
                }

                Text(verbatim: recommendation.name)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)

                HStack(spacing: 8) {
                    HStack(spacing: 3) {
                        Image(systemName: "memorychip")
                            .themedFont(.tiny)
                            .accessibilityHidden(true)
                        if isMoE {
                            Text("RAM: ~\(MetricFormat.storage(recommendation.countedBytes)) (Cache)", bundle: .module)
                        } else {
                            Text("RAM: ~\(MetricFormat.storage(recommendation.countedBytes))", bundle: .module)
                        }
                    }
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.appSecondary)
                    .help(Text("Estimated unified memory footprint", bundle: .module))

                    Text(verbatim: "-")
                        .themedFont(.tiny)
                        .foregroundStyle(.tertiary)

                    HStack(spacing: 3) {
                        Image(systemName: "internaldrive")
                            .themedFont(.tiny)
                        Text("Disk: \(MetricFormat.storage(recommendation.installBytes))", bundle: .module)
                    }
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.appSecondary)
                    .help(Text("On-disk installation size", bundle: .module))

                    if let minRate = recommendation.toksPerSecondMin, let maxRate = recommendation.toksPerSecondMax {
                        Text(verbatim: "-")
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)
                        Text(verbatim: "\(Int(minRate))-\(Int(maxRate)) tok/s")
                            .themedFont(.tiny).monospacedDigit()
                            .foregroundStyle(.appAccent)
                    }
                }
            }

            Spacer(minLength: 8)

            // Install Button
            Button {
                model.installModel(alias: recommendation.alias, library: library)
            } label: {
                Label { Text("Install & Start", bundle: .module) } icon: { Image(systemName: "arrow.down.circle.fill") }
            }
            .appAction(.secondary)
            .disabled(model.isRunning || !model.canInstall(alias: recommendation.alias))
            .help(Text("Downloads the model and starts a new chat with it", bundle: .module))
            .accessibilityLabel(Text("Install and start \(recommendation.alias)", bundle: .module))
        }
        .padding(12)
        .background(
            .appSurface,
            in: RoundedRectangle(cornerRadius: 12)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(
                    isMoE ? AnyShapeStyle(Color.purple.opacity(0.3)) : AnyShapeStyle(.appBorder),
                    lineWidth: isMoE ? 1 : 0.5
                )
        )
    }

    /// Determines whether a recommendation refers to a Mixture-of-Experts architecture.
    static func isMoEModel(_ rec: ModelRecommendation) -> Bool {
        // One source of truth: the descriptor's exact family set. Alias
        // substrings mislabeled dense models (ornith, ternary) as MoE.
        return rec.slotCacheSlots > 0 || ModelFeatureDescriptor.isKnownMoEFamily(rec.family)
    }
}

/// Visual badge reflecting the hardware fit verdict of a model recommendation.
struct RecommendationVerdictBadge: View {
    let verdict: ModelRecommendation.FitVerdict

    /// The same words the fit pill uses, so a verdict reads the same in
    /// every surface (`ModelFitPresentation` owns the mapping).
    private var verdictHelp: Text {
        Text(LocalizedStringKey(ModelFitPresentation.of(verdict).help), bundle: .module)
    }

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
            .help(verdictHelp)
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
            .help(verdictHelp)
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
            .help(verdictHelp)
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
            .help(verdictHelp)
        case .unknown:
            EmptyView()
        }
    }
}
