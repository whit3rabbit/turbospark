import SwiftUI
import TurboSpark

/// Left-column model card in the Model Hub list.
struct ModelCardView: View {
    let alias: String
    let name: String
    let family: String
    let downloadBytes: UInt64
    let isInstalled: Bool
    let isActive: Bool
    let isDownloading: Bool
    let downloadFraction: Double?
    let recommendation: ModelRecommendation?
    let isSelected: Bool
    let onSelect: () -> Void

    private var visuals: ModelFamilyVisuals {
        ModelFamilyVisuals.resolve(alias: alias, family: family, name: name)
    }

    var body: some View {
        Button(action: onSelect) {
            HStack(alignment: .center, spacing: 10) {
                avatarIcon

                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .center, spacing: 4) {
                        Text(alias)
                            .font(.system(.body, design: .rounded).weight(.semibold))
                            .foregroundStyle(.primary)
                            .lineLimit(1)

                        Image(systemName: "checkmark.seal.fill")
                            .font(.caption2)
                            .foregroundStyle(Color.accentColor)
                            .help("Verified model in TurboSpark catalog")
                            .accessibilityHidden(true)

                        Spacer(minLength: 2)

                        if isDownloading {
                            HStack(spacing: 4) {
                                ProgressView().controlSize(.mini)
                                if let fraction = downloadFraction {
                                    Text("\(Int(fraction * 100))%")
                                        .font(.caption2.monospacedDigit().weight(.medium))
                                        .foregroundStyle(Color.accentColor)
                                }
                            }
                            .accessibilityElement(children: .ignore)
                            .accessibilityLabel("Downloading")
                            .accessibilityValue(downloadFraction.map { "\(Int($0 * 100)) percent" } ?? "in progress")
                        } else if isActive {
                            Text("Active")
                                .font(.caption2.weight(.bold))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1.5)
                                .background(Color.accentColor.opacity(0.18), in: Capsule())
                                .foregroundStyle(Color.accentColor)
                        } else if isInstalled {
                            HStack(spacing: 3) {
                                Circle().fill(.green).frame(width: 5, height: 5)
                                Text("Installed")
                                    .font(.caption2.weight(.medium))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    Text(name)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)

                    HStack(spacing: 4) {
                        HStack(spacing: 3) {
                            Image(systemName: visuals.iconSystemName)
                                .font(.system(size: 8, weight: .semibold))
                            Text(visuals.parameterTag)
                        }
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1.5)
                        .background(Color(nsColor: .quaternaryLabelColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.secondary)

                        Text(visuals.formatLabel)
                            .font(.caption2.weight(.medium))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1.5)
                            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 4))
                            .foregroundStyle(.secondary)

                        if let rec = recommendation {
                            verdictBadge(rec.verdict)
                        }

                        Spacer(minLength: 2)

                        Text(MetricFormat.storage(downloadBytes))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .contentShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .background(
            isSelected
                ? Color.accentColor.opacity(0.14)
                : Color.clear,
            in: RoundedRectangle(cornerRadius: 10)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(
                    isSelected ? Color.accentColor.opacity(0.4) : Color.clear,
                    lineWidth: 1
                )
        )
        .accessibilityLabel("\(alias), \(visuals.family)")
        .accessibilityValue(cardAccessibilityValue)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var cardAccessibilityValue: String {
        var parts: [String] = [name]
        if isActive { parts.append("Active") }
        if isInstalled && !isActive { parts.append("Installed") }
        if isDownloading { parts.append("Downloading") }
        if let rec = recommendation {
            switch rec.verdict {
            case .resident: parts.append("Fits memory")
            case .streams: parts.append("Streams fast")
            case .tight: parts.append("Tight fit")
            case .refused: parts.append("Too large")
            case .unknown: break
            }
        }
        parts.append(MetricFormat.storage(downloadBytes))
        return parts.joined(separator: ", ")
    }

    private var avatarIcon: some View {
        ModelLogoView(visuals: visuals, size: 36, cornerRadius: 8)
    }

    @ViewBuilder
    private func verdictBadge(_ verdict: ModelRecommendation.FitVerdict) -> some View {
        switch verdict {
        case .resident:
            HStack(spacing: 2) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 7, weight: .bold))
                Text("Resident")
            }
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 4)
            .padding(.vertical, 1.5)
            .background(Color.green.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.green)
        case .streams:
            HStack(spacing: 2) {
                Image(systemName: "bolt.fill")
                    .font(.system(size: 7, weight: .bold))
                Text("Streams")
            }
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 4)
            .padding(.vertical, 1.5)
            .background(Color.blue.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.blue)
        case .tight:
            HStack(spacing: 2) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 7, weight: .bold))
                Text("Tight")
            }
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 4)
            .padding(.vertical, 1.5)
            .background(Color.orange.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.orange)
        case .refused:
            HStack(spacing: 2) {
                Image(systemName: "xmark.octagon.fill")
                    .font(.system(size: 7, weight: .bold))
                Text("Too Large")
            }
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 4)
            .padding(.vertical, 1.5)
            .background(Color.red.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.red)
        case .unknown:
            EmptyView()
        }
    }

}
