import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Left-column model card in the Model Hub list.
@MainActor
struct ModelCardView: View {
    let alias: String
    let name: String
    let family: String
    /// The catalog row's own `status`: `verified`, `runs` or `caveat`.
    let status: String
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
                        Text(verbatim: alias)
                            .themedFont(.base, weight: .semibold)
                            .foregroundStyle(.appText)
                            .lineLimit(1)

                        statusSeal

                        Spacer(minLength: 2)

                        if isDownloading {
                            HStack(spacing: 4) {
                                ProgressView().controlSize(.mini)
                                if let fraction = downloadFraction {
                                    Text(verbatim: "\(Int(fraction * 100))%")
                                        .themedFont(.tiny, weight: .medium).monospacedDigit()
                                        .foregroundStyle(Color.accentColor)
                                }
                            }
                            .accessibilityElement(children: .ignore)
                            .accessibilityLabel(Text("Downloading", bundle: .module))
                            .accessibilityValue(
                                downloadFraction.map { Text(verbatim: MetricFormat.percent($0 * 100)) }
                                    ?? Text("In progress", bundle: .module))
                        } else if isActive {
                            Text("Active", bundle: .module)
                                .themedFont(.tiny, weight: .bold)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1.5)
                                .background(Color.accentColor.opacity(0.18), in: Capsule())
                                .foregroundStyle(Color.accentColor)
                        } else if isInstalled {
                            HStack(spacing: 3) {
                                Circle().fill(.appSuccess).frame(width: 5, height: 5)
                                Text("Installed", bundle: .module)
                                    .themedFont(.tiny, weight: .medium)
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                    }

                    Text(verbatim: name)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                        .truncationMode(.tail)

                    // The tags wrap onto a second line in a narrow column (or
                    // at a large text size) instead of being clipped; the
                    // size stays on the trailing edge of the first line.
                    HStack(alignment: .top, spacing: 4) {
                        FlowLayout(spacing: 4, lineSpacing: 3) {
                            metadataTag {
                                HStack(spacing: 3) {
                                    Image(systemName: visuals.iconSystemName)
                                        .themedFont(.micro, weight: .semibold)
                                        .accessibilityHidden(true)
                                    Text(verbatim: visuals.parameterTag)
                                }
                            }
                            metadataTag { Text(verbatim: visuals.formatLabel) }
                                .help(Text(verbatim: visuals.formatLabel))

                            if let rec = recommendation {
                                verdictBadge(rec.verdict)
                            }
                        }

                        Spacer(minLength: 2)

                        Text(verbatim: MetricFormat.storage(downloadBytes))
                            .themedFont(.tiny).monospacedDigit()
                            .lineLimit(1)
                            .fixedSize()
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
        .help(Text(verbatim: "\(alias) (\(name))"))
        .accessibilityLabel(Text(verbatim: "\(alias), \(visuals.family)"))
        .accessibilityValue(cardAccessibilityValue)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    /// Spoken summary of the card. Names and sizes are data and stay
    /// verbatim; the state words come from the catalog.
    private var cardAccessibilityValue: Text {
        var parts: [Text] = [Text(verbatim: name)]
        if let statusAccessibilityText { parts.append(statusAccessibilityText) }
        if isActive { parts.append(Text("Active", bundle: .module)) }
        if isInstalled && !isActive { parts.append(Text("Installed", bundle: .module)) }
        if isDownloading { parts.append(Text("Downloading", bundle: .module)) }
        if let rec = recommendation {
            switch rec.verdict {
            case .resident: parts.append(Text("Fits Memory", bundle: .module))
            case .streams: parts.append(Text("Streams", bundle: .module))
            case .tight: parts.append(Text("Tight fit", bundle: .module))
            case .refused: parts.append(Text("Too large", bundle: .module))
            case .unknown: break
            }
        }
        parts.append(Text(verbatim: MetricFormat.storage(downloadBytes)))
        return parts.dropFirst().reduce(parts[0]) { $0 + Text(verbatim: ", ") + $1 }
    }

    /// A small neutral tag in the card's metadata row.
    private func metadataTag<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .themedFont(.tiny, weight: .medium)
            .lineLimit(1)
            .truncationMode(.tail)
            .padding(.horizontal, 4)
            .padding(.vertical, 1.5)
            .background(.appElevated, in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(.appSecondary)
    }

    private var avatarIcon: some View {
        ModelLogoView(visuals: visuals, size: 36, cornerRadius: 8)
    }

    /// The catalog's own verdict on the row.
    ///
    /// This used to be an unconditional `checkmark.seal.fill` captioned
    /// "Verified model in TurboSpark catalog" on EVERY row, which is a claim
    /// about a field the view never read: `models.json` marks rows `verified`,
    /// `runs` or `caveat`, and one of the shipped rows is a caveat. An
    /// always-on badge is the same failure as an always-true filter -- it
    /// looks like information and carries none.
    @ViewBuilder
    private var statusSeal: some View {
        switch status {
        case "verified":
            Image(systemName: "checkmark.seal.fill")
                .themedFont(.tiny)
                .foregroundStyle(.appAccent)
                .help(Text("Verified: this port has run this row end to end", bundle: .module))
                .accessibilityHidden(true)
        case "caveat":
            Image(systemName: "exclamationmark.triangle.fill")
                .themedFont(.tiny)
                .foregroundStyle(.appWarning)
                .help(Text("Runs with a caveat; see the model notes.", bundle: .module))
                .accessibilityHidden(true)
        default:
            EmptyView()
        }
    }

    /// Spoken form of the status seal, which is decorative for sighted users.
    private var statusAccessibilityText: Text? {
        switch status {
        case "verified": return Text("Verified", bundle: .module)
        case "caveat": return Text("Caveat", bundle: .module)
        default: return nil
        }
    }

    @ViewBuilder
    private func verdictBadge(_ verdict: ModelRecommendation.FitVerdict) -> some View {
        ModelFitVerdictPill(verdict: verdict, compact: true)
    }
}
