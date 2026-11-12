import SwiftUI
import TurboSpark

/// One live Hugging Face row in the Discover list.
///
/// **THE BADGES ARE ADVISORY AND SAY SO.** A row is one list request: it knows
/// the weight format the repository advertises and whether this port knows its
/// architecture name. It does not know the model runs, so nothing here is
/// green. Only the check in the detail pane produces a verdict.
@MainActor
struct HubResultRowView: View {
    let result: HubSearchResult
    let isSelected: Bool
    let onSelect: () -> Void

    @Environment(\.appTheme) private var theme

    var body: some View {
        Button(action: onSelect) {
            VStack(alignment: .leading, spacing: 4) {
                Text(verbatim: result.repoId)
                    .themedFont(.base, weight: .semibold)
                    .foregroundStyle(.appText)
                    .lineLimit(1)
                    .truncationMode(.middle)

                FlowLayout(spacing: 4, lineSpacing: 3) {
                    if result.hasGguf { tag { Text(verbatim: "GGUF") } }
                    if result.hasMlx { tag { Text(verbatim: "MLX") } }
                    archBadge
                    if result.gated { gatedBadge }
                    if !result.bundledAliases.isEmpty { catalogBadge }
                }

                HStack(spacing: 10) {
                    if let downloads = result.downloads {
                        stat(systemImage: "arrow.down.circle", value: downloads)
                    }
                    if let likes = result.likes {
                        stat(systemImage: "heart", value: likes)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .contentShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .background(
            isSelected ? AnyShapeStyle(.appAccent.opacity(0.14)) : AnyShapeStyle(Color.clear),
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(isSelected ? AnyShapeStyle(.appAccent.opacity(0.4)) : AnyShapeStyle(Color.clear), lineWidth: 1)
        )
        .help(Text(verbatim: result.repoId))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text(verbatim: result.repoId))
        .accessibilityValue(accessibilitySummary)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    // MARK: - Badges

    @ViewBuilder private var archBadge: some View {
        switch result.archHint {
        case .recognized:
            tag {
                HStack(spacing: 3) {
                    Image(systemName: "cpu").themedFont(.micro, weight: .bold).accessibilityHidden(true)
                    Text("Architecture recognized", bundle: .module)
                }
            }
            .help(Text("This app knows this architecture. Select the model to check whether it runs.", bundle: .module))
        case .planned:
            warningTag(symbol: "hourglass", Text("Not supported yet", bundle: .module))
                .help(Text(verbatim: result.archNeeds ?? ""))
        case .unrecognized:
            warningTag(symbol: "exclamationmark.triangle.fill", Text("Architecture not supported", bundle: .module))
        case .unknown:
            EmptyView()
        }
    }

    private var gatedBadge: some View {
        warningTag(symbol: "lock.fill", Text("Gated", bundle: .module))
            .help(Text("This repository needs a Hugging Face sign-in or an accepted licence.", bundle: .module))
    }

    private var catalogBadge: some View {
        tag {
            HStack(spacing: 3) {
                Image(systemName: "checkmark.seal").themedFont(.micro, weight: .bold).accessibilityHidden(true)
                Text("In catalog", bundle: .module)
            }
        }
        .help(Text("This repository is also in the curated catalog.", bundle: .module))
    }

    private func tag<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .themedFont(.tiny, weight: .medium)
            .foregroundStyle(.appSecondary)
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 4))
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(.appBorder, lineWidth: 0.5))
    }

    /// Pairs a glyph with the words, so a warning never rests on color alone.
    private func warningTag(symbol: String, _ text: Text) -> some View {
        HStack(spacing: 3) {
            Image(systemName: symbol).themedFont(.micro, weight: .bold).accessibilityHidden(true)
            text
        }
        .themedFont(.tiny, weight: .semibold)
        .foregroundStyle(.appWarning)
        .padding(.horizontal, 5)
        .padding(.vertical, 1.5)
        .background(.appWarningTint, in: RoundedRectangle(cornerRadius: 4))
    }

    private func stat(systemImage: String, value: UInt64) -> some View {
        HStack(spacing: 3) {
            Image(systemName: systemImage).themedFont(.micro).accessibilityHidden(true)
            Text(verbatim: value.formatted(.number.notation(.compactName)))
                .themedFont(.tiny).monospacedDigit()
        }
        .foregroundStyle(.appSecondary)
    }

    /// Spoken summary: format, architecture hint and gating, in words.
    private var accessibilitySummary: Text {
        var parts: [Text] = []
        if result.hasGguf { parts.append(Text(verbatim: "GGUF")) }
        if result.hasMlx { parts.append(Text(verbatim: "MLX")) }
        switch result.archHint {
        case .recognized: parts.append(Text("Architecture recognized", bundle: .module))
        case .planned: parts.append(Text("Not supported yet", bundle: .module))
        case .unrecognized: parts.append(Text("Architecture not supported", bundle: .module))
        case .unknown: break
        }
        if result.gated { parts.append(Text("Gated", bundle: .module)) }
        guard var summary = parts.first else { return Text(verbatim: "") }
        for part in parts.dropFirst() { summary = summary + Text(verbatim: ", ") + part }
        return summary
    }
}
