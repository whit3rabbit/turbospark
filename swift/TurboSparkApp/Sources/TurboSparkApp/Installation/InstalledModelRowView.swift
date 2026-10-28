import AppKit
import SwiftUI
import TurboSpark

/// A row item in the ModelManager master list representing an installed model.
///
/// Where the model is, and whether it can be opened, come from `rowModel`
/// (requirements 3.1, 3.7 and 5.3): a badge with the library, a glyph AND a word
/// for "Not connected" and "Missing", a faded logo and secondary text for those
/// rows, and the load-time estimate for a model on an external drive. The view
/// draws those strings and nothing else.
struct InstalledModelRowView: View {
    let modelItem: InstalledModel
    let rowModel: InstalledModelRowModel
    let isSelected: Bool
    let isLoaded: Bool
    let isFavorite: Bool
    let tags: [String]
    let onSelect: () -> Void

    private var descriptor: ModelFeatureDescriptor {
        ModelFeatureDescriptor.resolve(installedModel: modelItem)
    }

    private var visuals: ModelFamilyVisuals {
        ModelFamilyVisuals.resolve(
            alias: modelItem.alias,
            family: modelItem.family,
            name: modelItem.alias
        )
    }

    var body: some View {
        Button(action: onSelect) {
            HStack(alignment: .center, spacing: 10) {
                ModelLogoView(visuals: visuals, size: 36, cornerRadius: 8)
                    // Not text, so fading it cannot cost any contrast; the
                    // words below stay at full strength.
                    .opacity(rowModel.isDimmed ? 0.5 : 1)

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 4) {
                        Text(verbatim: modelItem.alias)
                            .themedFont(.base, weight: .semibold)
                            // Secondary only for a model that cannot be opened; any
                            // other row keeps the primary text it always had.
                            .foregroundStyle(rowModel.isDimmed ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.primary))
                            .lineLimit(1)

                        if isFavorite {
                            Image(systemName: "star.fill")
                                .themedFont(.micro)
                                .foregroundStyle(.appWarning)
                        }

                        Spacer(minLength: 2)

                        if isLoaded {
                            HStack(spacing: 3) {
                                Circle().fill(.appSuccess).frame(width: 5, height: 5)
                                Text("Loaded", bundle: .module)
                                    .themedFont(.tiny, weight: .bold)
                                    .foregroundStyle(.appSuccess)
                            }
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(.appSuccessTint, in: Capsule())
                        }
                    }

                    // The state of a model that cannot be opened: a glyph AND a
                    // word, so it never rests on color. Drawn above the badges
                    // so a narrow list and a long translation still show it.
                    if let status = rowModel.status, let text = rowModel.statusText {
                        HStack(alignment: .firstTextBaseline, spacing: 4) {
                            Image(systemName: status.glyph)
                                .themedFont(.tiny)
                                .accessibilityHidden(true)
                            Text(verbatim: text)
                                .themedFont(.tiny, weight: .semibold)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .foregroundStyle(.appText)
                    }

                    // Feature & Source Badges. Same density rule as the
                    // Discover cards: badges wrap instead of clipping, size
                    // stays on the trailing edge of the first line.
                    HStack(alignment: .top, spacing: 4) {
                        FlowLayout(spacing: 4, lineSpacing: 3) {
                            locationOrSourceBadge

                            if descriptor.routingType == .moe {
                                ModelFeatureBadgeView.moe(details: "MoE", style: .compact)
                            } else {
                                ModelFeatureBadgeView.dense(details: "Dense", style: .compact)
                            }

                            ModelFeatureBadgeView.quant(descriptor.quantFormat, style: .compact)

                            if descriptor.speculativeDrafter == .dynamicSloth {
                                ModelFeatureBadgeView.dynamicSloth(style: .compact)
                            } else if descriptor.speculativeDrafter == .mtp {
                                ModelFeatureBadgeView.mtp(style: .compact)
                            }
                        }

                        Spacer(minLength: 2)

                        HStack(spacing: 2) {
                            Image(systemName: "internaldrive")
                                .themedFont(.micro)
                                .foregroundStyle(.tertiary)
                                .accessibilityHidden(true)
                            Text(verbatim: MetricFormat.storage(modelItem.installBytes))
                                .themedFont(.tiny).monospacedDigit()
                                .foregroundStyle(.appSecondary)
                        }
                        .fixedSize()
                    }

                    // How long the drive will take (requirement 5.3). Only for a
                    // model on an external drive whose link has a known rate.
                    if let estimate = rowModel.estimateText {
                        HStack(alignment: .firstTextBaseline, spacing: 4) {
                            Image(systemName: "clock")
                                .themedFont(.micro)
                                .accessibilityHidden(true)
                            Text(verbatim: estimate)
                                .themedFont(.tiny)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .foregroundStyle(.appSecondary)
                    }

                    if !tags.isEmpty {
                        HStack(spacing: 3) {
                            ForEach(tags.prefix(3), id: \.self) { tag in
                                Text(verbatim: tag)
                                    .themedFont(.micro, weight: .medium)
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 1)
                                    .background(Color.teal.opacity(0.12), in: RoundedRectangle(cornerRadius: 3))
                                    .foregroundStyle(Color.teal)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .background(
            isSelected
                ? Color.teal.opacity(0.14)
                : Color.clear,
            in: RoundedRectangle(cornerRadius: 8)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    isSelected ? Color.teal.opacity(0.4) : Color.clear,
                    lineWidth: 1
                )
        )
        .help(Text(verbatim: rowModel.accessibilityLabel))
        .accessibilityElement(children: .combine)
        // One element: name, library, status, size, load-time estimate, in that
        // order (the model builds it), then the state words below as its value.
        .accessibilityLabel(rowModel.accessibilityLabel)
        .accessibilityValue(accessibilityDescription)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    /// A model the engine lists says which library holds it. A model only a scan
    /// found keeps the source badge (TurboSpark, LM Studio, Custom Folder): the
    /// source label of a library model would read "Custom Folder", which it is
    /// not. A long library name is cut at the tail, with the full name on hover.
    @ViewBuilder private var locationOrSourceBadge: some View {
        if let text = rowModel.locationText, let glyph = rowModel.locationGlyph {
            HStack(spacing: 3) {
                Image(systemName: glyph)
                    .themedFont(.micro)
                    .accessibilityHidden(true)
                Text(verbatim: text)
                    .themedFont(.micro, weight: .semibold)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            // Short enough that the badge, the size and the logo fit the list
            // column at its narrowest (260 pt): a longer name would push the
            // size off the row's trailing edge.
            .frame(maxWidth: 120, alignment: .leading)
            .padding(.horizontal, 4)
            .padding(.vertical, 1.5)
            .background(.appElevated, in: RoundedRectangle(cornerRadius: 3))
            .foregroundStyle(.appSecondary)
            .help(Text(verbatim: text))
        } else {
            Text(LocalizedStringKey(descriptor.storageSource.shortLabel), bundle: .module)
                .themedFont(.micro, weight: .semibold)
                .lineLimit(1)
                .padding(.horizontal, 4)
                .padding(.vertical, 1.5)
                .background(.appElevated, in: RoundedRectangle(cornerRadius: 3))
                .foregroundStyle(.appSecondary)
        }
    }

    /// Spoken row state: state words from the catalog, data verbatim. The name,
    /// library, status, size and estimate are the label.
    private var accessibilityDescription: Text {
        var parts: [Text] = []
        if isLoaded { parts.append(Text("Loaded", bundle: .module)) }
        if isFavorite { parts.append(Text("Favorite", bundle: .module)) }
        parts.append(Text(LocalizedStringKey(descriptor.routingType.label), bundle: .module))
        if !tags.isEmpty {
            parts.append(Text("Tags:", bundle: .module) + Text(verbatim: " " + tags.joined(separator: ", ")))
        }
        return parts.dropFirst().reduce(parts[0]) { $0 + Text(verbatim: ", ") + $1 }
    }
}
