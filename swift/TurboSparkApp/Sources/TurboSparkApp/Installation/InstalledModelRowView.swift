import AppKit
import SwiftUI
import TurboSpark

/// A row item in the ModelManager master list representing an installed model.
struct InstalledModelRowView: View {
    let modelItem: InstalledModel
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

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 4) {
                        Text(verbatim: modelItem.alias)
                            .themedFont(.base, weight: .semibold)
                            .lineLimit(1)

                        if isFavorite {
                            Image(systemName: "star.fill")
                                .themedFont(.micro)
                                .foregroundStyle(Color.yellow)
                        }

                        Spacer(minLength: 2)

                        if isLoaded {
                            HStack(spacing: 3) {
                                Circle().fill(Color.green).frame(width: 5, height: 5)
                                Text("Loaded", bundle: .module)
                                    .themedFont(.tiny, weight: .bold)
                                    .foregroundStyle(Color.green)
                            }
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(Color.green.opacity(0.12), in: Capsule())
                        }
                    }

                    // Feature & Source Badges. Same density rule as the
                    // Discover cards: badges wrap instead of clipping, size
                    // stays on the trailing edge of the first line.
                    HStack(alignment: .top, spacing: 4) {
                        FlowLayout(spacing: 4, lineSpacing: 3) {
                            Text(LocalizedStringKey(descriptor.storageSource.shortLabel), bundle: .module)
                                .themedFont(.micro, weight: .semibold)
                                .lineLimit(1)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1.5)
                                .background(.appElevated, in: RoundedRectangle(cornerRadius: 3))
                                .foregroundStyle(.appSecondary)

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
        .help(Text(verbatim: modelItem.alias))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(modelItem.alias)
        .accessibilityValue(accessibilityDescription)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    /// Spoken row summary: state words from the catalog, data verbatim.
    private var accessibilityDescription: Text {
        var parts: [Text] = []
        if isLoaded { parts.append(Text("Loaded", bundle: .module)) }
        if isFavorite { parts.append(Text("Favorite", bundle: .module)) }
        parts.append(Text(LocalizedStringKey(descriptor.routingType.label), bundle: .module))
        parts.append(Text(verbatim: MetricFormat.storage(modelItem.installBytes)))
        if !tags.isEmpty {
            parts.append(Text("Tags:", bundle: .module) + Text(verbatim: " " + tags.joined(separator: ", ")))
        }
        return parts.dropFirst().reduce(parts[0]) { $0 + Text(verbatim: ", ") + $1 }
    }
}
