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
                        Text(modelItem.alias)
                            .font(.system(.body, design: .rounded).weight(.semibold))
                            .lineLimit(1)

                        if isFavorite {
                            Image(systemName: "star.fill")
                                .font(.system(size: 9))
                                .foregroundStyle(Color.yellow)
                        }

                        Spacer(minLength: 2)

                        if isLoaded {
                            HStack(spacing: 3) {
                                Circle().fill(Color.green).frame(width: 5, height: 5)
                                Text("Loaded")
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(Color.green)
                            }
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(Color.green.opacity(0.12), in: Capsule())
                        }
                    }

                    // Feature & Source Badges
                    HStack(spacing: 4) {
                        Text(descriptor.storageSource.shortLabel)
                            .font(.system(size: 8, weight: .semibold))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1.5)
                            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 3))
                            .foregroundStyle(.secondary)

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

                        Spacer(minLength: 2)

                        HStack(spacing: 2) {
                            Image(systemName: "internaldrive")
                                .font(.system(size: 8))
                                .foregroundStyle(.tertiary)
                            Text(MetricFormat.storage(modelItem.installBytes))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }

                    if !tags.isEmpty {
                        HStack(spacing: 3) {
                            ForEach(tags.prefix(3), id: \.self) { tag in
                                Text(tag)
                                    .font(.system(size: 8, weight: .medium))
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
    }
}
