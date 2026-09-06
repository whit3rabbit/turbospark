import AppKit
import SwiftUI
import TurboSpark

/// Hero header for an installed model detail pane.
struct InstalledModelHeroHeaderView: View {
    let installedModel: InstalledModel
    let visuals: ModelFamilyVisuals
    let descriptor: ModelFeatureDescriptor
    let isCurrentlyLoaded: Bool
    let isCurrentlySelected: Bool
    let isFavorite: Bool
    let onToggleFavorite: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            ModelLogoView(visuals: visuals, size: 56, cornerRadius: 14)

            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text(installedModel.alias)
                        .font(.title2.weight(.bold))
                        .lineLimit(2)

                    Text("INSTALLED")
                        .font(.system(size: 9, weight: .bold))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.teal.opacity(0.16), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(Color.teal)

                    Button(action: onToggleFavorite) {
                        Image(systemName: isFavorite ? "star.fill" : "star")
                            .font(.system(size: 14))
                            .foregroundStyle(isFavorite ? Color.yellow : Color.secondary)
                    }
                    .buttonStyle(.plain)
                    .help(isFavorite ? "Remove from Favorites" : "Mark as Favorite")
                    .accessibilityLabel(isFavorite ? "Favorited" : "Favorite")

                    Spacer(minLength: 4)

                    statusBadge
                }

                HStack(spacing: 8) {
                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .font(.caption2)
                            .foregroundStyle(visuals.accentColor)
                            .accessibilityHidden(true)
                        Text(visuals.family)
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                    Text("\u{2022}")
                        .foregroundStyle(.tertiary)
                        .accessibilityHidden(true)

                    Text(MetricFormat.storage(installedModel.installBytes))
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)

                    Text("\u{2022}")
                        .foregroundStyle(.tertiary)
                        .accessibilityHidden(true)

                    Text(descriptor.storageSource.shortLabel)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                }

                if !installedModel.repo.isEmpty && installedModel.repo != "local" {
                    if let url = URL(string: "https://huggingface.co/\(installedModel.repo)") {
                        Link(destination: url) {
                            HStack(spacing: 3) {
                                Text(installedModel.repo)
                                Image(systemName: "arrow.up.right.square")
                                    .font(.system(size: 9))
                            }
                            .font(.caption.monospaced())
                            .foregroundStyle(Color.accentColor)
                        }
                        .help("Open \(installedModel.repo) on Hugging Face: \(url.absoluteString)")
                        .accessibilityLabel("Hugging Face repository \(installedModel.repo)")
                        .accessibilityHint("Opens this model's page on Hugging Face in your web browser")
                        .accessibilityAddTraits(.isLink)
                    } else {
                        Text(installedModel.repo)
                            .font(.caption.monospaced())
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        if isCurrentlyLoaded {
            HStack(spacing: 4) {
                Circle().fill(Color.green).frame(width: 7, height: 7)
                Text("Loaded in Memory")
                    .font(.caption2.weight(.bold))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(Color.green.opacity(0.16), in: Capsule())
            .foregroundStyle(Color.green)
        } else if isCurrentlySelected {
            HStack(spacing: 4) {
                Circle().fill(Color.accentColor).frame(width: 6, height: 6)
                Text("Selected")
                    .font(.caption2.weight(.semibold))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(Color.accentColor.opacity(0.14), in: Capsule())
            .foregroundStyle(Color.accentColor)
        } else {
            HStack(spacing: 4) {
                Circle().fill(Color.secondary).frame(width: 5, height: 5)
                Text("Ready to Load")
                    .font(.caption2.weight(.medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.4), in: Capsule())
            .foregroundStyle(.secondary)
        }
    }
}
