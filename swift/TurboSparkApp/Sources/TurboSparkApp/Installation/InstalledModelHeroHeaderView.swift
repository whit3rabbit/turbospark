import AppKit
import SwiftUI
import TurboSpark

/// Hero header for an installed model detail pane.
struct InstalledModelHeroHeaderView: View {
    let installedModel: InstalledModel
    /// Where the model is and whether it can be opened (task 6.4).
    let rowModel: InstalledModelRowModel
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
                        .themedFont(.title2, weight: .bold)
                        .lineLimit(2)

                    Text("INSTALLED", bundle: .module)
                        .themedFont(.micro, weight: .bold)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.teal.opacity(0.16), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(Color.teal)

                    Button(action: onToggleFavorite) {
                        Image(systemName: isFavorite ? "star.fill" : "star")
                            .themedFont(.callout)
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
                            .themedFont(.tiny)
                            .foregroundStyle(visuals.accentColor)
                            .accessibilityHidden(true)
                        Text(visuals.family)
                    }
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)

                    Text(verbatim: "\u{2022}")
                        .foregroundStyle(.tertiary)
                        .accessibilityHidden(true)

                    Text(MetricFormat.storage(installedModel.installBytes))
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)

                    Text(verbatim: "\u{2022}")
                        .foregroundStyle(.tertiary)
                        .accessibilityHidden(true)

                    locationLabel
                }

                // How long the drive will take to read the model (5.3).
                if let estimate = rowModel.estimateText {
                    HStack(spacing: 4) {
                        Image(systemName: "clock")
                            .themedFont(.tiny)
                            .accessibilityHidden(true)
                        Text(verbatim: estimate)
                            .themedFont(.small)
                    }
                    .foregroundStyle(.appSecondary)
                }

                if !installedModel.repo.isEmpty && installedModel.repo != "local" {
                    if ModelFeatureDescriptor.isHuggingFaceRepo(installedModel.repo),
                       let url = URL(string: "https://huggingface.co/\(installedModel.repo)") {
                        Link(destination: url) {
                            HStack(spacing: 3) {
                                Text(installedModel.repo)
                                Image(systemName: "arrow.up.right.square")
                                    .themedFont(.micro)
                            }
                            .themedCode(.small)
                            .foregroundStyle(Color.accentColor)
                        }
                        .help(Text("Open \(installedModel.repo) on Hugging Face: \(url.absoluteString)", bundle: .module))
                        .accessibilityLabel(
                            Text("Hugging Face source", bundle: .module)
                                + Text(verbatim: " \(installedModel.repo)"))
                        .accessibilityAddTraits(.isLink)
                    } else {
                        Text(installedModel.repo)
                            .themedCode(.small)
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }
                }
            }
        }
    }

    /// The library that holds the model, or for a model only a scan found, the
    /// source it was found in. A library model's source label would read
    /// "Custom Folder", which it is not.
    @ViewBuilder
    private var locationLabel: some View {
        if let text = rowModel.locationText {
            HStack(spacing: 4) {
                if let glyph = rowModel.locationGlyph {
                    Image(systemName: glyph)
                        .themedFont(.tiny)
                        .accessibilityHidden(true)
                }
                Text(verbatim: text)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .themedFont(.small, weight: .medium)
            .foregroundStyle(.appSecondary)
            .help(Text(verbatim: text))
        } else {
            Text(LocalizedStringKey(descriptor.storageSource.shortLabel), bundle: .module)
                .themedFont(.small, weight: .medium)
                .foregroundStyle(.appSecondary)
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        if let status = rowModel.status, let text = rowModel.statusText {
            // A glyph AND a word: the two states differ in both.
            HStack(spacing: 4) {
                Image(systemName: status.glyph)
                    .themedFont(.tiny)
                    .accessibilityHidden(true)
                Text(verbatim: text)
                    .themedFont(.tiny, weight: .semibold)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(.appElevated, in: Capsule())
            .overlay(Capsule().stroke(.appBorder, lineWidth: 1))
            .foregroundStyle(.appText)
        } else if isCurrentlyLoaded {
            HStack(spacing: 4) {
                Circle().fill(Color.green).frame(width: 7, height: 7)
                Text("Loaded in Memory", bundle: .module)
                    .themedFont(.tiny, weight: .bold)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(Color.green.opacity(0.16), in: Capsule())
            .foregroundStyle(Color.green)
        } else if isCurrentlySelected {
            HStack(spacing: 4) {
                Circle().fill(Color.accentColor).frame(width: 6, height: 6)
                Text("Selected", bundle: .module)
                    .themedFont(.tiny, weight: .semibold)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(Color.accentColor.opacity(0.14), in: Capsule())
            .foregroundStyle(Color.accentColor)
        } else {
            HStack(spacing: 4) {
                Circle().fill(Color.secondary).frame(width: 5, height: 5)
                Text("Ready to Load", bundle: .module)
                    .themedFont(.tiny, weight: .medium)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(.appElevated, in: Capsule())
            .foregroundStyle(.appSecondary)
        }
    }
}
