import SwiftUI
import TurboSpark

/// Technical specifications card displaying architecture, quantization, dimensions, and path.
struct InstalledModelTechnicalSpecsView: View {
    let visuals: ModelFamilyVisuals
    let descriptor: ModelFeatureDescriptor
    let installedModel: InstalledModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label { Text("Technical Specifications", bundle: .module) } icon: { Image(systemName: "info.circle") }
                .themedFont(.small, weight: .semibold)

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    Text("Architecture", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                    Text(visuals.architectureType).themedFont(.small, weight: .medium)
                }
                GridRow {
                    Text("Quantization", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                    Text(descriptor.quantFormat).themedFont(.small, weight: .medium)
                }
                GridRow {
                    Text("Format & Package", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                    Text(descriptor.format.rawValue).themedFont(.small, weight: .medium)
                }
                if let ctx = descriptor.contextLimit {
                    GridRow {
                        Text("Trained Context", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                        Text("\(ctx) tokens", bundle: .module).themedFont(.small, weight: .medium).monospacedDigit()
                    }
                }
                if let layers = descriptor.layerCount {
                    GridRow {
                        Text("Layers", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                        Text("\(layers) layers", bundle: .module).themedFont(.small, weight: .medium).monospacedDigit()
                    }
                }
                if let hidden = descriptor.hiddenSize {
                    GridRow {
                        Text("Hidden Dimension", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                        Text("\(hidden)", bundle: .module).themedFont(.small, weight: .medium).monospacedDigit()
                    }
                }
                if let vocab = descriptor.vocabSize {
                    GridRow {
                        Text("Vocabulary Size", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                        Text("\(vocab)", bundle: .module).themedFont(.small, weight: .medium).monospacedDigit()
                    }
                }
                GridRow {
                    Text("Disk Path", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                    Button {
                        ModelStorageManager.revealInFinder(path: installedModel.path)
                    } label: {
                        Text(installedModel.path)
                            .themedCode(.small)
                            .lineLimit(2)
                            .foregroundStyle(Color.accentColor)
                    }
                    .buttonStyle(.plain)
                    .help("Reveal in Finder: \(installedModel.path)")
                    .accessibilityLabel("Disk path: \(installedModel.path)")
                    .accessibilityHint("Opens Finder to reveal model files")
                    .accessibilityAddTraits(.isLink)
                }
                if !installedModel.installedOn.isEmpty {
                    GridRow {
                        Text("Installed On", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                        Text(installedModel.installedOn).themedFont(.small, weight: .medium)
                    }
                }
            }
        }
        .padding(14)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }
}
