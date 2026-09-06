import SwiftUI
import TurboSpark

/// Technical specifications card displaying architecture, quantization, dimensions, and path.
struct InstalledModelTechnicalSpecsView: View {
    let visuals: ModelFamilyVisuals
    let descriptor: ModelFeatureDescriptor
    let installedModel: InstalledModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Technical Specifications", systemImage: "info.circle")
                .font(.subheadline.weight(.semibold))

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    Text("Architecture").font(.caption).foregroundStyle(.secondary)
                    Text(visuals.architectureType).font(.caption.weight(.medium))
                }
                GridRow {
                    Text("Quantization").font(.caption).foregroundStyle(.secondary)
                    Text(descriptor.quantFormat).font(.caption.weight(.medium))
                }
                GridRow {
                    Text("Format & Package").font(.caption).foregroundStyle(.secondary)
                    Text(descriptor.format.rawValue).font(.caption.weight(.medium))
                }
                if let ctx = descriptor.contextLimit {
                    GridRow {
                        Text("Trained Context").font(.caption).foregroundStyle(.secondary)
                        Text("\(ctx) tokens").font(.caption.monospacedDigit().weight(.medium))
                    }
                }
                if let layers = descriptor.layerCount {
                    GridRow {
                        Text("Layers").font(.caption).foregroundStyle(.secondary)
                        Text("\(layers) layers").font(.caption.monospacedDigit().weight(.medium))
                    }
                }
                if let hidden = descriptor.hiddenSize {
                    GridRow {
                        Text("Hidden Dimension").font(.caption).foregroundStyle(.secondary)
                        Text("\(hidden)").font(.caption.monospacedDigit().weight(.medium))
                    }
                }
                if let vocab = descriptor.vocabSize {
                    GridRow {
                        Text("Vocabulary Size").font(.caption).foregroundStyle(.secondary)
                        Text("\(vocab)").font(.caption.monospacedDigit().weight(.medium))
                    }
                }
                GridRow {
                    Text("Disk Path").font(.caption).foregroundStyle(.secondary)
                    Button {
                        ModelStorageManager.revealInFinder(path: installedModel.path)
                    } label: {
                        Text(installedModel.path)
                            .font(.caption.monospaced())
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
                        Text("Installed On").font(.caption).foregroundStyle(.secondary)
                        Text(installedModel.installedOn).font(.caption.weight(.medium))
                    }
                }
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }
}
