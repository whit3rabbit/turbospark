import SwiftUI
import TurboSpark

/// Image installs live inside the installed-model manager because they have
/// the same lifecycle: inspect, select, and remove local artifacts.
@MainActor
struct ImageModelManagerView: View {
    @ObservedObject var model: AppModel

    @State private var imageToDelete: ImageInstalledModel?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                recommendationCard
                installedSection
                curatedSection
            }
            .frame(maxWidth: 860, alignment: .leading)
            .frame(maxWidth: .infinity)
            .padding(24)
        }
        .confirmationDialog(
            "Delete image model?",
            isPresented: Binding(
                get: { imageToDelete != nil },
                set: { if !$0 { imageToDelete = nil } }),
            presenting: imageToDelete
        ) { image in
            Button("Delete \(image.alias)", role: .destructive) {
                model.deleteImageModel(image)
                imageToDelete = nil
            }
            Button("Cancel", role: .cancel) {}
        } message: { image in
            Text("This removes \(image.path) from the shared TurboSpark image store.", bundle: .module)
        }
        .task {
            model.refreshTelemetry()
        }
    }

    private var recommendationCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .foregroundStyle(.appAccent)
                Text("Curated Z-Image models", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                Spacer()
                if let telemetry = model.telemetry {
                    Text(verbatim: MetricFormat.memory(telemetry.physicalMemoryBytes))
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(.appSecondary)
                }
            }
            Text(
                "Pinned MLX variants with completed install gates. The first choice is sized for this Mac; the others are available when you want a different quality or disk tradeoff.",
                bundle: .module
            )
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)

            ForEach(model.recommendedZImageSources) { source in
                curatedRow(source, recommended: true)
            }

            HStack {
                Text(
                    model.hasInstalledZImageModel
                        ? "A curated Z-Image install is available."
                        : "No curated Z-Image model is downloaded yet.",
                    bundle: .module
                )
                .themedFont(.tiny)
                .foregroundStyle(
                    model.hasInstalledZImageModel ? Color.green : Color.secondary)
                Spacer()
                Button {
                    model.activeSection = .images
                } label: {
                    Label {
                        Text("Open Images", bundle: .module)
                    } icon: {
                        Image(systemName: "photo.on.rectangle")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(16)
        .background(.appAccent.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .stroke(.appAccent.opacity(0.2), lineWidth: 0.5)
        }
    }

    private var installedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Installed image models", bundle: .module)
                .themedFont(.base, weight: .semibold)
            if model.imageModels.isEmpty {
                Text("Nothing is installed. Choose a curated variant above to get started.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else {
                ForEach(model.imageModels) { image in
                    installedRow(image)
                }
            }
        }
    }

    private var curatedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("All curated variants", bundle: .module)
                .themedFont(.base, weight: .semibold)
            ForEach(model.imageCatalog) { source in
                curatedRow(source, recommended: false)
            }
        }
    }

    private func installedRow(_ image: ImageInstalledModel) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "photo.fill")
                .foregroundStyle(.appAccent)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 3) {
                Text(image.alias)
                    .themedFont(.small, weight: .medium)
                Text(verbatim: "\(image.quantization)  |  \(image.width) x \(image.height)")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            if model.imageModelPath == image.path {
                Label("Selected", systemImage: "checkmark.circle.fill")
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.green)
            } else {
                Button {
                    model.selectImageModel(image)
                } label: {
                    Text("Use", bundle: .module)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            Button(role: .destructive) {
                imageToDelete = image
            } label: {
                Label {
                    Text("Delete", bundle: .module)
                } icon: {
                    Image(systemName: "trash")
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .accessibilityHint("Removes this image model from the shared image store")
        }
        .padding(10)
        .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8).stroke(.appBorder, lineWidth: 0.5)
        }
    }

    private func curatedRow(_ source: ImageCatalogEntry, recommended: Bool) -> some View {
        let installed = model.imageModels.first { $0.alias == source.alias }
        let isTested = AppModel.testedZImageAliases.contains(source.alias)
        return HStack(spacing: 10) {
            Image(systemName: installed == nil ? "arrow.down.circle" : "checkmark.circle.fill")
                .foregroundStyle(installed == nil ? Color.accentColor : Color.green)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(source.alias)
                        .themedFont(.small, weight: .medium)
                    if recommended {
                        Text("Recommended", bundle: .module)
                            .themedFont(.micro, weight: .bold)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(.appAccent.opacity(0.14), in: Capsule())
                            .foregroundStyle(.appAccent)
                    }
                    if isTested {
                        Text("Install-tested", bundle: .module)
                            .themedFont(.micro, weight: .medium)
                            .foregroundStyle(.green)
                    }
                }
                Text(verbatim: "\(source.quantization)  |  \(source.modelID)")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            if let installed {
                Button {
                    model.selectImageModel(installed)
                } label: {
                    Text("Use", bundle: .module)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                Button {} label: {
                    Text("Installed", bundle: .module)
                }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(true)
            } else {
                Button {
                    model.installImageModel(source)
                } label: {
                    Label {
                        Text("Download", bundle: .module)
                    } icon: {
                        Image(systemName: "icloud.and.arrow.down")
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(model.isInstallingImageModel || !isTested)
            }
        }
        .padding(10)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8).stroke(.appBorder, lineWidth: 0.5)
        }
    }
}

/// First-run prompt for machines without a curated Z-Image install.
@MainActor
struct ImageModelRecommendationSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("Set up local image generation", systemImage: "photo.on.rectangle.angled")
                    .themedFont(.title3, weight: .semibold)
                Spacer()
                Button { dismiss() } label: {
                    Text("Later", bundle: .module)
                }
                    .keyboardShortcut(.cancelAction)
            }
            Text(
                "TurboSpark did not find a downloaded Z-Image model. These pinned MLX variants have completed install gates and are matched to this Mac's memory.",
                bundle: .module
            )
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)

            ForEach(model.recommendedZImageSources) { source in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(source.alias)
                            .themedFont(.base, weight: .medium)
                        Spacer()
                        Button {
                            model.installImageModel(source)
                        } label: {
                            Text("Download", bundle: .module)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(model.isInstallingImageModel)
                    }
                    Text(verbatim: "\(source.quantization)  |  \(source.modelID)")
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                .padding(10)
                .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
            }

            HStack {
                Spacer()
                Button {
                    dismiss()
                    model.activeSection = .modelManager
                } label: {
                    Text("Manage image models", bundle: .module)
                }
                .buttonStyle(.link)
            }
        }
        .padding(24)
        .frame(width: 560)
        .task {
            model.refreshTelemetry()
        }
    }
}
