import SwiftUI
import TurboSpark

/// Image installs live inside the installed-model manager because they have
/// the same lifecycle: inspect, select, and remove local artifacts.
@MainActor
struct ImageModelManagerView: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme

    @State private var imageToDelete: ImageInstalledModel?
    /// The library the user picked for the downloads on this page. Nil follows
    /// the default (requirement 2.2). Per install: not remembered across launches.
    @State private var destination: String?

    /// The `library:` the installs take for the menu's choice: nil for This Mac.
    private var chosenLibrary: String? {
        model.installDestinations.libraryForInstall(selection: destination)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                // Where a download goes: only when there is a choice (2.1).
                InstallDestinationPicker(model: model, selection: $destination)
                recommendationCard
                installedSection
                curatedSection
            }
            .frame(maxWidth: 860, alignment: .leading)
            .frame(maxWidth: .infinity)
            .padding(24)
        }
        .confirmationDialog(
            deleteDialogTitle,
            isPresented: Binding(
                get: { imageToDelete != nil },
                set: { if !$0 { imageToDelete = nil } }),
            presenting: imageToDelete
        ) { image in
            if model.isManagedImageModel(image) {
                Button(role: .destructive) {
                    model.deleteImageModel(image)
                    imageToDelete = nil
                } label: {
                    Text("Delete \(image.alias)", bundle: .module)
                }
            } else {
                Button(role: .destructive) {
                    model.deleteImageModel(image)
                    imageToDelete = nil
                } label: {
                    Text("Remove \(image.alias) from the list", bundle: .module)
                }
            }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: { image in
            if model.isManagedImageModel(image) {
                Text("This removes \(image.path) from the shared TurboSpark image store.", bundle: .module)
            } else {
                Text(
                    "The files stay on disk at \(image.path). You can restore it under Settings > Models & Storage.",
                    bundle: .module)
            }
        }
        .task {
            model.refreshTelemetry()
        }
    }

    private var deleteDialogTitle: Text {
        imageToDelete.map { model.isManagedImageModel($0) } ?? true
            ? Text("Delete image model?", bundle: .module)
            : Text("Remove image model?", bundle: .module)
    }

    private var recommendationCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .foregroundStyle(.appAccent)
                Text("Image Models", bundle: .module)
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

            ForEach(model.recommendedImageModelSources) { source in
                curatedRow(source, recommended: true)
            }

            HStack {
                Group {
                    if model.hasInstalledMLXImageModel {
                        Text("A curated image model is installed.", bundle: .module)
                    } else {
                        Text("No curated image model is downloaded yet.", bundle: .module)
                    }
                }
                .themedFont(.tiny)
                .foregroundStyle(
                    model.hasInstalledMLXImageModel ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))
                Spacer()
                Button {
                    model.activeSection = .images
                } label: {
                    Label {
                        Text("Open Image Generation", bundle: .module)
                    } icon: {
                        Image(systemName: "photo.on.rectangle")
                    }
                }
                .appAction(.secondary, size: .small)
            }
        }
        .padding(16)
        .background(.appAccent.opacity(0.08), in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(theme.border, lineWidth: 0.5)
        }
    }

    private var installedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Installed image models", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
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
                .accessibilityAddTraits(.isHeader)
            ForEach(model.imageDownloadChoices) { source in
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
                Text(verbatim: image.alias)
                    .themedFont(.small, weight: .medium)
                Text(verbatim: "\(image.quantization)  |  \(image.width) x \(image.height)")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            if !AppModel.supportsMLXImageModel(modelID: image.modelID) {
                Label { Text("MLX required", bundle: .module) } icon: { Image(systemName: "exclamationmark.triangle") }
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.appSecondary)
            } else if model.imageModelPath == image.path {
                Label { Text("Selected", bundle: .module) } icon: { Image(systemName: "checkmark.circle.fill") }
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.green)
            } else {
                Button {
                    model.selectImageModel(image)
                } label: {
                    Text("Use", bundle: .module)
                }
                .appAction(.secondary, size: .small)
            }
            // Only the shared store is ours to delete from. A model found in an
            // additional model folder is dropped from the list and its files
            // stay where they are, the same rule the text models follow.
            let managed = model.isManagedImageModel(image)
            // Only an install the engine lists can be moved (task 6.11); the sheet
            // is hosted by the Installed Models list around this page.
            if let subject = MoveSubject.make(image: image, isManaged: managed) {
                Button {
                    model.presentMoveSheet(for: subject)
                } label: {
                    Label { Text("Move to...", bundle: .module) } icon: { Image(systemName: "arrow.left.arrow.right.circle") }
                }
                .appAction(.secondary, size: .small)
                .disabled(model.modelsBeingChanged.contains(subject.installId))
                .accessibilityValue(Text(verbatim: UntrustedText.display(image.alias)))
            }
            Button(role: managed ? .destructive : nil) {
                imageToDelete = image
            } label: {
                Label {
                    if managed {
                        Text("Delete", bundle: .module)
                    } else {
                        Text("Remove", bundle: .module)
                    }
                } icon: {
                    Image(systemName: managed ? "trash" : "eye.slash")
                }
            }
            .appAction(managed ? .destructive : .secondary, size: .small)
            .accessibilityHint(
                managed
                    ? Text("Removes this image model from the shared image store", bundle: .module)
                    : Text("Hides this image model without deleting its files", bundle: .module))
        }
        .padding(10)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous).stroke(.appBorder, lineWidth: 0.5)
        }
    }

    private func curatedRow(_ source: ImageCatalogEntry, recommended: Bool) -> some View {
        let installed = model.imageModels.first { $0.alias == source.alias }
        let isTested = AppModel.testedImageModelAliases.contains(source.alias)
        return HStack(spacing: 10) {
            Image(systemName: installed == nil ? "arrow.down.circle" : "checkmark.circle.fill")
                .foregroundStyle(.appAccent)
                .frame(minWidth: 22)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(verbatim: source.alias)
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
                            .foregroundStyle(.appAccent)
                    }
                }
                Text(verbatim: "\(source.quantization)  |  \(source.modelID)")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                // The Qwen Research License allows research and evaluation use
                // only and asks derivative products to say they are built with
                // Qwen. Show it before the download, in the license's words.
                if source.resolvedFamily.hasPrefix("qwen-image-2.1") {
                    Text("Qwen Research License: research and evaluation use only, no commercial use. Built with Qwen.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(AppAttention.color)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer()
            if let installed {
                Button {
                    model.selectImageModel(installed)
                } label: {
                    Text("Use", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                Button {} label: {
                    Text("Installed", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .disabled(true)
            } else {
                Button {
                    model.installImageModel(source, library: chosenLibrary)
                } label: {
                    Label {
                        Text("Download", bundle: .module)
                    } icon: {
                        Image(systemName: "icloud.and.arrow.down")
                    }
                }
                .appAction(.secondary, size: .small)
                .disabled(!isTested || !model.canInstallImageModel(alias: source.alias))
                .help(!isTested || !model.canInstallImageModel(alias: source.alias) ? Text("Only supported MLX image models can be installed in the app.", bundle: .module) : Text("Download", bundle: .module))
            }
        }
        .padding(10)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous).stroke(.appBorder, lineWidth: 0.5)
        }
    }
}

/// First-run prompt for machines without a curated Z-Image install.
@MainActor
struct ImageModelRecommendationSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    /// The library the user picked for the download. Nil follows the default
    /// (requirement 2.2). Per install: not remembered across launches.
    @State private var destination: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label {
                    Text("Set up local image generation", bundle: .module)
                } icon: {
                    Image(systemName: "photo.on.rectangle.angled")
                }
                .themedFont(.title3, weight: .semibold)
                Spacer()
                Button { dismiss() } label: {
                    Text("Later", bundle: .module)
                }
                .appAction(.secondary)
                .keyboardShortcut(.cancelAction)
            }
            Text(
                "TurboSpark did not find a downloaded Z-Image model. These pinned MLX variants have completed install gates and are matched to this Mac's memory.",
                bundle: .module
            )
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)

            // Where the download goes: only when there is a choice (2.1).
            InstallDestinationPicker(model: model, selection: $destination)

            ForEach(model.recommendedImageModelSources) { source in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(verbatim: source.alias)
                            .themedFont(.base, weight: .medium)
                        Spacer()
                        Button {
                            model.installImageModel(
                                source,
                                library: model.installDestinations.libraryForInstall(selection: destination))
                        } label: {
                            Text("Download", bundle: .module)
                        }
                        // The first source is the one sized for this Mac, so it
                        // is the sheet's one primary action.
                        .appAction(
                            source.id == model.recommendedImageModelSources.first?.id ? .primary : .secondary,
                            size: .small)
                        .disabled(!model.canInstallImageModel(alias: source.alias))
                    }
                    Text(verbatim: "\(source.quantization)  |  \(source.modelID)")
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                .padding(10)
                .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            }

            HStack {
                Spacer()
                Button {
                    dismiss()
                    model.activeSection = .modelManager
                } label: {
                    Text("Manage image models", bundle: .module)
                }
                .appAction(.quiet)
            }
        }
        .padding(24)
        .frame(minWidth: 480, idealWidth: 560, maxWidth: 640)
        .task {
            model.refreshTelemetry()
        }
    }
}
