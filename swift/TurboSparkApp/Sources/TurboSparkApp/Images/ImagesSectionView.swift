import AppKit
import SwiftUI
import TurboSpark

/// Creation and organization share the same saved outputs and image actions.
@MainActor
struct ImagesSectionView: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    @State private var importing = false
    @State private var search = ""
    @State private var selecting = false
    @State private var selection: Set<UUID> = []
    @State private var preview: AppArtifact?
    @AppStorage("TurboSpark.imageModelRecommendationSeen")
    private var imageModelRecommendationSeen = false
    @State private var showingImageModelRecommendation = false

    /// The mode lives on the model because the panel beside the rail switches
    /// it (`ImagesSidebarPanel`); this pane has no switcher of its own.
    private var organizing: Bool { model.imagesMode == .organize }

    private var artifacts: [AppArtifact] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        guard organizing, !query.isEmpty else { return model.savedImageArtifacts }
        return model.savedImageArtifacts.filter {
            ($0.imageRequest?.prompt ?? $0.title).localizedStandardContains(query)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if organizing { organizationToolbar }
            gallery
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay {
                    if model.imageGenerationTask != nil {
                        generationProgress
                    }
                }
            if !organizing {
                ImageComposerView(model: model, importing: $importing)
            }
        }
        .background(.appPage)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
            if case let .success(url) = result { model.imageModelPathText = url.path }
        }
        .sheet(item: $preview) { artifact in
            ImageGalleryCarousel(model: model, initialID: artifact.id) { artifact in
                reusePrompt(artifact)
            }
        }
        .sheet(isPresented: $showingImageModelRecommendation) {
            ImageModelRecommendationSheet(model: model)
        }
        .onChange(of: model.shouldRecommendImageModel, initial: true) { _, shouldRecommend in
            guard shouldRecommend, !imageModelRecommendationSeen else { return }
            imageModelRecommendationSeen = true
            showingImageModelRecommendation = true
        }
        .onChange(of: model.imagesMode) { selection.removeAll(); selecting = false }
        .onChange(of: search) { selection.removeAll() }
        .onChange(of: model.generating) {
            if model.generating && model.imageGenerationTask != nil {
                model.imagesMode = .create
                preview = nil
            }
        }
        .onChange(of: model.savedImageArtifacts.map(\.id)) {
            selection.formIntersection(Set(model.savedImageArtifacts.map(\.id)))
        }
    }

    /// Title and count. The Create and Organize switch moved to the panel
    /// beside the rail (`ImagesSidebarPanel`), and the rail's Images menu
    /// offers both while that panel is hidden.
    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Image Generation", bundle: .module)
                .themedFont(.title2, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Text("\(model.savedImageArtifacts.count) image(s)", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 24).padding(.vertical, 18)
    }

    private var organizationToolbar: some View {
        AppAdaptiveHeader {
            AppSearchField(text: $search, prompt: Text("Search images", bundle: .module))
        } actions: {
            if selecting {
                Text(verbatim: String(selection.count)).monospacedDigit()
                    .foregroundStyle(.appSecondary)
                Button { selection = Set(artifacts.map(\.id)) } label: {
                    Text("Select all", bundle: .module)
                }
                .appAction(.secondary)
                Button(role: .destructive) {
                    selection.subtract(model.trashGeneratedImages(ids: selection))
                } label: {
                    Label { Text("Move to Trash", bundle: .module) } icon: { Image(systemName: "trash") }
                }
                .appAction(.destructive)
                .disabled(selection.isEmpty)
            }
            Button { selecting.toggle(); selection.removeAll() } label: {
                if selecting { Text("Done", bundle: .module) }
                else { Text("Select", bundle: .module) }
            }
            .appAction(.secondary)
            .disabled(model.savedImageArtifacts.isEmpty)
        }
        .themedFont(.small).padding(.horizontal, 24).padding(.vertical, 12)
    }

    @ViewBuilder
    private var gallery: some View {
        if artifacts.isEmpty && model.imageGenerationTask == nil
            && (organizing || model.imageJob?.result == nil || model.imageJob?.savedPath != nil) {
            emptyGallery
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    if !organizing { unsavedResult }
                    ForEach(dayGroups, id: \.day) { group in
                        VStack(alignment: .leading, spacing: 14) {
                            Text(group.day, format: .dateTime.month(.wide).day().year())
                                .themedFont(.small, weight: .medium).foregroundStyle(.appSecondary)
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: organizing ? 160 : 230), spacing: 14)], spacing: 20) {
                                ForEach(group.images) { artifact in
                                    ImageGalleryCard(
                                        model: model, artifact: artifact, compact: organizing,
                                        selecting: selecting, selected: selection.contains(artifact.id),
                                        open: {
                                            if selecting {
                                                if !selection.insert(artifact.id).inserted { selection.remove(artifact.id) }
                                            } else { preview = artifact }
                                        }, reuse: { reusePrompt(artifact) })
                                }
                            }
                        }
                    }
                }
                .padding(24)
            }
        }
    }

    @ViewBuilder
    private var emptyGallery: some View {
        if organizing {
            AppEmptyState(
                systemImage: "square.grid.2x2",
                title: Text("No images found", bundle: .module),
                message: Text("Your saved images appear here.", bundle: .module)
            ) {
                if !search.isEmpty {
                    Button { search = "" } label: { Text("Clear search", bundle: .module) }
                        .appAction(.secondary)
                }
            }
        } else {
            AppEmptyState(
                systemImage: "photo.badge.plus",
                title: Text("Create your first image", bundle: .module),
                message: Text(
                    "Describe an idea below. Your images will be saved here automatically.",
                    bundle: .module)
            ) {
                if model.imageModelPath.isEmpty, let source = model.imageDownloadChoices.first {
                    Button {
                        model.installImageModel(source)
                    } label: {
                        Label {
                            Text("Download", bundle: .module)
                                + Text(verbatim: " \(ImageModelPresentation.family(source.modelID))")
                                + Text(verbatim: " \(ImageModelPresentation.quantization(source.quantization))")
                        } icon: {
                            Image(systemName: "arrow.down.circle.fill")
                        }
                    }
                    .appAction(.primary)
                    .disabled(!model.canInstallImageModel(alias: source.alias))
                }
            }
        }
    }

    private var dayGroups: [(day: Date, images: [AppArtifact])] {
        Dictionary(grouping: artifacts) { Calendar.current.startOfDay(for: $0.createdAt) }
            .map { (day: $0.key, images: $0.value) }
            .sorted { $0.day > $1.day }
    }

    private var generationProgress: some View {
        VStack(spacing: 10) {
            Text("Image \(model.imageBatchIndex) of \(model.imageBatchCount)", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
            progressStageLabel
                .themedFont(.small, weight: .medium)
                .lineLimit(1)
            HStack(spacing: 10) {
                AppProgressBar(value: model.imageProgressFraction, label: progressStageLabel)
                    .frame(minWidth: 160, idealWidth: 280, maxWidth: 320)
                if let job = model.imageJob, job.total > 0 {
                    Text(verbatim: "\(job.completed)/\(job.total)")
                        .themedFont(.tiny)
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }
            }
        }
        .padding(18)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12).stroke(.appBorder, lineWidth: 0.75)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        .accessibilityElement(children: .combine)
    }

    /// A `Text`, not a view builder, so it also labels the progress bar.
    private var progressStageLabel: Text {
        switch model.imageJob?.stage {
        case "downloading_model": Text("Downloading model files", bundle: .module)
        case "loading_model": Text("Loading model", bundle: .module)
        case "loading_tokenizer": Text("Loading tokenizer", bundle: .module)
        case "loading_text_encoder": Text("Loading text encoder", bundle: .module)
        case "loading_transformer": Text("Loading transformer", bundle: .module)
        case "loading_vae": Text("Loading VAE", bundle: .module)
        case "text_encoder": Text("Encoding prompt", bundle: .module)
        case "transformer": Text("Denoising", bundle: .module)
        case "vae_decoder": Text("Decoding image", bundle: .module)
        case "png_encode": Text("Saving image", bundle: .module)
        default: Text("Loading model", bundle: .module)
        }
    }

    @ViewBuilder
    private var unsavedResult: some View {
        if let job = model.imageJob, job.savedPath == nil, let result = job.result,
           let image = NSImage(data: result.png) {
            VStack(alignment: .leading, spacing: 12) {
                Image(nsImage: image).resizable().scaledToFit().frame(maxHeight: 260)
                Text("This image has not been saved.", bundle: .module).themedFont(.small)
                HStack(spacing: AppMetrics.actionSpacing) {
                    Button { model.saveImage() } label: { Text("Save", bundle: .module) }
                        .appAction(.secondary, size: .small)
                    Button(role: .destructive) { model.discardUnsavedImage() } label: {
                        Text("Remove", bundle: .module)
                    }
                    .appAction(.destructive, size: .small)
                }
                .disabled(model.imageGenerationTask != nil)
            }
        }
    }

    private func reusePrompt(_ artifact: AppArtifact) {
        model.writePromptTextDirectly(artifact.imageRequest?.prompt ?? artifact.title)
        model.imagesMode = .create
    }
}
