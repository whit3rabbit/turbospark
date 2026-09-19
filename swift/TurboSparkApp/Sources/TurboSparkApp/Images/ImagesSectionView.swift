import AppKit
import SwiftUI
import TurboSpark

/// Creation and organization share the same saved outputs and image actions.
@MainActor
struct ImagesSectionView: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    @State private var organizing = false
    @State private var importing = false
    @State private var search = ""
    @State private var selecting = false
    @State private var selection: Set<UUID> = []
    @State private var preview: AppArtifact?

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
            gallery.frame(maxWidth: .infinity, maxHeight: .infinity)
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
        .onChange(of: organizing) { selection.removeAll(); selecting = false }
        .onChange(of: search) { selection.removeAll() }
        .onChange(of: model.generating) {
            if model.generating && model.imageGenerationTask != nil {
                organizing = false
                preview = nil
            }
        }
        .onChange(of: model.savedImageArtifacts.map(\.id)) {
            selection.formIntersection(Set(model.savedImageArtifacts.map(\.id)))
        }
    }

    private var header: some View {
        HStack(spacing: 20) {
            Text("Images", bundle: .module).themedFont(.title2, weight: .semibold)
            Picker(selection: $organizing) {
                Text("Create", bundle: .module).tag(false)
                Text("Organize", bundle: .module).tag(true)
            } label: { Text("Images", bundle: .module) }
            .labelsHidden().pickerStyle(.segmented).frame(width: 210)
            Spacer(minLength: 0)
            Text("\(model.savedImageArtifacts.count) image(s)", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 24).padding(.vertical, 18)
    }

    private var organizationToolbar: some View {
        HStack(spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(.appSecondary)
                TextField(text: $search) { Text("Search images", bundle: .module) }
                    .textFieldStyle(.plain)
                if !search.isEmpty {
                    Button { search = "" } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain)
                        .help(Text("Clear search", bundle: .module))
                        .accessibilityLabel(Text("Clear search", bundle: .module))
                }
            }
            .padding(10).background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
            .frame(maxWidth: 360)
            Spacer(minLength: 0)
            if selecting {
                Text(verbatim: String(selection.count)).monospacedDigit()
                Button { selection = Set(artifacts.map(\.id)) } label: {
                    Text("Select all", bundle: .module)
                }
                Button(role: .destructive) {
                    selection.subtract(model.trashGeneratedImages(ids: selection))
                } label: {
                    Label { Text("Move to Trash", bundle: .module) } icon: { Image(systemName: "trash") }
                }
                .disabled(selection.isEmpty)
            }
            Button { selecting.toggle(); selection.removeAll() } label: {
                if selecting { Text("Done", bundle: .module) }
                else { Text("Select", bundle: .module) }
            }
            .disabled(model.savedImageArtifacts.isEmpty)
        }
        .themedFont(.small).padding(.horizontal, 24).padding(.vertical, 12)
    }

    @ViewBuilder
    private var gallery: some View {
        if artifacts.isEmpty && (organizing || model.imageJob?.result == nil || model.imageJob?.savedPath != nil) {
            VStack(spacing: 14) {
                Image(systemName: organizing ? "square.grid.2x2" : "photo.badge.plus")
                    .themedFont(.hero).foregroundStyle(.appAccent)
                Text(organizing ? "No images found" : "Create your first image", bundle: .module)
                    .themedFont(.title2, weight: .semibold)
                Text(organizing ? "Your saved images appear here." : "Describe an idea below. Your images will be saved here automatically.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
                    .multilineTextAlignment(.center).frame(maxWidth: 380)
                if organizing && !search.isEmpty {
                    Button { search = "" } label: { Text("Clear search", bundle: .module) }
                } else if !organizing && model.imageModelPath.isEmpty && !model.isInstallingImageModel {
                    if let source = model.imageDownloadChoices.first {
                        Button {
                            model.installImageModel(source)
                        } label: {
                            Label {
                                HStack(spacing: 4) {
                                    Text("Download", bundle: .module)
                                    Text(ImageModelPresentation.family(source.modelID))
                                    Text(ImageModelPresentation.quantization(source.quantization))
                                }
                            } icon: {
                                Image(systemName: "arrow.down.circle.fill")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.regular)
                        .padding(.top, 4)
                    }
                }
            }
            .padding(32)
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

    private var dayGroups: [(day: Date, images: [AppArtifact])] {
        Dictionary(grouping: artifacts) { Calendar.current.startOfDay(for: $0.createdAt) }
            .map { (day: $0.key, images: $0.value) }
            .sorted { $0.day > $1.day }
    }

    @ViewBuilder
    private var unsavedResult: some View {
        if let job = model.imageJob, job.savedPath == nil, let result = job.result,
           let image = NSImage(data: result.png) {
            VStack(alignment: .leading, spacing: 12) {
                Image(nsImage: image).resizable().scaledToFit().frame(maxHeight: 260)
                Text("This image has not been saved.", bundle: .module).themedFont(.small)
                HStack {
                    Button { model.saveImage() } label: { Text("Save", bundle: .module) }
                    Button(role: .destructive) { model.discardUnsavedImage() } label: {
                        Text("Remove", bundle: .module)
                    }
                }
                .disabled(model.imageGenerationTask != nil)
            }
        }
    }

    private func reusePrompt(_ artifact: AppArtifact) {
        model.writePromptTextDirectly(artifact.imageRequest?.prompt ?? artifact.title)
        organizing = false
    }
}
