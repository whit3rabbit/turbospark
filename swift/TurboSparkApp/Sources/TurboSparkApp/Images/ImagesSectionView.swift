import AppKit
import SwiftUI

/// The canonical image-generation destination.
///
/// Image jobs are transient, while the gallery is rebuilt from the active
/// profile's saved artifact rows. That keeps the gallery useful after relaunch
/// without restoring an interrupted native image session.
@MainActor
struct ImagesSectionView: View {
    @ObservedObject var model: AppModel

    private enum Tab: String, CaseIterable, Identifiable {
        case create
        case gallery

        var id: String { rawValue }
        var title: String { rawValue.capitalized }
    }

    @Environment(\.appTheme) private var theme
    @State private var tab: Tab = .create
    @State private var advancedExpanded = false
    @State private var isImportingImageModel = false
    @State private var selectedGalleryIndex: Int?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if tab == .create {
                createView
            } else {
                galleryView
            }
        }
        .background(.appPage)
        .fileImporter(
            isPresented: $isImportingImageModel,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let url = urls.first {
                model.imageModelPathText = url.path
            }
        }
        .sheet(isPresented: Binding(
            get: { selectedGalleryIndex != nil },
            set: { if !$0 { selectedGalleryIndex = nil } }
        )) {
            if let selectedGalleryIndex {
                ImageGalleryCarousel(
                    model: model,
                    artifacts: model.savedImageArtifacts,
                    initialIndex: selectedGalleryIndex)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Images", bundle: .module)
                    .font(theme.ui(.title2, weight: .semibold))
                Text(verbatim:
                    tab == .create
                        ? "Create images with a verified local image install."
                        : "Saved images from this profile.")
                    .font(theme.ui(.small))
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Picker(selection: $tab) {
                ForEach(Tab.allCases) { tab in
                    Text(verbatim: tab.title).tag(tab)
                }
            } label: { Text(verbatim: "Image view") }
            .pickerStyle(.segmented)
            .frame(width: 190)
            .accessibilityLabel("Image view")
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    private var createView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                modelPicker

                VStack(alignment: .leading, spacing: 8) {
                    Text(verbatim: "Prompt")
                        .font(theme.ui(.title3, weight: .semibold))
                    TextEditor(text: $model.promptText)
                        .font(theme.ui(.base))
                        .frame(minHeight: 150)
                        .padding(8)
                        .background(
                            Color.primary.opacity(theme.isDark ? 0.1 : 0.05),
                            in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .stroke(.appBorder, lineWidth: 1)
                        }
                    Text(verbatim: "Describe the image you want to create.")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.secondary)
                }

                DisclosureGroup(isExpanded: $advancedExpanded) {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text(verbatim: "Seed")
                            Spacer()
                            TextField("Random", text: $model.imageSeedText)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 180)
                                .multilineTextAlignment(.trailing)
                        }
                        settingRow("Size", value: model.imageSizeLabel)
                        settingRow("Scheduler steps", value: String(model.imageSchedulerSteps))
                        settingRow(
                            "Quantization",
                            value: model.selectedImageModel?.quantization ?? "Unknown")
                        Text(verbatim: "Size, steps, and quantization are read from the selected install.")
                            .font(theme.ui(.tiny))
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 8)
                } label: {
                    Text(verbatim: "Advanced settings")
                }
                .font(theme.ui(.small, weight: .medium))

                generationControls
                currentResult
            }
            .frame(maxWidth: 820)
            .frame(maxWidth: .infinity)
            .padding(28)
        }
    }

    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(verbatim: "Image model")
                .font(theme.ui(.title3, weight: .semibold))
            Menu {
                if model.imageModels.isEmpty {
                    Text(verbatim: "No installed image models")
                } else {
                    ForEach(model.imageModels) { imageModel in
                        Button {
                            model.selectImageModel(imageModel)
                        } label: {
                            Label(
                                imageModel.alias,
                                systemImage: model.imageModelPath == imageModel.path
                                    ? "checkmark" : "photo")
                        }
                    }
                }
                Divider()
                Button {
                    isImportingImageModel = true
                } label: {
                    Text(verbatim: "Choose side-loaded install folder…")
                }
            } label: {
                HStack {
                    Image(systemName: "photo.on.rectangle")
                    Text(selectedImageModelLabel)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                            .font(theme.ui(.small, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(
                    Color.primary.opacity(theme.isDark ? 0.1 : 0.05),
                    in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
            .menuStyle(.borderlessButton)
            .frame(maxWidth: 420)
            .disabled(model.isRunning || model.isInstallingModel)
        }
    }

    private var generationControls: some View {
        HStack(spacing: 10) {
            if let job = model.imageJob, job.status == .waiting || job.status == .generating {
                ProgressView(value: model.imageProgressFraction)
                    .frame(width: 160)
                Text(job.stage ?? "Generating")
                    .font(theme.ui(.small))
                    .foregroundStyle(.secondary)
                Button { model.cancelImageGeneration() } label: {
                    Text(verbatim: "Cancel")
                }
                    .buttonStyle(.bordered)
            } else {
                Button {
                    model.generateImage()
                } label: {
                    Label("Generate", systemImage: "wand.and.stars")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canGenerateImage || model.isInGhostChat)
                .keyboardShortcut(.return, modifiers: .command)
            }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private var currentResult: some View {
        if let job = model.imageJob, let result = job.result,
           let image = NSImage(data: result.png) {
            VStack(alignment: .leading, spacing: 12) {
                Text(job.status == .completed ? "Preview" : "Result")
                    .font(theme.ui(.title3, weight: .semibold))
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 720, maxHeight: 560)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .shadow(radius: 8)
                HStack(spacing: 10) {
                    if job.savedPath == nil {
                        Button { model.saveImage() } label: {
                            Text(verbatim: "Save to gallery")
                        }
                            .buttonStyle(.borderedProminent)
                    } else {
                        Label("Saved to this profile", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                    Button { model.regenerateImage() } label: {
                        Text(verbatim: "Regenerate")
                    }
                        .buttonStyle(.bordered)
                        .disabled(job.status != .completed || model.isRunning)
                    Text(verbatim: "Seed " + String(job.options.seed))
                        .font(theme.code(.tiny))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 8)
        }
    }

    private var galleryView: some View {
        let artifacts = model.savedImageArtifacts
        return Group {
            if artifacts.isEmpty {
                ContentUnavailableView {
                    Label("No saved images", systemImage: "photo.on.rectangle.angled")
                } description: {
                    Text(verbatim: "Generate an image and save it to see it here.")
                }
            } else {
                ScrollView {
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 170), spacing: 16)],
                        spacing: 18) {
                        ForEach(Array(artifacts.enumerated()), id: \.element.id) { index, artifact in
                            ImageGalleryThumbnail(artifact: artifact) {
                                selectedGalleryIndex = index
                            }
                        }
                    }
                    .padding(24)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var selectedImageModelLabel: String {
        if let selected = model.selectedImageModel { return selected.alias }
        if !model.imageModelPath.isEmpty {
            return URL(fileURLWithPath: model.imageModelPath).lastPathComponent
        }
        return "Select image model"
    }

    private func settingRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
        }
    }
}

private struct ImageGalleryThumbnail: View {
    let artifact: AppArtifact
    let onOpen: () -> Void
    @Environment(\.appTheme) private var theme

    var body: some View {
        Button(action: onOpen) {
            VStack(alignment: .leading, spacing: 8) {
                Group {
                    if let path = artifact.path, let image = NSImage(contentsOfFile: path) {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFill()
                    } else {
                        Image(systemName: "photo")
                            .font(theme.ui(.hero))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
                .frame(height: 150)
                .frame(maxWidth: .infinity)
                .clipped()
                .background(Color.primary.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                Text(artifact.imageRequest?.prompt ?? artifact.title)
                    .font(theme.ui(.small))
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Open image")
    }
}

private struct ImageGalleryCarousel: View {
    @ObservedObject var model: AppModel
    let artifacts: [AppArtifact]
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appTheme) private var theme
    @State private var index: Int

    init(model: AppModel, artifacts: [AppArtifact], initialIndex: Int) {
        self.model = model
        self.artifacts = artifacts
        _index = State(initialValue: initialIndex)
    }

    private var artifact: AppArtifact? {
        guard artifacts.indices.contains(index) else { return nil }
        return artifacts[index]
    }

    var body: some View {
        VStack(spacing: 14) {
            HStack {
                Text(verbatim: "Gallery")
                    .font(theme.ui(.callout, weight: .semibold))
                Spacer()
                Button { dismiss() } label: {
                    Text(verbatim: "Done")
                }
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 20)
            .padding(.top, 16)

            if let artifact, let path = artifact.path,
               let image = NSImage(contentsOfFile: path) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 900, maxHeight: 650)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                Text(artifact.imageRequest?.prompt ?? artifact.title)
                    .font(theme.ui(.base))
                    .lineLimit(3)
                    .frame(maxWidth: 760, alignment: .leading)

                HStack {
                    Button {
                        index -= 1
                    } label: {
                        Label("Previous", systemImage: "chevron.left")
                    }
                    .disabled(index == 0)

                    Text(verbatim: String(index + 1) + " of " + String(artifacts.count))
                        .font(theme.ui(.small, systemDesign: .monospaced))
                        .foregroundStyle(.secondary)

                    Button {
                        index += 1
                    } label: {
                        Label("Next", systemImage: "chevron.right")
                    }
                    .labelStyle(.titleAndIcon)
                    .disabled(index == artifacts.count - 1)

                    if artifact.imageRequest != nil {
                        Button {
                            model.regenerateImage(from: artifact)
                            dismiss()
                        } label: {
                            Text(verbatim: "Regenerate")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isRunning)
                    }
                }
            }
        }
        .frame(minWidth: 720, minHeight: 620)
        .padding(.bottom, 16)
    }
}
