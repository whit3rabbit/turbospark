import SwiftUI
import TurboSpark

@MainActor
struct ImageComposerView: View {
    @ObservedObject var model: AppModel
    @Binding var importing: Bool
    @State private var expanded = false

    private var active: Bool { model.imageGenerationTask != nil }

    var body: some View {
        VStack(spacing: 0) {
            if model.imageModelPath.isEmpty && !model.isInstallingImageModel
                && !model.savedImageArtifacts.isEmpty && !expanded {
                ImageModelDownloadOffer(model: model, importing: $importing)
            }
            VStack(spacing: 12) {
                // The prompt is the pane's multi-line editor, so it takes the
                // shared editor surface every other editor uses.
                TextField(text: $model.promptText, axis: .vertical) {
                    Text("Describe the image you want to create.", bundle: .module)
                }
                .textFieldStyle(.plain)
                .themedFont(.base)
                .lineLimit(2...5)
                .appEditorSurface(minHeight: 56)
                .accessibilityLabel(Text("Image prompt", bundle: .module))

                controlRow
            }

            if expanded {
                ImageGenerationSettings(model: model, importing: $importing)
                    .padding(18)
                    .background(
                        .appSurface,
                        in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                            .stroke(.appBorder, lineWidth: 1))
                    .padding(.top, 10)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
            guidance
        }
        .frame(maxWidth: 980)
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .frame(maxWidth: .infinity)
        .background(.appPage)
    }

    private var controlRow: some View {
        HStack(spacing: 12) {
            // Named "Image settings" rather than "Settings" so it cannot be
            // read as the app's Settings window.
            AppChip(
                Text("Image settings", bundle: .module),
                systemImage: "slider.horizontal.3",
                isOn: expanded
            ) {
                TSMotion.animate(TSMotion.pane) { expanded.toggle() }
            }

            ImageStyleMenu(model: model)
                .appMenuAction(size: .small)

            summary
                .foregroundStyle(.appSecondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 0)
            if model.imageSession != nil {
                Button {
                    model.unloadImageModel()
                } label: {
                    Label { Text("Unload", bundle: .module) } icon: { Image(systemName: "eject.fill") }
                }
                .appAction(.quiet, size: .small)
                .disabled(!model.canUnloadImageModel)
                .help(Text("Unload Model", bundle: .module))
                .accessibilityLabel(Text("Unload Model", bundle: .module))
            }
            if active {
                Button { model.cancelImageGeneration() } label: {
                    Label { Text("Stop", bundle: .module) } icon: { Image(systemName: "stop.fill") }
                }
                .appAction(.secondary)
                .disabled(model.isCancellationPending)
            } else {
                Button { model.generateImage() } label: {
                    Label { Text("Generate", bundle: .module) } icon: { Image(systemName: "arrow.up") }
                }
                .appAction(.primary)
                .disabled(!model.canGenerateImage || model.isInGhostChat)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .themedFont(.small)
    }

    /// Names only controls that are on screen: the Download action (empty
    /// gallery or the offer above the prompt) and the Image settings chip,
    /// whose panel holds the model and quantization pickers. An older build
    /// sent people to Settings, which has no image model picker.
    @ViewBuilder
    private var guidance: some View {
        if model.isInGhostChat {
            Text("Image generation is unavailable in a temporary chat.", bundle: .module)
                .themedFont(.tiny).foregroundStyle(.appSecondary).padding(.top, 8)
        } else if model.imageModelPath.isEmpty {
            Button {
                TSMotion.animate(TSMotion.pane) { expanded = true }
            } label: {
                Text("Download an image model, or choose one in Image settings.", bundle: .module)
            }
            .appAction(.quiet, size: .small)
            .foregroundStyle(.appAccent)
            .padding(.top, 8)
        }
    }

    /// Model, quantization, size and count. Model names, sizes and
    /// quantization labels are data and stay verbatim; the words around
    /// them come from the catalog.
    private var summary: Text {
        let unsupportedSelection = !model.imageModelPath.isEmpty
            && !model.hasSupportedSelectedImageModel
        let name = model.selectedImageModel.map { ImageModelPresentation.family($0.modelID) }
            ?? (model.imageModelPath.isEmpty ? "" : URL(fileURLWithPath: model.imageModelPath).lastPathComponent)
        var parts: [Text] = []
        if !name.isEmpty {
            parts.append(Text(verbatim: name))
        } else if !model.hasInstalledMLXImageModel {
            parts.append(Text(verbatim: "Z-Image Turbo"))
        }
        let quant = model.selectedImageModel.map { ImageModelPresentation.quantization($0.quantization) } ?? ""
        if unsupportedSelection {
            parts.append(Text("MLX required", bundle: .module))
        } else if !quant.isEmpty {
            parts.append(Text(verbatim: quant))
        } else if !model.hasInstalledMLXImageModel {
            parts.append(Text("Download", bundle: .module))
        }
        if !model.imageModelPath.isEmpty {
            parts.append(Text(verbatim: model.imageSizeLabel))
        }
        parts.append(Text("\(model.imageCount) image(s)", bundle: .module))
        return parts.dropFirst().reduce(parts[0]) { $0 + Text(verbatim: " / ") + $1 }
    }
}
