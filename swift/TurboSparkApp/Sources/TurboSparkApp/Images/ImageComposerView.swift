import SwiftUI
import TurboSpark

@MainActor
struct ImageComposerView: View {
    @ObservedObject var model: AppModel
    @Binding var importing: Bool
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.appTheme) private var theme
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @State private var expanded = false

    private var active: Bool { model.imageGenerationTask != nil }

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        VStack(spacing: 0) {
            if model.imageModelPath.isEmpty && !model.isInstallingImageModel
                && !model.savedImageArtifacts.isEmpty && !expanded {
                ImageModelDownloadOffer(model: model, importing: $importing)
            }
            if active { progress.padding(.bottom, 12) }
            VStack(spacing: 12) {
                TextField(text: $model.promptText, axis: .vertical) {
                    Text("Describe the image you want to create.", bundle: .module)
                }
                .textFieldStyle(.plain)
                .themedFont(.base)
                .lineLimit(2...5)
                .accessibilityLabel(Text("Image prompt", bundle: .module))

                HStack(spacing: 12) {
                    Button {
                        withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.2)) {
                            expanded.toggle()
                        }
                    } label: {
                        Label {
                            Text("Settings", bundle: .module)
                        } icon: { Image(systemName: "slider.horizontal.3") }
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(expanded ? theme.accent : theme.secondaryText)
                    .accessibilityValue(Text(expanded ? "Expanded" : "Collapsed", bundle: .module))
                    .help(Text("Image settings", bundle: .module))
                    .accessibilityLabel(Text("Image settings", bundle: .module))

                    Text(summary)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 0)
                    if model.imageSession != nil {
                        Button {
                            model.unloadImageModel()
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "eject.fill")
                                Text("Unload", bundle: .module)
                            }
                        }
                        .buttonStyle(TSPressScaleStyle(scale: 0.94))
                        .foregroundStyle(.appSecondary)
                        .disabled(!model.canUnloadImageModel)
                        .help(Text("Unload Model", bundle: .module))
                        .accessibilityLabel(Text("Unload Model", bundle: .module))
                    }
                    if active {
                        Button { model.cancelImageGeneration() } label: {
                            Label { Text("Stop", bundle: .module) } icon: { Image(systemName: "stop.fill") }
                        }
                        .disabled(model.isCancellationPending)
                    } else {
                        Button { model.generateImage() } label: {
                            Label { Text("Generate", bundle: .module) } icon: { Image(systemName: "arrow.up") }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canGenerateImage || model.isInGhostChat)
                        .keyboardShortcut(.return, modifiers: .command)
                    }
                }
                .themedFont(.small)
            }
            .padding(16)
            .background(theme.composerBackground, in: RoundedRectangle(cornerRadius: 16))
            .overlay { RoundedRectangle(cornerRadius: 16).stroke(.appBorder, lineWidth: 1) }

            if expanded {
                settings
                    .padding(18)
                    .background(.appSurface, in: UnevenRoundedRectangle(
                        bottomLeadingRadius: 14, bottomTrailingRadius: 14))
                    .padding(.horizontal, 10)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
            if model.isInGhostChat {
                Text("Image generation is unavailable in a temporary chat.", bundle: .module)
                    .themedFont(.tiny).foregroundStyle(.appSecondary).padding(.top, 8)
            } else if model.imageModelPath.isEmpty {
                Button { expanded = true } label: {
                    Text("Choose an image model in Settings to get started.", bundle: .module)
                }
                .buttonStyle(.plain).themedFont(.tiny).foregroundStyle(.appAccent).padding(.top, 8)
            }
        }
        .frame(maxWidth: 980)
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .frame(maxWidth: .infinity)
        .background(.appPage)
    }

    private var summary: String {
        let name = model.selectedImageModel.map { ImageModelPresentation.family($0.modelID) }
            ?? (model.imageModelPath.isEmpty ? "" : URL(fileURLWithPath: model.imageModelPath).lastPathComponent)
        let quant = model.selectedImageModel.map { ImageModelPresentation.quantization($0.quantization) } ?? ""
        let parts = [
            name.isEmpty ? (model.hasInstalledZImageModel ? "" : "Z-Image Turbo") : name,
            quant.isEmpty ? (model.hasInstalledZImageModel ? "" : String(localized: "Download", bundle: .module)) : quant,
            model.imageModelPath.isEmpty ? "" : model.imageSizeLabel,
            String(localized: "\(model.imageCount) image(s)", bundle: .module)
        ].filter { !$0.isEmpty }
        return parts.joined(separator: " / ")
    }

    private var settings: some View {
        ImageGenerationSettings(model: model, importing: $importing)
    }

    private var progress: some View {
        HStack(spacing: 12) {
            ProgressView(value: model.imageProgressFraction).frame(width: 100)
            Text("Image \(model.imageBatchIndex) of \(model.imageBatchCount)", bundle: .module)
            Text(stageDescription)
                .foregroundStyle(.appSecondary).lineLimit(1)
            Spacer(minLength: 0)
        }
        .themedFont(.small)
        .accessibilityElement(children: .combine)
    }

    private var stageDescription: String {
        guard let job = model.imageJob, let stage = job.stage else {
            return String(localized: "Waiting", bundle: .module)
        }
        switch stage {
        case "text_encoder":
            return "Encoding prompt"
        case "transformer":
            if job.total > 0 {
                return "Denoising step \(job.completed)/\(job.total)"
            }
            return "Denoising"
        case "vae_decoder":
            return "Decoding image"
        case "png_encode":
            return "Saving image"
        default:
            return stage.capitalized
        }
    }
}
