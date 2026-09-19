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
            if active || model.isInstallingImageModel { progress.padding(.bottom, 12) }
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
            if model.isInstallingImageModel {
                ProgressView(value: model.imageInstallProgressFraction)
                    .frame(width: 120)
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.imageInstallAlias ?? String(localized: "Download", bundle: .module))
                        .themedFont(.small, weight: .medium)
                        .lineLimit(1)
                    if let stage = model.imageInstallStage {
                        Text(verbatim: stage)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
                Button { model.cancelImageInstall() } label: {
                    Text("Cancel", bundle: .module)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                ProgressView(value: model.imageProgressFraction).frame(width: 100)
                Text("Image \(model.imageBatchIndex) of \(model.imageBatchCount)", bundle: .module)
                Text(model.imageJob?.stage ?? String(localized: "Waiting", bundle: .module))
                    .foregroundStyle(.appSecondary).lineLimit(1)
                Spacer(minLength: 0)
            }
        }
        .themedFont(.small)
        .accessibilityElement(children: .combine)
    }
}
