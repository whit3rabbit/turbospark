import SwiftUI
import TurboSpark

@MainActor
struct ImageComposerView: View {
    @ObservedObject var model: AppModel
    @Binding var importing: Bool
    @State private var expanded = false
    /// View-owned: a suggestion belongs to this field and dies with it.
    @StateObject private var enhancer = ImagePromptEnhancer()

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

                enhanceStatus
                controlRow
            }
            .tsAnimation(TSMotion.pane, value: enhancer.phase)

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
        .onChange(of: model.promptText) { _, text in enhancer.promptChanged(to: text) }
        // A rewrite nobody can see must not keep the chat session busy.
        .onDisappear { enhancer.dismiss() }
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

            enhanceButton

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
                Button {
                    // Generating is the user's own choice of prompt; an
                    // open suggestion would compete with the image for the GPU.
                    enhancer.dismiss()
                    model.generateImage()
                } label: {
                    Label { Text("Generate", bundle: .module) } icon: { Image(systemName: "arrow.up") }
                }
                .appAction(.primary)
                .disabled(!model.canGenerateImage || model.isInGhostChat)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .themedFont(.small)
    }

    // MARK: - Enhance

    /// Local only: `model.session` is the on-device chat session, never a
    /// remote provider, so the prompt does not leave the Mac.
    private var enhanceAvailability: ImagePromptEnhancement.Availability {
        ImagePromptEnhancement.availability(
            prompt: model.promptText,
            textModelLoaded: model.session != nil,
            // The same idleness the chat title sidecar waits for, plus load
            // and eject, which own the session while they run.
            textModelBusy: model.generating || model.submitting || model.pendingToolCall != nil
                || model.titleGenerationTask != nil || model.opening || model.isEjecting,
            imageGenerating: active,
            temporaryChat: model.isInGhostChat)
    }

    private var enhanceHelp: Text {
        switch enhanceAvailability {
        case .ready:
            Text("Rewrite the prompt in more detail with the loaded chat model. It runs on this Mac.", bundle: .module)
        case .emptyPrompt:
            Text("Type a prompt to enhance.", bundle: .module)
        case .noTextModel:
            Text("Load a local chat model to enhance the prompt.", bundle: .module)
        case .textModelBusy:
            Text("The chat model is busy. Try again when it finishes.", bundle: .module)
        case .imageGenerating:
            Text("Wait for the image to finish.", bundle: .module)
        case .temporaryChat:
            Text("Image generation is unavailable in a temporary chat.", bundle: .module)
        }
    }

    private var enhanceButton: some View {
        Button(action: startEnhance) {
            Label { Text("Enhance", bundle: .module) } icon: { Image(systemName: "wand.and.stars") }
        }
        .appAction(.quiet, size: .small)
        .disabled(enhanceAvailability != .ready || enhancer.isRunning)
        .help(enhanceHelp)
    }

    private func startEnhance() {
        guard enhanceAvailability == .ready, let session = model.session else { return }
        enhancer.start(
            prompt: model.promptText, completion: ImagePromptEnhancement.localCompletion(session))
    }

    /// Progress, the suggestion with Accept and Dismiss, a failure note, or
    /// Undo after an accept. Nothing here starts an image.
    @ViewBuilder
    private var enhanceStatus: some View {
        switch enhancer.phase {
        case .running:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Enhancing the prompt with the local chat model...", bundle: .module)
                    .foregroundStyle(.appSecondary)
                Spacer(minLength: 0)
                Button { enhancer.dismiss() } label: { Text("Cancel", bundle: .module) }
                    .appAction(.quiet, size: .small)
                    .keyboardShortcut(.cancelAction)
            }
            .themedFont(.small)
            .accessibilityElement(children: .contain)
        case .suggestion(_, let text):
            VStack(alignment: .leading, spacing: 8) {
                Text("Suggested prompt", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .accessibilityAddTraits(.isHeader)
                Text(verbatim: text)
                    .themedFont(.base)
                    .foregroundStyle(.appText)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 8) {
                    Spacer(minLength: 0)
                    Button { enhancer.dismiss() } label: { Text("Dismiss", bundle: .module) }
                        .appAction(.quiet, size: .small)
                        .keyboardShortcut(.cancelAction)
                    Button(action: acceptSuggestion) { Text("Accept", bundle: .module) }
                        .appAction(.secondary, size: .small)
                }
            }
            .appCardSurface(padding: 12)
            .accessibilityElement(children: .contain)
            .onAppear { announce(AppStrings.localized("Suggested prompt ready.")) }
        case .failed:
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.circle").foregroundStyle(.appWarning)
                    .accessibilityHidden(true)
                Text("The chat model did not return a usable prompt.", bundle: .module)
                    .foregroundStyle(.appSecondary)
                Spacer(minLength: 0)
                Button { enhancer.dismiss() } label: { Text("Dismiss", bundle: .module) }
                    .appAction(.quiet, size: .small)
                    .keyboardShortcut(.cancelAction)
            }
            .themedFont(.small)
            .onAppear { announce(AppStrings.localized("The chat model did not return a usable prompt.")) }
        case .idle:
            if enhancer.undoPrompt != nil {
                HStack(spacing: 8) {
                    Text("Prompt replaced with the suggestion.", bundle: .module)
                        .foregroundStyle(.appSecondary)
                    Spacer(minLength: 0)
                    Button {
                        if let previous = enhancer.undoAccept() { model.promptText = previous }
                    } label: {
                        Label { Text("Undo", bundle: .module) } icon: { Image(systemName: "arrow.uturn.backward") }
                    }
                    .appAction(.quiet, size: .small)
                }
                .themedFont(.small)
            }
        }
    }

    private func acceptSuggestion() {
        guard let text = enhancer.accept() else { return }
        model.promptText = text
    }

    private func announce(_ message: String) {
        _ = AccessibilityNotification.Announcement(message).post()
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
