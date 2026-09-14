import AppKit
import SwiftUI
import TurboSpark

/// Downloads one user-selected Hugging Face control vector for an installed
/// model, validates it, and turns it into the app's named preset.
struct SteeringVectorImportSheet: View {
    @ObservedObject var model: AppModel
    let installedModel: InstalledModel
    @Environment(\.dismiss) private var dismiss

    @State private var repo = ""
    @State private var file = ""
    @State private var revision = "main"
    @State private var name = "Refusal direction"
    @State private var scale = String(AppSteeringPreset.defaultScale)
    @State private var isDownloading = false
    @State private var errorMessage: String?

    private var descriptor: ModelFeatureDescriptor {
        ModelFeatureDescriptor.resolve(installedModel: installedModel)
    }

    private var source: SteeringVectorSource {
        SteeringVectorSource(repo: repo, file: file, revision: revision)
    }

    private var trimmedName: String {
        let value = name.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? "Downloaded direction" : value
    }

    private var parsedScale: Double? {
        guard let value = Double(scale), value.isFinite else { return nil }
        return value
    }

    private var validationError: String? {
        if let sourceError = source.validationError { return sourceError }
        guard let parsedScale else { return "Strength must be a finite number." }
        guard abs(parsedScale) <= 2 else { return "Strength must be between -2 and 2." }
        return nil
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            form
            Divider()
            footer
        }
        .frame(minWidth: 560, minHeight: 470)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(verbatim: "Add steering for " + installedModel.alias)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Text(verbatim: "Download a GGUF control vector, check its shape, and select it for the next model load.")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button { dismiss() } label: {
                Text("Done", bundle: .module)
            }
                .keyboardShortcut(.cancelAction)
        }
        .padding(16)
    }

    private var form: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if !descriptor.isSteeringReady {
                    Label("This model family does not support live steering.", systemImage: "xmark.octagon.fill")
                        .foregroundStyle(.red)
                }

                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(verbatim: "Hugging Face source")
                            .themedFont(.small, weight: .semibold)
                        TextField("owner/repository", text: $repo)
                            .textFieldStyle(.roundedBorder)
                        TextField("path/to/vector.gguf", text: $file)
                            .textFieldStyle(.roundedBorder)
                        TextField("Revision", text: $revision)
                            .textFieldStyle(.roundedBorder)
                    }
                }

                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(verbatim: "Preset")
                            .themedFont(.small, weight: .semibold)
                        TextField("Direction name", text: $name)
                            .textFieldStyle(.roundedBorder)
                        HStack {
                            Text(verbatim: "Ablate strength")
                            Spacer()
                            TextField("0.3", text: $scale)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 90)
                        }
                    }
                }

                Text(verbatim: "Ablation is an activation edit. It can be used on an ordinary model, including one that was not pre-abliterated. The vector still needs to come from this checkpoint, and this check cannot prove that semantic match.")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)

                VStack(alignment: .leading, spacing: 6) {
                    Text(verbatim: "Community sources")
                        .themedFont(.small, weight: .semibold)
                    Text(verbatim: "TurboSpark accepts GGUF control vectors here. Research .pt axes need offline conversion first.")
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 12) {
                            sourceLink(
                                title: "Ready-made GGUF vectors",
                                urlString: "https://huggingface.co/jukofyork/creative-writing-control-vectors-v3.0",
                                hint: "Opens a community repository of ready-made GGUF control vectors in your browser")
                            sourceLink(
                                title: "Vector generator",
                                urlString: "https://github.com/jukofyork/control-vectors",
                                hint: "Opens the community control-vector generator instructions in your browser")
                        }
                        HStack(spacing: 12) {
                            sourceLink(
                                title: "Research axes (.pt)",
                                urlString: "https://huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors",
                                hint: "Opens a research dataset of PyTorch axes and activation captures in your browser")
                            sourceLink(
                                title: "All community sources",
                                urlString: "https://huggingface.co/models?search=control%20vector",
                                hint: "Opens Hugging Face control-vector search in your browser")
                        }
                    }
                }
            }
            .padding(16)
        }
    }

    private func sourceLink(title: String, urlString: String, hint: String) -> some View {
        Button {
            guard let url = URL(string: urlString) else { return }
            NSWorkspace.shared.open(url)
        } label: {
            Text(verbatim: title)
        }
        .buttonStyle(.link)
        .themedFont(.small)
        .accessibilityHint(hint)
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let validationError {
                Label(validationError, systemImage: "info.circle")
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Spacer()
                Button { dismiss() } label: {
                    Text("Cancel", bundle: .module)
                }
                    .keyboardShortcut(.cancelAction)
                Button {
                    downloadAndUse()
                } label: {
                    if isDownloading {
                        ProgressView()
                            .controlSize(.small)
                        Text(verbatim: "Checking vector...")
                    } else {
                        Label {
                            Text(verbatim: "Download and Use")
                        } icon: {
                            Image(systemName: "arrow.down.circle.fill")
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isDownloading || validationError != nil || !descriptor.isSteeringReady || model.generating || model.opening)
            }
        }
        .padding(16)
    }

    private func downloadAndUse() {
        guard !isDownloading, validationError == nil, let parsedScale else { return }
        let requestedSource = source
        let requestedName = trimmedName
        isDownloading = true
        errorMessage = nil

        Task {
            do {
                let result = try await SteeringVectorDownloader.download(
                    source: requestedSource,
                    expectedHidden: descriptor.hiddenSize,
                    expectedLayers: descriptor.layerCount)

                if let existing = model.steeringPresets.first(where: {
                    URL(fileURLWithPath: $0.vectorPath).standardizedFileURL.path
                        == URL(fileURLWithPath: result.path).standardizedFileURL.path
                }) {
                    model.selectSteeringPreset(existing.id)
                } else {
                    let preset = AppSteeringPreset(
                        name: requestedName,
                        vectorPath: result.path,
                        mode: .ablate,
                        scale: parsedScale,
                        notes: "Downloaded from " + requestedSource.identity,
                        vectorHidden: result.info.hidden,
                        vectorSpannedLayers: result.info.spannedLayers)
                    model.upsertSteeringPreset(preset)
                    model.selectSteeringPreset(preset.id)
                }
                model.setSteeringEnabled(true)
                model.showToast(
                    "Vector downloaded and selected. Reload the model to apply steering.",
                    style: .success,
                    duration: 7.0)
                isDownloading = false
                dismiss()
            } catch {
                errorMessage = error.localizedDescription
                isDownloading = false
            }
        }
    }
}
