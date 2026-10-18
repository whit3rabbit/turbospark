import SwiftUI
import TurboSpark

/// Group variants by model identity, with each family's curated exports
/// sharing one row.
enum ImageModelPresentation {
    static func family(_ modelID: String) -> String {
        let lowered = modelID.lowercased()
        if lowered.contains("qwen-image-2.1-turbo") {
            return "Qwen-Image 2.1 Turbo"
        }
        if lowered.contains("z-image-turbo") || lowered.contains("z-image") {
            return "Z-Image Turbo"
        }
        return modelID
    }

    static func quantization(_ value: String) -> String {
        if let range = value.range(of: "bits-", options: .backwards),
           let bits = Int(value[range.upperBound...]) { return "\(bits)-bit" }
        if value.lowercased().contains("int4") { return "4-bit" }
        if value.lowercased().contains("fp16") { return "FP16" }
        return value
    }
}

@MainActor
struct ImageModelControls: View {
    @ObservedObject var model: AppModel
    @Binding var importing: Bool
    @State private var browsingFamily: String?
    /// The library the user picked for a download from this panel. Nil follows
    /// the default (requirement 2.2). Per install: not remembered across launches.
    @State private var destination: String?

    /// The `library:` the downloads take for the menu's choice: nil for This Mac.
    private var chosenLibrary: String? {
        model.installDestinations.libraryForInstall(selection: destination)
    }

    private var families: [String] {
        let set = Set(
            model.imageModels
                .filter { AppModel.supportsMLXImageModel(modelID: $0.modelID) }
                .map { ImageModelPresentation.family($0.modelID) }
                + model.imageDownloadChoices.map { ImageModelPresentation.family($0.modelID) })
        if set.isEmpty {
            return ["Z-Image Turbo"]
        }
        return set.sorted()
    }

    private var family: String? {
        browsingFamily ?? model.selectedImageModel.map { ImageModelPresentation.family($0.modelID) }
            ?? families.first ?? "Z-Image Turbo"
    }

    private var installed: [ImageInstalledModel] {
        model.imageModels.filter {
            AppModel.supportsMLXImageModel(modelID: $0.modelID)
                && ImageModelPresentation.family($0.modelID) == family
        }
    }

    private var sources: [ImageCatalogEntry] {
        model.imageDownloadChoices.filter { entry in
            ImageModelPresentation.family(entry.modelID) == family
                && !model.imageModels.contains(where: { $0.alias == entry.alias })
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            controlRow
            // Where a download goes: only when there is a choice (2.1), and only
            // while a download is on offer here.
            if !sources.isEmpty {
                InstallDestinationPicker(model: model, selection: $destination)
            }
        }
        .themedFont(.small)
        .disabled(model.isRunning)
        .onChange(of: model.imageModelPath) { browsingFamily = nil }
    }

    private var controlRow: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Model", bundle: .module).themedFont(.tiny, weight: .medium)
                Menu {
                    ForEach(families, id: \.self) { name in
                        Button {
                            browsingFamily = name
                            if let first = model.imageModels.first(where: {
                                AppModel.supportsMLXImageModel(modelID: $0.modelID)
                                    && ImageModelPresentation.family($0.modelID) == name
                            }) { model.selectImageModel(first) }
                        } label: { Text(verbatim: name) }
                    }
                    Divider()
                    Button { importing = true } label: { Text("Choose Folder...", bundle: .module) }
                } label: {
                    if let family {
                        Text(verbatim: family).lineLimit(1)
                    } else {
                        Text("Choose Folder...", bundle: .module).lineLimit(1)
                    }
                }
                .appMenuAction(size: .small)
                .help(Text("Model", bundle: .module))
                .accessibilityLabel(Text("Model", bundle: .module))
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .leading, spacing: 8) {
                Text("Quantization", bundle: .module).themedFont(.tiny, weight: .medium)
                Menu {
                    ForEach(installed) { item in
                        Button { model.selectImageModel(item) } label: {
                            Label {
                                Text(verbatim: ImageModelPresentation.quantization(item.quantization) + " (" + item.alias + ")")
                            } icon: {
                                Image(systemName: model.imageModelPath == item.path ? "checkmark" : "internaldrive")
                            }
                        }
                    }
                    if !installed.isEmpty && !sources.isEmpty { Divider() }
                    ForEach(sources) { source in
                        Button { model.installImageModel(source, library: chosenLibrary) } label: {
                            Label {
                                Text("Download", bundle: .module)
                                    + Text(verbatim: " " + ImageModelPresentation.quantization(source.quantization))
                            } icon: { Image(systemName: "arrow.down.circle") }
                        }
                        .disabled(!model.canInstallImageModel(alias: source.alias))
                    }
                } label: {
                    if let selected = model.selectedImageModel,
                       AppModel.supportsMLXImageModel(modelID: selected.modelID),
                       ImageModelPresentation.family(selected.modelID) == family {
                        Text(verbatim: ImageModelPresentation.quantization(selected.quantization))
                    } else if let firstSource = sources.first {
                        Text(verbatim: ImageModelPresentation.quantization(firstSource.quantization))
                    } else {
                        Text("Download", bundle: .module)
                    }
                }
                .appMenuAction(size: .small)
                .disabled(installed.isEmpty && sources.isEmpty)
                .help(Text("Quantization", bundle: .module))
                .accessibilityLabel(Text("Quantization", bundle: .module))
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if installed.isEmpty, let source = sources.first {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Download", bundle: .module)
                        .themedFont(.tiny).hidden().accessibilityHidden(true)
                    Button {
                        model.installImageModel(source, library: chosenLibrary)
                    } label: {
                        Label {
                            Text("Download", bundle: .module)
                        } icon: {
                            Image(systemName: "arrow.down.circle.fill")
                        }
                    }
                    // Secondary: Generate is the composer's one primary action.
                    .appAction(.secondary, size: .small)
                    .disabled(!model.canInstallImageModel(alias: source.alias))
                }
            }

            if model.imageSession != nil {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Model", bundle: .module)
                        .themedFont(.tiny).hidden().accessibilityHidden(true)
                    Button {
                        model.unloadImageModel()
                    } label: {
                        Label {
                            Text("Unload", bundle: .module)
                        } icon: {
                            Image(systemName: "eject.fill")
                        }
                    }
                    .appAction(.secondary, size: .small)
                    .disabled(!model.canUnloadImageModel)
                    .help(Text("Unload Model", bundle: .module))
                    .accessibilityLabel(Text("Unload Model", bundle: .module))
                }
            }
        }
    }
}
