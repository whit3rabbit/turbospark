import SwiftUI
import TurboSpark

/// Group variants by model identity, with Z-Image's curated exports sharing one row.
enum ImageModelPresentation {
    static func family(_ modelID: String) -> String {
        if modelID.lowercased().contains("z-image-turbo") { return "Z-Image Turbo" }
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

    private var families: [String] {
        Set(model.imageModels.map { ImageModelPresentation.family($0.modelID) }
            + model.imageCatalog.map { ImageModelPresentation.family($0.modelID) }).sorted()
    }

    private var family: String? {
        browsingFamily ?? model.selectedImageModel.map { ImageModelPresentation.family($0.modelID) }
            ?? families.first
    }

    private var installed: [ImageInstalledModel] {
        model.imageModels.filter { ImageModelPresentation.family($0.modelID) == family }
    }

    private var sources: [ImageCatalogEntry] {
        model.imageCatalog.filter { entry in
            ImageModelPresentation.family(entry.modelID) == family
                && AppModel.testedZImageAliases.contains(entry.alias)
                && !model.imageModels.contains(where: { $0.alias == entry.alias })
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Model", bundle: .module).themedFont(.tiny, weight: .medium)
                Menu {
                    ForEach(families, id: \.self) { name in
                        Button {
                            browsingFamily = name
                            if let first = model.imageModels.first(where: {
                                ImageModelPresentation.family($0.modelID) == name
                            }) { model.selectImageModel(first) }
                        } label: { Text(name) }
                    }
                    Divider()
                    Button { importing = true } label: { Text("Choose Folder...", bundle: .module) }
                } label: {
                    Text(family ?? String(localized: "Choose Folder...", bundle: .module))
                        .lineLimit(1)
                }
                .help(Text("Model", bundle: .module))
                .accessibilityLabel(Text("Model", bundle: .module))
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .leading, spacing: 8) {
                Text("Quantization", bundle: .module).themedFont(.tiny, weight: .medium)
                Menu {
                    ForEach(installed) { item in
                        Button { model.selectImageModel(item) } label: {
                            Label(
                                ImageModelPresentation.quantization(item.quantization) + " (" + item.alias + ")",
                                systemImage: model.imageModelPath == item.path ? "checkmark" : "internaldrive")
                        }
                    }
                    if !installed.isEmpty && !sources.isEmpty { Divider() }
                    ForEach(sources) { source in
                        Button { model.installImageModel(source) } label: {
                            Label {
                                Text("Download", bundle: .module)
                                Text(ImageModelPresentation.quantization(source.quantization))
                            } icon: { Image(systemName: "arrow.down.circle") }
                        }
                    }
                } label: {
                    if let selected = model.selectedImageModel,
                       ImageModelPresentation.family(selected.modelID) == family {
                        Text(ImageModelPresentation.quantization(selected.quantization))
                    } else {
                        Text("Download", bundle: .module)
                    }
                }
                .disabled(installed.isEmpty && sources.isEmpty)
                .help(Text("Quantization", bundle: .module))
                .accessibilityLabel(Text("Quantization", bundle: .module))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .menuStyle(.borderlessButton)
        .themedFont(.small)
        .disabled(model.isRunning || model.isInstallingModel || model.isInstallingImageModel)
        .onChange(of: model.imageModelPath) { browsingFamily = nil }
    }
}
