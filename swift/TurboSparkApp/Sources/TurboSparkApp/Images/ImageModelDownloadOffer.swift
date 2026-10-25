import SwiftUI
import TurboSpark

@MainActor
struct ImageModelDownloadOffer: View {
    @ObservedObject var model: AppModel
    @Binding var importing: Bool
    /// The library the user picked for the download. Nil follows the default
    /// (requirement 2.2). Per install: not remembered across launches.
    @State private var destination: String?

    /// The `library:` the downloads take for the menu's choice: nil for This Mac.
    private var chosenLibrary: String? {
        model.installDestinations.libraryForInstall(selection: destination)
    }

    var body: some View {
        if let source = model.imageDownloadChoices.first {
            VStack(alignment: .leading, spacing: 8) {
                offerRow(source)
                // Where the download goes: only when there is a choice (2.1).
                InstallDestinationPicker(model: model, selection: $destination)
            }
            .disabled(model.isRunning)
            .padding(12)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 1)
            }
            .padding(.bottom, 8)
        }
    }

    private func offerRow(_ source: ImageCatalogEntry) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "photo.badge.arrow.down").foregroundStyle(.appAccent)
            VStack(alignment: .leading, spacing: 4) {
                Text(verbatim: ImageModelPresentation.family(source.modelID))
                    .themedFont(.small, weight: .semibold)
                HStack(spacing: 6) {
                    Text(verbatim: ImageModelPresentation.quantization(source.quantization))
                    Text("Recommended", bundle: .module)
                }
                .themedFont(.tiny).foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 8)
            Button { model.installImageModel(source, library: chosenLibrary) } label: {
                Label { Text("Download", bundle: .module) } icon: { Image(systemName: "arrow.down.circle") }
            }
            .appAction(.primary)
            .disabled(!model.canInstallImageModel(alias: source.alias))
            Menu {
                ForEach(model.imageDownloadChoices) { source in
                    Button { model.installImageModel(source, library: chosenLibrary) } label: {
                        Text(verbatim: ImageModelPresentation.quantization(source.quantization) + " (" + source.alias + ")")
                    }
                    .disabled(!model.canInstallImageModel(alias: source.alias))
                }
                Divider()
                Button { importing = true } label: { Text("Choose Folder...", bundle: .module) }
            } label: {
                Label { Text("More download options", bundle: .module) } icon: { Image(systemName: "ellipsis") }
                    .labelStyle(.iconOnly)
            }
            .appMenuAction()
            .help(Text("More download options", bundle: .module))
        }
    }
}
