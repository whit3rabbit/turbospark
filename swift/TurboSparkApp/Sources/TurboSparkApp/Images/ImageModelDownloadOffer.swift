import SwiftUI

@MainActor
struct ImageModelDownloadOffer: View {
    @ObservedObject var model: AppModel
    @Binding var importing: Bool

    var body: some View {
        if let source = model.imageDownloadChoices.first {
            HStack(spacing: 12) {
                Image(systemName: "photo.badge.arrow.down").foregroundStyle(.appAccent)
                VStack(alignment: .leading, spacing: 4) {
                    Text(ImageModelPresentation.family(source.modelID))
                        .themedFont(.small, weight: .semibold)
                    HStack(spacing: 6) {
                        Text(ImageModelPresentation.quantization(source.quantization))
                        Text("Recommended", bundle: .module)
                    }
                    .themedFont(.tiny).foregroundStyle(.appSecondary)
                }
                Spacer(minLength: 8)
                Button { model.installImageModel(source) } label: {
                    Label { Text("Download", bundle: .module) } icon: { Image(systemName: "arrow.down.circle") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canInstallImageModel(alias: source.alias))
                Menu {
                    ForEach(model.imageDownloadChoices) { source in
                        Button { model.installImageModel(source) } label: {
                            Text(ImageModelPresentation.quantization(source.quantization) + " (" + source.alias + ")")
                        }
                        .disabled(!model.canInstallImageModel(alias: source.alias))
                    }
                    Divider()
                    Button { importing = true } label: { Text("Choose Folder...", bundle: .module) }
                } label: { Image(systemName: "ellipsis") }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                .help(Text("Options", bundle: .module))
                .accessibilityLabel(Text("Options", bundle: .module))
            }
            .disabled(model.isRunning)
            .padding(12)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
            .overlay { RoundedRectangle(cornerRadius: 12).stroke(.appBorder, lineWidth: 1) }
            .padding(.bottom, 8)
        }
    }
}
