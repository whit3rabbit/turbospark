import SwiftUI

@MainActor
struct ImageGenerationSettings: View {
    @ObservedObject var model: AppModel
    @Binding var importing: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            ImageModelControls(model: model, importing: $importing)
            Divider()
            style
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 28) {
                    count
                    size
                    Spacer(minLength: 0)
                    seed
                }
                VStack(alignment: .leading, spacing: 16) {
                    HStack(alignment: .top, spacing: 24) { count; Spacer(); seed }
                    size
                }
            }
            .disabled(model.imageGenerationTask != nil)
        }
        .themedFont(.small)
    }

    private var style: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Style", bundle: .module).themedFont(.tiny, weight: .medium)
            ImageStyleMenu(model: model)
        }
    }

    private var count: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Number of images", bundle: .module).themedFont(.tiny, weight: .medium)
            Picker(selection: $model.imageCount) {
                ForEach(1...4, id: \.self) { count in Text(verbatim: String(count)).tag(count) }
            } label: { Text("Number of images", bundle: .module) }
            .labelsHidden().pickerStyle(.segmented).frame(width: 150)
        }
    }

    private var size: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Image size", bundle: .module).themedFont(.tiny, weight: .medium)
            Picker(selection: $model.imageResolution) {
                ForEach(ImageResolutionPreset.allCases) { resolution in
                    Text(resolution.label).tag(resolution)
                }
            } label: {
                Text("Image size", bundle: .module)
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(minWidth: 130, alignment: .leading)
        }
    }

    private var seed: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Seed", bundle: .module).themedFont(.tiny, weight: .medium)
            TextField(text: $model.imageSeedText) { Text("Random", bundle: .module) }
                .textFieldStyle(.roundedBorder).frame(width: 130)
                .accessibilityLabel(Text("Seed", bundle: .module))
        }
    }
}
