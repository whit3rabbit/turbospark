import SwiftUI
import TurboSpark

/// Offers the pinned embedding models under an embedding-model field:
/// Download, then Use, which fills the field with the model's alias. The
/// same row serves Memory settings and the server's Advanced settings, so
/// both describe the model the same way. Models this Mac cannot run (Core
/// ML exports need macOS 15) are not offered.
@MainActor
struct RecommendedEmbeddingModelRow: View {
    @ObservedObject var model: AppModel
    @Binding var selection: String
    var disabled: Bool = false
    var onSelect: () -> Void = {}

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(model.embeddingProfiles.filter(\.canRun)) { profile in
                row(profile)
            }
        }
        .onAppear { model.refreshEmbeddingProfiles() }
    }

    private func row(_ profile: EmbeddingProfile) -> some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Recommended", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                Text(verbatim: profile.displayName)
                    .themedFont(.small, weight: .medium)
                Text(verbatim: ByteCountFormatter.string(
                    fromByteCount: Int64(clamping: profile.downloadBytes), countStyle: .file))
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            .accessibilityElement(children: .combine)
            Spacer()
            if profile.installedPath == nil {
                Button {
                    model.enqueueEmbeddingDownload(profile)
                } label: { Text("Download", bundle: .module) }
                .accessibilityLabel(Text("Download", bundle: .module) + Text(verbatim: " \(profile.displayName)"))
                .appAction(.secondary, size: .small)
                .disabled(disabled || !model.canQueueModelDownload(.embedding(alias: profile.alias)))
                .appPointerCursor()
            } else if selection == profile.alias {
                Label {
                    Text("Installed", bundle: .module)
                } icon: {
                    Image(systemName: "checkmark.circle")
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            } else {
                Button {
                    selection = profile.alias
                    onSelect()
                } label: { Text("Use", bundle: .module) }
                .accessibilityLabel(Text("Use", bundle: .module) + Text(verbatim: " \(profile.displayName)"))
                .appAction(.secondary, size: .small)
                .disabled(disabled)
                .appPointerCursor()
            }
        }
    }
}
