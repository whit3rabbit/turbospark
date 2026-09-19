import AppKit
import SwiftUI

@MainActor
struct ImageGalleryCarousel: View {
    @ObservedObject var model: AppModel
    let reuse: (AppArtifact) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var selectedID: UUID

    init(model: AppModel, initialID: UUID, reuse: @escaping (AppArtifact) -> Void) {
        self.model = model
        self.reuse = reuse
        _selectedID = State(initialValue: initialID)
    }

    private var artifacts: [AppArtifact] { model.savedImageArtifacts }
    private var index: Int { artifacts.firstIndex { $0.id == selectedID } ?? 0 }
    private var artifact: AppArtifact? { artifacts.indices.contains(index) ? artifacts[index] : nil }

    var body: some View {
        VStack(spacing: 18) {
            HStack {
                Text("Preview", bundle: .module).themedFont(.title3, weight: .semibold)
                Spacer()
                if let artifact {
                    Menu {
                        ImageGalleryActions(model: model, artifact: artifact) {
                            reuse(artifact); dismiss()
                        }
                    } label: { Image(systemName: "ellipsis") }
                    .menuStyle(.borderlessButton).fixedSize()
                    .help(Text("Image actions", bundle: .module))
                    .accessibilityLabel(Text("Image actions", bundle: .module))
                }
                Button { dismiss() } label: { Text("Done", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
            }
            if let artifact {
                if let url = artifact.url, let image = NSImage(contentsOf: url) {
                    Image(nsImage: image).resizable().scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    ContentUnavailableView {
                        Label { Text("Image unavailable", bundle: .module) } icon: { Image(systemName: "photo") }
                    }
                }
                Text(artifact.imageRequest?.prompt ?? artifact.title)
                    .themedFont(.base).textSelection(.enabled).lineLimit(4)
                if let request = artifact.imageRequest {
                    HStack {
                        Text(verbatim: "\(request.width) x \(request.height)")
                        Text("Seed", bundle: .module)
                        Text(verbatim: String(request.seed)).textSelection(.enabled)
                    }
                    .themedFont(.tiny).foregroundStyle(.appSecondary)
                }
                HStack(spacing: 20) {
                    Button { selectedID = artifacts[index - 1].id } label: {
                        Image(systemName: "chevron.left")
                    }
                    .disabled(index == 0).keyboardShortcut(.leftArrow, modifiers: [])
                    .help(Text("Previous", bundle: .module))
                    .accessibilityLabel(Text("Previous", bundle: .module))
                    Text(verbatim: "\(index + 1) / \(artifacts.count)").monospacedDigit()
                    Button { selectedID = artifacts[index + 1].id } label: {
                        Image(systemName: "chevron.right")
                    }
                    .disabled(index + 1 >= artifacts.count).keyboardShortcut(.rightArrow, modifiers: [])
                    .help(Text("Next", bundle: .module))
                    .accessibilityLabel(Text("Next", bundle: .module))
                    Spacer()
                    Button { reuse(artifact); dismiss() } label: { Text("Use prompt", bundle: .module) }
                        .buttonStyle(.borderedProminent)
                }
                .themedFont(.small)
            }
        }
        .padding(24)
        .frame(minWidth: 580, idealWidth: 800, maxWidth: 960, minHeight: 520, idealHeight: 720)
        .background(.appPage)
        .onChange(of: artifacts.map(\.id)) { old, new in
            if new.isEmpty { dismiss() }
            else if !new.contains(selectedID) {
                let previous = old.firstIndex(of: selectedID) ?? 0
                selectedID = new[min(previous, new.count - 1)]
            }
        }
    }
}
