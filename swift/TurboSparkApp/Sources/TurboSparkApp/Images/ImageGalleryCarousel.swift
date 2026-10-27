import AppKit
import SwiftUI

@MainActor
struct ImageGalleryCarousel: View {
    @ObservedObject var model: AppModel
    let reuse: (AppArtifact) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var selectedID: UUID
    @State private var confirmingTrash = false

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
            AppAdaptiveHeader {
                Text("Preview", bundle: .module)
                    .themedFont(.title3, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
            } actions: {
                if let artifact {
                    Menu {
                        ImageGalleryActions(model: model, artifact: artifact, reuse: {
                            reuse(artifact); dismiss()
                        }, requestTrash: { confirmingTrash = true })
                    } label: {
                        Label { Text("Image actions", bundle: .module) } icon: { Image(systemName: "ellipsis") }
                            .labelStyle(.iconOnly)
                    }
                    .appMenuAction()
                    .help(Text("Image actions", bundle: .module))
                    .accessibilityLabel(Text("Image actions", bundle: .module))
                }
                Button { dismiss() } label: { Text("Done", bundle: .module) }
                    .appAction(.secondary)
                    .keyboardShortcut(.cancelAction)
            }
            if let artifact {
                if let url = artifact.url, let image = NSImage(contentsOf: url) {
                    Image(nsImage: image).resizable().scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
                        .accessibilityLabel(Text(verbatim: artifact.imageRequest?.prompt ?? artifact.title))
                } else {
                    ContentUnavailableView {
                        Label { Text("Image unavailable", bundle: .module) } icon: { Image(systemName: "photo") }
                    }
                }
                Text(verbatim: artifact.imageRequest?.prompt ?? artifact.title)
                    .themedFont(.base).textSelection(.enabled).lineLimit(4)
                    .help(Text(verbatim: artifact.imageRequest?.prompt ?? artifact.title))
                if let request = artifact.imageRequest {
                    HStack {
                        Text(verbatim: "\(request.width) x \(request.height)")
                        Text("Seed", bundle: .module)
                        Text(verbatim: String(request.seed)).textSelection(.enabled)
                    }
                    .themedFont(.tiny).foregroundStyle(.appSecondary)
                }
                HStack(spacing: 20) {
                    AppIconButton("chevron.left", label: Text("Previous", bundle: .module)) {
                        selectedID = artifacts[index - 1].id
                    }
                    .appAction(.secondary)
                    .disabled(index == 0).keyboardShortcut(.leftArrow, modifiers: [])
                    Text(verbatim: "\(index + 1) / \(artifacts.count)").monospacedDigit()
                    AppIconButton("chevron.right", label: Text("Next", bundle: .module)) {
                        selectedID = artifacts[index + 1].id
                    }
                    .appAction(.secondary)
                    .disabled(index + 1 >= artifacts.count).keyboardShortcut(.rightArrow, modifiers: [])
                    Spacer()
                    Button { reuse(artifact); dismiss() } label: { Text("Use prompt", bundle: .module) }
                        .appAction(.primary)
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
        .confirmationDialog(Text("Move to Trash", bundle: .module), isPresented: $confirmingTrash) {
            Button(role: .destructive) {
                if let artifact {
                    model.trashGeneratedImages(ids: [artifact.id])
                    dismiss()
                }
            } label: {
                Text("Move to Trash", bundle: .module)
            }
            Button(role: .cancel) {
                confirmingTrash = false
            } label: {
                Text("Cancel", bundle: .module)
            }
        } message: {
            Text("Are you sure you want to move this image to the Trash?", bundle: .module)
        }
    }
}
