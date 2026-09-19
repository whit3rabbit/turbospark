import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct ImageGalleryCard: View {
    @ObservedObject var model: AppModel
    let artifact: AppArtifact
    let compact: Bool
    let selecting: Bool
    let selected: Bool
    let open: () -> Void
    let reuse: () -> Void
    @State private var thumbnail: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack(alignment: .topTrailing) {
                Button(action: open) {
                    Color.clear
                        .aspectRatio(1, contentMode: .fit)
                        .overlay {
                            if let thumbnail {
                                Image(nsImage: thumbnail).resizable().scaledToFit()
                            } else {
                                VStack(spacing: 8) {
                                    Image(systemName: "photo").themedFont(.title)
                                    Text("Image unavailable", bundle: .module).themedFont(.tiny)
                                }
                                .foregroundStyle(.appSecondary)
                            }
                        }
                        .background(.appSurface)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay {
                            if selected {
                                RoundedRectangle(cornerRadius: 12).stroke(.appAccent, lineWidth: 3)
                            }
                        }
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(Text("Open image", bundle: .module))
                .accessibilityLabel(Text("Open image", bundle: .module))
                .accessibilityValue(Text(artifact.imageRequest?.prompt ?? artifact.title))

                if selecting {
                    Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                        .themedFont(.title3).foregroundStyle(.appAccent)
                        .padding(8).background(.appPage, in: Circle()).padding(8)
                        .allowsHitTesting(false)
                } else {
                    Menu {
                        ImageGalleryActions(model: model, artifact: artifact, reuse: reuse)
                    } label: { Image(systemName: "ellipsis").themedFont(.small, weight: .bold) }
                    .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                    .padding(9).background(.appPage, in: Circle()).padding(8)
                    .help(Text("Image actions", bundle: .module))
                    .accessibilityLabel(Text("Image actions", bundle: .module))
                }
            }
            Text(artifact.imageRequest?.prompt ?? artifact.title)
                .themedFont(.small).lineLimit(compact ? 1 : 2)
                .frame(maxWidth: .infinity, alignment: .leading)
            if !compact, let request = artifact.imageRequest {
                Text(verbatim: "\(request.width) x \(request.height)")
                    .themedFont(.tiny).foregroundStyle(.appSecondary)
            }
        }
        .contextMenu { ImageGalleryActions(model: model, artifact: artifact, reuse: reuse) }
        .task(id: artifact.contentKey) {
            // Decode a thumbnail once per revision, not the full PNG on each progress tick.
            thumbnail = nil
            guard let path = artifact.path,
                  let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
                  let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceThumbnailMaxPixelSize: 600,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                  ] as CFDictionary) else { return }
            thumbnail = NSImage(cgImage: image, size: .zero)
        }
    }
}

@MainActor
struct ImageGalleryActions: View {
    @ObservedObject var model: AppModel
    let artifact: AppArtifact
    let reuse: () -> Void

    var body: some View {
        Button(action: reuse) {
            Label { Text("Use prompt", bundle: .module) } icon: { Image(systemName: "text.bubble") }
        }
        Button { model.regenerateImage(from: artifact) } label: {
            Label { Text("Regenerate", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
        }
        .disabled(!model.canStartImageGeneration || model.imageModelPath.isEmpty || artifact.imageRequest == nil)
        Divider()
        Button { export() } label: {
            Label { Text("Save As...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.up") }
        }
        .disabled(!artifact.existsOnDisk)
        Button {
            if let path = artifact.path {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
            }
        } label: {
            Label { Text("Show in Finder", bundle: .module) } icon: { Image(systemName: "folder") }
        }
        .disabled(!artifact.existsOnDisk)
        Divider()
        Button(role: .destructive) { model.trashGeneratedImages(ids: [artifact.id]) } label: {
            Label { Text("Move to Trash", bundle: .module) } icon: { Image(systemName: "trash") }
        }
    }

    private func export() {
        guard let path = artifact.path else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = URL(fileURLWithPath: path).lastPathComponent
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        do {
            try Data(contentsOf: URL(fileURLWithPath: path)).write(to: destination, options: .atomic)
        } catch { model.showToast(error.localizedDescription, style: .error) }
    }
}
