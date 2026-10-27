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
    @State private var confirmingTrash = false

    private var openLabel: Text {
        selecting ? Text("Select", bundle: .module) : Text("Open image", bundle: .module)
    }

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
                        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
                        .overlay {
                            if selected {
                                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous).stroke(.appAccent, lineWidth: 3)
                            }
                        }
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(openLabel)
                .accessibilityLabel(openLabel)
                .accessibilityValue(Text(artifact.imageRequest?.prompt ?? artifact.title))
                .accessibilityAddTraits(selected ? .isSelected : [])

                if selecting {
                    Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                        .themedFont(.title3).foregroundStyle(.appAccent)
                        .padding(8).background(.appPage, in: Circle()).padding(8)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                } else {
                    Menu {
                        ImageGalleryActions(model: model, artifact: artifact, reuse: reuse, requestTrash: { confirmingTrash = true })
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
                .help(Text(verbatim: artifact.imageRequest?.prompt ?? artifact.title))
            if !compact, let request = artifact.imageRequest {
                Text(verbatim: "\(request.width) x \(request.height)")
                    .themedFont(.tiny).foregroundStyle(.appSecondary)
            }
        }
        .contextMenu { ImageGalleryActions(model: model, artifact: artifact, reuse: reuse, requestTrash: { confirmingTrash = true }) }
        .confirmationDialog(Text("Move to Trash", bundle: .module), isPresented: $confirmingTrash) {
            Button(role: .destructive) {
                model.trashGeneratedImages(ids: [artifact.id])
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
        .task(id: artifact.contentKey) {
            // Decode a thumbnail once per revision, not the full PNG on each progress tick.
            thumbnail = nil
            guard let url = artifact.url,
                  let source = CGImageSourceCreateWithURL(url as CFURL, nil),
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
    var requestTrash: (() -> Void)? = nil

    var body: some View {
        Button(action: reuse) {
            Label { Text("Use prompt", bundle: .module) } icon: { Image(systemName: "text.bubble") }
        }
        Button { model.regenerateImage(from: artifact) } label: {
            Label { Text("Regenerate", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
        }
        .disabled(
            !model.canStartImageGeneration
                || !model.hasSupportedSelectedImageModel
                || model.imageModelPath.isEmpty
                || artifact.imageRequest == nil)
        Divider()
        Button { export() } label: {
            Label { Text("Save As...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.up") }
        }
        .disabled(!artifact.existsOnDisk)
        Button {
            if let url = artifact.url {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
        } label: {
            Label { Text("Show in Finder", bundle: .module) } icon: { Image(systemName: "folder") }
        }
        .disabled(!artifact.existsOnDisk)
        Divider()
        Button(role: .destructive) {
            if let requestTrash {
                requestTrash()
            } else {
                model.trashGeneratedImages(ids: [artifact.id])
            }
        } label: {
            Label { Text("Move to Trash", bundle: .module) } icon: { Image(systemName: "trash") }
        }
    }

    private func export() {
        guard let url = artifact.url else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = artifact.fileName
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        do {
            try Data(contentsOf: url).write(to: destination, options: .atomic)
        } catch { model.showToast(error.localizedDescription, style: .error) }
    }
}
