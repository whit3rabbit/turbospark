import AppKit
import SwiftUI
import TurboSpark

/// Action bar for an installed model (Load/Unload, Start Chat, Reveal in Finder, Delete).
struct InstalledModelActionBarView: View {
    let installedModel: InstalledModel
    let isCurrentlyLoaded: Bool
    let canUnloadModel: Bool
    let canDeleteModel: Bool
    let onLoad: () -> Void
    let onUnload: () -> Void
    let onStartChat: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            if isCurrentlyLoaded {
                Button(action: onUnload) {
                    Label("Unload", systemImage: "eject.fill")
                        .themedFont(.small, weight: .medium)
                        .frame(height: 28)
                        .padding(.horizontal, 12)
                }
                .buttonStyle(.plain)
                .background(Color.orange.opacity(0.16), in: RoundedRectangle(cornerRadius: 6))
                .foregroundStyle(Color.orange)
                .disabled(!canUnloadModel)
                .help("Eject model and release unified memory")
            } else {
                Button(action: onLoad) {
                    Label("Load Model", systemImage: "bolt.fill")
                        .themedFont(.small, weight: .medium)
                        .frame(height: 28)
                        .padding(.horizontal, 12)
                }
                .buttonStyle(.plain)
                .background(Color.accentColor.opacity(0.18), in: RoundedRectangle(cornerRadius: 6))
                .foregroundStyle(Color.accentColor)
                .help("Load model into memory slot cache for inference")
            }

            Button(action: onStartChat) {
                Label("Start Chat", systemImage: "bubble.left.and.bubble.right.fill")
                    .themedFont(.small, weight: .medium)
                    .frame(height: 28)
                    .padding(.horizontal, 12)
            }
            .buttonStyle(.plain)
            .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 6))
            .foregroundStyle(.white)
            .help("Switch to Chat view with this model")

            Button {
                ModelStorageManager.revealInFinder(path: installedModel.path)
            } label: {
                Label("Reveal", systemImage: "folder")
                    .themedFont(.small, weight: .medium)
                    .frame(height: 28)
                    .padding(.horizontal, 10)
            }
            .buttonStyle(.plain)
            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            .foregroundStyle(.appText)
            .help("Show model files in Finder: \(installedModel.path)")
            .accessibilityLabel("Reveal in Finder")
            .accessibilityHint("Opens Finder showing the model files at \(installedModel.path)")
            .accessibilityAddTraits(.isLink)

            Spacer()

            Button(role: .destructive, action: onDelete) {
                Label("Delete", systemImage: "trash")
                    .themedFont(.small, weight: .medium)
                    .frame(height: 28)
                    .padding(.horizontal, 10)
            }
            .buttonStyle(.plain)
            .background(Color.red.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
            .foregroundStyle(Color.red)
            // state#73: this pane offered Delete during a load, where the
            // model file is mapped by an open still in flight.
            .disabled(!canDeleteModel)
            .help("Delete model files from disk")
        }
    }
}
