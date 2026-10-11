import AppKit
import SwiftUI
import TurboSpark

/// Action bar for an installed model (Load/Unload, Start Chat, Reveal in Finder, Delete).
///
/// Drawn with the shared action roles: Start Chat is the one primary, Load
/// and Unload are secondary, Reveal is quiet, Delete is destructive. A
/// `FlowLayout` lets the row wrap at large text sizes instead of clipping.
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
        HStack(alignment: .top, spacing: 10) {
            FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: 6) {
                if isCurrentlyLoaded {
                    Button(action: onUnload) {
                        Label { Text("Unload", bundle: .module) } icon: { Image(systemName: "eject.fill") }
                    }
                    .appAction(.secondary)
                    .disabled(!canUnloadModel)
                    .help(Text("Eject model and release unified memory", bundle: .module))
                } else {
                    Button(action: onLoad) {
                        Label { Text("Load Model", bundle: .module) } icon: { Image(systemName: "bolt.fill") }
                    }
                    .appAction(.secondary)
                    .help(Text("Load model into memory slot cache for inference", bundle: .module))
                }

                Button(action: onStartChat) {
                    Label { Text("Start Chat", bundle: .module) } icon: { Image(systemName: "bubble.left.and.bubble.right.fill") }
                }
                .appAction(.primary)
                .help(Text("Switch to Chat view with this model", bundle: .module))
                .accessibilityLabel(Text("Start chat with \(installedModel.alias)", bundle: .module))

                Button {
                    ModelStorageManager.revealInFinder(path: installedModel.path)
                } label: {
                    Label { Text("Reveal", bundle: .module) } icon: { Image(systemName: "folder") }
                }
                .appAction(.quiet)
                .help(Text("Reveal in Finder", bundle: .module) + Text(verbatim: ": \(installedModel.path)"))
                .accessibilityLabel(Text("Reveal in Finder", bundle: .module))
                .accessibilityAddTraits(.isLink)
            }

            Spacer(minLength: 0)

            Button(role: .destructive, action: onDelete) {
                Label { Text("Delete", bundle: .module) } icon: { Image(systemName: "trash") }
            }
            .appAction(.destructive)
            // state#73: this pane offered Delete during a load, where the
            // model file is mapped by an open still in flight.
            .disabled(!canDeleteModel)
            .help(Text("Delete model from disk", bundle: .module))
            .accessibilityLabel(Text("Delete \(installedModel.alias)", bundle: .module))
        }
    }
}
