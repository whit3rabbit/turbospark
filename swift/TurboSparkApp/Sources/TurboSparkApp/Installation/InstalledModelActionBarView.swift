import AppKit
import SwiftUI
import TurboSpark

/// Action bar for an installed model (Load/Unload, Start Chat, Reveal in Finder, Delete).
///
/// Drawn with the shared action roles: Start Chat is the one primary, Load
/// and Unload are secondary, Reveal is quiet, Delete is destructive. A
/// `FlowLayout` lets the row wrap at large text sizes instead of clipping.
///
/// **A MODEL THAT CANNOT BE OPENED OFFERS NO WAY TO OPEN IT** (task 6.4;
/// requirements 3.1 and 3.7). Load, Start Chat and Reveal are disabled with the
/// reason in their tooltips ("Connect T7 to load"). The proper replacement of
/// Load with a connect statement and a remove action is task 6.7; this only keeps
/// an offline row from offering something the existing code cannot do.
///
/// **WHAT LEAVES THE LIST DEPENDS ON THE ROW.** A model on a drive that is away
/// offers no removal: the engine refuses a delete there (the recorded path may
/// now be another drive) and forgets a row only with its library connected (the
/// marker would keep listing it). The row says to connect the drive. A model
/// whose folder is gone has nothing left to delete, so it offers only "Remove
/// from list". Every other row offers Delete.
struct InstalledModelActionBarView: View {
    let installedModel: InstalledModel
    let rowModel: InstalledModelRowModel
    let isCurrentlyLoaded: Bool
    let canUnloadModel: Bool
    let canDeleteModel: Bool
    /// A delete or a remove for this row is already running.
    let isChanging: Bool
    let onLoad: () -> Void
    let onUnload: () -> Void
    let onStartChat: () -> Void
    let onDelete: () -> Void
    let onRemoveFromList: () -> Void

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
                    .disabled(rowModel.loadBlockedReason != nil)
                    .help(
                        rowModel.loadBlockedReason.map { Text(verbatim: $0) }
                            ?? Text("Load model into memory slot cache for inference", bundle: .module))
                }

                Button(action: onStartChat) {
                    Label { Text("Start Chat", bundle: .module) } icon: { Image(systemName: "bubble.left.and.bubble.right.fill") }
                }
                .appAction(.primary)
                .disabled(rowModel.loadBlockedReason != nil)
                .help(
                    rowModel.loadBlockedReason.map { Text(verbatim: $0) }
                        ?? Text("Switch to Chat view with this model", bundle: .module))
                .accessibilityLabel(Text("Start chat with \(installedModel.alias)", bundle: .module))

                // The recorded path of a model that is away or missing is not
                // somewhere Finder can show.
                Button {
                    ModelStorageManager.revealInFinder(path: installedModel.path)
                } label: {
                    Label { Text("Reveal", bundle: .module) } icon: { Image(systemName: "folder") }
                }
                .appAction(.quiet)
                .disabled(rowModel.status != nil)
                .help(
                    rowModel.loadBlockedReason.map { Text(verbatim: $0) }
                        ?? (Text("Reveal in Finder", bundle: .module) + Text(verbatim: ": \(installedModel.path)")))
                .accessibilityLabel(Text("Reveal in Finder", bundle: .module))
                .accessibilityAddTraits(.isLink)
            }

            Spacer(minLength: 0)

            if rowModel.offersRemoveFromList {
                Button(role: .destructive, action: onRemoveFromList) {
                    Label { Text("Remove from list", bundle: .module) } icon: { Image(systemName: "minus.circle") }
                }
                .appAction(.destructive)
                .disabled(isChanging)
                .help(Text("TurboSpark stops listing this model. Its files are not deleted.", bundle: .module))
                .accessibilityValue(Text(verbatim: UntrustedText.display(installedModel.alias)))
            }
            if rowModel.offersDelete {
                Button(role: .destructive, action: onDelete) {
                    Label { Text("Delete", bundle: .module) } icon: { Image(systemName: "trash") }
                }
                .appAction(.destructive)
                // state#73: this pane offered Delete during a load, where the
                // model file is mapped by an open still in flight.
                .disabled(!canDeleteModel || isChanging)
                .help(Text("Delete model from disk", bundle: .module))
                .accessibilityLabel(Text("Delete \(installedModel.alias)", bundle: .module))
            }
        }
    }
}
