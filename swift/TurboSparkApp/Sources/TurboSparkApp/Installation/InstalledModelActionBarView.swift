import AppKit
import SwiftUI
import TurboSpark

/// Action bar for an installed model (Load/Unload, Start Chat, Reveal in Finder, Delete).
///
/// Drawn with the shared action roles: Start Chat is the one primary, Load
/// and Unload are secondary, Reveal is quiet, Delete is destructive. A
/// `FlowLayout` lets the row wrap at large text sizes instead of clipping.
///
/// **WHAT IS OFFERED COMES FROM `InstalledModelActionPlan`, NOT FROM THIS VIEW**
/// (task 6.7; requirements 3.2, 3.7, 5.4 and 5.5). A model whose drive is away has
/// no Load button: a statement naming the drive stands in its place, with a glyph
/// and the sentence, and there is no Start Chat. A model whose folder is gone says
/// so and offers Remove from list. A model the drive will make slow keeps a
/// warning on screen, and its load action reads "Load anyway": it runs the very
/// load Load runs, because advice never prevents a load, and Start Chat waits
/// until the model is loaded so it cannot skip the warning.
///
/// **THE NOTICE COMES BEFORE THE BUTTONS** in reading order, so VoiceOver says
/// what is wrong before it offers what can be done. It is its own row, because a
/// `FlowLayout` places a subview at its ideal width and a sentence has to wrap.
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
    /// "Load anyway": the same load as `onLoad`, with the warning already read.
    let onLoadAnyway: () -> Void
    let onUnload: () -> Void
    let onStartChat: () -> Void
    let onDelete: () -> Void
    let onRemoveFromList: () -> Void
    /// Opens the move sheet (task 6.11). The sheet gives the reason when a move is
    /// refused, so the button is not hidden for a model that is loaded.
    let onMove: () -> Void

    private var plan: InstalledModelActionPlan {
        InstalledModelActionPlan(rowModel: rowModel, isCurrentlyLoaded: isCurrentlyLoaded)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let notice = plan.notice {
                LoadNoticeView(notice: notice)
            }
            buttons
        }
    }

    private var buttons: some View {
        HStack(alignment: .top, spacing: 10) {
            FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: 6) {
                if plan.loadControl == .unload {
                    Button(action: onUnload) {
                        Label { Text("Unload", bundle: .module) } icon: { Image(systemName: "eject.fill") }
                    }
                    .appAction(.secondary)
                    .disabled(!canUnloadModel)
                    .help(Text("Eject model and release unified memory", bundle: .module))
                } else if plan.loadControl == .load {
                    Button(action: onLoad) {
                        Label { Text("Load Model", bundle: .module) } icon: { Image(systemName: "bolt.fill") }
                    }
                    .appAction(.secondary)
                    .help(Text("Load model into memory slot cache for inference", bundle: .module))
                } else if plan.loadControl == .loadAnyway {
                    // The one primary action of a warned model: Start Chat waits.
                    Button(action: onLoadAnyway) {
                        Label { Text("Load anyway", bundle: .module) } icon: { Image(systemName: "bolt.fill") }
                    }
                    .appAction(.primary)
                    .help(plan.notice.map { Text(verbatim: $0.text) } ?? Text("Load anyway", bundle: .module))
                }

                startChatButton

                // The recorded path of a model that is away or missing is not
                // somewhere Finder can show.
                Button {
                    ModelStorageManager.revealInFinder(path: installedModel.path)
                } label: {
                    Label { Text("Reveal", bundle: .module) } icon: { Image(systemName: "folder") }
                }
                .appAction(.quiet)
                .disabled(!plan.revealEnabled)
                .help(
                    rowModel.loadBlockedReason.map { Text(verbatim: $0) }
                        ?? (Text("Reveal in Finder", bundle: .module) + Text(verbatim: ": \(installedModel.path)")))
                .accessibilityLabel(Text("Reveal in Finder", bundle: .module))
                .accessibilityAddTraits(.isLink)

                // A real install on This Mac or a connected drive can move to
                // another library; the sheet says why when it cannot yet.
                if plan.offersMove {
                    Button(action: onMove) {
                        Label { Text("Move to...", bundle: .module) } icon: { Image(systemName: "arrow.left.arrow.right.circle") }
                    }
                    .appAction(.secondary)
                    .disabled(isChanging)
                    .accessibilityLabel(Text("Move to...", bundle: .module))
                    .accessibilityValue(Text(verbatim: UntrustedText.display(installedModel.alias)))
                }
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

    /// Start Chat is the primary action, except for a warned model that is not
    /// loaded yet: there it is disabled with the warning as its reason, and
    /// "Load anyway" is the primary.
    @ViewBuilder
    private var startChatButton: some View {
        switch plan.startChat {
        case .hidden:
            EmptyView()
        case .enabled:
            Button(action: onStartChat) {
                Label { Text("Start Chat", bundle: .module) } icon: { Image(systemName: "bubble.left.and.bubble.right.fill") }
            }
            .appAction(.primary)
            .help(Text("Switch to Chat view with this model", bundle: .module))
            .accessibilityLabel(Text("Start chat with \(installedModel.alias)", bundle: .module))
        case .disabled(let reason):
            Button(action: onStartChat) {
                Label { Text("Start Chat", bundle: .module) } icon: { Image(systemName: "bubble.left.and.bubble.right.fill") }
            }
            .appAction(.secondary)
            .disabled(true)
            .help(Text(verbatim: reason))
            .accessibilityLabel(Text("Start chat with \(installedModel.alias)", bundle: .module))
        }
    }
}

/// A statement or a warning of the action bar: a glyph and a sentence, never
/// color alone. The glyph is decoration and VoiceOver skips it; the sentence is
/// read as one element.
private struct LoadNoticeView: View {
    let notice: InstalledModelActionPlan.Notice

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: notice.glyph)
                .themedFont(.small)
                .foregroundStyle(notice.kind == .warning ? Color.orange : Color.secondary)
                .accessibilityHidden(true)
            Text(verbatim: notice.text)
                .themedFont(.small)
                .foregroundStyle(.appText)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            .appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
    }
}
