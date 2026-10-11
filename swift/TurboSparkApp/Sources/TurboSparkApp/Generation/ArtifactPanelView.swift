import AppKit
import SwiftUI

/// Which content the artifact panel is showing. Built fresh by `RootView`
/// from the resolved claimant, so every field is re-read per model change
/// and a rewritten file bumps the panel's identity rather than going stale.
enum ArtifactPanelSource: Equatable {
    case artifact(UUID)
    case inlinePreview
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The right-hand artifact panel: the rendered document where the format
/// allows it, the source otherwise.
///
/// The `.html` arm is the sandbox (`ArtifactWebView`): offline until the
/// user grants THIS content network, ephemeral storage, file reads scoped
/// to the artifact's own folder. Everything else renders in-process with no
/// sandbox question to ask.
@MainActor
struct ArtifactPanelView: View {
    @ObservedObject var model: AppModel
    let source: ArtifactPanelSource

    private enum ViewMode { case preview, source }

    @State private var viewMode: ViewMode = .preview
    @State private var sourceText: String?
    @State private var sourceReadFailed = false
    @State private var isBannerDismissed = false
    @State private var isMaximized = false
    @State private var isCopied = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if showsNetworkBanner {
                networkBanner
                Divider()
            }
            content
        }
        .background(TurboSparkTheme.barBackgroundColor)
        .onExitCommand { close() }
        .sheet(isPresented: $isMaximized) { maximizedSheet }
        .onChange(of: identityKey) { _, _ in resetPerContentState() }
        .onAppear { resetPerContentState() }
        .task(id: identityKey) { loadSourceText() }
    }

    // MARK: - Resolution

    private var resolvedArtifact: AppArtifact? {
        switch source {
        case .artifact(let id): return model.artifact(id: id)
        case .inlinePreview: return nil
        }
    }

    private var resolvedPreview: ArtifactHTMLPreview? {
        switch source {
        case .artifact: return nil
        case .inlinePreview: return model.htmlPreview
        }
    }

    /// Reload key. The artifact's `contentKey` moves once per rewrite (never
    /// per streamed token); an inline preview's id moves when its html does.
    private var identityKey: String {
        switch source {
        case .artifact(let id):
            return "artifact:\(resolvedArtifact.map { $0.contentKey } ?? id.uuidString)"
        case .inlinePreview:
            return "inline:\(resolvedPreview?.id.uuidString ?? "gone")"
        }
    }

    /// The webview's own identity, which also carries the network grant: a
    /// grant flip rebuilds the view, so the offline content-rule list the
    /// first (blocked) load installed never meets a page that is allowed to
    /// fetch. Rules are enforced below the navigation delegate, so no
    /// delegate decision could lift them inside the old webview.
    private var webViewIdentity: String {
        "\(identityKey)#net:\(networkAllowed)"
    }

    private var title: String {
        resolvedArtifact?.title ?? resolvedPreview?.title ?? AppStrings.localized("Preview")
    }

    private var subtitle: String {
        if let artifact = resolvedArtifact { return artifact.detailText }
        if resolvedPreview != nil { return AppStrings.localized("HTML \u{2022} in memory") }
        return AppStrings.localized("No longer available")
    }

    private var symbolName: String {
        resolvedArtifact?.symbolName ?? "curlybraces.square"
    }

    private var isHTML: Bool {
        if resolvedArtifact != nil { return resolvedArtifact?.renderKind == .html }
        return resolvedPreview != nil
    }

    private var networkAllowed: Bool {
        if let artifact = resolvedArtifact { return model.isNetworkAllowed(for: artifact) }
        if let preview = resolvedPreview { return model.isNetworkAllowed(for: preview) }
        return false
    }

    private var showsNetworkBanner: Bool {
        isHTML && !networkAllowed && !isBannerDismissed
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: symbolName)
                .themedFont(.callout)
                .foregroundStyle(.appAccent)
                .help(subtitle)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(subtitle)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 4)

            Picker(selection: $viewMode) {
                Text("Preview", bundle: .module).tag(ViewMode.preview)
                Text("Source", bundle: .module).tag(ViewMode.source)
            } label: { Text("View", bundle: .module) }
            .pickerStyle(.segmented)
            .controlSize(.mini)
            .fixedSize()
            .help(Text("Switch between the rendered page and its source", bundle: .module))

            headerButton(
                isCopied ? "checkmark" : "doc.on.doc",
                help: isCopied ? Text("Copied", bundle: .module) : Text("Copy source", bundle: .module),
                label: isCopied ? Text("Copied source", bundle: .module) : Text("Copy source", bundle: .module))
            {
                copySource()
            }
            .disabled(sourceText == nil)

            if resolvedArtifact?.existsOnDisk == true {
                if let artifact = resolvedArtifact,
                   artifact.path.flatMap(ManagedAssetStore.assetID(from:)) == nil {
                    headerButton(
                        "lock.doc",
                        help: Text("Copy \(title) into this encrypted profile", bundle: .module),
                        label: Text("Copy into profile", bundle: .module))
                    {
                        model.copyArtifactIntoProfile(id: artifact.id)
                    }
                }
                headerButton(
                    "arrow.down.to.line",
                    help: Text("Save a copy of \(title)", bundle: .module),
                    label: Text("Save a copy", bundle: .module))
                {
                    exportArtifact()
                }
                headerButton(
                    "folder",
                    help: Text("Reveal \(title) in Finder", bundle: .module),
                    label: Text("Reveal in Finder", bundle: .module))
                {
                    if let url = resolvedArtifact?.url {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                }
            } else if resolvedPreview != nil {
                headerButton(
                    "arrow.down.to.line",
                    help: Text("Save the html to a file", bundle: .module),
                    label: Text("Save", bundle: .module))
                {
                    exportPreview()
                }
            }

            headerButton(
                "arrow.up.left.and.arrow.down.right",
                help: Text("Open a larger preview", bundle: .module),
                label: Text("Open a larger preview", bundle: .module))
            {
                isMaximized = true
            }

            headerButton("xmark", help: Text("Close preview", bundle: .module), label: Text("Close preview", bundle: .module)) {
                close()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    /// `label` names the action for VoiceOver and is followed by the
    /// artifact's title, so a reader moving between panels knows which
    /// document the control acts on.
    private func headerButton(
        _ systemImage: String,
        help: Text,
        label: Text,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .themedFont(.tiny, weight: .semibold)
                .frame(width: 22, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(Color.secondary)
        .help(help)
        .accessibilityLabel(label + Text(verbatim: " " + title))
    }

    // MARK: - Network banner

    private var networkBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "wifi.slash")
                .themedFont(.tiny, weight: .semibold)
                .foregroundStyle(Color.secondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text("Network is off for this preview", bundle: .module)
                    .themedFont(.tiny, weight: .medium)
                Text("Pages load only local files from the artifact's own folder.", bundle: .module)
                    .themedFont(.micro)
                    .foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 4)
            Button {
                model.allowNetworkForCurrentPreview()
            } label: {
                Text("Allow", bundle: .module)
            }
            .appAction(.secondary, size: .mini)
            .help(Text("Let this preview load remote resources", bundle: .module))
            Button {
                isBannerDismissed = true
            } label: {
                Image(systemName: "xmark")
                    .themedFont(.micro, weight: .semibold)
                    .frame(width: 16, height: 16)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.appSecondary)
            .help(Text("Dismiss", bundle: .module))
            .accessibilityLabel(Text("Dismiss network banner", bundle: .module))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.primary.opacity(0.03))
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if viewMode == .source {
            sourceView
        } else if let artifact = resolvedArtifact {
            artifactPreview(artifact)
        } else if let preview = resolvedPreview {
            ArtifactWebView(document: .inline(html: preview.html), networkAllowed: networkAllowed)
                .id(webViewIdentity)
        } else {
            unavailableView("This preview is no longer available.")
        }
    }

    @ViewBuilder
    private func artifactPreview(_ artifact: AppArtifact) -> some View {
        switch artifact.renderKind {
        case .html:
            if artifact.existsOnDisk, let url = artifact.url {
                ArtifactWebView(
                    document: .file(page: url, readAccessFolder: url.deletingLastPathComponent()),
                    networkAllowed: networkAllowed)
                    .id(webViewIdentity)
            } else {
                unavailableView("The file is no longer at its written path.")
            }
        case .markdown:
            // An oversized or non-UTF-8 file would otherwise render as an
            // empty page, which reads as an empty file.
            if sourceReadFailed {
                unavailableView("The source could not be read as text.")
            } else {
                ScrollView {
                    ChatMessageMarkdownView(sourceText ?? "")
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        case .text:
            if sourceReadFailed {
                unavailableView("The source could not be read as text.")
            } else {
                monospacedSourceView
            }
        case .image:
            if let url = artifact.url, let image = NSImage(contentsOf: url) {
                ImagePreviewView(image: image)
            } else {
                unavailableView("The image is no longer at its written path.")
            }
        case .pdf:
            if artifact.existsOnDisk, let url = artifact.url {
                PDFDocumentView(url: url)
            } else {
                unavailableView("The file is no longer at its written path.")
            }
        case .opaque:
            if artifact.existsOnDisk, let url = artifact.url {
                VStack(spacing: 0) {
                    QuickLookPreviewView(url: url)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .accessibilityLabel(Text("Quick Look preview of \(title)", bundle: .module))
                    Divider()
                    Button {
                        NSWorkspace.shared.open(url)
                    } label: {
                        Text("Open with default app", bundle: .module)
                    }
                    .appAction(.secondary, size: .small)
                    .padding(.vertical, 6)
                }
            } else {
                unavailableView("The file is no longer at its written path.")
            }
        }
    }

    private var sourceView: some View {
        Group {
            if sourceReadFailed {
                unavailableView("The source could not be read as text.")
            } else {
                monospacedSourceView
            }
        }
    }

    private var monospacedSourceView: some View {
        ScrollView {
            Text(sourceText ?? "")
                .themedCode(.callout)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func unavailableView(_ reason: LocalizedStringKey) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "eye.slash")
                .themedFont(.title2)
                .foregroundStyle(.quaternary)
            Text(title)
                .themedFont(.base, weight: .medium)
            Text(reason, bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
            Button {
                close()
            } label: {
                Text("Close", bundle: .module)
            }
            .appAction(.secondary, size: .small)
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Maximized sheet

    private var maximizedSheet: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: symbolName)
                    .foregroundStyle(.appAccent)
                Text(title)
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(1)
                Spacer()
                Button {
                    isMaximized = false
                } label: {
                    Image(systemName: "xmark")
                        .themedFont(.tiny, weight: .semibold)
                        .frame(width: 22, height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.appSecondary)
                .help(Text("Close", bundle: .module))
                .accessibilityLabel(Text("Close maximized preview", bundle: .module))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Divider()
            content
        }
        .frame(width: 1100, height: 800)
        .background(TurboSparkTheme.barBackgroundColor)
    }

    // MARK: - Actions

    private func resetPerContentState() {
        viewMode = .preview
        isBannerDismissed = false
        sourceReadFailed = false
        loadSourceText()
    }

    private func loadSourceText() {
        sourceText = nil
        if let preview = resolvedPreview {
            sourceText = preview.html
            return
        }
        guard let artifact = resolvedArtifact, artifact.existsOnDisk, let url = artifact.url else {
            return
        }
        // Source view only; a multi-gigabyte binary read would be a freeze,
        // and nothing worth reading as source is anywhere near this bound.
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        let byteSize = attributes?[.size] as? Int
        guard byteSize ?? 0 <= 4_000_000 else {
            sourceReadFailed = true
            return
        }
        sourceText = try? String(contentsOf: url, encoding: .utf8)
        if sourceText == nil {
            // Not valid utf8 (a real binary): the Source arm has nothing to
            // say, but the Preview arm may still render it.
            sourceReadFailed = artifact.renderKind == .text || artifact.renderKind == .markdown
        }
    }

    private func copySource() {
        guard let sourceText else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(sourceText, forType: .string)
        TSMotion.animate(.easeInOut(duration: 0.15)) { isCopied = true }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            TSMotion.animate(.easeOut(duration: 0.15)) { isCopied = false }
        }
    }

    private func exportArtifact() {
        guard let artifact = resolvedArtifact, let url = artifact.url else { return }
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = artifact.fileName
        if panel.runModal() == .OK, let destination = panel.url {
            // NSSavePanel already confirmed Replace with the user, but
            // copyItem refuses an existing destination. Copy to a sibling
            // temp name first so a failed copy cannot destroy the old file.
            do {
                let fm = FileManager.default
                if fm.fileExists(atPath: destination.path) {
                    let staging = destination.deletingLastPathComponent()
                        .appendingPathComponent(".turbospark-save-\(UUID().uuidString)")
                    try fm.copyItem(at: url, to: staging)
                    _ = try fm.replaceItemAt(destination, withItemAt: staging)
                } else {
                    try fm.copyItem(at: url, to: destination)
                }
            } catch {
                model.showToast(AppStrings.localized("Could not save a copy: \(error.localizedDescription)"), style: .error)
            }
        }
    }

    private func exportPreview() {
        guard let preview = resolvedPreview else { return }
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = "preview.html"
        if panel.runModal() == .OK, let destination = panel.url {
            do {
                try preview.html.write(to: destination, atomically: true, encoding: .utf8)
            } catch {
                model.showToast(AppStrings.localized("Could not export the preview: \(error.localizedDescription)"), style: .error)
            }
        }
    }

    private func close() {
        switch source {
        case .artifact: model.dismissArtifact()
        case .inlinePreview: model.dismissHTMLPreview()
        }
    }
}
