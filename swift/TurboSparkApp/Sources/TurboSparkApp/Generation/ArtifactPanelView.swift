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
    @Environment(\.appTheme) private var theme
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
        .background(TurboSparkTheme.barBackgroundColor(isDark: theme.isDark))
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
        AppAdaptiveHeader {
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
                        .help(title)
                    Text(subtitle)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                }
            }
        } actions: {
            Picker(selection: $viewMode) {
                Text("Preview", bundle: .module).tag(ViewMode.preview)
                Text("Source", bundle: .module).tag(ViewMode.source)
            } label: { Text("View", bundle: .module) }
            .pickerStyle(.segmented)
            .controlSize(.mini)
            .fixedSize()
            .help(Text("Switch between the rendered page and its source", bundle: .module))

            AppIconButton(
                isCopied ? "checkmark" : "doc.on.doc",
                label: isCopied ? Text("Copied source", bundle: .module) : Text("Copy source", bundle: .module)
            ) {
                copySource()
            }
            .disabled(sourceText == nil)
            .help(sourceText == nil ? Text("Source text is not available", bundle: .module) : (isCopied ? Text("Copied", bundle: .module) : Text("Copy source", bundle: .module)))

            if resolvedArtifact?.existsOnDisk == true {
                if let artifact = resolvedArtifact,
                   artifact.path.flatMap(ManagedAssetStore.assetID(from:)) == nil {
                    AppIconButton("lock.doc", label: Text("Copy into profile", bundle: .module)) {
                        model.copyArtifactIntoProfile(id: artifact.id)
                    }
                    .help(Text("Copy \(title) into this encrypted profile", bundle: .module))
                }
                AppIconButton("arrow.down.to.line", label: Text("Save a copy", bundle: .module)) {
                    exportArtifact()
                }
                .help(Text("Save a copy of \(title)", bundle: .module))
                AppIconButton("folder", label: Text("Reveal in Finder", bundle: .module)) {
                    if let url = resolvedArtifact?.url {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                }
                .help(Text("Reveal \(title) in Finder", bundle: .module))
            } else if resolvedPreview != nil {
                AppIconButton("arrow.down.to.line", label: Text("Save", bundle: .module)) {
                    exportPreview()
                }
                .help(Text("Save the html to a file", bundle: .module))
            }

            AppIconButton("arrow.up.left.and.arrow.down.right", label: Text("Open a larger preview", bundle: .module)) {
                isMaximized = true
            }
            .help(Text("Open a larger preview", bundle: .module))

            AppIconButton("xmark", label: Text("Close preview", bundle: .module)) {
                close()
            }
            .help(Text("Close preview", bundle: .module))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - Network banner

    private var networkBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "wifi.slash")
                .themedFont(.tiny, weight: .semibold)
                .foregroundStyle(.appSecondary)
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
            AppIconButton("xmark", label: Text("Dismiss network banner", bundle: .module)) {
                isBannerDismissed = true
            }
            .help(Text("Dismiss", bundle: .module))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.appElevated)
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
                .foregroundStyle(.appSecondary)
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
                AppIconButton("xmark", label: Text("Close maximized preview", bundle: .module)) {
                    isMaximized = false
                }
                .help(Text("Close", bundle: .module))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Divider()
            content
        }
        .frame(minWidth: 700, idealWidth: 1100, maxWidth: 1400, minHeight: 500, idealHeight: 800, maxHeight: 1000)
        .background(TurboSparkTheme.barBackgroundColor(isDark: theme.isDark))
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
        TSMotion.animate(TSMotion.select) { isCopied = true }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            TSMotion.animate(TSMotion.select) { isCopied = false }
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
