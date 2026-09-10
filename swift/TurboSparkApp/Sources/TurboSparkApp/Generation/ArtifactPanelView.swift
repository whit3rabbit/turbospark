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
        resolvedArtifact?.title ?? resolvedPreview?.title ?? "Preview"
    }

    private var subtitle: String {
        if let artifact = resolvedArtifact { return artifact.detailText }
        if resolvedPreview != nil { return "HTML • in memory" }
        return "No longer available"
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

            Picker("View", selection: $viewMode) {
                Text("Preview", bundle: .module).tag(ViewMode.preview)
                Text("Source", bundle: .module).tag(ViewMode.source)
            }
            .pickerStyle(.segmented)
            .controlSize(.mini)
            .fixedSize()
            .help("Switch between the rendered page and its source")

            headerButton(
                isCopied ? "checkmark" : "doc.on.doc",
                help: isCopied ? "Copied" : "Copy source",
                label: isCopied ? "Copied source" : "Copy source")
            {
                copySource()
            }
            .disabled(sourceText == nil)

            if resolvedArtifact?.existsOnDisk == true {
                headerButton(
                    "arrow.down.to.line",
                    help: "Save a copy of \(title)",
                    label: "Save a copy")
                {
                    exportArtifact()
                }
                headerButton(
                    "folder",
                    help: "Reveal \(title) in Finder",
                    label: "Reveal in Finder")
                {
                    if let url = resolvedArtifact?.url {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                }
            } else if resolvedPreview != nil {
                headerButton(
                    "arrow.down.to.line",
                    help: "Save the html to a file",
                    label: "Save")
                {
                    exportPreview()
                }
            }

            headerButton(
                "arrow.up.left.and.arrow.down.right",
                help: "Open a larger preview",
                label: "Open a larger preview")
            {
                isMaximized = true
            }

            headerButton("xmark", help: "Close preview", label: "Close preview") {
                close()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private func headerButton(
        _ systemImage: String,
        help: String,
        label: String,
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
        .accessibilityLabel("\(label) \(title)")
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
            .controlSize(.mini)
            .help("Let this preview load remote resources")
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
            .help("Dismiss")
            .accessibilityLabel("Dismiss network banner")
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
            ScrollView {
                ChatMessageMarkdownView(sourceText ?? "")
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        case .text:
            monospacedSourceView
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
                        .accessibilityLabel("Quick Look preview of \(title)")
                    Divider()
                    Button {
                        NSWorkspace.shared.open(url)
                    } label: {
                        Text("Open with default app", bundle: .module)
                    }
                    .controlSize(.small)
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
            .controlSize(.small)
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
                .help("Close")
                .accessibilityLabel("Close maximized preview")
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
        withAnimation(.easeInOut(duration: 0.15)) { isCopied = true }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            withAnimation(.easeOut(duration: 0.15)) { isCopied = false }
        }
    }

    private func exportArtifact() {
        guard let artifact = resolvedArtifact, let url = artifact.url else { return }
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = artifact.fileName
        if panel.runModal() == .OK, let destination = panel.url {
            try? FileManager.default.copyItem(at: url, to: destination)
        }
    }

    private func exportPreview() {
        guard let preview = resolvedPreview else { return }
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = "preview.html"
        if panel.runModal() == .OK, let destination = panel.url {
            try? preview.html.write(to: destination, atomically: true, encoding: .utf8)
        }
    }

    private func close() {
        switch source {
        case .artifact: model.dismissArtifact()
        case .inlinePreview: model.dismissHTMLPreview()
        }
    }
}
