import AppKit
import SwiftUI
import TurboSpark

@MainActor
struct HumanizeView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: HumanizeController
    @State private var showChanges = true
    @State private var difference: HumanizeDiff?
    @State private var copied = false
    @State private var confirmingDelete = false

    private var installedModel: InstalledModel? {
        model.installed.first { $0.alias == HumanizeController.modelAlias }
    }
    private var downloadPending: Bool {
        model.isModelDownloadPending(.catalog(alias: HumanizeController.modelAlias))
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            GeometryReader { geometry in
                ScrollView {
                    VStack(spacing: 16) {
                        modelToolbar
                        workspace(height: geometry.size.width >= 768 ? max(440, geometry.size.height - 220) : 820)
                        contextControls
                        feedback
                    }
                    .padding(24)
                }
            }
        }
        .background(.appPage)
        .themedFont(.base)
        .task(id: ContextKey(draft: controller.draft, path: installedModel?.path)) {
            do { try await Task.sleep(for: .milliseconds(180)) }
            catch { return }
            await controller.refreshContext()
        }
        .task(id: DiffKey(source: controller.sourceDraft, output: controller.rewrite, busy: controller.isBusy)) {
            difference = nil
            copied = false
            guard !controller.isBusy, !controller.rewrite.isEmpty else { return }
            let source = controller.sourceDraft, output = controller.rewrite
            let comparison = await Task.detached(priority: .userInitiated) {
                HumanizeDiff.compare(source: source, rewrite: output)
            }.value
            guard !Task.isCancelled else { return }
            difference = comparison
        }
        .confirmationDialog(Text("Delete Model", bundle: .module), isPresented: $confirmingDelete) {
            Button(role: .destructive) {
                if let installedModel { model.deleteModel(installedModel) }
            } label: { Text("Delete Model", bundle: .module) }
        } message: {
            Text("This will remove the downloaded model files from disk.", bundle: .module)
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "text.badge.checkmark")
                .themedFont(.title2).foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            Text("Humanize", bundle: .module)
                .themedFont(.title2, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Spacer()
            Text("Local rewriting", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 24).padding(.vertical, 18)
    }

    private var modelToolbar: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 16) { modelIdentity; Spacer(); modelActions }
            VStack(alignment: .leading, spacing: 12) { modelIdentity; modelActions }
        }
        .padding(16)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
    }

    private var modelIdentity: some View {
        HStack(spacing: 12) {
            Image(systemName: "cpu")
                .themedFont(.title2).foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                Text(verbatim: "Humanizer 12B").themedFont(.callout, weight: .semibold)
                HStack(spacing: 8) {
                    Text(verbatim: "8-bit").foregroundStyle(.appSecondary)
                    if controller.isBusy {
                        ProgressView().controlSize(.mini)
                        statusLabel
                    } else if controller.engineLoaded {
                        Label { Text("Loaded", bundle: .module) } icon: { Image(systemName: "checkmark.circle.fill") }
                            .foregroundStyle(.appAccent)
                    } else if controller.isInstalled {
                        Text("Not loaded", bundle: .module).foregroundStyle(.appSecondary)
                    } else if downloadPending {
                        Text("Downloads", bundle: .module).foregroundStyle(.appSecondary)
                    } else {
                        Text(verbatim: "12.7 GB").foregroundStyle(.appSecondary)
                    }
                }
                .themedFont(.small)
            }
        }
    }

    private var modelActions: some View {
        HStack(spacing: 10) {
            if controller.isInstalled {
                if controller.engineLoaded {
                    Button { controller.unload() } label: {
                        Label { Text("Unload", bundle: .module) } icon: { Image(systemName: "eject") }
                    }
                    .disabled(controller.isBusy)
                } else {
                    Button { controller.load() } label: {
                        Label { Text("Load Model", bundle: .module) } icon: { Image(systemName: "play") }
                    }
                    .disabled(controller.isBusy)
                }
                Button(role: .destructive) { confirmingDelete = true } label: {
                    Label { Text("Delete Model", bundle: .module) } icon: { Image(systemName: "trash") }
                }
                .disabled(controller.isBusy || model.isRunning || model.opening)
            } else if downloadPending {
                Button { model.isDownloadManagerExpanded = true } label: {
                    Label { Text("Downloads", bundle: .module) } icon: { Image(systemName: "arrow.down.circle") }
                }
            } else {
                Button { controller.installHandler() } label: {
                    Label { Text("Download", bundle: .module) } icon: { Image(systemName: "arrow.down.circle") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canQueueModelDownload(.catalog(alias: HumanizeController.modelAlias)))
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.regular)
    }

    private func workspace(height: CGFloat) -> some View {
        GeometryReader { geometry in
            let wide = geometry.size.width >= 720
            let layout = wide ? AnyLayout(HStackLayout(spacing: 16)) : AnyLayout(VStackLayout(spacing: 16))
            let panelWidth = wide ? (geometry.size.width - 16) / 2 : geometry.size.width
            let panelHeight = wide ? geometry.size.height : (geometry.size.height - 16) / 2
            layout {
                originalPanel.frame(width: panelWidth, height: panelHeight)
                resultPanel.frame(width: panelWidth, height: panelHeight)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        // Both cards receive exactly the same size, including header and footer.
        .frame(height: height)
    }

    private var originalPanel: some View {
        VStack(spacing: 0) {
            panelHeader(title: "Original", number: "01", text: controller.draft)
            Divider()
            ZStack(alignment: .topLeading) {
                if controller.draft.isEmpty {
                    Text("Paste a draft to rewrite.", bundle: .module)
                        .foregroundStyle(.appSecondary)
                        .padding(.horizontal, 21).padding(.vertical, 24)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $controller.draft)
                    .themedFont(.base).lineSpacing(6)
                    .scrollContentBackground(.hidden)
                    .padding(16)
                    .disabled(controller.isBusy)
                    .accessibilityLabel(Text("Original", bundle: .module))
                    .accessibilityIdentifier("humanize.draft")
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            HStack(spacing: 12) {
                Button {
                    if let text = NSPasteboard.general.string(forType: .string) { controller.draft = text }
                } label: {
                    Label { Text("Paste", bundle: .module) } icon: { Image(systemName: "document.on.clipboard") }
                }
                .disabled(controller.isBusy)
                Button { controller.draft = "" } label: { Text("Clear", bundle: .module) }
                    .disabled(controller.draft.isEmpty || controller.isBusy)
                Spacer(minLength: 0)
            }
            .buttonStyle(.borderless)
            .padding(.horizontal, 16).frame(height: 48)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.appSecondary.opacity(0.18)))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var resultPanel: some View {
        VStack(spacing: 0) {
            panelHeader(title: "Rewrite", number: "02", text: controller.rewrite)
            Divider()
            ScrollView {
                if controller.rewrite.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "text.alignleft").themedFont(.title2).accessibilityHidden(true)
                        Text("Your rewrite will appear here.", bundle: .module)
                    }
                    .foregroundStyle(.appSecondary)
                    .frame(maxWidth: .infinity).padding(.top, 52)
                } else {
                    Text(highlightedRewrite)
                        .themedFont(.base).lineSpacing(6)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .padding(20)
                        .accessibilityLabel(Text("Rewrite", bundle: .module))
                        .accessibilityIdentifier("humanize.rewrite")
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            HStack(spacing: 12) {
                Toggle(isOn: $showChanges) { Text("Show changes", bundle: .module) }
                    .toggleStyle(.checkbox)
                    .disabled(controller.rewrite.isEmpty || controller.isBusy)
                Spacer(minLength: 0)
                Button {
                    NSPasteboard.general.clearContents()
                    copied = NSPasteboard.general.setString(controller.rewrite, forType: .string)
                } label: {
                    Label {
                        if copied { Text("Copied", bundle: .module) }
                        else { Text("Copy", bundle: .module) }
                    } icon: { Image(systemName: copied ? "checkmark" : "doc.on.doc") }
                }
                .buttonStyle(.bordered)
                .disabled(controller.rewrite.isEmpty)
            }
            .padding(.horizontal, 16).frame(height: 48)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.appSecondary.opacity(0.18)))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func panelHeader(title: LocalizedStringKey, number: String, text: String) -> some View {
        HStack(spacing: 8) {
            Text(number).themedFont(.small).monospacedDigit().foregroundStyle(.appSecondary)
            Text(title, bundle: .module).themedFont(.callout, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 0)
            Text("\(text.count) characters", bundle: .module)
                .themedFont(.small).monospacedDigit().foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 16).frame(height: 48)
    }

    private var highlightedRewrite: AttributedString {
        var text = AttributedString(controller.rewrite)
        guard showChanges, !controller.isBusy, let difference else { return text }
        for range in difference.inserted {
            guard let stringRange = Range(range, in: controller.rewrite),
                  let attributedRange = Range(stringRange, in: text) else { continue }
            text[attributedRange].backgroundColor = Color.accentColor.opacity(0.22)
            // Underlining keeps additions distinguishable without color.
            text[attributedRange].underlineStyle = .single
        }
        return text
    }

    private var contextControls: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 20) { contextMeter; Spacer(); runButton }
            VStack(alignment: .leading, spacing: 12) {
                contextMeter
                HStack { Spacer(); runButton }
            }
        }
    }

    private var contextMeter: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let plan = controller.currentPlan {
                Text("Estimated draft: \(plan.tokens) tokens", bundle: .module)
                    .themedFont(.small).monospacedDigit()
                    .foregroundStyle(plan.fits ? Color.secondary : Color.orange)
            } else {
                Text("Context window", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                Text(verbatim: "8,192 tokens").themedFont(.small).monospacedDigit()
            }
        }
        .help(Text("Includes draft tokens, 256 tokens reserved for the prompt, and the output limit. The engine checks the exact prompt before rewriting.", bundle: .module))
    }

    private var runButton: some View {
        HStack {
            if controller.isBusy {
                Button { controller.cancel() } label: { Text("Stop", bundle: .module) }
                    .keyboardShortcut(".", modifiers: .command)
            }
            Button { controller.run() } label: {
                Label {
                    if controller.rewrite.isEmpty { Text("Humanize", bundle: .module) }
                    else { Text("Rewrite again", bundle: .module) }
                } icon: { Image(systemName: "arrow.right") }
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(!controller.canRun)
            .keyboardShortcut(.return, modifiers: .command)
        }
    }

    @ViewBuilder private var statusLabel: some View {
        switch controller.phase {
        case .idle: EmptyView()
        case .loadingModel: Text("Loading the Humanizer model...", bundle: .module)
        case .running:
            if let progress = controller.prefillProgress {
                Text("Reading the text...", bundle: .module)
                ProgressView(value: progress).frame(width: 60)
            } else { Text("Rewriting...", bundle: .module) }
        }
    }

    @ViewBuilder private var feedback: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let plan = controller.currentPlan, !plan.fits {
                Text("Shorten the draft to fit the context window.", bundle: .module)
                    .foregroundStyle(.orange)
            }
            if let message = controller.errorMessage ?? controller.contextError {
                Text(message).foregroundStyle(.red).textSelection(.enabled)
            }
            if controller.wasTruncated {
                Text("The rewrite stopped at the length limit and may be cut off.", bundle: .module)
                    .foregroundStyle(.orange)
            }
            if controller.stopReason == .cancelled {
                Text("Stopped. The rewrite is incomplete.", bundle: .module).foregroundStyle(.appSecondary)
            }
            if !controller.rewrite.isEmpty, controller.draft != controller.sourceDraft {
                Text("The draft has changed. Highlights compare with the version used for this rewrite.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
            if let checks = controller.checks { HumanizeAuditView(checks: checks) }
        }
        .themedFont(.small)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private struct ContextKey: Equatable { let draft: String; let path: String? }
    private struct DiffKey: Equatable { let source: String; let output: String; let busy: Bool }
}

/// The automatic checks on a rewrite, as plain sentences. Every line is a
/// pointer for the proofreader, never a claim that the rewrite is wrong.
struct HumanizeAuditView: View {
    let checks: HumanizeChecks

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Check before you use it", bundle: .module)
                .themedFont(.callout, weight: .semibold)
            if checks.isClean {
                Text("The automatic checks found nothing. Read it before you use it.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            } else {
                ForEach(HumanizeAuditLines.lines(for: checks), id: \.self) { line in
                    Label(line, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }
}

/// Localized sentences for an audit, built from its structured fields (the
/// engine's own `warnings` are English only).
enum HumanizeAuditLines {
    static func lines(for audit: HumanizeAudit) -> [String] {
        var checks = HumanizeChecks()
        checks.add(audit)
        return lines(for: checks)
    }

    static func lines(for checks: HumanizeChecks) -> [String] {
        var out: [String] = []
        if !checks.droppedNumbers.isEmpty {
            let numbers = checks.droppedNumbers.joined(separator: ", ")
            out.append(String(localized: "Numbers missing from the rewrite: \(numbers)", bundle: .module))
        }
        if !checks.inventedNumbers.isEmpty {
            let numbers = checks.inventedNumbers.joined(separator: ", ")
            out.append(String(localized: "Numbers not in the original: \(numbers)", bundle: .module))
        }
        if checks.languageDrift {
            out.append(String(
                localized: "The rewrite looks like a different language than the original.",
                bundle: .module))
        }
        if checks.reuseFlagged {
            out.append(String(
                localized: "The rewrite copies long phrases from the original.", bundle: .module))
        }
        return out
    }
}
