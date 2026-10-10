import AppKit
import SwiftUI
import TurboSpark

/// Draft and rewrite side by side at exactly the same size, built from the
/// chat composer's parts: the model bar in the header, composer cards, chips
/// in the footers, and the run button between the cards. Advanced rules open
/// as a sidebar.
@MainActor
struct HumanizeView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: HumanizeController
    @Environment(\.appTheme) private var theme
    @AppStorage("TurboSpark.humanizeAdvancedVisible") private var showAdvanced = false
    @State private var showChanges = true
    @State private var difference: HumanizeDiff?
    /// The draft card shows what the rewrite cut until the user goes back to
    /// editing.
    @State private var reviewing = false
    @State private var copied = false
    @State private var confirmingDelete = false
    @FocusState private var draftFocused: Bool

    private var installedModel: InstalledModel? {
        model.installed.first { $0.alias == HumanizeController.modelAlias }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(alignment: .top, spacing: 16) {
                VStack(spacing: 12) {
                    workspace
                    feedback
                }
                if showAdvanced {
                    HumanizeAdvancedPanel(controller: controller) {
                        withAnimation(TSMotion.pane) { showAdvanced = false }
                    }
                    .frame(width: 300)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
            .padding(20)
        }
        .background(.appPage)
        .font(theme.ui(.base))
        .task(id: ContextKey(draft: controller.draft, path: installedModel?.path)) {
            do { try await Task.sleep(for: .milliseconds(180)) }
            catch { return }
            await controller.refreshContext()
        }
        .task(id: DiffKey(source: controller.sourceDraft, output: controller.rewrite,
                          segments: controller.segments, busy: controller.isBusy)) {
            difference = nil
            copied = false
            guard !controller.isBusy, !controller.segments.isEmpty else { return }
            let source = controller.sourceDraft, output = controller.rewrite
            let segments = controller.segments
            let comparison = await Task.detached(priority: .userInitiated) {
                HumanizeDiff.compare(source: source, rewrite: output, segments: segments)
            }.value
            guard !Task.isCancelled else { return }
            withAnimation(TSMotion.hover) {
                difference = comparison
                reviewing = true
            }
        }
        .confirmationDialog(Text("Delete Model", bundle: .module), isPresented: $confirmingDelete) {
            Button(role: .destructive) {
                if let installedModel { model.deleteModel(installedModel) }
            } label: { Text("Delete Model", bundle: .module) }
        } message: {
            Text("This will remove the downloaded model files from disk.", bundle: .module)
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "text.badge.checkmark")
                .symbolRenderingMode(.monochrome)
                .themedFont(.title2).foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            Text("Humanize", bundle: .module)
                .themedFont(.title2, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 12)
            HumanizeModelBar(model: model, controller: controller) { confirmingDelete = true }
            HumanizeChip(title: "Advanced", systemImage: "sidebar.right", isOn: showAdvanced) {
                withAnimation(TSMotion.pane) { showAdvanced.toggle() }
            }
        }
        .padding(.horizontal, 24).padding(.vertical, 12)
    }

    // MARK: - Workspace

    private var workspace: some View {
        GeometryReader { geometry in
            let wide = geometry.size.width >= 600
            let rail: CGFloat = wide ? 76 : 68
            let panel = wide
                ? CGSize(width: max(0, (geometry.size.width - rail) / 2), height: geometry.size.height)
                : CGSize(width: geometry.size.width, height: max(0, (geometry.size.height - rail) / 2))
            let layout = wide ? AnyLayout(HStackLayout(spacing: 0)) : AnyLayout(VStackLayout(spacing: 0))
            layout {
                draftPanel.frame(width: panel.width, height: panel.height)
                rewriteRail(wide: wide)
                    .frame(width: wide ? rail : geometry.size.width, height: wide ? geometry.size.height : rail)
                resultPanel.frame(width: panel.width, height: panel.height)
            }
        }
        // Both cards receive exactly the same size, header and footer included.
        .frame(minHeight: 440, maxHeight: .infinity)
    }

    private var showsMarkedDraft: Bool {
        reviewing && showChanges && difference != nil && !controller.isBusy
            && controller.draft == controller.sourceDraft
    }

    private var draftPanel: some View {
        HumanizeCard(focused: draftFocused && !showsMarkedDraft) {
            panelTitle(number: "01", title: "Draft")
            Spacer(minLength: 8)
            if controller.hasDraft {
                Text("Words: \(Self.words(in: controller.draft))", bundle: .module).modifier(StatStyle())
            }
        } content: {
            if showsMarkedDraft {
                ScrollView {
                    Text(markedDraft)
                        .lineSpacing(6)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .padding(.horizontal, 21).padding(.vertical, 18)
                        .accessibilityLabel(Text("Draft", bundle: .module))
                }
                .transition(.opacity)
            } else {
                ZStack(alignment: .topLeading) {
                    if controller.draft.isEmpty {
                        Text("Paste a draft to rewrite.", bundle: .module)
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 21).padding(.vertical, 18)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: $controller.draft)
                        .font(theme.ui(.base)).lineSpacing(6)
                        .scrollContentBackground(.hidden)
                        .focused($draftFocused)
                        .padding(.horizontal, 16).padding(.vertical, 18)
                        .disabled(controller.isBusy)
                        .accessibilityLabel(Text("Draft", bundle: .module))
                        .accessibilityIdentifier("humanize.draft")
                }
                .transition(.opacity)
            }
        } footer: {
            HumanizeChip(title: "Paste", systemImage: "doc.on.clipboard") {
                if let text = NSPasteboard.general.string(forType: .string) {
                    controller.draft = text
                    reviewing = false
                }
            }
            .disabled(controller.isBusy)
            HumanizeChip(title: "Clear", systemImage: "eraser") {
                controller.draft = ""
                reviewing = false
            }
            .disabled(controller.draft.isEmpty || controller.isBusy)
            if showsMarkedDraft {
                HumanizeChip(title: "Edit", systemImage: "pencil") {
                    withAnimation(TSMotion.hover) { reviewing = false }
                    draftFocused = true
                }
            }
            Spacer(minLength: 8)
            if let plan = controller.currentPlan {
                Text("Tokens: \(plan.tokens)", bundle: .module)
                    .modifier(StatStyle())
                    .foregroundStyle(plan.fits ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(Color.orange))
                    .help(Text("The model reads up to 8,192 tokens at a time. Longer drafts are rewritten in parts at paragraph breaks.", bundle: .module))
            }
        }
    }

    private var resultPanel: some View {
        HumanizeCard {
            panelTitle(number: "02", title: "Rewrite")
            Spacer(minLength: 8)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { rewriteStats; changesChip(iconOnly: false) }
                changesChip(iconOnly: false)
                changesChip(iconOnly: true)
            }
        } content: {
            if controller.rewrite.isEmpty {
                if controller.isBusy {
                    VStack(alignment: .leading, spacing: 18) {
                        HumanizeSkeleton()
                        progressStatus
                    }
                    .padding(.horizontal, 21).padding(.vertical, 22)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .transition(.opacity)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "text.alignleft").themedFont(.title2).accessibilityHidden(true)
                        Text("Your rewrite will appear here.", bundle: .module)
                    }
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                ScrollView {
                    Text(highlightedRewrite)
                        .lineSpacing(6)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .padding(.horizontal, 21).padding(.vertical, 18)
                        .contentTransition(.opacity)
                        .accessibilityLabel(Text("Rewrite", bundle: .module))
                        .accessibilityIdentifier("humanize.rewrite")
                }
                .transition(.opacity)
            }
        } footer: {
            if controller.isBusy {
                if !controller.rewrite.isEmpty { progressStatus }
            } else {
                if let clean = checksClean { checksChip(clean: clean) }
                if let stats = controller.stats {
                    Text(verbatim: Self.statsLine(stats))
                        .font(theme.ui(.tiny)).monospacedDigit().foregroundStyle(.appSecondary)
                        .lineLimit(1).layoutPriority(-1)
                }
            }
            Spacer(minLength: 8)
            if !controller.rewrite.isEmpty, !controller.isBusy {
                HumanizeChip(title: "Regenerate", systemImage: "arrow.clockwise") { controller.run() }
                    .disabled(!controller.canRun)
            }
            HumanizeChip(title: copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc",
                         isOn: copied) {
                NSPasteboard.general.clearContents()
                copied = NSPasteboard.general.setString(controller.rewrite, forType: .string)
            }
            .disabled(controller.rewrite.isEmpty || controller.isBusy)
        }
        .animation(TSMotion.hover, value: controller.rewrite.isEmpty)
    }

    @ViewBuilder private var rewriteStats: some View {
        if !controller.rewrite.isEmpty {
            HStack(spacing: 6) {
                Text("Words: \(Self.words(in: controller.rewrite))", bundle: .module)
                if let difference, !controller.isBusy {
                    separatorDot
                    Text("\(Int((difference.changedShare * 100).rounded()))% changed", bundle: .module)
                        .contentTransition(.numericText())
                }
            }
            .modifier(StatStyle())
            .fixedSize()
        }
    }

    private func changesChip(iconOnly: Bool) -> some View {
        HumanizeChip(title: "Show changes", systemImage: "highlighter", isOn: showChanges, iconOnly: iconOnly) {
            withAnimation(TSMotion.hover) { showChanges.toggle() }
        }
        .disabled(controller.rewrite.isEmpty || controller.isBusy)
    }

    /// The run button between the cards; a stop target while a rewrite runs.
    private func rewriteRail(wide: Bool) -> some View {
        let busy = controller.isBusy
        let ready = busy || controller.canRun
        let layout = wide ? AnyLayout(VStackLayout(spacing: 8)) : AnyLayout(HStackLayout(spacing: 12))
        return layout {
            HumanizeRunButton(
                busy: busy, ready: ready, prefill: busy ? controller.prefillProgress : nil,
                arrow: wide ? "arrow.right" : "arrow.down",
                invite: controller.canRun && controller.rewrite.isEmpty
            ) {
                if busy { controller.cancel() } else { controller.run() }
            }
            .disabled(!ready)
            .keyboardShortcut(busy ? KeyEquivalent(".") : .return, modifiers: .command)
            .help(busy ? Text("Stop", bundle: .module) : Text("Rewrite", bundle: .module))
            .accessibilityLabel(busy ? Text("Stop", bundle: .module) : Text("Rewrite", bundle: .module))
            .accessibilityIdentifier("humanize.run")
            VStack(spacing: 4) {
                Group {
                    if busy { Text("Stop", bundle: .module) } else { Text("Rewrite", bundle: .module) }
                }
                .font(theme.ui(.micro, weight: .semibold))
                .textCase(.uppercase)
                .tracking(1)
                if let part = controller.partProgress {
                    ModelLoadProgressBar(
                        fraction: Double(part.current - 1) / Double(part.total), tint: theme.accent)
                        .frame(width: 40)
                    Text(verbatim: "\(part.current)/\(part.total)").font(theme.ui(.micro)).monospacedDigit()
                } else {
                    Text(verbatim: busy ? "\u{2318}." : "\u{2318}\u{21A9}").font(theme.ui(.micro))
                }
            }
            .foregroundStyle(.appSecondary)
            .lineLimit(1)
            .fixedSize()
            .accessibilityHidden(true)
        }
        .animation(TSMotion.select, value: busy)
    }

    // MARK: - Pieces

    private func panelTitle(number: String, title: LocalizedStringKey) -> some View {
        HStack(spacing: 8) {
            Text(verbatim: number)
                .font(theme.ui(.tiny, weight: .semibold)).monospacedDigit().foregroundStyle(.appSecondary)
            Text(title, bundle: .module).font(theme.ui(.callout, weight: .semibold))
                .accessibilityAddTraits(.isHeader)
        }
        .fixedSize()
    }

    private var separatorDot: some View {
        Text(verbatim: "\u{00B7}").accessibilityHidden(true)
    }

    @ViewBuilder private var progressStatus: some View {
        HStack(spacing: 8) {
            ModelStatusDot(state: .opening, accent: theme.accent)
            switch controller.phase {
            case .idle: EmptyView()
            case .loadingModel: Text("Loading the Humanizer model...", bundle: .module)
            case .running:
                if controller.prefillProgress != nil {
                    Text("Reading the text...", bundle: .module)
                } else {
                    Text("Rewriting...", bundle: .module)
                }
            }
            if let part = controller.partProgress {
                Text("Part \(part.current) of \(part.total)", bundle: .module).monospacedDigit()
            }
        }
        .font(theme.ui(.small))
        .foregroundStyle(.appSecondary)
        .lineLimit(1)
    }

    /// Nil until something was checked: the model's audit, the user's rules,
    /// or both.
    private var checksClean: Bool? {
        guard controller.checks != nil || controller.ruleFindings != nil else { return nil }
        return (controller.checks?.isClean ?? true) && (controller.ruleFindings?.isEmpty ?? true)
    }

    private var warningLines: [String] {
        (controller.checks.map(HumanizeAuditLines.lines(for:)) ?? [])
            + (controller.ruleFindings.map(HumanizeAuditLines.lines(for:)) ?? [])
    }

    private func checksChip(clean: Bool) -> some View {
        let tint: Color = clean ? .green : .orange
        return Label {
            if clean { Text("Checks passed", bundle: .module) } else { Text("Check before you use it", bundle: .module) }
        } icon: {
            Image(systemName: clean ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
        }
        .font(theme.ui(.small, weight: .medium))
        .foregroundStyle(tint)
        .lineLimit(1)
        .padding(.horizontal, 8).padding(.vertical, 5)
        .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .fixedSize()
        .help(clean
              ? String(localized: "The automatic checks found nothing. Read it before you use it.", bundle: .module)
              : warningLines.joined(separator: "\n"))
        .transition(.scale(scale: 0.9).combined(with: .opacity))
    }

    @ViewBuilder private var feedback: some View {
        let plan = controller.currentPlan
        VStack(alignment: .leading, spacing: 8) {
            if let message = controller.errorMessage ?? controller.contextError {
                Label(message, systemImage: "xmark.octagon").foregroundStyle(.red).textSelection(.enabled)
            } else if let plan, !plan.fits {
                Label(HumanizeError.passageTooLong.localizedDescription, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
            if let plan, plan.parts.count > 1 {
                Label { Text("Long draft: rewritten in parts at paragraph breaks.", bundle: .module) }
                    icon: { Image(systemName: "rectangle.split.3x1") }
                    .foregroundStyle(.appSecondary)
            }
            if controller.wasTruncated {
                Label { Text("The rewrite stopped at the length limit and may be cut off.", bundle: .module) }
                    icon: { Image(systemName: "scissors") }
                    .foregroundStyle(.orange)
            }
            if controller.stopReason == .cancelled {
                Text("Stopped. The rewrite is incomplete.", bundle: .module).foregroundStyle(.appSecondary)
            }
            if !controller.rewrite.isEmpty, !controller.isBusy, controller.draft != controller.sourceDraft {
                Text("The draft has changed. Highlights compare with the version used for this rewrite.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
            ForEach(warningLines, id: \.self) { line in
                Label(line, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
        }
        .font(theme.ui(.small))
        .frame(maxWidth: .infinity, alignment: .leading)
        .animation(TSMotion.hover, value: warningLines)
    }

    // MARK: - Highlighting

    private var markedDraft: AttributedString {
        let source = controller.sourceDraft
        var text = AttributedString(source)
        for range in difference?.deleted ?? [] {
            guard let bounds = Range(range, in: source), let marked = Range(bounds, in: text) else { continue }
            text[marked].foregroundColor = Color.red.opacity(0.85)
            text[marked].strikethroughStyle = Text.LineStyle(pattern: .solid, color: .red)
        }
        return text
    }

    private var highlightedRewrite: AttributedString {
        let output = controller.rewrite
        var text = AttributedString(output)
        if controller.isBusy {
            // A caret at the end while the model writes.
            var caret = AttributedString("\u{258D}")
            caret.foregroundColor = theme.accent
            text.append(caret)
            return text
        }
        if showChanges, let difference {
            for range in difference.inserted {
                guard let bounds = Range(range, in: output), let marked = Range(bounds, in: text) else { continue }
                text[marked].backgroundColor = theme.accent.opacity(0.2)
                // Underlining keeps additions distinguishable without color.
                text[marked].underlineStyle = Text.LineStyle(pattern: .solid, color: theme.accent)
            }
        }
        // Words to avoid show whether or not changes are highlighted: they are
        // warnings, not a view of the diff.
        for range in controller.ruleFindings?.avoidRanges ?? [] {
            guard let bounds = Range(range, in: output), let marked = Range(bounds, in: text) else { continue }
            text[marked].backgroundColor = Color.orange.opacity(0.22)
            text[marked].underlineStyle = Text.LineStyle(pattern: .dash, color: .orange)
        }
        return text
    }

    static func words(in text: String) -> Int {
        var count = 0
        text.enumerateSubstrings(in: text.startIndex..., options: [.byWords, .substringNotRequired]) { _, _, _, _ in
            count += 1
        }
        return count
    }

    static func statsLine(_ stats: HumanizeStats) -> String {
        var pieces = ["\(stats.tokens) tok"]
        if let rate = stats.tokensPerSecond { pieces.append(String(format: "%.1f tok/s", rate)) }
        pieces.append(String(format: "%.1fs", stats.seconds))
        return pieces.joined(separator: " \u{00B7} ")
    }

    private struct ContextKey: Equatable { let draft: String; let path: String? }
    private struct DiffKey: Equatable {
        let source: String
        let output: String
        let segments: [HumanizeSegment]
        let busy: Bool
    }
}

private struct StatStyle: ViewModifier {
    @Environment(\.appTheme) private var theme

    func body(content: Content) -> some View {
        content.font(theme.ui(.small)).monospacedDigit().foregroundStyle(.appSecondary).lineLimit(1)
    }
}

/// Localized sentences for the checks, built from their structured fields
/// (the engine's own `warnings` are English only). Every line is a pointer
/// for the proofreader, never a claim that the rewrite is wrong.
enum HumanizeAuditLines {
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

    static func lines(for findings: HumanizeRuleFindings) -> [String] {
        var out: [String] = []
        if !findings.missingProtected.isEmpty {
            let terms = findings.missingProtected.joined(separator: ", ")
            out.append(String(localized: "Protected terms missing from the rewrite: \(terms)", bundle: .module))
        }
        if !findings.avoidFound.isEmpty {
            let terms = findings.avoidFound.joined(separator: ", ")
            out.append(String(localized: "Words to avoid found in the rewrite: \(terms)", bundle: .module))
        }
        return out
    }
}
