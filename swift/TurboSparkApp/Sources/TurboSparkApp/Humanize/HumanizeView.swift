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
    @State private var confirmingClearAll = false
    @State private var confirmingClearRewrite = false
    @FocusState private var draftFocused: Bool

    /// From the loadable list: a Humanizer on a drive that is away cannot be
    /// opened, counted or deleted from here.
    private var installedModel: InstalledModel? {
        model.loadableModels.first { $0.alias == HumanizeController.modelAlias }
    }

    var body: some View {
        VStack(spacing: 0) {
            HumanizeHeader(model: model, controller: controller, showAdvanced: $showAdvanced) {
                confirmingDelete = true
            }
            Divider()
            HStack(alignment: .top, spacing: 16) {
                VStack(spacing: 12) {
                    workspace
                    HumanizeFeedback(controller: controller)
                }
                if showAdvanced {
                    let width = HumanizeMetrics.advancedWidth(theme)
                    HumanizeAdvancedPanel(controller: controller) {
                        TSMotion.animate(TSMotion.pane) { showAdvanced = false }
                    }
                    // Flexible rather than a fixed 300: the HStack sizes the
                    // panel first, so it keeps its width while the window has
                    // room, gives way to the cards in a narrow one, and widens
                    // with the text size so its labels wrap instead of clip.
                    .frame(minWidth: HumanizeMetrics.advancedMinWidth, idealWidth: width, maxWidth: width)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
            .padding(20)
        }
        .background { backdrop }
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
            TSMotion.animate(TSMotion.hover) {
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

    // MARK: - Workspace

    private var workspace: some View {
        GeometryReader { geometry in
            let scale = HumanizeMetrics.textScale(theme)
            let breakpoint = (640 * scale).rounded()
            let wide = geometry.size.width >= breakpoint
            let rail: CGFloat = wide ? (96 * scale).rounded() : (72 * scale).rounded()
            let panel = wide
                ? CGSize(width: max(0, (geometry.size.width - rail) / 2), height: geometry.size.height)
                : CGSize(width: geometry.size.width, height: max(0, (geometry.size.height - rail) / 2))
            // The footers carry three or four chips: below this width the
            // Clear chips keep only their glyph (the name stays the tooltip
            // and the VoiceOver label) so a footer reflows instead of clipping.
            let tight = panel.width < (380 * scale).rounded()
            let layout = wide ? AnyLayout(HStackLayout(spacing: 0)) : AnyLayout(VStackLayout(spacing: 0))
            layout {
                draftPanel(tight: tight).frame(width: panel.width, height: panel.height)
                    .tsEntrance()
                HumanizeRewriteRail(controller: controller, wide: wide, canClearAll: canClearAll, onClearAll: clearAll)
                    .frame(width: wide ? rail : geometry.size.width, height: wide ? geometry.size.height : rail)
                    .tsEntrance(delay: 0.02)
                resultPanel(tight: tight).frame(width: panel.width, height: panel.height)
                    .tsEntrance(delay: 0.04)
            }
        }
        // Both cards receive exactly the same size, header and footer included.
        .frame(minHeight: 440, maxHeight: .infinity)
        .confirmationDialog(Text("Clear Both", bundle: .module), isPresented: $confirmingClearAll) {
            Button(role: .destructive) {
                performClearAll()
            } label: { Text("Clear Both", bundle: .module) }
        } message: {
            Text("This removes the draft and the rewrite.", bundle: .module)
        }
        .confirmationDialog(Text("Clear Rewrite", bundle: .module), isPresented: $confirmingClearRewrite) {
            Button(role: .destructive) {
                controller.clearRewrite()
                reviewing = false
            } label: { Text("Clear Rewrite", bundle: .module) }
        } message: {
            Text("This removes the rewrite.", bundle: .module)
        }
    }

    private var canClearAll: Bool {
        !controller.isBusy && (!controller.draft.isEmpty || !controller.rewrite.isEmpty)
    }

    /// A rewrite is the one thing a clear cannot give back (every run is a new
    /// sample), so it alone asks first. A lone draft is as easy to restore as
    /// to clear, and goes without a dialog like the Clear Draft chip.
    private func clearAll() {
        if controller.rewrite.isEmpty { performClearAll() } else { confirmingClearAll = true }
    }

    private func performClearAll() {
        controller.clearAll()
        reviewing = false
    }

    private var showsMarkedDraft: Bool {
        reviewing && showChanges && difference != nil && !controller.isBusy
            && controller.draft == controller.sourceDraft
    }

    private func draftPanel(tight: Bool) -> some View {
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
                            .foregroundStyle(.appSecondary)
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
            AppChip(Text("Paste", bundle: .module), systemImage: "doc.on.clipboard") {
                if let text = NSPasteboard.general.string(forType: .string) {
                    controller.draft = text
                    reviewing = false
                }
            }
            .disabled(controller.isBusy)
            AppChip(Text("Clear Draft", bundle: .module), systemImage: "eraser", iconOnly: tight) {
                controller.draft = ""
                reviewing = false
            }
            .disabled(controller.draft.isEmpty || controller.isBusy)
            .accessibilityIdentifier("humanize.clearDraft")
            if showsMarkedDraft {
                AppChip(Text("Edit", bundle: .module), systemImage: "pencil") {
                    TSMotion.animate(TSMotion.hover) { reviewing = false }
                    draftFocused = true
                }
            }
            Spacer(minLength: 8)
            if let plan = controller.currentPlan {
                Text("Tokens: \(plan.tokens)", bundle: .module)
                    .modifier(StatStyle())
                    .foregroundStyle(plan.fits ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(AppAttention.color))
                    .help(Text("The model reads up to 8,192 tokens at a time. Longer drafts are rewritten in parts at paragraph breaks.", bundle: .module))
            }
        }
    }

    private func resultPanel(tight: Bool) -> some View {
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
                        HumanizeProgressStatus(controller: controller)
                    }
                    .padding(.horizontal, 21).padding(.vertical, 22)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .transition(.opacity)
                } else {
                    AppEmptyState(
                        systemImage: "text.alignleft",
                        title: Text("Your rewrite will appear here.", bundle: .module)
                    )
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
                if !controller.rewrite.isEmpty { HumanizeProgressStatus(controller: controller) }
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
                AppChip(Text("Regenerate", bundle: .module), systemImage: "arrow.clockwise") { controller.run() }
                    .disabled(!controller.canRun)
            }
            AppChip(copied ? Text("Copied", bundle: .module) : Text("Copy", bundle: .module),
                    systemImage: copied ? "checkmark" : "doc.on.doc", isOn: copied) {
                NSPasteboard.general.clearContents()
                copied = NSPasteboard.general.setString(controller.rewrite, forType: .string)
            }
            .disabled(controller.rewrite.isEmpty || controller.isBusy)
            // Last, so it sits at the card's outer edge like Clear Draft does
            // on the other side of the run button.
            AppChip(Text("Clear Rewrite", bundle: .module), systemImage: "eraser", iconOnly: tight) {
                confirmingClearRewrite = true
            }
            .disabled(controller.rewrite.isEmpty || controller.isBusy)
            .accessibilityIdentifier("humanize.clearRewrite")
        }
        .tsAnimation(TSMotion.hover, value: controller.rewrite.isEmpty)
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
        AppChip(Text("Show changes", bundle: .module), systemImage: "highlighter", isOn: showChanges,
                iconOnly: iconOnly) {
            TSMotion.animate(TSMotion.hover) { showChanges.toggle() }
        }
        .disabled(controller.rewrite.isEmpty || controller.isBusy)
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

    /// Nil until something was checked: the model's audit, the user's rules,
    /// or both.
    private var checksClean: Bool? {
        guard controller.checks != nil || controller.ruleFindings != nil else { return nil }
        return (controller.checks?.isClean ?? true) && (controller.ruleFindings?.isEmpty ?? true)
    }

    private func checksChip(clean: Bool) -> some View {
        let tint: Color = clean ? .green : AppAttention.color
        return Label {
            if clean { Text("Checks passed", bundle: .module) } else { Text("Check before you use it", bundle: .module) }
        } icon: {
            Image(systemName: clean ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
        }
        .font(theme.ui(.small, weight: .medium))
        .foregroundStyle(tint)
        .lineLimit(1)
        .padding(.horizontal, 8).padding(.vertical, 5)
        .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .fixedSize()
        .help(clean
              ? Text("The automatic checks found nothing. Read it before you use it.", bundle: .module)
              : Text(verbatim: HumanizeAuditLines.warnings(for: controller).joined(separator: "\n")))
        .transition(.scale(scale: 0.9).combined(with: .opacity))
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
            text[marked].backgroundColor = AppAttention.color.opacity(0.22)
            text[marked].underlineStyle = Text.LineStyle(pattern: .dash, color: AppAttention.color)
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

    /// The welcome hero's wash: a faint accent glow behind the page, matching Voice Cloning.
    private var backdrop: some View {
        RadialGradient(
            colors: [theme.accent.opacity(theme.isDark ? 0.10 : 0.07), .clear],
            center: UnitPoint(x: 0.5, y: 0.0), startRadius: 0, endRadius: 620)
            .ignoresSafeArea()
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}

private struct StatStyle: ViewModifier {
    @Environment(\.appTheme) private var theme

    func body(content: Content) -> some View {
        content.font(theme.ui(.small)).monospacedDigit().foregroundStyle(.appSecondary).lineLimit(1)
    }
}
