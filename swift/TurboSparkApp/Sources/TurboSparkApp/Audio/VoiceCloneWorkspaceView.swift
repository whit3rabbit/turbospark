import SwiftUI

/// The Voice Cloning page: voice setup in a leading column (reference,
/// engine cards, model) beside the work itself (steps, script composer, live
/// progress, the latest take and every take). One column when the window or
/// the text size leaves too little width for two.
///
/// The look follows the welcome screen: a faint radial accent wash, cards
/// that assemble with a short stagger, the composer's radius and focus glow,
/// and a glowing primary action.
@MainActor
struct VoiceCloneWorkspaceView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let description: Text
    let showPromptLibrary: () -> Void
    let savePrompt: () -> Void

    @Environment(\.appTheme) private var theme
    @FocusState private var scriptFocused: Bool
    @FocusState private var transcriptFocused: Bool
    @FocusState private var consentFocused: Bool
    @AccessibilityFocusState private var focusedStep: VoiceCloneStep?
    @StateObject private var attention = VoiceCloneAttention()

    private var scale: CGFloat { AppMetrics.textScale(theme) }
    private var missing: [VoiceCloneStep] { controller.voiceCloneMissingSteps }

    var body: some View {
        GeometryReader { geometry in
            ScrollViewReader { proxy in
                if geometry.size.width >= (840 * scale).rounded() {
                    twoColumns(proxy)
                } else {
                    oneColumn(proxy)
                }
            }
        }
        .background { backdrop }
        .onChange(of: controller.speechRun == nil) { _, finished in
            guard finished else { return }
            _ = AccessibilityNotification.Announcement.post(.init(controller.error ?? controller.status))
        }
    }

    // MARK: Layouts

    private func twoColumns(_ proxy: ScrollViewProxy) -> some View {
        HStack(alignment: .top, spacing: 24) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) { setupCards(proxy) }
                    .padding(.vertical, 24)
            }
            .frame(minWidth: (300 * scale).rounded(), idealWidth: (360 * scale).rounded(), maxWidth: (440 * scale).rounded())
            ScrollView {
                VStack(alignment: .leading, spacing: 20) { mainContent(proxy) }
                    .padding(.vertical, 24)
            }
            .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, 24)
        .frame(maxWidth: 1440)
        .frame(maxWidth: .infinity)
    }

    private func oneColumn(_ proxy: ScrollViewProxy) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                description.foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
                stepsStrip(proxy)
                setupCards(proxy)
                workContent(proxy)
            }
            .padding(24)
        }
    }

    /// Reference, engine cards and model. The model comes first until one
    /// is chosen, because the engine decides which cards follow.
    @ViewBuilder private func setupCards(_ proxy: ScrollViewProxy) -> some View {
        if controller.selectedProfile == nil { modelCard.tsEntrance() }
        VoiceReferenceCard(controller: controller, attention: attention, missing: missing)
            .id(VoiceCloneStep.voice)
            .accessibilityFocused($focusedStep, equals: .voice)
            .tsEntrance(delay: 0.04)
        VoiceCloneEngineCards(controller: controller, attention: attention, missing: missing,
                              transcriptFocused: $transcriptFocused, consentFocused: $consentFocused,
                              needsReference: { show([.voice], proxy: proxy) })
            .tsEntrance(delay: 0.08)
        if controller.selectedProfile != nil { modelCard.tsEntrance(delay: 0.12) }
    }

    private func mainContent(_ proxy: ScrollViewProxy) -> some View {
        Group {
            description.foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
            stepsStrip(proxy).tsEntrance()
            workContent(proxy)
        }
    }

    @ViewBuilder private func workContent(_ proxy: ScrollViewProxy) -> some View {
        VoiceCloneScriptComposer(controller: controller, attention: attention, missing: missing,
                                 scriptFocused: $scriptFocused,
                                 showPromptLibrary: showPromptLibrary, savePrompt: savePrompt,
                                 generate: { attemptRun(proxy) },
                                 showMissing: { show(missing, proxy: proxy) })
            .id(VoiceCloneStep.script)
            .tsEntrance(delay: 0.04)
        AudioScriptSections(controller: controller)
        Group {
            if let run = controller.speechRun, controller.isBusy {
                VoiceCloneProgressCard(controller: controller, run: run)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .tsAnimation(TSMotion.pane, value: controller.speechRun == nil)
        if let item = controller.workspaceResult, item.status != .processing {
            AudioItemDetailView(controller: controller, item: item)
                .appCardSurface()
                .transition(.opacity)
        }
        VoiceCloneTakesList(controller: controller)
            .tsEntrance(delay: 0.08)
    }

    private func stepsStrip(_ proxy: ScrollViewProxy) -> some View {
        VoiceCloneStepsStrip(steps: attention.flagged(controller.voiceCloneSteps, missing: missing)) { step in
            // A step that still needs the user also gets the ping, so the
            // strip doubles as "show me where".
            if missing.contains(step) { attention.show([step]) }
            go(to: step, proxy: proxy)
        }
    }

    /// Generate is pressable while only the user's own input is missing.
    /// With everything in place it runs; otherwise it shows what is missing.
    private func attemptRun(_ proxy: ScrollViewProxy) {
        guard !controller.canRun else {
            attention.reset()
            controller.run()
            return
        }
        guard controller.voiceCloneRunNeedsInput, !missing.isEmpty else { return }
        show(missing, proxy: proxy)
    }

    /// Flags `steps`, moves to the first, and says so for VoiceOver, which
    /// would otherwise hear nothing from a press that starts no run.
    private func show(_ steps: [VoiceCloneStep], proxy: ScrollViewProxy) {
        guard let first = steps.first else { return }
        attention.show(steps)
        go(to: first, proxy: proxy)
        let names = steps.map(VoiceCloneAttention.name(of:)).joined(separator: ", ")
        _ = AccessibilityNotification.Announcement.post(.init(AppStrings.localized("Cannot generate yet. Needs attention: \(names)")))
    }

    /// Scrolls to a step's section and moves focus into it: keyboard focus
    /// for the editors, VoiceOver focus for the cards.
    private func go(to step: VoiceCloneStep, proxy: ScrollViewProxy) {
        TSMotion.animate(TSMotion.pane) { proxy.scrollTo(step, anchor: .center) }
        switch step {
        case .script: scriptFocused = true
        case .transcript: transcriptFocused = true
        case .consent:
            consentFocused = true
            focusedStep = step
        default: focusedStep = step
        }
    }

    // MARK: Pieces

    private var modelCard: some View {
        AudioModelPickerView(model: model, controller: controller).appCardSurface()
    }

    /// The welcome hero's wash: a faint accent glow behind the page.
    private var backdrop: some View {
        RadialGradient(
            colors: [theme.accent.opacity(theme.isDark ? 0.10 : 0.07), .clear],
            center: UnitPoint(x: 0.5, y: 0.0), startRadius: 0, endRadius: 620)
            .ignoresSafeArea()
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}
