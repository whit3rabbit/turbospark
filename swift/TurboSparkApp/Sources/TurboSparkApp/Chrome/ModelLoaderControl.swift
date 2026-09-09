import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The top bar's model loader: pick a model, watch it open, eject it.
///
/// Replaces the round-1-era control that read `gptoss-20b [play] Load` with
/// no status line and no eject affordance until a session already existed.
/// The pill now always carries three things in a fixed order -- a status dot,
/// the alias, and a secondary line naming the state -- so the control's
/// appearance does not reflow as the state changes underneath it.
///
/// Load and eject stay separate hit targets rather than one toggle, because
/// the two have different costs: an accidental eject discards a warm session
/// (weights, KV cache and expert slots) that takes seconds to rebuild.
///
/// ## What the secondary line says, per state
///
/// | state | secondary line |
/// | --- | --- |
/// | nothing installed | `No model installed` (primary), no secondary |
/// | installed, none selected | `Select a model`, no secondary |
/// | selected, not open | `Not loaded` |
/// | installing | `installStageText`, with a progress bar |
/// | opening | `Loading...`, with an indeterminate bar |
/// | open | `Ready -- 65,536 ctx` |
///
/// ## The context figure is not abbreviated
///
/// The board drew `Ready - 64K ctx`. This renders `Ready -- 65,536 ctx`
/// instead, deliberately: `65,536` is a rung the user picked by name in the
/// Essentials ladder, and `64K` is a different string for the same number
/// that they then have to reconcile. `ContextLadderPicker` prints
/// `rung.context.formatted()`; this prints
/// `model.resolvedContextTokens.formatted()`, so the two agree character for
/// character. If an abbreviation is wanted later it belongs in
/// `MetricFormat` as a shared `MetricFormat.contextTokens(_:)` used by BOTH
/// surfaces, never spelled once here.
///
/// Note this reads `resolvedContextTokens` and not `info.maxContext`, which
/// is what the old file showed. They differ whenever the user has set a
/// window below the checkpoint's trained maximum: `maxContext` is the
/// checkpoint's ceiling, `resolvedContextTokens` is the window this session
/// actually opened with, which is what "ctx" in the chrome should mean.
@MainActor
struct ModelLoaderControl: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @ScaledMetric private var barHeight: CGFloat = 34

    var body: some View {
        HStack(spacing: 0) {
            chooser
            // Only once a session exists: the accepted levels are read off
            // the checkpoint's own template at open, so before that there is
            // nothing honest to put in the menu.
            if model.reasoningPickerEnabled {
                separator
                reasoningPicker
            }
            if model.session != nil {
                separator
                ejectButton
            } else if model.canLoadModel {
                separator
                loadButton
            }
        }
        .frame(height: barHeight)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay {
            Capsule().stroke(strokeColor, lineWidth: 0.5)
        }
        .animation(TSMotion.select, value: model.session != nil)
        .animation(TSMotion.hover, value: loaderState)
        .fixedSize()
    }

    /// The pill's own border picks up the accent while a session is live, so
    /// "a model is loaded" is legible from the shape of the chrome and not
    /// only from reading the words.
    private var strokeColor: Color {
        model.session != nil
            ? theme.accent.opacity(0.32)
            : TurboSparkTheme.hairlineColor
    }

    // MARK: - Chooser

    private var chooser: some View {
        Menu {
            installedModelsSection

            Divider()

            Button {
                model.activeSection = .modelManager
            } label: {
                Label("Manage installed models...", systemImage: "internaldrive")
            }

            Button {
                model.activeSection = .modelHub
            } label: {
                Label("Discover new models...", systemImage: "shippingbox")
            }

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label("Open model folder...", systemImage: "folder")
            }

            if model.canReloadModel {
                Divider()
                Button {
                    model.reloadModel()
                } label: {
                    Label("Reload model", systemImage: "arrow.clockwise")
                }
            }
        } label: {
            HStack(spacing: 8) {
                ModelStatusDot(state: loaderState, accent: theme.accent)

                VStack(alignment: .leading, spacing: 1) {
                    Text(primaryText)
                        .font(theme.ui(.small, weight: .semibold))
                        .lineLimit(1)
                        .foregroundStyle(.primary)

                    if let secondaryText {
                        Text(secondaryText)
                            .font(theme.ui(.micro))
                            .lineLimit(1)
                            .foregroundStyle(.secondary)
                    }
                }

                // The bar occupies the same slot the chevron would, so the
                // pill does not change width when an install starts.
                if let progress = progressFraction {
                    ModelLoadProgressBar(fraction: progress, tint: theme.accent)
                        .frame(width: 42)
                } else if isIndeterminate {
                    ModelLoadProgressBar(fraction: nil, tint: theme.accent)
                        .frame(width: 42)
                } else {
                    Image(systemName: "chevron.up.chevron.down")
                        .font(theme.ui(.micro, weight: .bold))
                        .foregroundStyle(.tertiary)
                        .accessibilityHidden(true)
                }
            }
            .padding(.leading, 10)
            .padding(.trailing, 9)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        // `.button` rather than `.borderlessButton`, which FLATTENS a custom
        // label on the macOS 14 SDK: AppKit renders a pull-down whose title
        // is the label's first Text plus its own indicator, and discards the
        // rest. The status dot, the secondary line and the progress bar were
        // all dropped, and `.menuIndicator(.hidden)` was ignored with it.
        // Verified by screenshot: `.borderlessButton` drew `<> gptoss-20b`,
        // `.button` draws the dot, `gptoss-20b`, `Not loaded` and the chevron.
        // The file this replaced had the same two-line VStack and had never
        // rendered its second line either, so this is older than the redesign.
        // `ToolApprovalDropdown` sidesteps the same limit a different way, by
        // being a Button with a popover rather than a Menu.
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .frame(maxWidth: 320)
        .help(chooserHelp)
        // Without this the two Texts are two elements, and VoiceOver reads
        // "Active model" twice.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Active model")
        .accessibilityValue(accessibilityValue)
        .accessibilityHint("Choose which installed model to use")
    }

    @ViewBuilder
    private var installedModelsSection: some View {
        if model.installed.isEmpty {
            Text("No models installed yet", bundle: .module)
        } else {
            Section("Installed") {
                ForEach(model.installed) { item in
                    Button {
                        model.openChatWithModel(item)
                    } label: {
                        if model.selected?.alias == item.alias {
                            Label("\(item.alias)  (\(item.family))", systemImage: "checkmark")
                        } else {
                            Text("\(item.alias)  (\(item.family))", bundle: .module)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Reasoning

    private var reasoningPicker: some View {
        Menu {
            Section("Reasoning Effort") {
                ForEach(model.availableReasoningLevels) { level in
                    let title = model.reasoningLabel(for: level)
                        + " - " + model.reasoningDescription(for: level)
                    Button {
                        model.setReasoning(level)
                    } label: {
                        if model.reasoning == level {
                            Label(title, systemImage: "checkmark")
                        } else {
                            Text(title)
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: model.reasoning != .off ? "brain.head.profile" : "brain")
                    .font(theme.ui(.tiny, weight: .semibold))
                    .foregroundStyle(model.reasoning != .off ? theme.accent : Color.secondary)

                Text(model.reasoningLabel(for: model.reasoning))
                    .font(theme.ui(.tiny, weight: .medium))
                    .foregroundStyle(model.reasoning != .off ? Color.primary : Color.secondary)

                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(.micro, weight: .bold))
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 8)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        // Same reason as the chooser above. NOT separately verified: this
        // arm only renders once a session is open, and the model would not
        // load on the tree it was checked against.
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Reasoning effort: \(model.reasoning.label). Click to change (takes effect immediately without reloading model).")
        .accessibilityLabel("Reasoning effort: \(model.reasoning.label)")
        .accessibilityHint("Select reasoning depth")
    }

    private var separator: some View {
        Rectangle()
            .fill(TurboSparkTheme.hairlineColor)
            .frame(width: 0.5, height: barHeight * 0.55)
    }

    // MARK: - Load / eject

    private var loadButton: some View {
        Button {
            model.loadModel()
        } label: {
            Label("Load", systemImage: "play.fill")
                .font(theme.ui(.tiny, weight: .semibold))
                .labelStyle(.titleAndIcon)
                .imageScale(.small)
                .foregroundStyle(theme.accent)
                .padding(.horizontal, 11)
                .frame(maxHeight: .infinity)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.94))
        .help("Load \(model.selected?.alias ?? "the selected model") into memory")
        .accessibilityLabel("Load model")
    }

    /// The board drew a pause glyph here. This keeps `eject.fill`.
    ///
    /// Pause implies the session is suspended and resumable at no cost, which
    /// is the opposite of what `unloadModel()` does: it frees the weights and
    /// the KV cache, and coming back means paying the full open again. The
    /// glyph should not describe a cheaper action than the one it performs.
    private var ejectButton: some View {
        Button {
            model.unloadModel()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "eject.fill")
                    .font(theme.ui(.tiny, weight: .semibold))
                Text("Eject", bundle: .module)
                    .font(theme.ui(.tiny, weight: .medium))
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal, 11)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.94))
        .disabled(!model.canUnloadModel)
        .help("Eject the loaded model and free its memory")
        .accessibilityLabel("Eject model")
        .accessibilityHint("Unloads the model and frees its memory. Loading it again takes several seconds.")
    }

    // MARK: - State

    /// The six states the control has to render, named once so the dot, the
    /// secondary line and the accessibility value cannot disagree about which
    /// one is current.
    enum LoaderState: Equatable {
        case noneInstalled
        case noneSelected
        case notLoaded
        case installing
        case opening
        case ready
    }

    private var loaderState: LoaderState {
        if model.isInstallingModel { return .installing }
        if model.opening { return .opening }
        if model.installed.isEmpty { return .noneInstalled }
        guard model.selected != nil else { return .noneSelected }
        return model.session != nil ? .ready : .notLoaded
    }

    private var primaryText: String {
        switch loaderState {
        case .noneInstalled: return "No model installed"
        case .noneSelected: return "Select a model"
        default: return model.selected?.alias ?? "Select a model"
        }
    }

    private var secondaryText: String? {
        switch loaderState {
        case .noneInstalled, .noneSelected:
            return nil
        case .installing:
            // The engine's own stage text, not a generic "Installing...":
            // downloading, verifying and converting have very different
            // durations and the user is entitled to know which one is slow.
            return model.installStageText ?? "Installing"
        case .opening:
            return "Loading..."
        case .notLoaded:
            return "Not loaded"
        case .ready:
            return "Ready -- \(model.resolvedContextTokens.formatted()) ctx"
        }
    }

    private var progressFraction: Double? {
        guard loaderState == .installing else { return nil }
        return model.installProgressFraction
    }

    private var isIndeterminate: Bool {
        loaderState == .opening
            || (loaderState == .installing && model.installProgressFraction == nil)
    }

    private var chooserHelp: String {
        switch loaderState {
        case .noneInstalled:
            return "No models are installed. Open the Model Hub to download one."
        case .noneSelected:
            return "Choose which installed model to use"
        case .notLoaded:
            return "Selected but not in memory. Press Load to open it."
        case .installing:
            return model.installStageText ?? "Installing this model"
        case .opening:
            return "Opening the model: reading weights and allocating the KV cache"
        case .ready:
            return "Loaded with a \(model.resolvedContextTokens.formatted()) token context window"
        }
    }

    private var accessibilityValue: String {
        guard let secondaryText else { return primaryText }
        return "\(primaryText), \(secondaryText)"
    }
}

/// The status dot in front of the alias.
///
/// Grey when there is nothing to run, accent when a session is live, and
/// pulsing accent while one is being built. The pulse is the ONLY animated
/// thing in the top bar chrome and it is state-driven rather than per token,
/// so it does not reintroduce the redraw cost that kept throughput out of
/// this bar.
private struct ModelStatusDot: View {
    let state: ModelLoaderControl.LoaderState
    let accent: Color

    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    private var isWorking: Bool {
        state == .installing || state == .opening
    }

    private var tint: Color {
        switch state {
        case .ready: return accent
        case .installing, .opening: return accent
        case .notLoaded: return .secondary
        case .noneInstalled, .noneSelected: return .secondary
        }
    }

    var body: some View {
        Circle()
            .fill(tint)
            .frame(width: 7, height: 7)
            .opacity(state == .ready || isWorking ? 1 : 0.5)
            .modifier(PulseIfWorking(active: isWorking && !reduceMotion))
            .accessibilityHidden(true)
    }

    /// Applied as a modifier rather than branching in `body`, so the
    /// `phaseAnimator` is torn down with the view when the state leaves
    /// `installing`/`opening` instead of being left running behind a static
    /// dot.
    private struct PulseIfWorking: ViewModifier {
        let active: Bool

        func body(content: Content) -> some View {
            if active {
                content.phaseAnimator([0.4, 1.0]) { view, opacity in
                    view.opacity(opacity)
                } animation: { _ in
                    .easeInOut(duration: 0.85)
                }
            } else {
                content
            }
        }
    }
}

/// Determinate or indeterminate hairline bar, sized by its caller.
///
/// Not `ProgressView`: the linear style on the 14 SDK brings its own vertical
/// padding and a minimum width that both break a 34pt pill, and the
/// indeterminate variant cannot be tinted to `theme.accent` without an
/// appearance override.
private struct ModelLoadProgressBar: View {
    /// `nil` renders the indeterminate sweep.
    let fraction: Double?
    let tint: Color

    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.12))

                if let fraction {
                    Capsule()
                        .fill(tint)
                        .frame(width: max(3, geometry.size.width * min(1, max(0, fraction))))
                        .animation(TSMotion.hover, value: fraction)
                } else if reduceMotion {
                    // A static half bar rather than a sweep: still reads as
                    // "working", with nothing in motion.
                    Capsule()
                        .fill(tint.opacity(0.5))
                        .frame(width: geometry.size.width * 0.5)
                } else {
                    Capsule()
                        .fill(tint)
                        .frame(width: geometry.size.width * 0.4)
                        .phaseAnimator([0.0, 1.0]) { view, phase in
                            view.offset(x: phase == 0 ? 0 : geometry.size.width * 0.6)
                        } animation: { _ in
                            .easeInOut(duration: 0.95)
                        }
                }
            }
        }
        .frame(height: 3)
        .accessibilityHidden(true)
    }
}
