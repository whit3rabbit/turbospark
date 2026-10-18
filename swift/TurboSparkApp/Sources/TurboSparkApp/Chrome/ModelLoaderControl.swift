import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The chat's model loader, under the prompt composer: pick a model, watch
/// it open, eject it.
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
/// | installing | `Installing`; progress lives in the Downloads panel |
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
    var density: Density = .regular

    // Both densities are declared and one is selected, rather than feeding
    // `density` into a single metric: a property wrapper's initial value
    // cannot read another stored property, and writing the init by hand to
    // get around that also forces spelling `_model` out longhand.
    @ScaledMetric private var regularBarHeight: CGFloat = 34
    @ScaledMetric private var compactBarHeight: CGFloat = 24

    private var barHeight: CGFloat {
        density == .compact ? compactBarHeight : regularBarHeight
    }

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
        .tsAnimation(TSMotion.select, value: model.session != nil)
        .tsAnimation(TSMotion.hover, value: loaderState)
        .fixedSize(horizontal: false, vertical: true)
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
                Label { Text("Manage installed models\u{2026}", bundle: .module) } icon: { Image(systemName: "internaldrive") }
            }

            Button {
                model.activeSection = .modelHub
            } label: {
                Label { Text("Discover new models\u{2026}", bundle: .module) } icon: { Image(systemName: "shippingbox") }
            }

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label { Text("Open model folder\u{2026}", bundle: .module) } icon: { Image(systemName: "folder") }
            }

            if model.canReloadModel {
                Divider()
                Button {
                    model.reloadModel()
                } label: {
                    Label { Text("Reload model", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                }
            }
        } label: {
            HStack(spacing: density == .compact ? 6 : 8) {
                ModelStatusDot(state: loaderState, accent: theme.accent)

                // Same two strings either way, so `Ready -- 65,536 ctx` stays
                // the checkpoint's own figure in both densities (see the
                // "not abbreviated" note at the top of this file). Compact
                // separates them by weight and colour rather than by a glyph,
                // which also keeps this arm free of any new localized string.
                if density == .compact {
                    HStack(spacing: 5) {
                        primaryText
                            .font(theme.ui(.tiny, weight: .semibold))
                            .lineLimit(1)
                            .foregroundStyle(.appText)

                        if let secondaryText {
                            secondaryText
                                .font(theme.ui(.tiny))
                                .lineLimit(1)
                                .truncationMode(.tail)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                } else {
                    VStack(alignment: .leading, spacing: 1) {
                        primaryText
                            .font(theme.ui(.small, weight: .semibold))
                            .lineLimit(1)
                            .foregroundStyle(.appText)

                        if let secondaryText {
                            secondaryText
                                .font(theme.ui(.micro))
                                .lineLimit(1)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                }

                if loaderState == .opening {
                    ModelLoadProgressBar(fraction: nil, tint: theme.accent)
                        .frame(width: 42)
                } else {
                    Image(systemName: "chevron.up.chevron.down")
                        .font(theme.ui(.micro, weight: .bold))
                        .foregroundStyle(.tertiary)
                        .accessibilityHidden(true)
                }
            }
            .padding(.leading, density == .compact ? 8 : 10)
            .padding(.trailing, density == .compact ? 7 : 9)
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
        // Honor the available width so long model aliases cannot draw
        // outside the capsule in a narrow conversation column.
        .frame(maxWidth: density == .compact ? 420 : 320)
        .fixedSize(horizontal: false, vertical: true)
        .help(chooserHelp)
        // Without this the two Texts are two elements, and VoiceOver reads
        // "Active model" twice.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Active model", bundle: .module))
        .accessibilityValue(accessibilityValue)
        .accessibilityHint(Text("Choose which installed model to use", bundle: .module))
    }

    /// The loadable list only (requirement 3.3): a model on a drive that is away
    /// is not offered here, and with nothing else to offer the menu says why.
    @ViewBuilder
    private var installedModelsSection: some View {
        let loadable = model.loadableModels
        if loadable.isEmpty {
            if let statement = model.notConnectedStatement {
                Text(verbatim: statement)
            } else {
                Text("No models installed yet", bundle: .module)
            }
        } else {
            Section(header: Text("Installed", bundle: .module)) {
                ForEach(loadable) { item in
                    Button {
                        model.openChatWithModel(item)
                    } label: {
                        // Alias and family are identifiers: nothing to translate.
                        if model.selected?.alias == item.alias {
                            Label {
                                Text(verbatim: "\(item.alias)  (\(item.family))")
                            } icon: {
                                Image(systemName: "checkmark")
                            }
                        } else {
                            Text(verbatim: "\(item.alias)  (\(item.family))")
                        }
                    }
                }
            }
        }
    }

    // MARK: - Reasoning

    private var reasoningPicker: some View {
        Menu {
            Section(header: Text("Reasoning Effort", bundle: .module)) {
                ForEach(model.availableReasoningLevels) { level in
                    let title = Text(LocalizedStringKey(model.reasoningLabel(for: level)), bundle: .module)
                        + Text(verbatim: " - ")
                        + Text(LocalizedStringKey(model.reasoningDescription(for: level)), bundle: .module)
                    Button {
                        model.setReasoning(level)
                    } label: {
                        if model.reasoning == level {
                            Label { title } icon: { Image(systemName: "checkmark") }
                        } else {
                            title
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: model.reasoning != .off ? "brain.head.profile" : "brain")
                    .font(theme.ui(.tiny, weight: .semibold))
                    .foregroundStyle(model.reasoning != .off ? theme.accent : theme.secondaryText)

                Text(LocalizedStringKey(model.reasoningLabel(for: model.reasoning)), bundle: .module)
                    .font(theme.ui(.tiny, weight: .medium))
                    .foregroundStyle(model.reasoning != .off ? theme.foreground : theme.secondaryText)

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
        .help(Text("Reasoning effort: \(reasoningName). Click to change (takes effect immediately without reloading model).", bundle: .module))
        .accessibilityLabel(Text("Reasoning effort: \(reasoningName)", bundle: .module))
        .accessibilityHint(Text("Select reasoning depth", bundle: .module))
    }

    /// The level's name in the UI language, for interpolating into a
    /// sentence. The binding's labels are English catalog keys.
    private var reasoningName: String {
        AppStrings.localized(String.LocalizationValue(model.reasoningLabel(for: model.reasoning)))
    }

    private var separator: some View {
        Rectangle()
            .fill(.appBorder)
            .frame(width: 0.5, height: barHeight * 0.55)
    }

    // MARK: - Load / eject

    private var loadButton: some View {
        Button {
            model.loadModel()
        } label: {
            Label { Text("Load", bundle: .module) } icon: { Image(systemName: "play.fill") }
                .font(theme.ui(.tiny, weight: .semibold))
                .labelStyle(.titleAndIcon)
                .imageScale(.small)
                .foregroundStyle(theme.accent)
                .padding(.horizontal, 11)
                .frame(maxHeight: .infinity)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.94))
        .help(model.selected.map { Text("Load \($0.alias) into memory", bundle: .module) }
            ?? Text("Load the selected model into memory", bundle: .module))
        .accessibilityLabel(Text("Load model", bundle: .module))
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
            .foregroundStyle(.appSecondary)
            .padding(.horizontal, 11)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.94))
        .disabled(!model.canUnloadModel)
        .help(Text("Eject the loaded model and free its memory", bundle: .module))
        .accessibilityLabel(Text("Eject model", bundle: .module))
        .accessibilityHint(Text("Unloads the model and frees its memory. Loading it again takes several seconds.", bundle: .module))
    }

    // MARK: - State

    /// How much vertical room the pill has. `.regular` is the two-line 34pt
    /// capsule this control was designed as; `.compact` is the one-line 24pt
    /// capsule the chat carries under its composer, where every point of
    /// height comes out of the transcript.
    enum Density: Equatable {
        case regular
        case compact
    }

    /// The six states the control has to render, named once so the dot, the
    /// secondary line and the accessibility value cannot disagree about which
    /// one is current.
    enum LoaderState: Equatable {
        case noneInstalled
        /// Models exist and none can be loaded because their drive is away. A
        /// statement naming the drive, never an error.
        case notConnected
        case noneSelected
        case notLoaded
        case installing
        case opening
        case ready
    }

    private var loaderState: LoaderState {
        // A download of some other model must not mask a live, ready session.
        if model.isInstallingModel && model.session == nil { return .installing }
        if model.opening { return .opening }
        if model.loadableModels.isEmpty {
            return model.notConnectedModels.isEmpty ? .noneInstalled : .notConnected
        }
        guard model.selected != nil else { return .noneSelected }
        return model.session != nil ? .ready : .notLoaded
    }

    private var primaryText: Text {
        switch loaderState {
        case .noneInstalled: return Text("No model installed", bundle: .module)
        case .notConnected: return Text(verbatim: model.notConnectedStatement ?? "")
        case .noneSelected: return Text("Select a model", bundle: .module)
        default:
            // The alias is an identifier, so it is never translated.
            return model.selected.map { Text(verbatim: $0.alias) } ?? Text("Select a model", bundle: .module)
        }
    }

    private var secondaryText: Text? {
        switch loaderState {
        case .noneInstalled, .notConnected, .noneSelected:
            return nil
        case .installing:
            // Installer logs can be arbitrarily wide. Keep the selector
            // compact and let the Downloads panel carry progress.
            return Text("Installing", bundle: .module)
        case .opening:
            return Text("Loading...", bundle: .module)
        case .notLoaded:
            return Text("Not loaded", bundle: .module)
        case .ready:
            return Text("Ready -- \(model.resolvedContextTokens.formatted()) ctx", bundle: .module)
        }
    }

    private var chooserHelp: Text {
        switch loaderState {
        case .noneInstalled:
            return Text("No models are installed. Open the Model Hub to download one.", bundle: .module)
        case .notConnected:
            // The same statement the pill shows: the way out is the cable.
            return Text(verbatim: model.notConnectedStatement ?? "")
        case .noneSelected:
            return Text("Choose which installed model to use", bundle: .module)
        case .notLoaded:
            return Text("Selected but not in memory. Press Load to open it.", bundle: .module)
        case .installing:
            // The stage line is the installer's own text.
            return model.installStageText.map { Text(verbatim: $0) }
                ?? Text("Installing this model", bundle: .module)
        case .opening:
            return Text("Opening the model: reading weights and allocating the KV cache", bundle: .module)
        case .ready:
            return Text("Loaded with a \(model.resolvedContextTokens.formatted()) token context window", bundle: .module)
        }
    }

    private var accessibilityValue: Text {
        guard let secondaryText else { return primaryText }
        return primaryText + Text(verbatim: ", ") + secondaryText
    }
}
