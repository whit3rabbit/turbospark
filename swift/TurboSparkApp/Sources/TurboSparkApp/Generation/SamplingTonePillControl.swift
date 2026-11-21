import SwiftUI

/// The composer's one-click sampling control: a pill naming the current tone
/// (Precise, Balanced, Creative, a saved preset, or Custom) that opens a
/// popover of the built-in tones and saved presets, with a way into the
/// Inspector's full Generation Sampling section.
///
/// It writes the same storage the Inspector does and nothing new: the
/// selected chat's override when it has one, the app-wide settings
/// otherwise, and the popover says which. It never CREATES an override, so a
/// composer click cannot silently fork one chat away from the defaults.
/// Reasoning is not here: the model loader's reasoning pill owns it.
@MainActor
struct SamplingTonePillControl: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var showsPopover = false

    private var current: AppSamplingSettings {
        model.selectedChat.samplingOverride ?? model.globalSamplingSettings()
    }

    private var match: SamplingPresetMatch {
        SamplingToneMatcher.match(current, savedPresets: model.samplingPresets)
    }

    var body: some View {
        Button {
            showsPopover.toggle()
        } label: {
            HStack(spacing: 3) {
                Image(systemName: "slider.horizontal.3")
                    .font(theme.ui(.tiny, weight: .semibold))
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
                matchName(match)
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .themedFont(.micro, weight: .bold)
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .fixedSize()
        .appPointerCursor()
        .disabled(model.isRunning)
        .help(model.isRunning
            ? Text("Sampling settings cannot be changed while a generation is running.", bundle: .module)
            : Text("Sampling preset", bundle: .module))
        .accessibilityLabel(Text("Sampling preset", bundle: .module))
        .accessibilityValue(matchName(match))
        .popover(isPresented: $showsPopover, attachmentAnchor: .point(.top), arrowEdge: .top) {
            SamplingTonePopoverView(model: model, current: current, match: match) {
                showsPopover = false
            }
        }
    }

    private func matchName(_ match: SamplingPresetMatch) -> Text {
        switch match {
        case .tone(let tone):
            return SamplingTonePopoverView.title(tone)
        case .saved(let id):
            // A preset name is user data, shown as typed.
            if let preset = model.samplingPresets.first(where: { $0.id == id }) {
                return Text(verbatim: preset.name)
            }
            return Text("Custom", bundle: .module)
        case .custom:
            return Text("Custom", bundle: .module)
        }
    }
}

/// The popover body. Rows are plain buttons so Tab, Space and VoiceOver work
/// as for any button, and Esc closes the popover natively.
@MainActor
struct SamplingTonePopoverView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let current: AppSamplingSettings
    let match: SamplingPresetMatch
    let onClose: () -> Void

    /// Same key and default as `RootView`, so "Customize..." opens the
    /// panel that holds the full Generation Sampling section.
    @AppStorage("TurboSpark.inspectorVisible") private var isInspectorVisible = false

    private var scope: SamplingPickScope {
        SamplingToneMatcher.scope(hasChatOverride: model.selectedChat.samplingOverride != nil)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Sampling", bundle: .module)
                    .font(theme.ui(.callout, weight: .semibold))
                    .foregroundStyle(.appText)
                scopeText
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            VStack(alignment: .leading, spacing: 2) {
                ForEach(SamplingTone.allCases) { tone in
                    row(
                        title: Self.title(tone),
                        detail: Self.numbers(tone.values),
                        isSelected: match == .tone(tone)
                    ) {
                        apply(SamplingToneMatcher.applying(tone, to: current))
                    }
                }
            }

            Text("Starting points, not tuned for each model. A small model can repeat itself or ramble at Creative.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            if !model.samplingPresets.isEmpty {
                Divider()
                Text("Saved presets", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(model.samplingPresets) { preset in
                        row(
                            title: Text(verbatim: preset.name),
                            detail: Self.numbers(preset.settings),
                            isSelected: match == .saved(preset.id)
                        ) {
                            apply(preset.settings)
                        }
                    }
                }
            }

            if match == .custom {
                // Custom is a state, not a choice: it shows what is running
                // so the user can see why no tone is checked.
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Image(systemName: "checkmark")
                        .themedFont(.small, weight: .semibold)
                        .foregroundStyle(.appAccentText)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Custom", bundle: .module)
                            .themedFont(.small, weight: .medium)
                            .foregroundStyle(.appText)
                        Self.numbers(current)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                    }
                }
                .padding(.horizontal, 6)
                .accessibilityElement(children: .combine)
                .accessibilityAddTraits(.isSelected)
            }

            Divider()

            Button {
                isInspectorVisible = true
                onClose()
            } label: {
                Text("Customize...", bundle: .module)
            }
            .appAction(.secondary, size: .small)
            .accessibilityHint(Text("Opens the Inspector's Generation Sampling section", bundle: .module))
        }
        .padding(14)
        .frame(minWidth: 260, idealWidth: 300, maxWidth: 360, alignment: .leading)
        .background(.appSurface)
    }

    private var scopeText: Text {
        switch scope {
        case .thisChat: return Text("Applies to this chat only", bundle: .module)
        case .allChats: return Text("Applies to all chats", bundle: .module)
        }
    }

    private func row(title: Text, detail: Text, isSelected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                // The check, not color, carries the selection.
                Image(systemName: "checkmark")
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appAccentText)
                    .opacity(isSelected ? 1 : 0)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 1) {
                    title
                        .themedFont(.small, weight: .medium)
                        .foregroundStyle(.appText)
                    detail
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .appPointerCursor()
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private func apply(_ settings: AppSamplingSettings) {
        switch scope {
        case .thisChat:
            model.setChatSamplingOverride(id: model.selectedChatID, settings: settings)
        case .allChats:
            model.applyGlobalSampling(settings)
        }
    }

    static func title(_ tone: SamplingTone) -> Text {
        switch tone {
        case .precise: return Text("Precise", bundle: .module)
        case .balanced: return Text("Balanced", bundle: .module)
        case .creative: return Text("Creative", bundle: .module)
        }
    }

    static func numbers(_ v: SamplingTone.Values) -> Text {
        let t = v.temperature.formatted(.number.precision(.fractionLength(2)))
        let p = v.topP.formatted(.number.precision(.fractionLength(2)))
        return Text("Temperature \(t), top-k \(v.topK), top-p \(p)", bundle: .module)
    }

    /// A full snapshot reads as its temperature plus whichever truncation
    /// knobs are on, so a disabled top-k is not shown as if it applied.
    static func numbers(_ s: AppSamplingSettings) -> Text {
        let t = s.temperature.formatted(.number.precision(.fractionLength(2)))
        let p = s.topP.formatted(.number.precision(.fractionLength(2)))
        switch (s.topKEnabled, s.topPEnabled) {
        case (true, true):
            return Text("Temperature \(t), top-k \(s.topK), top-p \(p)", bundle: .module)
        case (true, false):
            return Text("Temperature \(t), top-k \(s.topK)", bundle: .module)
        case (false, true):
            return Text("Temperature \(t), top-p \(p)", bundle: .module)
        case (false, false):
            return Text("Temperature \(t)", bundle: .module)
        }
    }
}
