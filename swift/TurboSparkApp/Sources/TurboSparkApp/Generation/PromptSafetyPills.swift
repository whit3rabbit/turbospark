import SwiftUI
import TurboSpark

/// Forge Guardrails status pill button with toggling and settings popover link.
struct ForgeGuardrailsPillControl: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        let isEnabled = model.effectiveForgeGuardrailsEnabled
        let isGlobalFixed = model.guardrailsMode == .alwaysOn || model.guardrailsMode == .alwaysOff
        // **THE INERT CONDITION IS "NO TOOLS WERE OFFERED", NOT "THE MODEL
        // LACKS TOOL MARKUP".** `inspect` accepts unconditionally when the
        // request carried no tools, so that is the one case where this toggle
        // provably changes nothing. Gating on the model's own markup would be
        // backwards: a checkpoint that frames no calls is exactly where the
        // rescue earns its keep, which is why `toolCalling` feeds the tooltip
        // and never the `disabled`.
        let inertReason = model.effectiveGuardrailsInertReason
        let isDisabled = isGlobalFixed || inertReason != nil

        HStack(spacing: 4) {
            Button {
                if !isDisabled {
                    model.setForgeGuardrailsEnabled(!isEnabled)
                }
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: isEnabled ? "shield.checkmark.fill" : "shield.slash")
                        .font(theme.ui(.tiny, weight: .semibold))
                        .foregroundStyle(isEnabled ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))

                    (isEnabled ? Text("Guardrails On", bundle: .module) : Text("Guardrails Off", bundle: .module))
                        .font(theme.ui(.small, weight: .medium))
                        .foregroundStyle(isEnabled ? AnyShapeStyle(.appText) : AnyShapeStyle(.appSecondary))
                        .lineLimit(1)
                }
            }
            .buttonStyle(.plain)
            // DISABLED WITH A REASON, NEVER HIDDEN (swift/CLAUDE.md Gotchas
            // 23 and 33): a missing control reads as a missing feature.
            .disabled(isDisabled)
            .help(Self.helpText(
                inertReason: inertReason,
                isGlobalFixed: isGlobalFixed,
                modeLabel: AppStrings.localized(String.LocalizationValue(model.guardrailsMode.label)),
                isEnabled: isEnabled,
                nativeReason: model.toolCallingNativeReason
            ))
            .accessibilityLabel(Text("Forge Guardrails", bundle: .module))
            .accessibilityValue(isEnabled ? Text("On", bundle: .module) : Text("Off", bundle: .module))
            // The inert reason is the model's own English diagnosis.
            .accessibilityHint(
                inertReason.map { Text($0) } ?? (isGlobalFixed
                    ? Text("Managed by global settings", bundle: .module)
                    : Text("Toggles tool-call guardrails for this context", bundle: .module)))

            Button {
                model.openSettings(tab: .engine)
            } label: {
                Image(systemName: "info.circle")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(Text("Tool-call guardrails rescue a call the decoder could not parse and check its arguments against the schema you sent. Click to open Settings.", bundle: .module))
            .accessibilityLabel(Text("Forge Guardrails information", bundle: .module))
            .accessibilityHint(Text("Opens Engine Settings to configure Guardrails", bundle: .module))
        }
    }
}

extension ForgeGuardrailsPillControl {
    /// One tooltip, assembled from the three things that can be true at once.
    ///
    /// A pure static so it can be asserted without a model (`swift/CLAUDE.md`
    /// Gotcha 26). The ORDER is the point: an inert control explains itself
    /// first, because a user looking at a greyed pill wants to know why it is
    /// grey before anything else. The fixed sentences come from the catalog
    /// in the UI language; the reasons passed in are the engine's own words.
    static func helpText(
        inertReason: String?,
        isGlobalFixed: Bool,
        modeLabel: String,
        isEnabled: Bool,
        nativeReason: String?
    ) -> String {
        if let inertReason { return inertReason }
        if isGlobalFixed { return AppStrings.localized("Forge Guardrails is fixed to \(modeLabel) in Settings") }
        var text =
            isEnabled
            ? AppStrings.localized("Forge Guardrails active: click to disable")
            : AppStrings.localized("Forge Guardrails disabled: click to enable")
        // A checkpoint that frames no calls of its own is the case the rescue
        // helps MOST, so this reads as a reason to leave guardrails on rather
        // than as a limitation.
        if let nativeReason {
            text += ". " + AppStrings.localized(
                "This model hands no framed call over (\(nativeReason)), so the rescue is what makes tool calls work here.")
        }
        return text
    }
}
