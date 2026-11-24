import SwiftUI

/// Sits above the composer while a local resource is past its warning line:
/// memory pressure, thermal state or a nearly full context window. The
/// readings are the status bar's own (`model.telemetry`, refreshed by the
/// chrome's existing 2 s poll, and the context meter's fraction), so this
/// adds no polling; it only moves the signal next to where the user is
/// typing. Which notice shows, and when a dismissed one comes back, is
/// `ResourceWarning`.
///
/// Modeled on `ModelLoadFailureBannerView` and `StallRecoveryBannerView`,
/// which win over it (it waits rather than stacking under them). A glyph and
/// a title carry the meaning, never the stroke color alone. No action carries
/// `.defaultAction`: Return belongs to the composer.
@MainActor
struct ResourceWarningStripView: View {
    @ObservedObject var model: AppModel
    /// Owned by the pane, which also prunes it when a resource recovers.
    /// Not persisted: a dismissal is "not now", not a setting.
    @Binding var dismissed: [ResourceWarning.Kind: ResourceWarning.Level]

    /// The context meter's reading (`StatusBarView.contextFill`), local
    /// chats only: a remote chat's window is the provider's, not this one.
    static func contextFraction(for model: AppModel) -> Double? {
        guard model.session != nil,
              model.remoteContextReading(chatID: model.selectedChatID) == nil else { return nil }
        let limit = model.resolvedContextTokens
        guard limit > 0 else { return nil }
        return min(1, Double(model.estimatedContextTokens) / Double(limit))
    }

    static func levels(for model: AppModel) -> [ResourceWarning.Kind: ResourceWarning.Level] {
        ResourceWarning.levels(
            hasSession: model.session != nil,
            memoryPressure: model.telemetry?.memoryPressure,
            thermalLevel: model.telemetry?.thermalLevel,
            contextFraction: contextFraction(for: model),
            isCompacting: model.isCompacting)
    }

    var body: some View {
        let notice = ResourceWarning.notice(
            levels: Self.levels(for: model), dismissed: dismissed,
            blockedByOtherStrip: model.modelLoadFailure != nil
                || model.generationStallNotice != nil
                || model.error != nil)
        if let notice {
            strip(notice)
                // Fires when the strip appears and when it switches to a
                // different resource or level, never per poll.
                .onChange(of: notice, initial: true) { _, shown in
                    let spoken = "\(Self.title(shown)). \(Self.detail(shown, percent: percent))"
                    _ = AccessibilityNotification.Announcement.post(.init(spoken))
                }
        }
    }

    private var percent: Int {
        Int(((Self.contextFraction(for: model) ?? 0) * 100).rounded())
    }

    private func strip(_ notice: ResourceWarning.Notice) -> some View {
        let tint: AppThemeColor = notice.level == .critical ? .appDanger : .appWarning
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: Self.symbol(notice))
                    .themedFont(.title3)
                    .foregroundStyle(tint)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(verbatim: Self.title(notice))
                        .themedFont(.small, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    Text(verbatim: Self.detail(notice, percent: percent))
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)

                Spacer(minLength: 8)

                AppIconButton("xmark", label: Text("Dismiss", bundle: .module)) {
                    dismiss(notice)
                }
                .appAction(.quiet, size: .small)
                .accessibilityHint(Text("Dismisses this message", bundle: .module))
            }
            if notice.kind != .thermal {
                HStack(spacing: 8) { action(notice) }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .fill(.appSurface)
                .overlay {
                    RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                        .stroke(tint, lineWidth: 1)
                }
        }
        .transition(.opacity.combined(with: .move(edge: .bottom)))
        .accessibilityElement(children: .contain)
        .accessibilityAction(named: Text("Dismiss", bundle: .module)) { dismiss(notice) }
    }

    private func dismiss(_ notice: ResourceWarning.Notice) {
        dismissed = ResourceWarning.dismissing(notice, from: dismissed)
    }

    /// The one existing fix per resource. Heat has none in the app (the fan
    /// readout in the status bar is opt-in hardware control), so it only
    /// informs.
    @ViewBuilder
    private func action(_ notice: ResourceWarning.Notice) -> some View {
        switch notice.kind {
        case .context:
            let blocker = model.contextMeterCompactBlocker
            Button {
                model.compactKeepingDraft()
            } label: {
                Label { Text("Compact now", bundle: .module) } icon: { Image(systemName: "rectangle.compress.vertical") }
                    .themedFont(.small, weight: .semibold)
            }
            .appAction(.primary, size: .small)
            .disabled(blocker != nil)
            .help(blocker?.helpText
                ?? Text("Summarize older messages now and keep recent turns verbatim", bundle: .module))
        case .memory:
            Button {
                model.unloadModel()
            } label: {
                Label { Text("Unload model", bundle: .module) } icon: { Image(systemName: "eject.fill") }
                    .themedFont(.small, weight: .semibold)
            }
            .appAction(.primary, size: .small)
            .disabled(!model.canUnloadModel)
            .help(model.canUnloadModel
                ? Text("Free this model's memory. Load it again from the model picker.", bundle: .module)
                : Text("Wait for the current turn to finish before unloading.", bundle: .module))
        case .thermal:
            EmptyView()
        }
    }

    private static func symbol(_ notice: ResourceWarning.Notice) -> String {
        switch notice.kind {
        case .memory: return "memorychip"
        case .thermal: return "thermometer.high"
        case .context: return "text.alignleft"
        }
    }

    static func title(
        _ notice: ResourceWarning.Notice, language: AppLanguage = AppStrings.currentLanguage
    ) -> String {
        switch (notice.kind, notice.level) {
        case (.memory, .warning): return AppStrings.localized("Memory is running low", language: language)
        case (.memory, .critical): return AppStrings.localized("Memory is critically low", language: language)
        case (.thermal, .warning): return AppStrings.localized("This Mac is running hot", language: language)
        case (.thermal, .critical): return AppStrings.localized("This Mac is overheating", language: language)
        case (.context, .warning): return AppStrings.localized("Context is almost full", language: language)
        case (.context, .critical): return AppStrings.localized("Context is nearly out of room", language: language)
        }
    }

    /// States the consequence, then the fix the strip offers.
    static func detail(
        _ notice: ResourceWarning.Notice, percent: Int,
        language: AppLanguage = AppStrings.currentLanguage
    ) -> String {
        switch (notice.kind, notice.level) {
        case (.memory, .warning):
            return AppStrings.localized(
                "macOS reports memory pressure, so replies are paced down. Unloading the model or closing other apps frees memory.",
                language: language)
        case (.memory, .critical):
            return AppStrings.localized(
                "macOS reports critical memory pressure. The model may stall or be stopped. Unload it or close other apps now.",
                language: language)
        case (.thermal, .warning):
            return AppStrings.localized(
                "macOS is slowing the chip to cool down, so replies will be slower than usual.",
                language: language)
        case (.thermal, .critical):
            return AppStrings.localized(
                "macOS is throttling hard. Letting the Mac rest before the next long reply helps it cool.",
                language: language)
        case (.context, _):
            return AppStrings.localized(
                "This chat fills \(percent)% of the context window. Compact it to keep room for replies.",
                language: language)
        }
    }
}
