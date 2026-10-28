import SwiftUI

/// Floating notification toast overlay presenting dismissible status, error, or confirmation banners.
public struct ToastOverlayView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    /// The app preference over the system one, the same resolution
    /// `RootView` uses. Reading the environment alone ignored the Reduce
    /// motion setting entirely (`swift/docs/SWIFT_SETTINGS_AUDIT.md`).
    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }
    @ScaledMetric private var iconSize: CGFloat = 16
    @ScaledMetric private var dismissButtonSize: CGFloat = 20

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        VStack(alignment: .trailing, spacing: 10) {
            if let toast = model.activeToast {
                toastCard(toast)
                    .transition(reduceMotion ? .opacity : .asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .scale(scale: 0.95).combined(with: .opacity)
                    ))
                    .id(toast.id)
                    .task(id: toast.id) {
                        try? await Task.sleep(for: .seconds(toast.duration))
                        guard !Task.isCancelled, model.activeToast?.id == toast.id else { return }
                        TSMotion.animate(TSMotion.hover) {
                            model.dismissToast()
                        }
                    }
            }
        }
        .frame(maxWidth: 420, alignment: .trailing)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func toastCard(_ toast: AppToast) -> some View {
        HStack(spacing: 10) {
            Image(systemName: toast.style.systemImage)
                .themedFont(fitting: iconSize, weight: .semibold)
                .foregroundStyle(toast.style.tintColor(in: theme))
                .accessibilityHidden(true)

            // The caller owns the message's localization.
            Text(verbatim: toast.message)
                .themedFont(.base, weight: .medium)
                .foregroundStyle(.appText)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

            AppIconButton("xmark", label: Text("Dismiss notification", bundle: .module)) {
                TSMotion.animate(TSMotion.hover) {
                    model.dismissToast()
                }
            }
            .appAction(.quiet, size: .small)
            .accessibilityHint(Text("Dismisses the current status message", bundle: .module))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .fill(.appPage)
                .shadow(color: .black.opacity(0.18), radius: 12, x: 0, y: 4)
                .overlay {
                    RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                        .stroke(.appBorder, lineWidth: 0.5)
                }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel(for: toast))
    }

    /// The severity is spoken first, in the UI language, because the icon
    /// that carries it visually is hidden from VoiceOver.
    private func accessibilityLabel(for toast: AppToast) -> Text {
        switch toast.style {
        case .info: return Text("Info: \(toast.message)", bundle: .module)
        case .success: return Text("Success: \(toast.message)", bundle: .module)
        case .warning: return Text("Warning: \(toast.message)", bundle: .module)
        case .error: return Text("Error: \(toast.message)", bundle: .module)
        }
    }
}
