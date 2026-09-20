import SwiftUI

/// Floating notification toast overlay presenting dismissible status, error, or confirmation banners.
public struct ToastOverlayView: View {
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
                        withAnimation(reduceMotion ? .none : .easeOut(duration: 0.2)) {
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
                .foregroundStyle(toast.style.tintColor)
                .accessibilityHidden(true)

            Text(toast.message)
                .themedFont(.base, weight: .medium)
                .foregroundStyle(.appText)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button {
                withAnimation(reduceMotion ? .none : .easeOut(duration: 0.15)) {
                    model.dismissToast()
                }
            } label: {
                Image(systemName: "xmark")
                    .themedFont(.tiny, weight: .bold)
                    .foregroundStyle(.appSecondary)
                    .frame(width: dismissButtonSize, height: dismissButtonSize)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(Text("Dismiss notification", bundle: .module))
            .accessibilityLabel("Dismiss notification")
            .accessibilityHint("Dismisses the current status message")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: 14)
                .fill(.appPage)
                .shadow(color: .black.opacity(0.18), radius: 12, x: 0, y: 4)
                .overlay {
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(.appBorder, lineWidth: 0.5)
                }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(toast.style.rawValue): \(toast.message)")
    }
}
