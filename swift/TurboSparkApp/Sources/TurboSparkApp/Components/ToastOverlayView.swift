import SwiftUI

/// Floating notification toast overlay presenting dismissible status, error, or confirmation banners.
public struct ToastOverlayView: View {
    @ObservedObject var model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @ScaledMetric private var iconSize: CGFloat = 16
    @ScaledMetric private var dismissButtonSize: CGFloat = 20

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        if let toast = model.activeToast {
            toastCard(toast)
                .transition(reduceMotion ? .opacity : .asymmetric(
                    insertion: .move(edge: .top).combined(with: .opacity),
                    removal: .scale(scale: 0.95).combined(with: .opacity)
                ))
                .id(toast.id)
                .task(id: toast.id) {
                    try? await Task.sleep(for: .seconds(toast.duration))
                    guard !Task.isCancelled else { return }
                    withAnimation(reduceMotion ? .none : .easeOut(duration: 0.2)) {
                        model.dismissToast()
                    }
                }
        }
    }

    private func toastCard(_ toast: AppToast) -> some View {
        HStack(spacing: 10) {
            Image(systemName: toast.style.systemImage)
                .font(.system(size: iconSize, weight: .semibold))
                .foregroundStyle(toast.style.tintColor)
                .accessibilityHidden(true)

            Text(toast.message)
                .font(.callout.weight(.medium))
                .foregroundStyle(.primary)
                .lineLimit(2)

            Button {
                withAnimation(reduceMotion ? .none : .easeOut(duration: 0.15)) {
                    model.dismissToast()
                }
            } label: {
                Image(systemName: "xmark")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
                    .frame(width: dismissButtonSize, height: dismissButtonSize)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Dismiss notification")
            .accessibilityLabel("Dismiss notification")
            .accessibilityHint("Dismisses the current status message")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            Capsule()
                .fill(Color(nsColor: .windowBackgroundColor))
                .shadow(color: .black.opacity(0.18), radius: 12, x: 0, y: 4)
                .overlay {
                    Capsule().stroke(Color.primary.opacity(0.12), lineWidth: 0.5)
                }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(toast.style.rawValue): \(toast.message)")
    }
}
