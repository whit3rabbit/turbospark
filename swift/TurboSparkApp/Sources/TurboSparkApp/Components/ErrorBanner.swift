import SwiftUI

/// Dismissible top error banner view presenting active inference or application errors with VoiceOver announcements.
struct ErrorBanner: View {
    @ObservedObject var model: AppModel

    var body: some View {
        if let error = model.error {
            HStack(spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(AppAttention.color)
                        .accessibilityHidden(true)
                    Text(verbatim: error)
                        .themedFont(.base)
                        .lineLimit(2)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(Text("Error: \(error)", bundle: .module))

                Spacer(minLength: 8)

                AppIconButton("xmark", label: Text("Dismiss error", bundle: .module)) {
                    model.error = nil
                }
                .appAction(.quiet, size: .small)
                .accessibilityHint(Text("Removes the error message", bundle: .module))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background {
                Capsule()
                    .fill(.appSurface)
                    .overlay {
                        Capsule().stroke(AppAttention.color, lineWidth: 1)
                    }
            }
            .accessibilityAction(named: Text("Dismiss error", bundle: .module)) {
                model.error = nil
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))
            .onAppear {
                _ = AccessibilityNotification.Announcement.post(.init(AppStrings.localized("Error: \(error)")))
            }
            .onChange(of: error) { _, newError in
                _ = AccessibilityNotification.Announcement.post(.init(AppStrings.localized("Error: \(newError)")))
            }
        }
    }
}
