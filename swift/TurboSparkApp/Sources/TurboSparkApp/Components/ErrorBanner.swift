import SwiftUI

/// Dismissible top error banner view presenting active inference or application errors with VoiceOver announcements.
struct ErrorBanner: View {
    @ObservedObject var model: AppModel

    var body: some View {
        if let error = model.error {
            HStack(spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .accessibilityHidden(true)
                    Text(verbatim: error)
                        .themedFont(.base)
                        .lineLimit(2)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(Text("Error: \(error)", bundle: .module))

                Spacer(minLength: 8)

                Button {
                    model.error = nil
                } label: {
                    Label { Text("Dismiss error", bundle: .module) } icon: { Image(systemName: "xmark") }
                        .labelStyle(.iconOnly)
                        .themedFont(.small, weight: .semibold)
                        .frame(width: 28, height: 28)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help(Text("Dismiss error", bundle: .module))
                .accessibilityLabel(Text("Dismiss error", bundle: .module))
                .accessibilityHint(Text("Removes the error message", bundle: .module))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background {
                Capsule()
                    .fill(.appSurface)
                    .overlay {
                        Capsule().stroke(.red.opacity(0.55), lineWidth: 1)
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
