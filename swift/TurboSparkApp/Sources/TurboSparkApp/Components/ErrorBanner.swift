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
                    Text(error)
                        .themedFont(.base)
                        .lineLimit(2)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Error: \(error)")

                Spacer(minLength: 8)

                Button {
                    model.error = nil
                } label: {
                    Label("Dismiss error", systemImage: "xmark")
                        .labelStyle(.iconOnly)
                        .themedFont(.small, weight: .semibold)
                        .frame(width: 28, height: 28)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Dismiss error")
                .accessibilityLabel("Dismiss error")
                .accessibilityHint("Removes the error message")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background {
                Capsule()
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .overlay {
                        Capsule().stroke(.red.opacity(0.55), lineWidth: 1)
                    }
            }
            .accessibilityAction(named: "Dismiss error") {
                model.error = nil
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))
            .onAppear {
                _ = AccessibilityNotification.Announcement.post(.init("Error: \(error)"))
            }
            .onChange(of: error) { _, newError in
                _ = AccessibilityNotification.Announcement.post(.init("Error: \(newError)"))
            }
        }
    }
}
