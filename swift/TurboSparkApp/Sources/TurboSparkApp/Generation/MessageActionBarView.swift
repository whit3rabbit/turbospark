import AppKit
import SwiftUI

/// Floating action bar containing copy, share, read out loud, and timestamp actions for a message.
///
/// The optional actions are the message-editing affordances (`AppModel+
/// MessageEditing.swift`): variant navigation, in-place edit, branch, and
/// retry. Each is nil where it does not apply -- the row decides, so this
/// view stays a dumb pill strip.
struct MessageActionBarView: View {
    let text: String
    let messageID: UUID
    let date: Date
    /// Steps to a neighbouring version of this message. Rendered with the
    /// position/count when the message has more than one version.
    var variantStep: ((Int) -> Void)? = nil
    var variantPosition: Int = 1
    var variantCount: Int = 1
    /// Opens the in-place editor (the last real user prompt).
    var editAction: (() -> Void)? = nil
    /// Opens the branch editor (an earlier user prompt).
    var branchAction: (() -> Void)? = nil
    /// Regenerates this response (the last prose reply).
    var retryAction: (() -> Void)? = nil
    /// Greyed while a turn runs; the model-side guards refuse anyway, this
    /// is the affordance half.
    var actionsDisabled: Bool = false

    var body: some View {
        HStack(spacing: 6) {
            if variantCount > 1, let variantStep {
                MessageVariantSwitcherView(
                    position: variantPosition, count: variantCount, step: variantStep)
            }

            MessageCopyButton(text: text)

            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                MessageShareButton(text: text)
                MessageSpeechButton(text: text, messageID: messageID)
            }

            if let editAction {
                MessagePillActionButton(
                    icon: "pencil", label: "Edit",
                    help: "Edit this message and regenerate the reply",
                    disabled: actionsDisabled, action: editAction)
            }

            if let branchAction {
                MessagePillActionButton(
                    icon: "arrow.triangle.branch", label: "Branch",
                    help: "Edit this message in a branched copy of the conversation",
                    disabled: actionsDisabled, action: branchAction)
            }

            if let retryAction {
                MessagePillActionButton(
                    icon: "arrow.clockwise", label: "Retry",
                    help: "Generate a new reply to the same prompt",
                    disabled: actionsDisabled, action: retryAction)
            }

            MessageTimestampBadge(date: date)
        }
    }
}

/// The shared pill button the editing affordances render as. Matches the
/// existing copy/share pills' look without restating them.
struct MessagePillActionButton: View {
    let icon: String
    let label: String
    let help: String
    let disabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .themedFont(.tiny, weight: .medium)
                    .accessibilityHidden(true)
                Text(label)
                    .themedFont(.tiny, weight: .medium)
            }
            .foregroundStyle(Color.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.4 : 1)
        .help(help)
        .accessibilityLabel("\(label) message")
    }
}

/// Share button opening the native macOS share picker.
struct MessageShareButton: View {
    let text: String

    var body: some View {
        ShareLink(item: text) {
            HStack(spacing: 4) {
                Image(systemName: "square.and.arrow.up")
                    .themedFont(.tiny, weight: .medium)
                    .accessibilityHidden(true)
                Text("Share", bundle: .module)
                    .themedFont(.tiny, weight: .medium)
            }
            .foregroundStyle(Color.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(Text("Share message", bundle: .module))
        .accessibilityLabel("Share message")
    }
}

/// Speech synthesis button triggering Apple's native AVSpeechSynthesizer.
struct MessageSpeechButton: View {
    let text: String
    let messageID: UUID
    @ObservedObject private var speechManager = AppSpeechSynthesizer.shared

    private var isSpeakingThis: Bool {
        speechManager.isSpeaking && speechManager.speakingMessageID == messageID
    }

    var body: some View {
        Button {
            speechManager.toggleSpeech(text: text, messageID: messageID)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isSpeakingThis ? "stop.fill" : "speaker.wave.2")
                    .themedFont(.tiny, weight: .medium)
                    .accessibilityHidden(true)
                Text(isSpeakingThis ? "Stop" : "Read")
                    .themedFont(.tiny, weight: .medium)
            }
            .foregroundStyle(isSpeakingThis ? TurboSparkTheme.accentColor : Color.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(
                isSpeakingThis ? TurboSparkTheme.accentColor.opacity(0.12) : Color.primary.opacity(0.04),
                in: RoundedRectangle(cornerRadius: 6, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(isSpeakingThis ? "Stop reading out loud" : "Read message out loud")
        .accessibilityLabel(isSpeakingThis ? "Stop reading message out loud" : "Read message out loud")
    }
}

/// Timestamp badge showing human-readable relative time and exact time on hover.
struct MessageTimestampBadge: View {
    let date: Date
    @State private var isHovered = false

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: "clock")
                .themedFont(.tiny, weight: .medium)
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            Text(isHovered ? MessageTimestampFormatter.standardString(for: date) : MessageTimestampFormatter.relativeString(for: date))
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .lineLimit(1)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(Color.primary.opacity(isHovered ? 0.08 : 0.04), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
        .help(MessageTimestampFormatter.exactString(for: date))
        .accessibilityLabel("Sent \(MessageTimestampFormatter.relativeString(for: date)), \(MessageTimestampFormatter.exactString(for: date))")
    }
}

/// Small copy button displayed beneath message results.
struct MessageCopyButton: View {
    let text: String
    @State private var isCopied = false

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            withAnimation(.easeInOut(duration: 0.15)) {
                isCopied = true
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                    .themedFont(.tiny, weight: .medium)
                    .accessibilityHidden(true)
                Text(isCopied ? "Copied" : "Copy")
                    .themedFont(.tiny, weight: .medium)
            }
            .foregroundStyle(isCopied ? TurboSparkTheme.accentColor : Color.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(Text("Copy message text", bundle: .module))
        .accessibilityLabel(isCopied ? "Copied" : "Copy message")
        .accessibilityHint("Copies this message to the clipboard")
        .task(id: isCopied) {
            guard isCopied else { return }
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                isCopied = false
            }
        }
    }
}
