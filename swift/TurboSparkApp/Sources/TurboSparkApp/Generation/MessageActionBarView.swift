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
    /// Pins an eligible user message as a standing instruction.
    var standingInstructionAction: (() -> Void)? = nil
    var isStandingInstruction: Bool = false
    /// Greyed while a turn runs; the model-side guards refuse anyway, this
    /// is the affordance half.
    var actionsDisabled: Bool = false

    var body: some View {
        HStack(spacing: 6) {
            if variantCount > 1, let variantStep {
                MessageVariantSwitcherView(
                    position: variantPosition, count: variantCount, step: variantStep)
            }

            if let standingInstructionAction {
                Button(action: standingInstructionAction) {
                    HStack(spacing: 4) {
                        Image(systemName: isStandingInstruction ? "pin.fill" : "pin")
                            .themedFont(.tiny, weight: .medium)
                            .accessibilityHidden(true)
                        standingInstructionTitle
                            .themedFont(.tiny, weight: .medium)
                    }
                    .foregroundStyle(
                        isStandingInstruction ? TurboSparkTheme.accentColor : Color.secondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .background(
                        Color.primary.opacity(0.04),
                        in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(actionsDisabled)
                .opacity(actionsDisabled ? 0.4 : 1)
                .help(standingInstructionTitle)
                .accessibilityLabel(standingInstructionTitle)
                .accessibilityIdentifier("message-standing-instruction-toggle")
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
                    accessibilityName: "Edit message",
                    disabled: actionsDisabled, action: editAction)
            }

            if let branchAction {
                MessagePillActionButton(
                    icon: "arrow.triangle.branch", label: "Branch",
                    help: "Edit this message in a branched copy of the conversation",
                    accessibilityName: "Branch message",
                    disabled: actionsDisabled, action: branchAction)
            }

            if let retryAction {
                MessagePillActionButton(
                    icon: "arrow.clockwise", label: "Retry",
                    help: "Generate a new reply to the same prompt",
                    accessibilityName: "Retry message",
                    disabled: actionsDisabled, action: retryAction)
            }

            MessageTimestampBadge(date: date)
        }
    }

    private var standingInstructionTitle: Text {
        isStandingInstruction
            ? Text("Unpin instruction", bundle: .module)
            : Text("Pin instruction", bundle: .module)
    }
}

/// The shared pill button the editing affordances render as. Matches the
/// existing copy/share pills' look without restating them.
///
/// `label`, `help` and `accessibilityName` are catalog keys; the name is its
/// own key rather than "label + message" so each language words it whole.
struct MessagePillActionButton: View {
    let icon: String
    let label: LocalizedStringKey
    let help: LocalizedStringKey
    let accessibilityName: LocalizedStringKey
    let disabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .themedFont(.tiny, weight: .medium)
                    .accessibilityHidden(true)
                Text(label, bundle: .module)
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
        .help(Text(help, bundle: .module))
        .accessibilityLabel(Text(accessibilityName, bundle: .module))
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
        .accessibilityLabel(Text("Share message", bundle: .module))
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
                (isSpeakingThis ? Text("Stop", bundle: .module) : Text("Read", bundle: .module))
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
        .help(isSpeakingThis
              ? Text("Stop reading out loud", bundle: .module)
              : Text("Read message out loud", bundle: .module))
        .accessibilityLabel(isSpeakingThis
                            ? Text("Stop reading message out loud", bundle: .module)
                            : Text("Read message out loud", bundle: .module))
    }
}

/// Timestamp badge showing human-readable relative time and exact time on hover.
struct MessageTimestampBadge: View {
    let date: Date
    @State private var isHovered = false
    @Environment(\.locale) private var locale

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: "clock")
                .themedFont(.tiny, weight: .medium)
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            Text(isHovered ? MessageTimestampFormatter.standardString(for: date, locale: locale) : MessageTimestampFormatter.relativeString(for: date, locale: locale))
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .lineLimit(1)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(Color.primary.opacity(isHovered ? 0.08 : 0.04), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .onHover { hovering in
            TSMotion.animate(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
        .help(MessageTimestampFormatter.exactString(for: date, locale: locale))
        .accessibilityLabel(Text(
            "Sent \(MessageTimestampFormatter.relativeString(for: date, locale: locale)), \(MessageTimestampFormatter.exactString(for: date, locale: locale))",
            bundle: .module))
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
            TSMotion.animate(.easeInOut(duration: 0.15)) {
                isCopied = true
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                    .themedFont(.tiny, weight: .medium)
                    .accessibilityHidden(true)
                (isCopied ? Text("Copied", bundle: .module) : Text("Copy", bundle: .module))
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
        .accessibilityLabel(isCopied ? Text("Copied", bundle: .module) : Text("Copy message", bundle: .module))
        .accessibilityHint(Text("Copies this message to the clipboard", bundle: .module))
        .task(id: isCopied) {
            guard isCopied else { return }
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            TSMotion.animate(.easeOut(duration: 0.15)) {
                isCopied = false
            }
        }
    }
}
