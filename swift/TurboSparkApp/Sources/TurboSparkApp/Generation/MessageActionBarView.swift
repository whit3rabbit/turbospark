import AppKit
import SwiftUI

/// Hover action row under a message: icon-only copy, share, read out loud
/// and the editing actions, then the timestamp.
///
/// The optional actions are the message-editing affordances (`AppModel+
/// MessageEditing.swift`): variant navigation, in-place edit, branch, and
/// retry. Each is nil where it does not apply -- the row decides, so this
/// view stays a dumb icon strip.
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
        HStack(spacing: 2) {
            if variantCount > 1, let variantStep {
                MessageVariantSwitcherView(
                    position: variantPosition, count: variantCount, step: variantStep)
            }

            if let standingInstructionAction {
                AppIconButton(
                    isStandingInstruction ? "pin.fill" : "pin",
                    label: standingInstructionTitle,
                    action: standingInstructionAction)
                    .buttonStyle(MessageIconActionStyle(isActive: isStandingInstruction))
                    .disabled(actionsDisabled)
                    .accessibilityIdentifier("message-standing-instruction-toggle")
            }

            MessageCopyButton(text: text)

            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                MessageShareButton(text: text)
                MessageSpeechButton(text: text, messageID: messageID)
            }

            if let editAction {
                MessageIconActionButton(
                    icon: "pencil",
                    name: "Edit message",
                    hint: "Edit this message and regenerate the reply",
                    disabled: actionsDisabled, action: editAction)
            }

            if let branchAction {
                MessageIconActionButton(
                    icon: "arrow.triangle.branch",
                    name: "Branch message",
                    hint: "Edit this message in a branched copy of the conversation",
                    disabled: actionsDisabled, action: branchAction)
            }

            if let retryAction {
                MessageIconActionButton(
                    icon: "arrow.clockwise",
                    name: "Retry message",
                    hint: "Generate a new reply to the same prompt",
                    disabled: actionsDisabled, action: retryAction)
            }

            MessageTimestampBadge(date: date)
                .padding(.leading, 4)
        }
    }

    private var standingInstructionTitle: Text {
        isStandingInstruction
            ? Text("Unpin instruction", bundle: .module)
            : Text("Pin instruction", bundle: .module)
    }
}

/// An icon-only message action. `name` is the tooltip and the VoiceOver
/// name (through `AppIconButton`, so the two cannot drift); `hint` keeps the
/// longer explanation the old text pill showed as its tooltip.
///
/// `name` and `hint` are catalog keys; the name is its own key rather than
/// "label + message" so each language words it whole.
struct MessageIconActionButton: View {
    let icon: String
    let name: LocalizedStringKey
    let hint: LocalizedStringKey
    let disabled: Bool
    let action: () -> Void

    var body: some View {
        AppIconButton(icon, label: Text(name, bundle: .module), action: action)
            .buttonStyle(MessageIconActionStyle())
            .disabled(disabled)
            .accessibilityHint(Text(hint, bundle: .module))
    }
}

/// The shared look of the message actions: a 24 pt quiet glyph with a hover
/// fill. Icon-only replaced text pills ("Pin instruction", "Copy", "Share",
/// "Read") that made one row of a short message wider than the message.
/// The frame grows with the text size so Extra Large keeps a real target.
struct MessageIconActionStyle: ButtonStyle {
    /// Accent ink for a state that is on: pinned, just copied, speaking.
    var isActive = false

    func makeBody(configuration: Configuration) -> some View {
        MessageIconActionBody(configuration: configuration, isActive: isActive)
    }
}

private struct MessageIconActionBody: View {
    let configuration: ButtonStyleConfiguration
    let isActive: Bool
    @Environment(\.appTheme) private var theme
    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovered = false

    var body: some View {
        let side = 24 * AppMetrics.textScale(theme)
        configuration.label
            .labelStyle(.iconOnly)
            .themedFont(.small, weight: .medium)
            .foregroundStyle(isActive ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))
            .frame(width: side, height: side)
            .background(
                theme.foreground.opacity(isHovered || configuration.isPressed ? 0.07 : 0),
                in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .contentShape(Rectangle())
            .opacity(isEnabled ? 1 : 0.4)
            .onHover { isHovered = $0 }
            .tsAnimation(TSMotion.hover, value: isHovered)
    }
}

/// Share button opening the native macOS share picker.
struct MessageShareButton: View {
    let text: String

    var body: some View {
        ShareLink(item: text) {
            Label {
                Text("Share message", bundle: .module)
            } icon: {
                Image(systemName: "square.and.arrow.up")
            }
        }
        .buttonStyle(MessageIconActionStyle())
        .help(Text("Share message", bundle: .module))
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
        AppIconButton(
            isSpeakingThis ? "stop.fill" : "speaker.wave.2",
            label: isSpeakingThis
                ? Text("Stop reading message out loud", bundle: .module)
                : Text("Read message out loud", bundle: .module)
        ) {
            speechManager.toggleSpeech(text: text, messageID: messageID)
        }
        .buttonStyle(MessageIconActionStyle(isActive: isSpeakingThis))
    }
}

/// Timestamp badge showing human-readable relative time and exact time on hover.
struct MessageTimestampBadge: View {
    let date: Date
    @State private var isHovered = false
    @Environment(\.locale) private var locale

    var body: some View {
        Text(isHovered ? MessageTimestampFormatter.standardString(for: date, locale: locale) : MessageTimestampFormatter.relativeString(for: date, locale: locale))
            .themedFont(.tiny)
            .foregroundStyle(.appSecondary)
            .lineLimit(1)
            .onHover { hovering in
                TSMotion.animate(TSMotion.hover) {
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
        AppIconButton(
            isCopied ? "checkmark" : "doc.on.doc",
            label: isCopied ? Text("Copied", bundle: .module) : Text("Copy message", bundle: .module)
        ) {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            TSMotion.animate(.easeInOut(duration: 0.15)) {
                isCopied = true
            }
        }
        .buttonStyle(MessageIconActionStyle(isActive: isCopied))
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
