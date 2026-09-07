import AppKit
import SwiftUI

/// Floating action bar containing copy, share, read out loud, and timestamp actions for a message.
struct MessageActionBarView: View {
    let text: String
    let messageID: UUID
    let date: Date

    var body: some View {
        HStack(spacing: 6) {
            MessageCopyButton(text: text)

            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                MessageShareButton(text: text)
                MessageSpeechButton(text: text, messageID: messageID)
            }

            MessageTimestampBadge(date: date)
        }
    }
}

/// Share button opening the native macOS share picker.
struct MessageShareButton: View {
    let text: String

    var body: some View {
        ShareLink(item: text) {
            HStack(spacing: 4) {
                Image(systemName: "square.and.arrow.up")
                    .themedFont(points: 11, weight: .medium)
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
        .help("Share message")
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
                    .themedFont(points: 11, weight: .medium)
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
                .themedFont(points: 10, weight: .medium)
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            Text(isHovered ? MessageTimestampFormatter.standardString(for: date) : MessageTimestampFormatter.relativeString(for: date))
                .themedFont(.tiny)
                .foregroundStyle(.secondary)
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
                    .themedFont(points: 11, weight: .medium)
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
        .help("Copy message text")
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
