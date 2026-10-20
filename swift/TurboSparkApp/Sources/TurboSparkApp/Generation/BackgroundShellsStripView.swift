import AppKit
import SwiftUI
import TurboSpark

/// The strip of RUNNING background shells for the selected chat, one row
/// each with a Kill button. `KillShell` has always existed for the model;
/// this is the user's equivalent -- a hung `yes` loop or dev server is
/// visible and endable without asking the assistant or quitting the app.
/// Finished shells are deliberately absent: their output reaches the model
/// through `BashOutput`, and a transcript row per finished background
/// command would grow without bound. Killed shells' descendants die with
/// them (`ProcessExecutor.terminateAndReap` is a tree kill).
struct BackgroundShellsStripView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    /// Same visibility rule as the agent strip: unscoped shells (a run
    /// whose chat was captured as nil) show everywhere; a chat-scoped shell
    /// shows only in its own chat.
    private var shellsForChat: [BackgroundShellSummary] {
        model.backgroundShellSummaries
            .filter { $0.chatID == nil || $0.chatID == model.selectedChatID }
    }

    var body: some View {
        let shells = shellsForChat
        if !shells.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(shells) { shell in
                    shellRow(shell)
                }
            }
        }
    }

    private func shellRow(_ shell: BackgroundShellSummary) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "terminal")
                .themedFont(.small, weight: .bold)
                .foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                if let description = shell.description, !description.isEmpty {
                    Text(description)
                        .themedFont(.small, weight: .semibold)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Text(shell.commandHead)
                        .font(theme.code(.small))
                        .foregroundStyle(theme.metadataForeground)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                } else {
                    Text(shell.commandHead)
                        .font(theme.code(.small))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }
            }
            Spacer(minLength: 8)
            TaskProgressFlameIcon(size: 14)
            Text(shell.id)
                .themedFont(.small, weight: .medium)
                .foregroundStyle(theme.metadataForeground)
            Text(shell.startedAt, style: .timer)
                .themedFont(.small)
                .foregroundStyle(theme.metadataForeground)
            Button(role: .destructive) {
                model.killBackgroundShell(id: shell.id)
            } label: {
                Image(systemName: "stop.fill")
                    .themedFont(.tiny)
            }
            .buttonStyle(.plain)
            .foregroundStyle(Color.red.opacity(0.8))
            .help(Text("Kill this background shell (its child processes are killed with it)", bundle: .module))
            .accessibilityLabel(Text("Kill background shell \(shell.id)", bundle: .module))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.appSurface.opacity(0.7))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(.appAccent.opacity(0.45), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text("Background shell \(shell.id) running, \(shell.commandHead)", bundle: .module))
    }
}
