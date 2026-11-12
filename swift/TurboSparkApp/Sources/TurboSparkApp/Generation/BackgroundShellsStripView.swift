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

    @State private var shellToKill: BackgroundShellSummary?

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
            .confirmationDialog(
                Text("Kill", bundle: .module),
                isPresented: Binding(
                    get: { shellToKill != nil },
                    set: { if !$0 { shellToKill = nil } }
                ),
                presenting: shellToKill
            ) { target in
                Button(role: .destructive) {
                    model.killBackgroundShell(id: target.id)
                } label: {
                    Text("Kill", bundle: .module)
                }
                Button(role: .cancel) {
                    shellToKill = nil
                } label: {
                    Text("Cancel", bundle: .module)
                }
            } message: { _ in
                Text("Kill this background shell (its child processes are killed with it)", bundle: .module)
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
                shellToKill = shell
            } label: {
                Image(systemName: "stop.fill")
                    .themedFont(.tiny)
            }
            .appAction(.destructive, size: .small)
            .help(Text("Kill this background shell (its child processes are killed with it)", bundle: .module))
            .accessibilityLabel(Text("Kill background shell \(shell.id)", bundle: .module))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.appSurface)
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(theme.border, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text("Background shell \(shell.id) running, \(shell.commandHead)", bundle: .module))
        .accessibilityAction(named: Text("Kill", bundle: .module)) {
            shellToKill = shell
        }
    }
}
