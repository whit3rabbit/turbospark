import SwiftUI

/// The `/diff`, `/log` and `/prs` sheet (qwen-code's GitDialog parity): one
/// container with three views, presented from `OutputPaneView` while
/// `model.showGitSheet` is true. Every command selects its own tab when it
/// runs, so switching commands mid-flight keeps the chrome.
struct GitInfoSheet: View {
    @Environment(\.appTheme) private var theme
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "arrow.triangle.branch")
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text("Git", bundle: .module)
                    .themedFont(.callout, weight: .semibold)
                Picker(selection: $model.gitInfoTab) {
                    // The tabs are named by the slash command that fills
                    // them (`/diff`, `/log`, `/prs`), so they stay as typed.
                    ForEach(AppModel.GitInfoTab.allCases) { tab in
                        Text(verbatim: tab.rawValue).tag(tab)
                    }
                } label: {
                    Text("Git view", bundle: .module)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(minWidth: 150, idealWidth: 220, maxWidth: 260)
                Spacer()
                if let branch = model.worktree?.currentBranch, !branch.isEmpty {
                    Text(branch)
                        .themedCode(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                AppIconButton("xmark.circle.fill", label: Text("Close git view", bundle: .module)) {
                    dismiss()
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)

            Divider()

            if let error = model.gitInfoError, !error.isEmpty {
                errorBanner(error)
            }

            ScrollView {
                Group {
                    switch model.gitInfoTab {
                    case .diff:
                        diffBody
                    case .log:
                        logBody
                    case .prs:
                        prsBody
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(minWidth: 540, idealWidth: 680, maxWidth: 900, minHeight: 400, idealHeight: 520, maxHeight: 800)
        .onChange(of: model.gitInfoTab) { _, tab in
            // Tabs are fetched by their slash command, never by this sheet,
            // so an authoritative empty state ("No commits yet.") would
            // otherwise describe data that was never asked for. Refetch an
            // empty tab; a populated one is left alone.
            model.refreshGitInfoTabIfEmpty(tab)
        }
    }

    @ViewBuilder
    private var diffBody: some View {
        if model.isLoadingGitInfo {
            AppProgressBar(value: nil, label: Text("Loading diff...", bundle: .module))
                .padding(.top, 24)
                .padding(.horizontal, 20)
        } else if model.gitDiffText.isEmpty {
            emptyState(Text("No working-tree changes against HEAD.", bundle: .module), systemImage: "checkmark.circle")
        } else {
            // Split at the same seam the text was assembled at: the --stat
            // block first, then the full diff. The diff lines get +/-/heading
            // coloring; the stat block reads better plain.
            VStack(alignment: .leading, spacing: 12) {
                if let range = model.gitDiffText.range(of: "\n\n") {
                    Text(String(model.gitDiffText[..<range.lowerBound]))
                        .themedCode(.tiny)
                        .foregroundStyle(.appSecondary)
                        .textSelection(.enabled)
                    diffLines(String(model.gitDiffText[range.upperBound...]))
                } else {
                    diffLines(model.gitDiffText)
                }
            }
        }
    }

    private func diffLines(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(text.split(separator: "\n", omittingEmptySubsequences: false).enumerated()), id: \.offset) { _, line in
                Text(String(line))
                    .themedCode(.tiny)
                    .foregroundStyle(lineColor(line))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .background(.appElevated)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private func lineColor(_ line: Substring) -> Color {
        if line.hasPrefix("+++") || line.hasPrefix("---") || line.hasPrefix("diff ") || line.hasPrefix("index ") {
            return theme.secondaryText
        }
        if line.hasPrefix("+") { return theme.success }
        if line.hasPrefix("-") { return theme.warning }
        if line.hasPrefix("@@") { return theme.accent }
        return theme.foreground
    }

    @ViewBuilder
    private var logBody: some View {
        if model.isLoadingGitInfo {
            AppProgressBar(value: nil, label: Text("Loading...", bundle: .module))
                .padding(.top, 24)
                .padding(.horizontal, 20)
        } else if model.gitCommits.isEmpty {
            emptyState(Text("No commits yet.", bundle: .module), systemImage: "clock")
        } else {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(model.gitCommits) { commit in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(commit.shortHash)
                            .themedCode(.tiny, weight: .semibold)
                            .foregroundStyle(.appAccent)
                            .textSelection(.enabled)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(commit.summary)
                                .themedFont(.small, weight: .medium)
                                .lineLimit(2)
                            Text(verbatim: "\(commit.author) - \(commit.relativeDate)")
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }
                        Spacer(minLength: 8)
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(commit.hash, forType: .string)
                            model.showToast(AppStrings.localized("Copied \(commit.shortHash)."), style: .success)
                        } label: {
                            Image(systemName: "doc.on.doc")
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }
                        .buttonStyle(.plain)
                        .help(Text("Copy full commit SHA", bundle: .module))
                        .accessibilityLabel(Text("Copy SHA for \(commit.summary)", bundle: .module))
                    }
                    .padding(.vertical, 3)
                }
            }
        }
    }

    @ViewBuilder
    private var prsBody: some View {
        if model.isLoadingGitInfo {
            AppProgressBar(value: nil, label: Text("Loading...", bundle: .module))
                .padding(.top, 24)
                .padding(.horizontal, 20)
        } else if model.gitPullRequests.isEmpty {
            emptyState(
                model.gitInfoError == nil
                    ? Text("No open pull requests.", bundle: .module)
                    : Text("No pull requests loaded.", bundle: .module),
                systemImage: "arrow.triangle.pull")
        } else {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(model.gitPullRequests) { pr in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        prStateIcon(pr.state)
                        Text(verbatim: "#\(pr.number)")
                            .themedCode(.tiny, weight: .semibold)
                            .foregroundStyle(.appAccent)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(pr.title)
                                .themedFont(.small, weight: .medium)
                                .lineLimit(2)
                            Text(pr.headBranch)
                                .themedCode(.tiny)
                                .foregroundStyle(.appSecondary)
                        }
                        Spacer(minLength: 8)
                        if let url = URL(string: pr.url) {
                            Button {
                                NSWorkspace.shared.open(url)
                            } label: {
                                Image(systemName: "arrow.up.forward.app")
                                    .themedFont(.tiny)
                                    .foregroundStyle(.appSecondary)
                            }
                            .buttonStyle(.plain)
                            .help(Text("Open in browser", bundle: .module))
                            .accessibilityLabel(Text("Open pull request #\(pr.number)", bundle: .module))
                        }
                    }
                    .padding(.vertical, 3)
                }
            }
        }
    }

    private func prStateIcon(_ state: String) -> some View {
        let normalized = state.uppercased()
        let (symbol, color): (String, Color) = {
            switch normalized {
            case "MERGED": return ("arrow.triangle.merge", .purple)
            case "CLOSED": return ("xmark.circle", theme.danger)
            default: return ("arrow.triangle.pull", theme.success)
            }
        }()
        // `gh` reports the state in English capitals; the three it can
        // return are named in the UI language, anything newer as sent.
        let stateText: Text = switch normalized {
        case "OPEN": Text("Open", bundle: .module)
        case "MERGED": Text("Merged", bundle: .module)
        case "CLOSED": Text("Closed", bundle: .module)
        default: Text(verbatim: state)
        }
        return Image(systemName: symbol)
            .themedFont(.small)
            .foregroundStyle(color)
            .help(stateText)
            .accessibilityLabel(Text("Pull request state", bundle: .module))
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(AppAttention.color)
                .accessibilityHidden(true)
            Text(message)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .lineLimit(3)
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(AppAttention.color.opacity(0.08))
    }

    private func emptyState(_ text: Text, systemImage: String) -> some View {
        AppEmptyState(systemImage: systemImage, title: text)
    }
}
