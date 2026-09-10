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
                Picker("", selection: $model.gitInfoTab) {
                    ForEach(AppModel.GitInfoTab.allCases) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 220)
                Spacer()
                if let branch = model.worktree?.currentBranch, !branch.isEmpty {
                    Text(branch)
                        .themedCode(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Close")
                .accessibilityLabel("Close git view")
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
        .frame(width: 680, height: 520)
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
            ProgressView()
                .padding(.top, 24)
        } else if model.gitDiffText.isEmpty {
            emptyState("No working-tree changes against HEAD.", systemImage: "checkmark.circle")
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
        .background(Color.primary.opacity(0.03))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private func lineColor(_ line: Substring) -> Color {
        if line.hasPrefix("+++") || line.hasPrefix("---") || line.hasPrefix("diff ") || line.hasPrefix("index ") {
            return .secondary
        }
        if line.hasPrefix("+") { return .green }
        if line.hasPrefix("-") { return .red }
        if line.hasPrefix("@@") { return TurboSparkTheme.accentColor }
        return .primary
    }

    @ViewBuilder
    private var logBody: some View {
        if model.isLoadingGitInfo {
            ProgressView()
                .padding(.top, 24)
        } else if model.gitCommits.isEmpty {
            emptyState("No commits yet.", systemImage: "clock")
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
                            Text("\(commit.author) - \(commit.relativeDate)", bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }
                        Spacer(minLength: 8)
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(commit.hash, forType: .string)
                            model.showToast("Copied \(commit.shortHash).", style: .success)
                        } label: {
                            Image(systemName: "doc.on.doc")
                                .themedFont(.tiny)
                                .foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                        .help("Copy full commit SHA")
                        .accessibilityLabel("Copy SHA for \(commit.summary)")
                    }
                    .padding(.vertical, 3)
                }
            }
        }
    }

    @ViewBuilder
    private var prsBody: some View {
        if model.isLoadingGitInfo {
            ProgressView()
                .padding(.top, 24)
        } else if model.gitPullRequests.isEmpty {
            emptyState(
                model.gitInfoError == nil ? "No open pull requests." : "No pull requests loaded.",
                systemImage: "arrow.triangle.pull")
        } else {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(model.gitPullRequests) { pr in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        prStateIcon(pr.state)
                        Text("#\(pr.number)", bundle: .module)
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
                                    .foregroundStyle(.tertiary)
                            }
                            .buttonStyle(.plain)
                            .help("Open in browser")
                            .accessibilityLabel("Open pull request #\(pr.number)")
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
            case "CLOSED": return ("xmark.circle", .red)
            default: return ("arrow.triangle.pull", .green)
            }
        }()
        return Image(systemName: symbol)
            .themedFont(.small)
            .foregroundStyle(color)
            .help(state)
            .accessibilityLabel("Pull request state \(state)")
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(.orange)
                .accessibilityHidden(true)
            Text(message)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .lineLimit(3)
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(Color.orange.opacity(0.08))
    }

    private func emptyState(_ text: String, systemImage: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: systemImage)
                .font(theme.ui(.title2))
                .foregroundStyle(.tertiary)
            Text(text)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.top, 40)
    }
}
