import SwiftUI

/// The workspace tab strip (ZP-21): one tab per open project, under the top
/// bar and above the working panes. It is a row of the top-bar band, not a
/// fifth band (DESIGN.md section 3): it answers "which project am I in", and
/// it is drawn only while two or more tabs are open, so the window is
/// unchanged for anyone who never opens a second tab.
///
/// The badge on each tab is the strongest ZP-02 state among that project's
/// chats (needs approval, generating, unread), shown as a glyph and read by
/// VoiceOver as text, never color alone.
struct WorkspaceTabStripView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ScaledMetric private var stripHeight: CGFloat = 34

    var body: some View {
        HStack(spacing: 4) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(model.workspaceTabs.tabs) { tab in
                        if let project = model.projects.first(where: { $0.id == tab.projectID }) {
                            WorkspaceTabItemView(
                                model: model, project: project,
                                isSelected: model.workspaceTabs.selectedProjectID == project.id)
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
            openMenu
        }
        .padding(.horizontal, 10)
        .frame(minHeight: stripHeight)
        .background(.appSurface)
        .overlay(alignment: .bottom) {
            Rectangle().fill(.appBorder).frame(height: AppChromeLayout.dividerWidth)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Workspace tabs", bundle: .module))
    }

    /// Projects without a tab, offered by the plus menu.
    private var closedProjects: [AppProject] {
        model.projects.filter { !model.workspaceTabs.contains($0.id) }
    }

    private var openMenu: some View {
        Menu {
            ForEach(closedProjects) { project in
                Button {
                    model.openWorkspaceTab(projectID: project.id)
                } label: {
                    Text(verbatim: project.name)
                }
            }
        } label: {
            Label { Text("Open Project in New Tab", bundle: .module) } icon: {
                Image(systemName: "plus")
            }
            .labelStyle(.iconOnly)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(closedProjects.isEmpty)
        .help(Text("Open Project in New Tab", bundle: .module))
    }
}

/// One tab: select button, status glyph, close button.
private struct WorkspaceTabItemView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let project: AppProject
    let isSelected: Bool
    @State private var isHovered = false

    private var badge: WorkspaceTabBadge { model.workspaceTabBadge(for: project.id) }

    var body: some View {
        HStack(spacing: 4) {
            Button {
                model.activateWorkspaceTab(projectID: project.id)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isSelected ? "folder.fill" : "folder")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                        .accessibilityHidden(true)
                    Text(verbatim: project.name)
                        .font(theme.ui(.small, weight: isSelected ? .semibold : nil))
                        .foregroundStyle(.appText)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    badgeGlyph
                }
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .help(Text("Switch to \(project.name)", bundle: .module))
            .accessibilityLabel(Text("Project: \(project.name)", bundle: .module))
            .accessibilityValue(badgeAccessibilityValue)
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)

            Button {
                model.closeWorkspaceTab(projectID: project.id)
            } label: {
                Label { Text("Close Tab", bundle: .module) } icon: {
                    Image(systemName: "xmark")
                }
                .labelStyle(.iconOnly)
                .font(theme.ui(.micro, weight: .medium))
                .foregroundStyle(.appSecondary)
                // Always laid out, so a hover does not shift the title; shown
                // on the selected tab, on hover, and to VoiceOver always.
                .opacity(isSelected || isHovered ? 1 : 0)
            }
            .buttonStyle(.plain)
            .help(Text("Close Tab", bundle: .module))
            .accessibilityLabel(Text("Close \(project.name) tab", bundle: .module))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .fill(isSelected ? AnyShapeStyle(.appElevated) : AnyShapeStyle(Color.clear))
        )
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(isSelected ? AnyShapeStyle(.appBorder) : AnyShapeStyle(Color.clear), lineWidth: 1)
        )
        .onHover { isHovered = $0 }
        .tsAnimation(TSMotion.hover, value: isHovered)
        .contextMenu {
            Button {
                model.closeWorkspaceTab(projectID: project.id)
            } label: {
                Text("Close Tab", bundle: .module)
            }
        }
    }

    @ViewBuilder
    private var badgeGlyph: some View {
        switch badge {
        case .pending:
            Image(systemName: "hand.raised.fill")
                .font(theme.ui(.tiny))
                .foregroundStyle(AppAttention.color)
                .help(Text("Needs Approval", bundle: .module))
                .accessibilityHidden(true)
        case .running:
            Image(systemName: "flame.fill")
                .font(theme.ui(.tiny))
                .foregroundStyle(theme.accent)
                .tsPulse(active: true)
                .help(Text("Generating", bundle: .module))
                .accessibilityHidden(true)
        case .unread:
            Circle()
                .fill(theme.accent)
                .frame(width: 8, height: 8)
                .help(Text("Unread", bundle: .module))
                .accessibilityHidden(true)
        case .none:
            EmptyView()
        }
    }

    private var badgeAccessibilityValue: Text {
        switch badge {
        case .pending: return Text("Needs Approval", bundle: .module)
        case .running: return Text("Generating", bundle: .module)
        case .unread: return Text("Unread", bundle: .module)
        case .none: return Text(verbatim: "")
        }
    }
}
