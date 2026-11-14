import SwiftUI

/// The reply row's sources chip (OC-09): a glyph and the count, shown only
/// when the turn used web sources. It opens `SourcesPopoverView`.
struct SourcesChipView: View {
    @ObservedObject var model: AppModel
    let sources: TurnSources
    @State private var showingPopover = false

    var body: some View {
        let count = sources.totalCount
        AppChip(
            Text("Sources \(count)", bundle: .module),
            systemImage: "link",
            isOn: showingPopover
        ) {
            showingPopover.toggle()
        }
        .help(Text("Show the web sources this reply used", bundle: .module))
        .accessibilityLabel(Text("Web sources used in this reply: \(count)", bundle: .module))
        .accessibilityHint(Text("Opens the list of sources", bundle: .module))
        .popover(isPresented: $showingPopover, arrowEdge: .bottom) {
            SourcesPopoverView(model: model, sources: sources) { showingPopover = false }
        }
    }
}

/// The deduplicated sources of one turn, grouped by domain. Each row opens
/// the page in the browser pane (a user action: the pane's user navigation
/// path, not the agent's approval ladder) and has a copy action.
struct SourcesPopoverView: View {
    @ObservedObject var model: AppModel
    let sources: TurnSources
    let dismiss: () -> Void
    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Sources", bundle: .module)
                .themedFont(.callout, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(sources.groups) { group in
                        VStack(alignment: .leading, spacing: 4) {
                            groupHeader(group)
                            ForEach(group.sources) { source in
                                SourceRowView(model: model, source: source, dismiss: dismiss)
                            }
                        }
                    }
                    if sources.overflowCount > 0 {
                        Text("\(sources.overflowCount) more sources not shown", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    }
                }
            }
            .frame(maxHeight: 360)

            if !model.canOpenSourcesInBrowserPane {
                Text("Turn on the browser in Settings to open sources in the browser pane.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        // minWidth/maxWidth rather than a fixed frame so Extra Large text and
        // long translations wrap instead of clipping (DESIGN.md, section 3).
        .frame(minWidth: 280, idealWidth: 360, maxWidth: 440, alignment: .leading)
        .padding(16)
    }

    private func groupHeader(_ group: TurnSources.Group) -> some View {
        HStack(spacing: 6) {
            Text(verbatim: group.domain)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Text(verbatim: "\(group.sources.count)")
                .themedFont(.small)
                .monospacedDigit()
                .foregroundStyle(.appSecondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}

private struct SourceRowView: View {
    @ObservedObject var model: AppModel
    let source: TurnSource
    let dismiss: () -> Void
    @Environment(\.appTheme) private var theme

    private var kindLabel: String {
        source.kind == .fetchedPage
            ? AppStrings.localized("Fetched page")
            : AppStrings.localized("Search result")
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Button {
                if model.openSourceInBrowserPane(source) { dismiss() }
            } label: {
                HStack(alignment: .top, spacing: 8) {
                    // Glyph plus the combined VoiceOver text tell a fetched
                    // page from a search result; never the glyph alone.
                    Image(systemName: source.kind == .fetchedPage ? "doc.text" : "magnifyingglass")
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(.appSecondary)
                        .padding(.top, 3)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(verbatim: source.title)
                            .themedFont(.small, weight: .medium)
                            .foregroundStyle(.appText)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                        Text(verbatim: source.domain + source.displayPath)
                            .themedCode(.tiny)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!model.canOpenSourcesInBrowserPane)
            .help(Text("Open in browser pane", bundle: .module))
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(Text(verbatim: "\(source.title), \(source.domain), \(kindLabel)"))
            .accessibilityHint(Text("Open in browser pane", bundle: .module))

            AppIconButton("doc.on.doc", label: Text("Copy URL", bundle: .module)) {
                model.copySourceURL(source)
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 2)
    }
}
