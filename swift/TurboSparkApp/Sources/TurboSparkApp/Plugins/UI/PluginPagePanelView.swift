import SwiftUI

/// The right-column panel that shows one plugin page. Chrome only: the page
/// itself, its failure card and its sandbox live in `PluginPageWebView.swift`.
@MainActor
struct PluginPagePanelView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var session: PluginPageSession

    var body: some View {
        VStack(spacing: 0) {
            AppAdaptiveHeader {
                HStack(spacing: 8) {
                    Image(systemName: "puzzlepiece.extension")
                        .themedFont(.callout)
                        .foregroundStyle(.appAccent)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(verbatim: session.surface.title)
                            .themedFont(.small, weight: .semibold)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .help(session.surface.title)
                            .accessibilityAddTraits(.isHeader)
                        Text(verbatim: session.pluginName)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                    }
                }
            } actions: {
                AppIconButton("xmark", label: Text("Close page", bundle: .module)) {
                    model.closePluginPage(session)
                }
                .help(Text("Close page", bundle: .module))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Divider()
            PluginPageContentView(session: session)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(TurboSparkTheme.barBackgroundColor(isDark: theme.isDark))
        .onExitCommand { model.closePluginPage(session) }
    }
}
