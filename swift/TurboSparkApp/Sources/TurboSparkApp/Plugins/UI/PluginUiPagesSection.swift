import SwiftUI

/// The "UI pages" block of a plugin's detail pane: the per-plugin opt-in and
/// one Open row per declared surface. Shown only when the manifest declares
/// surfaces. Pages stay off until the person turns the plugin's switch on.
@MainActor
struct PluginUiPagesSection: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var optIn: PluginUiOptInStore
    let plugin: LoadedPlugin

    var body: some View {
        let surfaces = plugin.manifest.uiSurfaces
        if !surfaces.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("UI pages", bundle: .module)
                    .font(theme.ui(.base, weight: .semibold))
                    .accessibilityAddTraits(.isHeader)
                Toggle(isOn: Binding(
                    get: { optIn.isEnabled(plugin.id) },
                    set: { model.setPluginUiOptIn($0, plugin: plugin) })
                ) {
                    Text("Allow this plugin to show UI pages", bundle: .module)
                }
                .settingsControl("Allow this plugin to show UI pages", pane: .plugins, timing: .immediate)
                ForEach(surfaces) { surface in
                    HStack {
                        Text(verbatim: surface.title)
                            .font(theme.ui(.base))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button {
                            model.openPluginPageReportingRefusal(pluginID: plugin.id, surfaceID: surface.id)
                        } label: {
                            Text("Open", bundle: .module)
                        }
                        .appAction(.secondary, size: .small)
                        .disabled(!optIn.isEnabled(plugin.id))
                        .help(optIn.isEnabled(plugin.id)
                            ? Text("Open this page beside the chat", bundle: .module)
                            : Text("Turn on UI pages for this plugin first.", bundle: .module))
                    }
                }
                Text("Pages run in a sandbox: no network, no files beyond the plugin's own, and no camera, microphone or location. Tools a page calls still go through the normal permission prompts.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
    }
}
