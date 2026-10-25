import SwiftUI

/// Settings > Browser. A grouped Form like General, one computed property
/// per `Section` for the macOS 14 type-checker reason documented on
/// `EngineSettingsPaneView.body`. The Settings shell already shows the
/// pane's title, so the pane starts with its first section.
@MainActor
struct BrowserSettingsPaneView: View {
    @StateObject private var viewModel: BrowserSettingsPaneViewModel
    @State private var showingFirstEnableOnboarding = false
    @State private var showingChromeImportPreview = false
    private let model: AppModel

    init(model: AppModel) {
        self.model = model
        let viewModel = BrowserSettingsPaneViewModel(
            settings: model.browserSettings, latest: { model.browserSettings }
        ) { settings in
            model.browserSettings = settings
            model.persistSettingsDebounced()
        }
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        Form {
            automationSection
            viewportSection
            bookmarksSection
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
        .alert(Text("Enable browser automation", bundle: .module), isPresented: $showingFirstEnableOnboarding) {
            Button(role: .cancel) {
                viewModel.cancelFirstEnable()
            } label: {
                Text("Cancel", bundle: .module)
            }
            Button {
                viewModel.confirmFirstEnable()
            } label: {
                Text("Enable browser automation", bundle: .module)
            }
        } message: {
            Text("Browser tools can navigate, click, type, take screenshots, and read page structure. They ask before acting. A project grant matches one exact HTTP(S) origin, and each project can hold 256 grants.", bundle: .module)
        }
        .sheet(isPresented: $showingChromeImportPreview) {
            ChromeBookmarkImportPreviewView(model: model)
                .frame(minWidth: 700, minHeight: 500)
                .appThemed()
        }
    }

    private var automationSection: some View {
        Section(header: Text("Browser Automation", bundle: .module)) {
            Toggle(
                isOn: Binding(
                    get: { viewModel.settings.enabled },
                    set: { enabled in
                        if viewModel.requestEnabled(enabled) == .onboardingRequired {
                            showingFirstEnableOnboarding = true
                        }
                    }
                )
            ) {
                Text("Enable browser automation", bundle: .module)
            }

            Picker(
                selection: Binding(
                    get: { viewModel.settings.dialogPolicy },
                    set: { viewModel.setDialogPolicy($0) }
                )
            ) {
                Text("Ask for each page dialog", bundle: .module).tag(BrowserDialogPolicy.ask)
                Text("Automatically accept page dialogs", bundle: .module).tag(BrowserDialogPolicy.autoAccept)
                Text("Automatically dismiss page dialogs", bundle: .module).tag(BrowserDialogPolicy.autoDismiss)
            } label: {
                Text("Dialog policy", bundle: .module)
            }
            .pickerStyle(.menu)

            // The same explanation the first-enable alert gives, kept in view
            // so the permission model is readable without toggling.
            Text("Browser tools can navigate, click, type, take screenshots, and read page structure. They ask before acting. A project grant matches one exact HTTP(S) origin, and each project can hold 256 grants.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }

    private var viewportSection: some View {
        Section(header: Text("Default viewport", bundle: .module)) {
            viewportRow(
                Text("Width", bundle: .module),
                unit: "px",
                value: Binding(
                    get: { viewModel.settings.defaultViewport.width },
                    set: { value in
                        viewModel.setDefaultViewport(
                            width: value,
                            height: viewModel.settings.defaultViewport.height,
                            zoom: viewModel.settings.defaultViewport.zoom
                        )
                    }
                )
            )

            viewportRow(
                Text("Height", bundle: .module),
                unit: "px",
                value: Binding(
                    get: { viewModel.settings.defaultViewport.height },
                    set: { value in
                        viewModel.setDefaultViewport(
                            width: viewModel.settings.defaultViewport.width,
                            height: value,
                            zoom: viewModel.settings.defaultViewport.zoom
                        )
                    }
                )
            )

            viewportZoomRow
        }
    }

    private var bookmarksSection: some View {
        Section(header: Text("Bookmarks", bundle: .module)) {
            Button {
                showingChromeImportPreview = true
            } label: {
                Label {
                    Text("Import Chrome bookmarks", bundle: .module)
                } icon: {
                    Image(systemName: "bookmark")
                }
            }
            .appAction(.secondary)
        }
    }

    // MARK: - Viewport rows

    /// The row label is the visible name; the field carries the same name
    /// for VoiceOver through its label closure. A width range rather than
    /// the old fixed 100pt, so a large text size or a long translation
    /// grows the field instead of clipping the number.
    private func viewportRow(
        _ label: Text,
        unit: String,
        value: Binding<Int>
    ) -> some View {
        LabeledContent {
            HStack(spacing: 6) {
                TextField(value: value, format: .number) {
                    label
                }
                .modifier(BrowserViewportFieldStyle())
                Self.unitText(unit)
            }
        } label: {
            label
        }
    }

    private var viewportZoomRow: some View {
        LabeledContent {
            HStack(spacing: 6) {
                TextField(
                    value: Binding(
                        get: { viewModel.settings.defaultViewport.zoom },
                        set: { zoom in
                            viewModel.setDefaultViewport(
                                width: viewModel.settings.defaultViewport.width,
                                height: viewModel.settings.defaultViewport.height,
                                zoom: zoom
                            )
                        }
                    ),
                    format: .number
                ) {
                    Text("Zoom", bundle: .module)
                }
                .modifier(BrowserViewportFieldStyle())
                // The multiplication sign, as the viewport popover shows it.
                Self.unitText("\u{00D7}")
            }
        } label: {
            Text("Zoom", bundle: .module)
        }
    }

    /// A unit glyph, not a word: nothing to translate.
    private static func unitText(_ unit: String) -> some View {
        Text(verbatim: unit)
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .fixedSize()
            .frame(minWidth: 18, alignment: .leading)
            .accessibilityHidden(true)
    }
}

private struct BrowserViewportFieldStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .labelsHidden()
            .textFieldStyle(.roundedBorder)
            .frame(minWidth: 80, idealWidth: 100, maxWidth: 140)
    }
}
