import SwiftUI

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
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Browser", bundle: .module)
                        .themedFont(.title2, weight: .semibold)
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
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Dialog policy", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                    Picker(
                        "Dialog policy",
                        selection: Binding(
                            get: { viewModel.settings.dialogPolicy },
                            set: { viewModel.setDialogPolicy($0) }
                        )
                    ) {
                        Text("Ask for each page dialog", bundle: .module).tag(BrowserDialogPolicy.ask)
                        Text("Automatically accept page dialogs", bundle: .module).tag(BrowserDialogPolicy.autoAccept)
                        Text("Automatically dismiss page dialogs", bundle: .module).tag(BrowserDialogPolicy.autoDismiss)
                    }
                    .pickerStyle(.menu)
                    .accessibilityLabel(Text("Dialog policy", bundle: .module))
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Default viewport", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                    HStack(spacing: 12) {
                        viewportField("Width") { value in
                            viewModel.setDefaultViewport(
                                width: value,
                                height: viewModel.settings.defaultViewport.height,
                                zoom: viewModel.settings.defaultViewport.zoom
                            )
                        } value: { viewModel.settings.defaultViewport.width }

                        viewportField("Height") { value in
                            viewModel.setDefaultViewport(
                                width: viewModel.settings.defaultViewport.width,
                                height: value,
                                zoom: viewModel.settings.defaultViewport.zoom
                            )
                        } value: { viewModel.settings.defaultViewport.height }

                        viewportZoomField
                    }
                }

                Button {
                    showingChromeImportPreview = true
                } label: {
                    Label {
                        Text("Import Chrome bookmarks", bundle: .module)
                    } icon: {
                        Image(systemName: "bookmark")
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(24)
        }
        .alert("Enable browser automation", isPresented: $showingFirstEnableOnboarding) {
            Button("Cancel", role: .cancel) {
                viewModel.cancelFirstEnable()
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

    private func viewportField(
        _ label: String,
        setValue: @escaping (Int) -> Void,
        value: @escaping () -> Int
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(LocalizedStringKey(label), bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
            TextField(
                label,
                value: Binding(get: value, set: setValue),
                format: .number
            )
            .textFieldStyle(.roundedBorder)
            .frame(width: 100)
        }
    }

    private var viewportZoomField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Zoom", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
            TextField(
                "Zoom",
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
            )
            .textFieldStyle(.roundedBorder)
            .frame(width: 100)
        }
    }
}
