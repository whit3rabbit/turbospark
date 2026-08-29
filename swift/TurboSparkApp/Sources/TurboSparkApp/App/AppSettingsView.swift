import SwiftUI

public struct AppSettingsView: View {
    @ObservedObject var model: AppModel
    @AppStorage(AppAppearance.storageKey)
    private var appearanceRawValue = AppAppearance.system.rawValue
    @AppStorage(AppTextSize.storageKey)
    private var textSizeRawValue = AppTextSize.standard.rawValue
    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        TabView {
            generalSettingsTab
                .tabItem {
                    Label("General", systemImage: "gearshape")
                }

            engineSettingsTab
                .tabItem {
                    Label("Engine", systemImage: "cpu")
                }
        }
        .frame(width: 520, height: 380)
        .padding(20)
    }

    private var generalSettingsTab: some View {
        Form {
            Section("Language & Localization") {
                Picker("Language", selection: $languageRawValue) {
                    ForEach(AppLanguage.allCases) { language in
                        Text(language.label).tag(language.rawValue)
                    }
                }
                .pickerStyle(.menu)

                if let keyboard = LanguageDetector.currentKeyboardLayoutName() {
                    HStack {
                        Text("Active Keyboard Layout:")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(keyboard)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("Appearance & Text Size") {
                Picker("Appearance Theme", selection: $appearanceRawValue) {
                    ForEach(AppAppearance.allCases) { appearance in
                        Label(appearance.label, systemImage: appearance.systemImage)
                            .tag(appearance.rawValue)
                    }
                }
                .pickerStyle(.segmented)

                Picker("Text Readability", selection: $textSizeRawValue) {
                    ForEach(AppTextSize.allCases) { size in
                        Text(size.label).tag(size.rawValue)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
        .formStyle(.grouped)
    }

    private var engineSettingsTab: some View {
        Form {
            Section("Generation Defaults") {
                HStack {
                    Text("Temperature")
                    Spacer()
                    Slider(value: $model.temperature, in: 0.0...1.5, step: 0.05)
                        .frame(width: 160)
                    Text(String(format: "%.2f", model.temperature))
                        .monospacedDigit()
                        .frame(width: 40, alignment: .trailing)
                }

                HStack {
                    Text("Top-P Sampling")
                    Spacer()
                    Toggle("", isOn: $model.topPEnabled)
                        .labelsHidden()
                    Slider(value: $model.topP, in: 0.1...1.0, step: 0.05)
                        .frame(width: 160)
                        .disabled(!model.topPEnabled)
                    Text(String(format: "%.2f", model.topP))
                        .monospacedDigit()
                        .frame(width: 40, alignment: .trailing)
                        .foregroundStyle(model.topPEnabled ? .primary : .secondary)
                }
            }

            Section("Speculative Decoding") {
                Picker("Speculation Mode", selection: $model.runtimeOptions.speculation) {
                    ForEach(AppSpeculationOption.allCases) { opt in
                        Text(opt.menuLabel).tag(opt)
                    }
                }
            }

        }
        .formStyle(.grouped)
    }
}
