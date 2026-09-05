import SwiftUI
import TurboSpark

public struct AppSettingsView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var appearanceManager = AppearanceManager.shared

    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    public enum SettingsTab: String, CaseIterable, Identifiable {
        case general = "General"
        case appearance = "Appearance"
        case shortcuts = "Keyboard shortcuts"
        case permissions = "Files & Permissions"
        case models = "Models & Storage"
        case engine = "Engine"
        case safety = "Safety & Steering"
        case mcp = "MCP Servers"
        case skills = "Skills"
        case agents = "Agents & Subagents"
        case hooks = "Hooks & Lifecycle"

        public var id: String { rawValue }

        public var title: String { rawValue }

        public var systemImage: String {
            switch self {
            case .general: return "gearshape"
            case .appearance: return "sun.max"
            case .shortcuts: return "keyboard"
            case .permissions: return "folder.badge.gearshape"
            case .models: return "cylinder.split.1x2"
            case .engine: return "cpu"
            case .safety: return "dial.medium"
            case .mcp: return "server.rack"
            case .skills: return "wand.and.stars"
            case .agents: return "person.2.badge.gearshape"
            case .hooks: return "link.badge.plus"
            }
        }

        public var category: String {
            switch self {
            case .general, .appearance, .shortcuts, .permissions:
                return "Personal"
            case .models, .engine, .safety, .mcp, .skills, .agents, .hooks:
                return "Engine & Coding"
            }
        }
    }

    @State private var selectedTab: SettingsTab = .appearance
    @State private var searchText: String = ""

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        HStack(spacing: 0) {
            settingsSidebar
                .frame(width: 220)
                .background(Color(nsColor: .windowBackgroundColor).opacity(0.85))

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(width: 1)

            VStack(spacing: 0) {
                // Top header for right pane
                HStack {
                    Text(selectedTab.title)
                        .font(.title3.weight(.semibold))
                    Spacer()
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
                .background(Color(nsColor: .windowBackgroundColor))

                Rectangle()
                    .fill(TurboSparkTheme.hairlineColor)
                    .frame(height: 1)

                tabContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(minWidth: 780, minHeight: 580)
    }

    // MARK: - Sidebar
    private var settingsSidebar: some View {
        VStack(spacing: 12) {
            // Search field
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .font(.caption)
                    .accessibilityHidden(true)
                TextField("Search settings...", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.caption)
                    .accessibilityLabel("Search settings")
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1)
            )
            .padding(.horizontal, 12)
            .padding(.top, 14)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    sidebarCategory(title: "Personal", tabs: [.general, .appearance, .shortcuts, .permissions])
                    sidebarCategory(title: "Engine & Coding", tabs: [.models, .engine, .mcp, .skills, .agents, .hooks])
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 12)
            }
        }
    }

    private func sidebarCategory(title: String, tabs: [SettingsTab]) -> some View {
        let filtered = tabs.filter { tab in
            searchText.isEmpty || tab.title.localizedCaseInsensitiveContains(searchText)
        }

        return VStack(alignment: .leading, spacing: 2) {
            if !filtered.isEmpty {
                Text(title)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .accessibilityAddTraits(.isHeader)

                ForEach(filtered) { tab in
                    let isSelected = selectedTab == tab
                    Button {
                        selectedTab = tab
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: tab.systemImage)
                                .font(.system(size: 13, weight: .medium))
                                .frame(width: 18)
                                .foregroundStyle(isSelected ? appearanceManager.activeAccentColor(isDark: false) : .secondary)
                                .accessibilityHidden(true)

                            Text(tab.title)
                                .font(.subheadline)
                                .foregroundStyle(isSelected ? .primary : .secondary)

                            Spacer()
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            isSelected ? Color.accentColor.opacity(0.12) : Color.clear,
                            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
                        )
                    }
                    .buttonStyle(.plain)
                    .help(tab.title)
                    .accessibilityLabel(tab.title)
                    .accessibilityHint("Switches settings view to \(tab.title)")
                    .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
                    .appPointerCursor()
                }
            }
        }
    }

    // MARK: - Tab Content
    @ViewBuilder
    private var tabContent: some View {
        switch selectedTab {
        case .appearance:
            AppearanceSettingsPaneView()
        case .general:
            generalSettingsTab
        case .shortcuts:
            shortcutsSettingsTab
        case .permissions:
            PermissionsSettingsPaneView(model: model)
        case .models:
            ModelsSettingsPaneView(model: model)
        case .engine:
            engineSettingsTab
        case .safety:
            SafetySettingsPaneView(model: model)
        case .mcp:
            mcpSettingsTab
        case .skills:
            skillsSettingsTab
        case .agents:
            agentsSettingsTab
        case .hooks:
            hooksSettingsTab
        }
    }

    private var skillsSettingsTab: some View {
        SkillsSettingsPaneView(model: model)
    }

    private var agentsSettingsTab: some View {
        AgentsSettingsPaneView(model: model)
    }

    private var hooksSettingsTab: some View {
        HooksSettingsPaneView(model: model)
    }

    private var mcpSettingsTab: some View {
        McpSettingsPaneView(model: model)
            .padding(16)
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

            Section("Text Size & Readability") {
                Picker("Text Readability", selection: $appearanceManager.textSize) {
                    ForEach(AppTextSize.allCases) { size in
                        Text(size.label).tag(size)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private var shortcutsSettingsTab: some View {
        Form {
            Section("Navigation & View") {
                shortcutRow(label: "Chat", shortcut: "⌘ 1")
                shortcutRow(label: "Files", shortcut: "⌘ 2")
                shortcutRow(label: "Installed Models", shortcut: "⌘ 3")
                shortcutRow(label: "Discover Models", shortcut: "⌘ 4")
                shortcutRow(label: "Toggle Chat Sidebar", shortcut: "⌃ ⌘ S")
                shortcutRow(label: "Toggle Inspector", shortcut: "⇧ ⌘ I")
            }

            Section("Text Size & Zoom") {
                shortcutRow(label: "Make Text Bigger", shortcut: "⌘ +")
                shortcutRow(label: "Make Text Smaller", shortcut: "⌘ -")
                shortcutRow(label: "Default Text Size", shortcut: "⌘ 0")
            }

            Section("Chat & Generation") {
                shortcutRow(label: "New Chat", shortcut: "⌘ N")
                shortcutRow(label: "Previous Chat", shortcut: "⌘ [")
                shortcutRow(label: "Next Chat", shortcut: "⌘ ]")
                shortcutRow(label: "Focus Prompt", shortcut: "⌘ L")
                shortcutRow(label: "Clear History", shortcut: "⇧ ⌘ K")
                shortcutRow(label: "Generate Response", shortcut: "⌘ ↩")
                shortcutRow(label: "Cancel Generation", shortcut: "⌘ .")
            }
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private func shortcutRow(label: String, shortcut: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(shortcut)
                .font(.caption.monospaced())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1)
                )
        }
    }

    /// **FIVE PROPERTIES RATHER THAN ONE EXPRESSION.** A `Form` holding
    /// all five `Section`s inline is ONE expression to the type checker,
    /// and it exceeded the solver budget on the SDK 14 toolchain that CI
    /// builds the app bundle with: "unable to type-check this expression
    /// in reasonable time". It compiles on a newer one, which is why
    /// nobody developing here saw it. Splitting is the fix the diagnostic
    /// itself asks for, and it is cheap insurance on any toolchain -- a
    /// SwiftUI container that grows a section at a time walks into this
    /// eventually.
    private var engineSettingsTab: some View {
        Form {
            systemPromptSection
            generationDefaultsSection
            reasoningEffortSection
            guardrailsSection
            speculationSection
            inProcessServerSection
        }
        .formStyle(.grouped)
        .padding(16)
        .onReceive(NotificationCenter.default.publisher(for: .openSettingsTab)) { notification in
            if let tab = notification.object as? SettingsTab {
                selectedTab = tab
            }
        }
    }

    /// A SEPARATE computed property, like every sibling section, for the
    /// type-checker reason spelled out above `engineSettingsTab`.
    private var systemPromptSection: some View {
        Section("Default System Prompt") {
            VStack(alignment: .leading, spacing: 6) {
                TextEditor(text: $model.defaultSystemPrompt)
                    .font(.system(.body, design: .monospaced))
                    .frame(minHeight: 100)
                    .onChange(of: model.defaultSystemPrompt) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text(
                    "Sent as the first system message of every conversation, ahead of any "
                    + "project rules. A chat with its own system prompt uses that instead. "
                    + "Leave empty for none."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Text("\(model.defaultSystemPrompt.count) characters")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var generationDefaultsSection: some View {
        Section("Generation Defaults") {
            HStack {
                Text("Temperature")
                Spacer()
                Slider(value: $model.temperature, in: 0.0...1.5, step: 0.05)
                    .frame(width: 160)
                    .onChange(of: model.temperature) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text(String(format: "%.2f", model.temperature))
                    .monospacedDigit()
                    .frame(width: 40, alignment: .trailing)
            }

            HStack {
                Text("Top-P Sampling")
                Spacer()
                Toggle("", isOn: $model.topPEnabled)
                    .labelsHidden()
                    .onChange(of: model.topPEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Slider(value: $model.topP, in: 0.1...1.0, step: 0.05)
                    .frame(width: 160)
                    .disabled(!model.topPEnabled)
                    .onChange(of: model.topP) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text(String(format: "%.2f", model.topP))
                    .monospacedDigit()
                    .frame(width: 40, alignment: .trailing)
                    .foregroundStyle(model.topPEnabled ? .primary : .secondary)
            }
        }
    }

    private var reasoningEffortSection: some View {
        Section("Thinking & Reasoning Effort") {
            Picker("Default Reasoning Level", selection: Binding(
                get: { model.reasoning },
                set: { model.setReasoning($0) }
            )) {
                // With a model loaded this is that checkpoint's own set.
                // With none it is the union, because this is the default
                // a FUTURE model inherits and clamping it to nothing
                // would leave a picker with one entry.
                ForEach(model.session == nil
                        ? GenerateOptions.Reasoning.allCases
                        : model.availableReasoningLevels) { level in
                    Text(model.reasoningLabel(for: level)).tag(level)
                }
            }
            .pickerStyle(.menu)

            Text("Controls internal chain-of-thought depth. The accepted levels belong to each checkpoint's own chat template, not to this app: Qwen 3.8 tops out at Extra High and refuses High, while gpt-oss is the other way round. A level a model cannot express is clamped to its nearest one on load, and your choice is remembered per model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var guardrailsSection: some View {
        Section("Forge Tool-Call Guardrails") {
            Picker("Guardrails Mode", selection: $model.guardrailsMode) {
                ForEach(AppGuardrailsMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .onChange(of: model.guardrailsMode) { _, _ in
                model.persistSettings()
            }

            Text(model.guardrailsMode.descriptionText)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var speculationSection: some View {
        Section("Speculative Decoding") {
            Picker("Speculation Mode", selection: $model.runtimeOptions.speculation) {
                ForEach(AppSpeculationOption.allCases) { opt in
                    Text(opt.menuLabel).tag(opt)
                }
            }
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.speculation) { _, _ in
                model.persistSettingsDebounced()
            }

            if model.runtimeOptions.speculation != .off {
                Picker("Speculative Drafter", selection: $model.runtimeOptions.speculativeDrafter) {
                    ForEach(AppSpeculativeDrafterOption.allCases) { drafter in
                        Text(drafter.menuLabel).tag(drafter)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: model.runtimeOptions.speculativeDrafter) { _, _ in
                    model.persistSettingsDebounced()
                }
            }
        }
    }

    private var inProcessServerSection: some View {
        Section("In-Process Server") {
            HStack {
                Toggle("Enable server", isOn: Binding(
                    get: { model.server != nil },
                    set: { $0 ? model.startServer() : model.stopServer() }
                ))
                .disabled(model.session == nil || model.serverBusy)

                if model.serverBusy {
                    ProgressView()
                        .controlSize(.small)
                        .padding(.leading, 4)
                }
            }

            // Both rows are READ BACK from the running server rather than
            // restated here. The address used to be half asserted (a
            // `127.0.0.1` literal beside the real port) and the auth state
            // was not shown at all, so a key of nothing but spaces trims
            // to empty, starts an unauthenticated server, and looked
            // identical to a key that took.
            if let info = model.serverInfo {
                let rows = ServerStatusRows(info: info, guardrails: model.serverStartedGuardrails)

                HStack {
                    Text("Address")
                    Spacer()
                    Text(rows.address)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }

                HStack {
                    Text("Auth")
                    Spacer()
                    Text(rows.authLabel)
                        .font(.caption)
                        .foregroundStyle(rows.authIsWarning ? Color.orange : Color.secondary)
                }
            }

            SecureField("API key (optional)", text: $model.serverAPIKeyInput)
                .disabled(model.server != nil)

            Text(
                "Serves the currently loaded model over OpenAI- and Anthropic-compatible "
                    + "HTTP endpoints on loopback, sharing the same engine this app's chat "
                    + "uses -- not a second copy of the model. Loading a different model, "
                    + "or unloading, stops the server. Loopback keeps it off the network "
                    + "and NOT off this machine: without an API key, any process running "
                    + "here can reach it."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}
