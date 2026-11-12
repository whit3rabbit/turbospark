# Components

## Overview
Shared SwiftUI views for the TurboSpark macOS app, mostly the Settings panes and the sheets, cards, and rows they host, plus a few chrome pieces (toasts, banners, composer button, mascot views). The views hold little logic: state and mutation go through `AppModel`, `AppearanceManager`, and singleton stores, and models such as `ModelLibraryRowModel` and `InstallDestinationModel` decide what is drawn so sighted and VoiceOver text cannot drift. Layout, theming, localization, and accessibility rules come from DESIGN.md.

## Interface

### Agents and skills
- Agents pane: [AgentsSettingsPaneView.swift:AgentsSettingsPaneView], [AgentsSettingsPaneView.swift:AgentScopeFilter], [AgentDetailView.swift:AgentDetailView], [AgentDetailView.swift:AgentBadge], [AgentEditorSheet.swift:AgentEditorSheet]
- Cross-agent import wizard: [AgentContentImportSheet.swift:AgentContentImportSheet], [AgentContentImportSheet.swift:ImportCategory], [AgentContentImportSheet.swift:scanMcpServers]
- Skills pane: [SkillsSettingsPaneView.swift:SkillsSettingsPaneView], [SkillsSettingsPaneView.swift:SkillScopeFilter], [SkillDetailView.swift:SkillDetailView], [SkillDetailView.swift:SkillScopeBadge], [SkillEditorSheet.swift:SkillEditorSheet], [SkillImportSheet.swift:SkillImportSheet]
- Agent Mode classifier hints: [AgentModeHintsSectionView.swift:AgentModeHintsSectionView]

### MCP servers
- [McpSettingsPaneView.swift:McpSettingsPaneView], [McpServerCardView.swift:McpServerCard], [McpServerEditorSheet.swift:McpServerEditorSheet]
- Editor sub-forms: [McpTransportFieldsView.swift:McpStdioTransportFields], [McpTransportFieldsView.swift:McpRemoteTransportFields]
- Catalog import: [McpImportSheet.swift:McpImportSheet], [McpCatalogSourceFormView.swift:McpCatalogSourceFormView]

### Extension pane shared parts and marketplaces
- [ExtensionPaneHeader.swift:ExtensionPaneLayout], [ExtensionPaneHeader.swift:ExtensionPaneHeader], [ExtensionPaneHeader.swift:ExtensionPaneIntro], [ExtensionPaneHeader.swift:ExtensionSplitView], [ExtensionPaneHeader.swift:ExtensionNoMatchesState], plus ExtensionImportMenu, ExtensionMoreMenu, ExtensionRefreshButton
- [ExtensionScopePicker.swift:ExtensionScopePicker], [ExtensionScopePicker.swift:ExtensionOverridePicker], [ExtensionScopePicker.swift:McpManagementView]
- [MarketplaceSourcesView.swift:MarketplaceSourcesView], [FetchedSourceSaveBar.swift:FetchedSourceSaveBar]
- Plugins: [PluginSettingsPaneView.swift:PluginSettingsPaneView], [PluginSettingsPaneView.swift:PluginContributionCounts], [PluginDetailView.swift:PluginDetailView], [PluginMarketplaceSheet.swift:PluginMarketplaceSheet]
- Hooks child directory: [Hooks](./Hooks/EGM.md)

### Models and storage
- Pane and card style: [ModelsSettingsPaneView.swift:ModelsSettingsPaneView], [ModelsSettingsPaneView.swift:modelsSettingsCard]
- Hugging Face: [HfAuthTokenCardView.swift:HfAuthTokenCardView], [HfMirrorEndpointRowView.swift:HfMirrorEndpointRowView]
- Libraries: [ModelLibrariesSectionView.swift:ModelLibrariesSectionView], [ModelLibraryRowView.swift:ModelLibraryRowView], [LibraryOfferBanner.swift:LibraryOfferBanner], [InstallDestinationPicker.swift:InstallDestinationPicker]
- Scan folders: [CustomModelFoldersSectionView.swift:CustomModelFoldersSectionView], [ScanFolderRowView.swift:ScanFolderRowView], [ExcludedScanPathsSectionView.swift:ExcludedScanPathsSectionView], [ModelProviderDiscoverySheet.swift:ModelProviderDiscoverySheet], [SettingsPathText.swift:SettingsPathText]
- Moving models: [ModelMoveSheet.swift:ModelMoveSheet], [ModelMoveSheet.swift:ModelMoveSheetHost] (via `View.modelMoveSheet(model:kinds:)`)
- Load policy: [LocalModelDefaultsSectionView.swift:LocalModelDefaultsSectionView], [LoadAnywayConfirmation.swift:LoadAnywayConfirmation]
- Embeddings: [EmbeddingModelPicker.swift:EmbeddingModelPicker], [EmbeddingModelPicker.swift:EmbeddingStatusLabel]
- Downloads: [DownloadManagerView.swift:DownloadManagerView], [ModelInstallToastView.swift:ModelInstallToastView]

### Appearance
- [AppearanceSettingsPaneView.swift:AppearanceSettingsPaneView], [AppearancePreferencesCardView.swift:AppearancePreferencesCardView], [ThemeConfigCardView.swift:ThemeConfigCardView], [ThemeGalleryView.swift:ThemeGalleryView], [ThemeGalleryView.swift:ThemeModeThumbnail]

### Privacy, network, and safety
- [PermissionsSettingsPaneView.swift:PermissionsSettingsPaneView], [PermissionsInfoSectionsView.swift:PermissionsSystemLinksSection], [PermissionsInfoSectionsView.swift:CommandGateSettingsSection], [PermissionsInfoSectionsView.swift:PermissionsSecurityCallout]
- [NetworkSettingsPaneView.swift:NetworkSettingsPaneView], [NetworkSettingsPaneView.swift:NetworkDomainListView]
- Steering: [SafetySettingsPaneView.swift:SafetySettingsPaneView], [SafetySettingsPaneView.swift:SteeringPresetRow], [SteeringPresetEditorSheet.swift:SteeringPresetEditorSheet], [SteeringTexts.swift:SteeringText]

### Profiles
- [ProfilesSettingsPaneView.swift:ProfilesSettingsPaneView], [ProfileSettingsSheets.swift:ProfileSecuritySheet], [ProfileSettingsSheets.swift:ProfileRenameSheet], [ProfileSettingsSheets.swift:ProfileOpenExportSheet], [ProfileSettingsSheets.swift:ProfileSheetParts], [ProfileBackupSheets.swift:ProfileEncryptedBackupSheet], [ProfileBackupSheets.swift:ProfileBackupImportSheet]

### Engine prompts and scheduling
- [SystemPromptSettingsSection.swift:SystemPromptSettingsSection], [PersonalitySettingsSection.swift:PersonalitySettingsSection], [SoulSettingsSection.swift:SoulSettingsSection], [CronJobsSettingsPaneView.swift:CronJobsSettingsPaneView]

### Chrome and primitives
- [GenerateControl.swift:GenerateControl], [ErrorBanner.swift:ErrorBanner], [ToastOverlayView.swift:ToastOverlayView], [FlowLayout.swift:FlowLayout]
- Mascots and glyphs (each with an asset cache): [WelcomeCharacterView.swift:WelcomeHeroView], [AgentsCharacterView.swift:TSIdlingAgentsSparkView], [MemoryCharacterView.swift:TSIdlingMemorySparkView], [SoulCharacterView.swift:TSIdlingSoulSparkView], [GhostGlyph.swift:GhostGlyph], [TaskProgressIndicatorView.swift:TaskProgressIndicatorView], [TaskProgressIndicatorView.swift:TaskProgressFlameIcon]

## Invariants
- Destructive actions (delete agent, skill, soul, MCP server, personality, theme, profile, library removal) go through a confirmation dialog [AgentsSettingsPaneView.swift:AgentsSettingsPaneView], [McpServerCardView.swift:McpServerCard], [ProfilesSettingsPaneView.swift:ProfilesSettingsPaneView].
- Folder and library scans stay off the main actor and out of `body`; superseded or cancelled results are never applied [CustomModelFoldersSectionView.swift:readOffMainActor], [ModelsSettingsPaneView.swift:ModelsSettingsPaneView], [ModelProviderDiscoverySheet.swift:ModelProviderDiscoverySheet].
- Library, scan-folder, and move views draw a precomputed row or sheet model only, and surface refusals in place as glyph plus sentence, never as an app error or toast [ModelLibraryRowView.swift:ModelLibraryRowView], [ScanFolderRowView.swift:ScanFolderRowView], [ModelLibrariesSectionView.swift:ModelLibrariesSectionView], [LibraryOfferBanner.swift:LibraryOfferBanner].
- `LibraryState` is not published by `AppModel`, so views that need it observe it directly [InstallDestinationPicker.swift:InstallDestinationPicker], [LibraryOfferBanner.swift:LibraryOfferBanner].
- Drive-supplied or remote-controlled text (library names, paths, manifest names) is drawn verbatim, never as a localization key [SettingsPathText.swift:SettingsPathText], [ModelLibraryRowView.swift:ModelLibraryRowView].
- The move sheet cannot be dismissed, and Escape is not bound to Cancel, while a move runs [ModelMoveSheet.swift:ModelMoveSheet].
- Fetching a marketplace never saves its source because the manifest name is remote-controlled; saving is explicit [FetchedSourceSaveBar.swift:FetchedSourceSaveBar], [McpImportSheet.swift:McpImportSheet], [SkillImportSheet.swift:SkillImportSheet].
- Imported or installed MCP servers are added disabled with auto-approve off, and name collisions are re-checked at install time [AgentContentImportSheet.swift:AgentContentImportSheet], [McpImportSheet.swift:McpImportSheet].
- Other agents' roots are scanned only for the default profile, and nothing is pre-selected [AgentContentImportSheet.swift:AgentContentImportSheet].
- Scope is a captured project id fixed at init, never an alias for the selected chat [ExtensionScopePicker.swift:ExtensionScopePicker], [SkillEditorSheet.swift:SkillEditorSheet], [McpImportSheet.swift:McpImportSheet].
- Remote MCP transport is not implemented; the unavailable notice must not be softened or relabeled [McpTransportFieldsView.swift:unavailableNotice]. Server name is identity, so duplicates are refused [McpServerEditorSheet.swift:McpServerEditorSheet].
- `allowed-tools` and agent `model` are enforced by nothing, so the skill and agent views must not show them as restrictions [SkillDetailView.swift:SkillDetailView], [AgentDetailView.swift:AgentDetailView].
- HF mirror endpoints the token must not be sent to (non-https, non-localhost http) are rejected, not silently reset [HfMirrorEndpointRowView.swift:saveMirrorEndpoint]. After token save or clear it is re-resolved because `HF_TOKEN` or the CLI cache can outrank the store [HfAuthTokenCardView.swift:HfAuthTokenCardView].
- Passphrases need at least `ProfileVaultCrypto.minimumPassphraseLength` characters, recent authentication lasts 300 s, and plaintext export needs the "not encrypted" acknowledgement and excludes Keychain secrets [ProfileSettingsSheets.swift:hasRecentAuthentication], [ProfileSettingsSheets.swift:ProfileOpenExportSheet], [ProfileBackupSheets.swift:ProfileEncryptedBackupSheet].
- The command-gate veto is off by default (classifier scores 0.70 held out), and `CommandGate.vetoEnabled` is a static mirrored in `@State` [PermissionsInfoSectionsView.swift:CommandGateSettingsSection].
- Network policy edits go only through `model.updateNetworkPolicy` and `addNetworkDomain`; the coverage section must state what each mode enforces [NetworkSettingsPaneView.swift:NetworkSettingsPaneView].
- Steering text only renders `AppSteeringPolicy` decisions and never claims "compatible"; the engine parses the GGUF header [SteeringTexts.swift:SteeringText], [SteeringPresetEditorSheet.swift:SteeringPresetEditorSheet].
- Status is glyph plus word, not color alone; decorative mascots are hidden from accessibility and motion honors `AppearanceManager.shouldReduceMotion` [EmbeddingModelPicker.swift:EmbeddingStatusLabel], [WelcomeCharacterView.swift:WelcomeHeroView], [ToastOverlayView.swift:ToastOverlayView].
- Settings-search anchors: the `preferenceRow(title:)` literal is read by `scripts/generate-settings-catalog.py`, and localized keys need one `Text` literal per case [AppearancePreferencesCardView.swift:preferenceRow], [LocalModelDefaultsSectionView.swift:guardTitle].
- Large pane bodies are split into separate properties or subviews to stay within the macOS 14 SDK type-checker budget [ProfilesSettingsPaneView.swift:ProfilesSettingsPaneView], [SafetySettingsPaneView.swift:SafetySettingsPaneView].

## Side Effects
- NSOpenPanel, NSSavePanel, and `fileImporter` prompts for folders and files [ModelsSettingsPaneView.swift:selectLmStudioFolder], [ThemeGalleryView.swift:exportFile], [SoulSettingsSection.swift:handleFileImport].
- Disk reads: other agents' skill, agent, and MCP config roots [AgentContentImportSheet.swift:scan]; model folders and providers [ModelProviderDiscoverySheet.swift:scan]; plugin ledger and directories [PluginMarketplaceSheet.swift:refreshInstalledIDs]; GGUF header [SteeringPresetEditorSheet.swift:readVector].
- Network and git: marketplace fetches and installs [McpImportSheet.swift:fetchCatalog], [PluginMarketplaceSheet.swift:loadSelected], [SkillImportSheet.swift:installRemoteEntry]; Hugging Face whoami validation [HfAuthTokenCardView.swift:triggerValidation]; MCP test connection spawns or contacts a server [McpServerEditorSheet.swift:runConnectionTest].
- Persisted state through `AppModel` and stores: settings, skills, agents, MCP servers, plugins, profiles, steering presets, souls, prompts, themes [McpSettingsPaneView.swift:McpSettingsPaneView], [SkillEditorSheet.swift:saveAction], [SoulSettingsSection.swift:save].
- Keychain and vault: HF token set and clear [HfAuthTokenCardView.swift:saveToken]; profile protection and quick unlock via `ProfileVaultCoordinator` [ProfileSettingsSheets.swift:apply].
- Profile switch saves state and relaunches the app [ProfilesSettingsPaneView.swift:ProfilesSettingsPaneView].
- Scheduler and model actions: `CronScheduler` enable and delete [CronJobsSettingsPaneView.swift:jobRow]; `model.run()` and `model.cancel()` [GenerateControl.swift:GenerateControl]; downloads pause, resume, retry, clear [ModelInstallToastView.swift:ModelInstallToastView].
- Pasteboard use for theme JSON and skill content [ThemeConfigCardView.swift:copyThemeJSON], [SkillDetailView.swift:instructions].
- Finder reveals and macOS Privacy pane links [PermissionsSettingsPaneView.swift:systemFolderRow], [PermissionsInfoSectionsView.swift:PermissionsSystemLinksSection].
- Bundled PNG and SVG assets loaded once from `Bundle.module` by the asset caches [GhostGlyph.swift:GhostGlyphAssetCache], [WelcomeCharacterView.swift:WelcomeCharacterAssetCache].
- Accessibility announcements [ErrorBanner.swift:ErrorBanner], [ModelInstallToastView.swift:ModelInstallToastView]; timed auto-dismiss of toasts [ToastOverlayView.swift:ToastOverlayView].

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
- For UI changes also run `make design-check` (see DESIGN.md).
