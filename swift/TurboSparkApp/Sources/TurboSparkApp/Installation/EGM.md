# Installation

## Overview
This directory holds the model discovery, install and management UI of the macOS app: the Installed workspace (text, image and audio kinds), the Discover hub (curated catalog, hardware-fit Recommended tab, live Hugging Face search and repo probing), and the first-run welcome screen. It also contains the pure logic those views rely on: install gating, filtering and sorting, alias suggestion, download ETA, feature descriptors and fit presentation. Engine work (probe, hub search, fit, context ladder) is reached through TurboSparkCatalog FFI and AppModel; views mostly draw state and delegate actions to AppModel or controllers. There are no child directories.

## Interface

Install gating and probing:
- `ModelInstallDecision` (.allowed, .confirm, .blocked) and `ModelInstallGate.decide(...)`, `diskHeadroomBytes`, `freeSpaceOnThisMac()`, `freeSpace(at:)` [ModelInstallGate.swift:ModelInstallGate]
- `HubInstallGating.decision(check:selectedVariant:destination:)` [HubInstallGating.swift:decision]
- `ModelProbeGating.sanitizeRepo(_:)`, `ModelProbeGating.installDecision(...)` [ModelProbeGating.swift:ModelProbeGating]
- `ModelProbeSheet(model:)` [ModelProbeSheet.swift:ModelProbeSheet]
- `ProbeReportCardView(report:)` [ProbeReportCardView.swift:ProbeReportCardView]
- `RepoVariant.pickerLabel` [RepoVariantLabel.swift:pickerLabel]
- `ModelAlias.suggest(repo:quantLabel:taken:)`, `isTaken(_:in:)`, `maxLength` [ModelAlias.swift:ModelAlias]
- `ShellQuote.single(_:)` [ShellQuote.swift:single]
- `DownloadETAEstimator.update(downloaded:total:)`, `reset()`, `DownloadETAObservation` [DownloadETAEstimator.swift:DownloadETAEstimator]

Hugging Face search:
- `HubSearchController` (`search(query:refresh:)`, `check(_:configuration:force:)`, `reset()`, `exactRepo(from:)`, `minimumQueryLength`) [HubSearchController.swift:HubSearchController]
- `HubSearchSectionView` [HubSearchSectionView.swift:HubSearchSectionView]
- `HubResultRowView` [HubResultRowView.swift:HubResultRowView]
- `HubResultDetailView(model:controller:repoId:result:)` [HubResultDetailView.swift:HubResultDetailView]

Discover hub (text catalog):
- `ModelHubView(model:)` [ModelHubView.swift:ModelHubView]
- `ModelHubFilter` (`apply(to:installedAliases:recommendations:)`, `recommendedPrecedes`, `recommendedSections`, `isNarrowed`) [ModelHubFilter.swift:ModelHubFilter]
- `ModelHubFilterBarView` [ModelHubFilterBarView.swift:ModelHubFilterBarView]
- `ModelCardView` [ModelCardView.swift:ModelCardView]
- `ModelDetailPaneView(model:entry:installedModel:recommendation:)` [ModelDetailPaneView.swift:ModelDetailPaneView]
- `ModelRecommendationRow`, `RecommendationVerdictBadge` [ModelRecommendationRow.swift:ModelRecommendationRow]
- `RecommendedSectionHeader(verdict:)` [RecommendedSectionHeader.swift:RecommendedSectionHeader]
- `CatalogSheet(model:)` [CatalogSheet.swift:CatalogSheet]

Welcome screen:
- `ModelInstallView`, `ModelRecommendationProbeProgress`, `ModelRecommendationLoadingView` [ModelInstallView.swift:ModelInstallView]
- `HardwareScanCard`, `MoESpotlightCard`, `ActiveInstallCard` [ModelInstallCards.swift:ActiveInstallCard]

Installed workspace (text models):
- `ModelManagerView(model:)` [ModelManagerView.swift:ModelManagerView]
- `ModelManagerHeaderView` [ModelManagerHeaderView.swift:ModelManagerHeaderView]
- `ModelManagerFilterBarView` [ModelManagerFilterBarView.swift:ModelManagerFilterBarView]
- `ModelManagerFilterPreferences` [ModelManagerFilterPreferences.swift:ModelManagerFilterPreferences]
- `ModelManagerView.filterModels(...)`, `groupModels(...)`, `selectedRow(path:in:)`, and the filter/sort/group enums [ModelManagerFilters.swift:filterModels]
- `InstalledSidebarPanel`, `InstalledModelKind` [InstalledSidebarPanel.swift:InstalledModelKind]
- `InstalledModelRowView` [InstalledModelRowView.swift:InstalledModelRowView]
- `InstalledModelDetailPaneView(model:installedModel:)` [InstalledModelDetailPaneView.swift:InstalledModelDetailPaneView]
- `InstalledModelActionBarView` [InstalledModelActionBarView.swift:InstalledModelActionBarView]
- `InstalledModelHeroHeaderView` [InstalledModelHeroHeaderView.swift:InstalledModelHeroHeaderView]
- `InstalledModelFeatureBadgesView` [InstalledModelFeatureBadgesView.swift:InstalledModelFeatureBadgesView]
- `InstalledModelMemoryFitCardView` [InstalledModelMemoryFitCardView.swift:InstalledModelMemoryFitCardView]
- `InstalledModelTechnicalSpecsView` [InstalledModelTechnicalSpecsView.swift:InstalledModelTechnicalSpecsView]
- `InstalledModelOrganizationCardView` [InstalledModelOrganizationCardView.swift:InstalledModelOrganizationCardView]
- `InstalledModelDeveloperCommandsView` (`chatCommand`, `serveCommand`) [InstalledModelDeveloperCommandsView.swift:InstalledModelDeveloperCommandsView]
- `ModelLocationPicker.choose(for:)` [ModelLocationPicker.swift:choose]

Steering (per-model detail pane):
- `ModelSteeringCardView` [ModelSteeringCardView.swift:ModelSteeringCardView]
- `SteeringTraitExtractSheet` [SteeringTraitExtractSheet.swift:SteeringTraitExtractSheet]
- `SteeringVectorImportSheet` [SteeringVectorImportSheet.swift:SteeringVectorImportSheet]

Model presentation (shared):
- `ModelFeatureDescriptor.resolve(installedModel:catalogEntry:sessionInfo:)`, `isMoE`, `isKnownMoEFamily`, `isHuggingFaceRepo`, `supportsKvQuant(atInstallPath:)`, `supportsMoEPrefillBank(atInstallPath:)` [ModelFeatureDescriptor.swift:ModelFeatureDescriptor]
- `ManifestFacts.parse(_:)`, `ManifestFactsCache.facts(forInstallPath:)` [ModelFeatureDescriptor.swift:ManifestFactsCache]
- `ModelFitPresentation.of(_:)`, `filterLabel(_:)`, `ModelFitVerdictPill` [ModelFitPresentation.swift:ModelFitPresentation]
- `ModelFamilyVisuals.resolve(alias:family:name:)`, `formatLabel(alias:name:)` [ModelFamilyVisuals.swift:ModelFamilyVisuals]
- `ModelFeatureBadgeView` and its presets [ModelFeatureBadgeView.swift:ModelFeatureBadgeView]
- `ModelLogoView`, `ModelLogoAssetCache.shared` [ModelLogoView.swift:ModelLogoAssetCache]
- `ModelHardwareFitCardView` [ModelHardwareFitCardView.swift:ModelHardwareFitCardView]
- `ModelTechnicalSpecsCardView`, `View.modelCardStyle()` [ModelTechnicalSpecsCardView.swift:ModelTechnicalSpecsCardView]
- `ContextLadderView`, `ThroughputRowView` [ContextLadderView.swift:ContextLadderView]
- `ModelWorkspaceTitle`, `ModelFilterMenu` [ModelWorkspaceTitle.swift:ModelWorkspaceTitle]

Audio and image models:
- `AudioInstalledCatalog.entries(...)`, `grouped(_:)`, `AudioInstalledEntry`, and `AudioWorkspaceController` helpers (`installedAudioEntries`, `entryState`, `speechToTextModes(using:)`, `localModelURL(_:)`) [AudioInstalledCatalog.swift:AudioInstalledCatalog]
- `AudioDiscoverCatalog.entries(profiles:searchText:sort:canRun:)`, `grouped(_:)` [AudioDiscoverCatalog.swift:AudioDiscoverCatalog]
- `AudioInstalledModelsView`, `AudioInstalledRow`, `AudioInstalledHeaderView` [AudioInstalledModelsView.swift:AudioInstalledModelsView]
- `AudioInstalledModelDetailView`, `AudioInstalledUsedByCard` [AudioInstalledModelDetailView.swift:AudioInstalledModelDetailView]
- `AudioDiscoverCatalogView` [AudioDiscoverCatalogView.swift:AudioDiscoverCatalogView]
- `ImageModelManagerView`, `ImageModelRecommendationSheet` [ImageModelManagerView.swift:ImageModelManagerView]

## Invariants
- Install permission is decided only in `ModelInstallGate.decide`, and AppModel.installModel/installRepo call it too, not only buttons. Check order: probe refusal, refused fit, unconnected destination, disk, tight fit. Low disk and tight fit are `.confirm`, never silent blocks. [ModelInstallGate.swift:decide]
- Callers (hub, probe sheet) decide check state first, then delegate to the gate; disk arithmetic is not duplicated. The probe sheet never allows an install without a probe report. [HubInstallGating.swift:decision] [ModelProbeGating.swift:installDecision]
- A `.confirm` decision shows an "Install anyway?" alert; `.blocked` never installs. [ModelDetailPaneView.swift:actionButtons] [HubResultDetailView.swift:requestInstall] [ModelProbeSheet.swift:requestInstall]
- `freeSpace` returns nil, never 0, when the volume cannot be queried, so a failed query is not read as a full disk. [ModelInstallGate.swift:freeSpace]
- Aliases must satisfy `AppModel.isSafeInstallAlias` (ASCII [a-z0-9._-], lower-cased, never empty, `.` or `..`) and are unique case-insensitively; the alias becomes the folder name and store key. [ModelAlias.swift:suggest]
- Text embedded in a pasted shell command is POSIX single-quoted through `ShellQuote.single` because aliases and prompts are attacker-influenced. [ShellQuote.swift:single] [InstalledModelDeveloperCommandsView.swift:chatCommand]
- Nothing hits the network on its own: Hugging Face search runs only on an explicit request, queries under `minimumQueryLength` reset, and a generation counter drops stale results. Header-read checks and probe results are keyed so late results cannot land on another repo. [HubSearchController.swift:search] [ModelHubView.swift:runHubSearch] [ModelProbeSheet.swift:runProbe]
- Blocking network and FFI calls run in `Task.detached`, never on the main actor; fit and ladder loads are cached in state, not computed in `body`, and dropped when `fitRecommendationConfigurationID` changes. [ModelProbeSheet.swift:runProbe] [InstalledModelDetailPaneView.swift:loadLadder] [ModelInstallView.swift:loadRecommendations]
- Fit and memory figures are engine-derived: the UI never multiplies or extrapolates the context ladder, guards figures on `countedSource != .unknown` (not on non-zero), and shows "unknown"/not-sized instead of zeros. [ContextLadderView.swift:ContextLadderView] [ProbeReportCardView.swift:fitSection] [InstalledModelMemoryFitCardView.swift:body] [ModelHardwareFitCardView.swift:fitIsKnown]
- MoE-ness comes from manifest `numExperts > 1` or an exact family in `knownMoEFamilies`, never alias substrings; unknown tool-call support stays nil (not true) and an unreadable manifest yields `supportsKvQuant` false. [ModelFeatureDescriptor.swift:isMoE] [ModelRecommendationRow.swift:isMoEModel]
- `ManifestFactsCache` trusts an entry only while manifest mtime and size match, is capped at 256 entries and is guarded by NSLock. [ModelFeatureDescriptor.swift:ManifestFactsCache]
- `ModelLogoAssetCache` is an unsynchronized dictionary: main thread only. Monochrome marks are an explicit allowlist. [ModelLogoView.swift:ModelLogoAssetCache]
- Fit verdict wording lives only in `ModelFitPresentation`; the `.unknown` verdict must render a pill in the shared pill (the recommendation badge intentionally renders nothing). Filter spellings are persisted and stay a separate string set. [ModelFitPresentation.swift:ModelFitVerdictPill] [ModelRecommendationRow.swift:RecommendationVerdictBadge]
- Format and capability filter options derive from `ModelFamilyVisuals`, the same table badges read; the format label is parsed from the catalog name, not hardcoded per family. [ModelHubFilter.swift:visuals] [ModelFamilyVisuals.swift:formatLabel]
- Status and warnings pair a glyph with text, never color alone; rows and badges are single accessibility elements; every UI string is a catalog key via LocalizedStringKey, while data (names, sizes, quant) renders verbatim. [InstalledModelRowView.swift:body] [ModelFeatureBadgeView.swift:titleText]
- Delete and remove always confirm. Removing a local folder, away model or list-only entry never deletes files; delete is disabled while a load maps the file or an install changes. [InstalledModelDetailPaneView.swift:removeDialog] [ImageModelManagerView.swift:installedRow] [AudioInstalledModelDetailView.swift:delete]
- Selection is looked up by path or id in the unfiltered list and returns nil when the model leaves it, so Load/Delete never rebind to another model. [ModelManagerFilters.swift:selectedRow] [ModelManagerView.swift:selectedModel]
- `ModelManagerView` observes `model.libraryState` directly because AppModel does not publish it; library names pass through `UntrustedText.display`, and location group ids use library id, not name. [ModelManagerView.swift:libraryState] [ModelManagerFilters.swift:locationGroups]
- Only group option and Available now persist (UserDefaults keys `TurboSpark.modelManager.groupOption` and `.availableOnly`); an unrecognized stored value reads as `.none`. Destination library choice is per sheet and never persisted. [ModelManagerFilterPreferences.swift:ModelManagerFilterPreferences] [CatalogSheet.swift:chosenLibrary]
- Steering UI is gated on `descriptor.isSteeringReady` and `AppSteeringPolicy.compatibility` (enabling only); strength must be finite with abs <= 2; extraction is blocked while the model is loaded; selection applies on next load. [ModelSteeringCardView.swift:presetRow] [SteeringTraitExtractSheet.swift:blockingReason] [SteeringVectorImportSheet.swift:validationError]
- Audio Discover offers only pinned `recommendedAliases` that exist, are runnable and pass `canRun`; installed rows are ordered group, task, name, offset, and engine on-disk verdicts override catalog state. Voice cloning rows must match the Voice Cloning page. [AudioDiscoverCatalog.swift:entries] [AudioInstalledCatalog.swift:entryState]
- `DownloadETAEstimator` ignores samples under 0.5 s apart, smooths 0.8/0.2, returns nil until rate > 1024 B/s, and `reset()` must run between download attempts. [DownloadETAEstimator.swift:update]
- Views stated `@MainActor` explicitly (macOS 14 SDK isolates only `body`). [ModelHubView.swift:ModelHubView] [ModelManagerFilterPreferences.swift:ModelManagerFilterPreferences]

## Side Effects
- Network: Hugging Face search and header checks via `TurboSparkCatalog.hubSearch`/`hubCheck`; repo probe and variants via `TurboSparkCatalog.probe`/`variants`; downloads through `model.installRepo`, `installModel`, `installImageModel`; steering vector download via `SteeringVectorDownloader.download`. [HubSearchController.swift:init] [ModelProbeSheet.swift:runProbe] [SteeringVectorImportSheet.swift:downloadAndUse]
- FFI reads: `TurboSparkCatalog.contextLadder`, `TurboSparkCatalog.available()`, `model.loadFitRecommendations`. [InstalledModelDetailPaneView.swift:loadLadder] [ModelFeatureDescriptor.swift:resolve] [ModelHubView.swift:loadRecommendations]
- Disk reads: `manifest.json` per install, volume capacity, store-root existence, `~/.lmstudio` presence, bundled logo images. [ModelFeatureDescriptor.swift:ManifestFactsCache.facts] [ModelInstallGate.swift:freeSpace] [ModelInstallView.swift:body] [ModelLogoView.swift:ModelLogoAssetCache.image]
- Disk writes: extracted steering vector at `SteeringTraitPresetFactory.outputURL`. [SteeringTraitExtractSheet.swift:startExtraction]
- Child process: `turbospark-steer` CLI via `SteeringTraitExtractionController`. [SteeringTraitExtractSheet.swift:startExtraction]
- Model state: load, unload, delete, remove-from-list, move, select, steering preset and enabled flag, STT mode and speaker labels, and `model.activeSection` switches. [InstalledModelDetailPaneView.swift:body] [ModelSteeringCardView.swift:presetRow] [AudioInstalledModelDetailView.swift:AudioInstalledUsedByCard]
- UserDefaults: group option and Available now. `ModelOrganizationStore.shared`: favorites, tags, notes (notes written on every keystroke). [ModelManagerFilterPreferences.swift:groupOption] [InstalledModelOrganizationCardView.swift:body]
- LM Studio detection toggle calls `persistSettings` and `refreshModels`. [ModelInstallView.swift:lmStudioDetectedBanner]
- UI and OS: modal `NSOpenPanel` folder picker, Finder reveal via `ModelStorageManager.revealInFinder`, `NSPasteboard.general` writes, `NSWorkspace` URL opens, toasts, VoiceOver announcements. [ModelLocationPicker.swift:choose] [InstalledModelDeveloperCommandsView.swift:cliSnippet] [HubSearchController.swift:announce]
- Security-scoped bookmark resolved with `.withoutUI`/`.withoutMounting` for local audio models. [AudioInstalledCatalog.swift:localModelURL]
- `AudioDiscoverCatalogView` and `AudioInstalledModelsView` call `controller.refreshModels()` on appear when empty. [AudioDiscoverCatalogView.swift:body] [AudioInstalledModelsView.swift:body]

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
- Design rules for UI changes: `make design-check`
