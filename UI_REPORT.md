# UI_REPORT.md

SwiftUI audit of the TurboSpark macOS app against [DESIGN.md](DESIGN.md).
Audit date: 2026-10-25. Report generated from a read-only review; no source
files were changed.

## Scope and method

- Target: every SwiftUI view-defining file under
  `swift/TurboSparkApp/Sources/TurboSparkApp` (299 files) across `App/`,
  `Audio/`, `Browser/`, `Chrome/`, `Components/` (+ `Hooks/`), `Diagnostics/`,
  `Files/`, `Generation/`, `Humanize/`, `Images/`, `Installation/`,
  `Presentation/`, `Server/`, and `Theme/`.
- Reference: [DESIGN.md](DESIGN.md) sections 1-9 (vocabulary, action
  hierarchy, layout, theming/type, accessibility, localization, menus,
  platform constraints).
- Method: 30 read-only review agents (at most 3 concurrent), each of which
  read DESIGN.md in full plus every assigned file end to end. Suspected
  localization defects were checked against
  `swift/TurboSparkApp/Localization/Localizable.xcstrings` before being rated,
  and ratchet-counted spellings were cross-checked against
  `DesignTokenRatchetTests`.
- This is a static code audit. Per DESIGN.md section 9, a fix is not verified
  until the touched screens are checked in the running app at Small/Default/
  Extra Large text, narrow window, light/dark/custom theme, keyboard only,
  VoiceOver, Reduce Motion/Transparency/Contrast, English/German/RTL. Nothing
  in this report replaces that pass.

## How to read this report

- **HIGH**: breaks a hard DESIGN.md rule with user-visible effect
  (unlocalized UI copy, destructive action without confirmation, wrong theme
  mode, data loss).
- **MEDIUM**: concrete vocabulary/layout/accessibility rule violation.
- **LOW**: minor drift or informational (off-token radii, file length).
- Severity labels were assigned by the reviewing agents and calibration
  varies slightly between batches (some rated ratchet-counted `.tertiary` /
  `.secondary` spellings HIGH, others MEDIUM). The defect-class groupings
  below are the reliable triage view; the per-file sections are the ground
  truth with line numbers.

## Headline numbers

| Metric | Count |
| :--- | :--- |
| View files audited | 299 |
| Files with no findings | 36 |
| Files with findings | 263 |
| Findings (as labeled) | 271 HIGH, 673 MEDIUM, 410 LOW |

The largest classes: theme-blind color spellings (~160 HIGH-labeled sites),
unlocalized fixed English (~73), destructive actions without a confirmation
dialog (~23), sheet focus/keyboard gaps, and hand-rolled copies of vocabulary
controls.

---

## Priority 1 - Localization: fixed English that never reaches the catalog

The catalog is the source of truth for 21 languages; each of these renders
English under every non-English language. Two files bypass the catalog
wholesale (`Text(verbatim:)` on all copy):

- `Audio/AudioMeetingNotesView.swift` - every user-facing string (18+ sites:
  "Generate Notes", "Re-generate", "Export MD", the empty-state paragraph,
  tooltips, "Generated with %@", the "Save Notes"/"Edit Notes" ternary), plus
  `.font(.system(size: 40))` on the empty-state glyph.
- `Audio/AudioCustomVocabularySheet.swift` - all copy verbatim (title, purpose,
  "Add New Term", "Done", prompts), an interpolated "\(...) terms" count, and
  `.font(.system(size: 28))`.

Other whole-feature clusters:

- `Audio/AudioSpeakerNamingSheet.swift:156,169` ("You (Microphone)", tooltip).
- `Audio/SpeechToTextWorkspaceView.swift:86,91` and
  `Audio/AudioTranscriptEditorView.swift:97,100` ("Vocabulary" header/help).
- `Audio/AudioLibraryView.swift:164-173` (detail tab picker rows).
- `Audio/AudioWorkspaceInputs.swift:117` - `MusicMemoryAdvisor.tierSummary`
  builds `String(format:)` English that reaches the UI.
- `Generation/ToolCallCardView.swift:470,489,741,848,348-356` - status labels
  through `Text(String)`, a ternary inside one `Text`, "\(...) lines"
  verbatim, and an efficiency narrative that also names the internal tool
  `recall_tool_output` in user copy.
- `Generation/WorktreeView.swift` - tab titles from `WorktreeTabMode.title`
  (not catalog keys), two `.help` String overloads, a toast literal, and an
  in-string plural ternary ("%lld %@" key missing).
- `Generation/ChatGitSheets.swift:156,164,210,240` - four keys missing from
  the catalog ("Copied %@.", "Copy SHA for %@", "Open pull request #%lld",
  "Pull request state %@").
- `Generation/WorktreeDiffView.swift:199` ("%lld unmodified lines"),
  `WorktreeTimelineView.swift:146` ("%lld files"),
  `WorktreeWorktreeListView.swift:43` ("%lld worktrees") - keys absent.
- `Generation/CollapsibleMessageContentView.swift:107-108` - bare-String
  `.help`/`.accessibilityLabel`, keys absent.
- `Generation/ToolGroupView.swift:42,130-131`, `ToolCodeCellView.swift:46,58`,
  `ToolResultOutputView.swift:38,184,202` - English plurals/labels through
  `Text(String)` or ternaries inside one `Text`.
- `Generation/TodoChecklistViews.swift:7-12,187` - "2/4 done (1 in progress)"
  and "+N completed" assembled in English; `ProjectChatSummaryView` renders
  the result.
- `Generation/StreamingStatusFooterView.swift:32,47` - "\(N) tokens" (key
  exists but unused) and status strings from `LoadingPhrases` (English-only
  literal list).
- `Generation/ChatSystemPromptSheet.swift:32-39` - multi-variant English
  sentences via `Text(String)`.
- `Generation/ChatTranscriptView.swift:122-148` - VoiceOver announcements from
  bare literals and a bundleless rotor key that resolves against `Bundle.main`.
- `Generation/ProjectToolApprovalSheet.swift:60,67` - `AppToolCategory.label`
  fixed English via `Text(verbatim:)`.
- `Browser/ChromeBookmarkImportPreviewView.swift:451,453` - two issue keys
  missing from the catalog.
- `Installation/ModelSteeringCardView.swift:132,154,203,220` - "or",
  "Enabled: %@", "%@ at %@", and a three-way verbatim ternary.
- `Installation/SteeringVectorImportSheet.swift:43-44,63,146-163` - validation
  errors, sheet title, and source-link titles/hints unlocalized.
- `Installation/InstalledModelHeroHeaderView.swift:40-41` - favorite
  `.help`/`.accessibilityLabel` through the String overloads; keys absent.
- `Installation/ImageModelManagerView.swift:263-267` - license note verbatim.
- `Presentation/ChatMessageMarkdownView.swift:323,363` - copy
  accessibilityLabel/announcement from bare Strings.
- `Humanize/HumanizeAdvancedPanel.swift:63` - rule-field placeholders verbatim.
- `Server/AudioAPIPaneView.swift:11` and `Server/ImageAPIPaneView.swift:9` -
  seeded editor drafts are English literals (need `AppStrings.localized`).
- Enum-backed values that are not registered in
  `testEnumBackedDisplayStringsAreCatalogKeys` although they are rendered
  through `Text(LocalizedStringKey)`:
  `AudioWorkspacePage` (Audio/AudioNavigationView.swift:13),
  `WorktreeTabMode` (Generation/WorktreeView.swift:87),
  `AppToolCategory` (Generation/ProjectToolApprovalSheet.swift).

Fix pattern: `Text("...", bundle: .module)` / `AppStrings.localized` at every
site, keys added in all 21 languages in the same change
(`make compile-strings`), and enum-backed display types added to the parity
test.

## Priority 2 - Destructive actions without a confirmation dialog
(and destructive red on removals the user can rebuild)

DESIGN.md: "A destructive role with no confirmation dialog is always a
defect" and "Removing an entry from a list the user can rebuild ... is a
quiet action with a neutral glyph (minus.circle), not a red trash can."

Confirmed destructive-with-no-dialog (data loss on one click):

| Site | Action |
| :--- | :--- |
| `App/MemorySettingsPaneView.swift:206` | Forget a memory claim |
| `Components/SoulSettingsSection.swift:109` | Delete saved soul |
| `Components/SafetySettingsPaneView.swift:227` | Delete steering preset |
| `Components/SystemPromptSettingsSection.swift:29` | Delete system prompt |
| `Components/PersonalitySettingsSection.swift:27` | Remove personality |
| `Components/McpServerCardView.swift:117` | Delete MCP server (verified: no dialog at any layer) |
| `Components/HfAuthTokenCardView.swift:233` | Remove stored token |
| `Components/ThemeGalleryView.swift:111` | Delete saved theme |
| `Components/Hooks/HookSourceDetailSheet.swift:392` | Delete custom hook |
| `Diagnostics/GenerationSamplingSection.swift:208,273` | Remove override; delete preset (trash icon) |
| `Generation/ProjectMcpServerRowView.swift:87` | Delete project MCP server (verified: no dialog in the chain) |
| `Generation/BackgroundShellsStripView.swift:70` | Kill background shell |
| `Images/ImagesSectionView.swift:107` + `Images/ImageGalleryCard.swift:125` | Move to Trash (one shared dialog should cover card menu, context menu, carousel) |
| `Installation/CatalogSheet.swift:146` | Delete model from disk (sibling pane confirms the same action) |
| `Server/ServerInspectorView.swift:73` | Delete saved server preset |
| `Components/MarketplaceSourcesView.swift:32` | Remove source - actually rebuildable, should be quiet, not destructive |

Role/styling defects where a dialog exists but the action is rebuildable
(should be quiet `minus.circle`, not red trash): `Components/ModelLibraryRowView.swift:222`,
`Components/PermissionsSettingsPaneView.swift:251` (authorized folder removal),
`Installation/InstalledModelActionBarView.swift:122`,
`Installation/AudioInstalledModelDetailView.swift:87`,
`Installation/ImageModelManagerView.swift:205`,
`Components/PluginMarketplaceSheet.swift:234,310`,
`Generation/ProjectPermissionsSectionView.swift:201` (allowlist origin remove),
`Generation/ProjectMcpSettingsSheet` trash glyph,
`Generation/ChatParitySheets.swift:94,127` (Stop/Kill),
`Files/FilesSectionView.swift:275,310` and `Files/FilePreviewView.swift:251`
(Detach), `Components/ModelInstallToastView.swift:59` and
`Audio/AudioModelDownloadControls.swift:91` + `Audio/AudioModelPickerView.swift:394`
(Cancel download; peers use a non-destructive secondary Cancel - align on one
convention), `Components/TypeSafePlaygroundCards.swift:23,53,88` (draft
removals), `Generation/OutputPaneView.swift:81` (Clear chat history: no role
and no dialog at all).

## Priority 3 - Theme-blind color spellings (largest class)

DESIGN.md section 4 forbids system colors and names the SwiftUI spellings
(`Color.accentColor`, `.secondary`, `.tertiary`, `Color.primary.opacity`,
`Color(nsColor:)`) that ignore custom themes; `DesignTokenRatchetTests` counts
them and its baselines may only go down.

1. Bare static `TurboSparkTheme.accentColor` / `.surfaceColor` /
   `.hairlineColor` read inside a view body (Gotcha 36: follows
   `NSApp.effectiveAppearance`, not `.preferredColorScheme`). Sites:
   `App/ProfileUnlockView.swift` (whole lock screen, see below),
   `Components/CronJobsSettingsPaneView.swift:127`,
   `Components/SkillsSettingsPaneView.swift:228`,
   `Components/McpServerCardView.swift:26`,
   `Components/PermissionsSettingsPaneView.swift:99`,
   `Components/AgentsCharacterView.swift:52,81`,
   `Components/MemoryCharacterView.swift:52,81`,
   `Components/SoulCharacterView.swift:52,81`,
   `Chrome/ModelLoaderControl.swift:84,99`, `Chrome/TopBarView.swift:240`,
   `Chrome/GenerationPhaseIndicator.swift:36`,
   `Diagnostics/UsageDashboardSection.swift:183`,
   `Diagnostics/ContextSelectionPopover.swift:133`,
   `Generation/InteractiveMarkdownTableView.swift:189,213`,
   `Generation/WorktreeView.swift:92,94,160,188,470`,
   `Generation/WorktreeFileListView.swift:101,122,233,247`,
   `Generation/WorktreeTimelineView.swift:58-228 (5 sites)`,
   `Generation/WorktreeDiffView.swift:220`, `Generation/ChatGitSheets.swift:125`,
   `Generation/ChatSearchOverlayView.swift:74`,
   `Generation/ChatSearchResultRowView.swift:16,59`,
   `Generation/PromptAttachmentsView.swift:43,83,88`,
   `Generation/ToolCallCardView.swift:1095`,
   `Generation/InteractiveQuestionCardView.swift:185,200`,
   `Generation/SubagentLiveCardView.swift:99`,
   `Generation/PromptSafetyPills.swift:31,129`,
   `Generation/SplitChatPanes.swift:62`,
   `Generation/ToolApprovalDropdown.swift:105,121`,
   `Generation/WorktreeCommitListView.swift:97,129`,
   `Generation/WorktreeWorktreeListView.swift:122,127`,
   `Generation/ProjectPickerView.swift:36`,
   `Generation/ProjectMcpServerRowView.swift:20`,
   `Server/ServerLoadedModelsView.swift:58`, `Server/APIPaneComponents.swift:303`,
   `Files/FilesSectionView.swift:214,244`,
   `Humanize/HumanizeChrome.swift:94,97`,
   `Theme/TurboSparkMotion.swift:200,231` (TSIdlingSparkView),
   `Installation/ContextLadderView.swift:78`,
   `Installation/InstalledModelMemoryFitCardView.swift:76`,
   `Installation/InstalledModelOrganizationCardView.swift:37-62`,
   `Installation/ModelTechnicalSpecsCardView.swift:48`,
   `Installation/InstalledModelTechnicalSpecsView.swift:60`,
   `Installation/ModelCardView.swift:49,62`,
   `Installation/InstalledModelHeroHeaderView.swift:168-175`,
   `Installation/ModelHardwareFitCardView.swift:32`,
   `Installation/ImageModelManagerView.swift:225,236`,
   `Installation/ModelInstallView.swift:68,91-92`.
2. System palette colors (`.red`, `.green`, `.orange`, `.yellow`, `.teal`,
   `.purple`, `.blue`, `.pink`) for status/warning/badges on themed surfaces.
   The warning spelling is `AppAttention.color`; success/error/status need
   theme-resolved tokens. Present in nearly every pane: Chrome telemetry
   readouts (StatusBarView, ChromeTelemetryView, FanReadoutView, TopBarView),
   ToolCallCardView (extensively), TodoChecklistViews, SubagentLiveCardView,
   ProjectMcpServerRowView, ToolResultOutputView, Hooks pane and sheets,
   Installation badges/verdicts (ModelFeatureBadgeView hard-codes a 13-color
   palette; ModelFitPresentation returns system colors that leak into
   ContextSelectionPopover, ProbeReportCardView, WorktreeFileListView),
   InstalledModelRowView (teal selection) vs ModelCardView (accent selection)
   - the two model lists draw selection differently; `theme.selection` exists.
3. Hard-coded opacities on themed borders/surfaces/text that bypass
   `surfaceOpacity` and Increase Contrast: `.appBorder.opacity(...)`
   (ratchet-counted; e.g. `KeyboardShortcutsSettingsPaneView.swift:75`,
   `AppearanceSettingsPaneView.swift:102`, `PermissionsInfoSectionsView.swift:123`,
   `AgentModeHintsSectionView.swift:59`, `MessageRowView.swift:94`,
   `ModelDetailPaneView.swift:242`, `ModelInstallCards.swift:63`,
   `ModelTechnicalSpecsCardView.swift:96`, `ToolCodeCellView.swift:32`,
   Hooks sheets), `.appSurface.opacity(...)` / `.appElevated.opacity(...)`
   / `.appPage.opacity(...)` (MessageRowView, SubagentLiveCardView,
   BackgroundShellsStripView, HumanizeChrome, AgentDetailView, SkillDetailView,
   ModelInstallCards, GoalBannerView uses `Color(nsColor:
   .controlBackgroundColor)` outright, AppSettingsView sidebar
   `.appPage.opacity(0.85)`, Hooks files), and `theme.foreground.opacity(...)`
   / `theme.secondaryText.opacity(...)` dimming (TopBarView,
   ChatSidebarGroupedProjectsView, GenerateControl).
4. Scene-level: `App/ProfileUnlockView.swift` is the locked-state root of both
   scenes and neither branch applies `.appThemed()`, so the lock screen
   renders the light fallback theme under every dark/custom theme; it also
   uses `.foregroundStyle(.secondary)`, `.red`, and `.background(.background)`.

## Priority 4 - Vocabulary drift

The vocabulary table in DESIGN.md section 2 is being re-implemented locally:

- `ProgressView` where `AppProgressBar` is required: `Components/ModelMoveSheet.swift:113,163`,
  `Components/AgentContentImportSheet.swift:170`, `Components/HfAuthTokenCardView.swift:152`,
  `Components/ModelLibraryRowView.swift:176`, `Components/LibraryOfferBanner.swift:171`,
  `Components/DownloadManagerView.swift:32`, `Components/McpServerCardView.swift:78`,
  `Generation/GoalBannerView.swift:60`, `Generation/ProfileMemoryCaptureSheet.swift:33`,
  `Generation/ChatGitSheets.swift:82,132,175`, `Generation/WorktreeView.swift:363,416`,
  `Generation/WorktreeTimelineView.swift:144,154`, `Generation/ChatParitySheets.swift:287`,
  `Generation/ProjectSettingsSheet.swift:337`, `Generation/ProjectMcpServerRowView.swift:48`,
  `Generation/ProjectMcpDetectionSectionView.swift:31`, `Generation/ChatActivityLine` ok,
  `Audio/AudioLiveTextView.swift:86`, `Audio/AudioRecordBar.swift:81`,
  `Audio/AudioSearchResultsView.swift:20`, `Server/APIPaneComponents.swift:60,177`,
  `Server/ServerMenuDashboardRows.swift:66`, `Server/TypeSafePlaygroundCards.swift:124,170`,
  `Server/TypeSafeBenchmarkView.swift:57`, `Installation/ModelProbeSheet.swift:104,253`,
  `Installation/SteeringVectorImportSheet.swift:206`, `Installation/ModelCardView.swift:45`,
  `Installation/AudioDiscoverCatalogView.swift:100`.
  A related duplication: three private micro meters exist
  (`StatusBarView.MeterBar`, `ChromeTelemetryView.TSMeterBar`,
  `ModelLoaderIndicators.ModelLoadProgressBar`) although `AppProgressBar`
  already has a `.compact` size.
- Hand-rolled icon-only buttons instead of `AppIconButton` (tooltip and
  VoiceOver name can drift): `Browser/BrowserPaneView.swift` (all toolbar
  controls), `Components/McpServerCardView.swift:77`, `Components/CronJobsSettingsPaneView.swift:168`,
  `Components/Hooks/*` (three different close buttons in one folder),
  `Components/ErrorBanner.swift:23`, `Generation/ArtifactPanelView.swift:216`,
  `Generation/ToastOverlayView.swift:59`, `Generation/PromptComposerFooter.swift:86`,
  `Generation/PromptAttachmentsView.swift:65`, `Server/ServerConsoleView.swift:212,254,328,369`,
  `Server/APIPaneComponents.swift:254`, `Installation/InstalledModelHeroHeaderView.swift:34`,
  `Installation/InstalledModelDeveloperCommandsView.swift:62`, plus ~10 more
  "(possible)" sites listed per file.
- Hand-rolled empty states instead of `AppEmptyState`: `Audio/AudioWorkspaceView.swift:115`,
  `Audio/AudioMeetingNotesView.swift:266`, `Audio/AudioCustomVocabularySheet.swift:149`,
  `Audio/AudioLibraryView.swift:53`, `Audio/AudioPromptLibraryView.swift:31`,
  `Components/AgentContentImportSheet.swift:177`, `Components/CustomModelFoldersSectionView.swift:76`,
  `Components/DownloadManagerView.swift:72`, `Components/SkillImportSheet.swift:218`,
  `Components/Hooks/HookSourceDetailSheet.swift:84`, `Hooks/HookOptionsConfigSheet.swift:44`,
  `Generation/ChatSidebarView.swift:338`, `Generation/ChatSidebarGroupedProjectsView.swift:441`,
  `Generation/WorktreeView.swift:390,441`, `Generation/WorktreeTimelineView.swift:35`,
  `Generation/WorktreeWorktreeListView.swift:52`, `Generation/SplitChatPanes.swift:118`,
  `Generation/Humanize/HumanizeView.swift:235`, `Images/ImageStylePicker.swift:171`,
  `Server/ServerConsoleView.swift:50`, `Server/ServerChartsView.swift:37`,
  `Server/TypeSafeAPIPaneView.swift:129`, `Components/ExcludedScanPathsSectionView.swift:22`.
  Missing entirely: `Components/SkillImportSheet.swift:176` (zero-skills list
  renders blank), `Browser/BrowserPaneView.swift:435` (no-tab workspace is a
  bare button).
- Hand-rolled search fields instead of `AppSearchField`:
  `Audio/AudioCustomVocabularySheet.swift:171`, `Generation/ChatSidebarView.swift:142`.
- Hand-rolled cards instead of `.appCardSurface(padding:)`:
  `Audio/AudioWorkspaceView.swift:149-230`, `Audio/AudioRecordingPanel.swift:25`,
  `Audio/AudioLibraryList.swift:94`, `Audio/AudioSearchResultsView.swift:74`,
  `Components/SkillDetailView.swift:73`, `Components/CronJobsSettingsPaneView.swift:224`,
  `Components/AgentDetailView.swift:137`, `Components/AgentModeHintsSectionView.swift:56`,
  `Components/Hooks/*` (groups, options), `Generation/InteractiveQuestionCardView.swift:53`,
  `Generation/ProjectPermissionsSectionView.swift:126-238 (3x in one file)`,
  `Server/TypeSafePlaygroundCards.swift:99,154`, `Server/ServerChartsView.swift:117`,
  `Server/ServerLiveChartsView.swift:42,65`, `Server/ServerInspectorView.swift:62`,
  `Installation/CatalogSheet.swift:154`, `Installation/ModelInstallCards.swift:56-205`,
  `Installation/ModelHubView.swift:138` + `ImageModelManagerView.swift:76` (same
  treatment duplicated across two files), `Installation/InstalledModelMemoryFitCardView.swift:57`,
  `Installation/InstalledModelFeatureBadgesView.swift:128`,
  `Installation/InstalledModelOrganizationCardView.swift:103`,
  `Installation/ModelTechnicalSpecsCardView.swift:92` (`modelCardStyle()` shared
  by other cards also hard-codes `.appBorder.opacity(0.5)` and radius 12),
  `Audio/RecordingAccessSheet.swift:219`.
- Bespoke button treatments instead of `.appAction`:
  `Components/McpServerEditorSheet.swift:247` (`.borderedProminent`),
  `Components/McpImportSheet.swift:146,280`, `Components/SkillImportSheet.swift:211`,
  `Components/Hooks/HookSourceDetailSheet.swift:159,317`,
  `Components/ProfileBackupSheets.swift:273` (`Select All` spelled three ways
  across three files), `Generation/InteractiveQuestionCardView.swift:66`,
  `Generation/ProjectToolApprovalSheet.swift:113`, `Generation/WorktreeView.swift:469`,
  `Generation/GoalBannerView.swift:42` (`.buttonStyle(.link)`),
  `Installation/ModelSteeringCardView.swift:180,222`, `Installation/SteeringVectorImportSheet.swift:217`,
  `Generation/ChatSearchResultRowView`, `Components/PluginMarketplaceSheet.swift:170-316`
  (six unstyled buttons against one styled sibling), plus the bespoke
  `chip()` in `Audio/MusicPromptGeneratorView.swift:324` re-implementing
  `AppChip` ~15 times.
- Bespoke menu treatments instead of `.appMenuAction()`:
  `Audio/AudioLibraryFilterBar.swift:116`, `Components/ProfilesSettingsPaneView.swift:200`,
  `Components/ThemeConfigCardView.swift:138`, `Components/ModelLibraryRowView? no`,
  `Generation/ChatSidebarChatRowView.swift:205`, `Generation/InteractiveMarkdownTableView.swift:162`,
  `Generation/ProjectPermissionsSectionView.swift:86`, `Generation/ChatShareButton.swift:20`,
  `Generation/ProjectMcpServerRowView.swift:93`, `Components/McpServerCardView.swift:122`.
- Filesystem paths not using `SettingsPathText` (code font, selectable, middle
  truncation, full path in `.help`): `Components/AgentContentImportSheet.swift:258`,
  `Components/PluginDetailView.swift:40`, `Components/AgentDetailView.swift:124`,
  `Generation/WorktreeDiffView.swift:102`, `Generation/WorktreeWorktreeListView.swift:95`,
  `Generation/ProjectToolApprovalSheet.swift:96`, `Generation/ToolApprovalSheet? n/a`,
  `Installation/ModelTechnicalSpecsCardView.swift:41` (not selectable),
  `Installation/InstalledModelTechnicalSpecsView.swift:54` (not selectable, wrong truncation),
  `Installation/AudioInstalledModelDetailView.swift:197`, `Audio/AudioModelPickerView.swift:350`.
- Idle mascot triplication: `AgentsCharacterView`, `MemoryCharacterView`,
  `SoulCharacterView` are three identical copies differing only in asset name
  and fallback glyph - extract one shared view (and give TSMotion a named slow
  idle curve).
- Dead/second spellings: `ConversationLayout.cardRadius` re-declares
  `AppMetrics.cardRadius`; `Components/ThemeGalleryView` and
  `Generation/InteractiveQuestionCardView` and others re-invent selection
  treatments; `Components/ProjectMcpApprovalSheet` and `Generation/ProjectMcpApprovalSheet`
  duplicate the approval summary rendering with the same unlocalized labels.

## Priority 5 - Motion

DESIGN.md section 5: TSMotion curves only, no inline durations, no
`repeatForever`, looping via phaseAnimator only while live.

- Bare `withAnimation` (ignores Reduce Motion) with inline durations:
  `Generation/ChatTranscriptView.swift:113,126,138`,
  `Generation/CollapsibleMessageContentView.swift:88`, `Generation/ToolGroupView.swift:69`,
  `Generation/WorktreeCommitListView.swift:29`, `Generation/ToolCodeCellView.swift:90,96`,
  `Generation/WorktreeFileListView.swift:182`, `Generation/WorktreeTimelineView.swift:74`,
  `Generation/WorktreeView.swift:80-182 (4 sites)`,
  `Generation/ToolResultOutputView.swift:31,218,224`,
  `Presentation/ChatMessageMarkdownView.swift:365,370`,
  `Components/TaskProgressIndicatorView.swift:97` (also
  `.repeatForever` from `withAnimation` - the explicitly banned pattern),
  `Files/AttachmentThumbnailStore.swift:139`.
- `.repeatForever` started from `.animation`: `Generation/WorktreeView.swift:173`
  (refresh spinner; should be `.tsPulse(active:)`).
- Inline durations inside `TSMotion.animate` / `.tsAnimation` (the wrapper is
  right, the duration is not a named curve): `App/RootView.swift:112-116 (5)`,
  `Chrome/StatusBarView.swift:88`, `Chrome/ModelLoaderIndicators.swift:53,103`,
  `Chrome/GenerationPhaseIndicator.swift:67`, `Generation/ArtifactPanelView.swift:468,471`,
  `Generation/MessageActionBarView.swift:226,236`, `Generation/ChatSidebarView.swift:105`,
  `Generation/ChatActivityLine` ok, `Generation/ContextUsageRingView.swift:72`,
  `Generation/SubagentLiveCardView.swift:112`, `Generation/OutputPaneView.swift:25,184`,
  `Components/Hooks/HookSourceDetailSheet.swift:295`, `Humanize/HumanizeChrome.swift:290`,
  `Humanize/HumanizeView` mascot loops (1.8s/2.4s inline; same in the three
  mascot copies and `Theme/TurboSparkMotion.swift:213,245`), `Components/ToastOverlayView.swift:34,60`,
  `Diagnostics/InspectorEssentialsSection.swift:211`.
- Live indicators without `.tsPulse`: `Audio/AudioControlsView.swift:187`
  (recording badge), `Generation/SplitChatPanes.swift:61` (static dot),
  `Chrome/ModelLoaderIndicators` (handled above).

## Priority 6 - Layout robustness

- Two-column breakpoints and column widths not scaled by
  `AppMetrics.textScale(theme)` (DESIGN.md: VoiceCloneWorkspaceView is the
  reference): `Audio/AudioWorkspaceView.swift:32,35`,
  `Audio/AudioLibraryView.swift:16-18`, `Chrome/TopBarView.swift:63`.
- Fixed `.frame(width:)` on pickers/search/labels/value readouts in rows
  (clips at Extra Large; the exact class that clipped the Agents scope
  picker): `Generation/ProjectPermissionsSectionView.swift:158,264` (175),
  `Generation/ProjectRulesSectionView.swift:51` (180),
  `Audio/MusicPromptGeneratorView.swift:87`, `Audio/AudioCustomVocabularySheet.swift:106`,
  `Diagnostics/GenerationSamplingSection.swift:91-160`, `Diagnostics/InspectorEssentialsSection.swift:233`,
  `Server/ServerConsoleView.swift:120,165,209`, `Server/ServerConnectCardView.swift:123`,
  `Server/TypeSafePlaygroundView.swift:67`, `Server/TypeSafePlaygroundCards.swift:76,169`,
  `Installation/ProbeReportCardView.swift:146-198`, `Installation/InstalledModelOrganizationCardView.swift:48`,
  `Installation/ImageGenerationSettings.swift:68`, `Installation/SteeringVectorImportSheet.swift:127`,
  `Components/Hooks/HookEditorSheet.swift:179`, `Generation/ContextLadderView.swift:57,60`,
  `Images/ImageGenerationSettings.swift:68`.
- Fixed sheet/window frames (should be min/ideal/max, scaled):
  `Audio/MusicPromptGeneratorView.swift:56` (640x720), `Audio/AudioPromptLibraryView.swift:72`,
  `Components/McpServerEditorSheet.swift:76`, `Components/McpImportSheet.swift:86`,
  `Components/SoulSettingsSection.swift:298` (360x190), `Components/SteeringPresetEditorSheet.swift:50`,
  `Components/SystemPromptSettingsSection.swift:129`, `Components/PersonalitySettingsSection.swift:109`,
  `Components/ModelProviderDiscoverySheet.swift:58`, `Generation/ChatGitSheets.swift:69`,
  `Generation/ChatDialogSheets.swift:85,190`, `Generation/ChatParitySheets.swift:366`,
  `Generation/ArtifactPanelView.swift:426` (1100x800 maximize; shared with
  `Files/FilePreviewView.swift:293`), `Generation/ChatSearchOverlayView.swift:73`,
  `Installation/ImageModelManagerView.swift:381`, `Installation/ModelProbeSheet` header,
  `Audio/AudioSpeakerNamingSheet.swift:59` (minWidth not scaled).
- `minimumScaleFactor` shrinking text: `Server/ServerLiveChartsView.swift:27`.
- Headers as bare `HStack + Spacer` that can truncate instead of
  `AppAdaptiveHeader`: pervasive but the confirmed-at-risk ones include
  `Audio/AudioModelPickerView.swift:67`, `Audio/AudioMeetingNotesView.swift:88`,
  `Audio/AudioWorkspaceInputs.swift:63,88`, `Audio/AudioWorkspaceView.swift:75`,
  `Components/PermissionsInfoSectionsView.swift:99` (heading+toggle, the exact
  DESIGN.md example), `Components/DownloadManagerView.swift:60`,
  `Components/AgentContentImportSheet.swift:112`, `Components/MarketplaceSourcesView.swift:17`,
  `Components/SkillEditorSheet.swift:31`, `Generation/ArtifactPanelView.swift:121`
  (7 trailing controls, title truncated with no tooltip),
  `Generation/SplitChatPanes.swift:59`, `Generation/ProjectMcpDetectionSectionView.swift:18`,
  `Generation/WorktreeWorktreeListView.swift:36`, `Generation/ChatSidebarProjectsSectionView.swift:21`,
  `Diagnostics/InspectorView.swift:39`, `Images/ImageStylePicker.swift:124`,
  `Images/ImageGalleryCarousel.swift:23`, `Installation/ModelProbeSheet.swift:80`,
  `Installation/SteeringVectorImportSheet.swift:60`, `Installation/InstalledModelHeroHeaderView.swift:18`,
  `Server/ServerConsoleView.swift:84`, `Components/Hooks` sheets.

## Priority 7 - Accessibility gaps

- Sheets that never focus their first field (DESIGN.md: "a sheet focuses its
  first field"): `Components/SkillEditorSheet.swift`, `Components/AgentEditorSheet.swift`,
  `Components/SoulSettingsSection.swift` (SoulNameSheet), `Components/SteeringPresetEditorSheet.swift`,
  `Components/McpServerEditorSheet.swift`, `Components/McpImportSheet.swift`,
  `Components/SkillImportSheet.swift`, `Components/ProfileBackupSheets.swift` (both),
  `Components/Hooks/HookEditorSheet.swift`, `Hooks/HookOptionsConfigSheet.swift`,
  `Server/BrowserDialogApprovalView.swift:58`, `App/ProfileUnlockView.swift:29`
  (SecureField), `Generation/ProjectSettingsSheet` (no shortcuts either).
- Sheets missing `.keyboardShortcut(.defaultAction)` on the primary action or
  `.cancelAction` on Cancel: `Components/AgentContentImportSheet.swift:288`,
  `Components/SkillImportSheet.swift:113`, `Components/PluginMarketplaceSheet.swift:36`,
  `Components/Hooks/HookEditorSheet.swift:221`, `Hooks/HookOptionsConfigSheet.swift:71`,
  `Generation/ProjectSettingsSheet.swift:405`, `Generation/ProfileMemoryCaptureSheet.swift:38`,
  `Server/ServerMenuDashboardView` ok.
- Sliders/steppers without `.accessibilityValue`: `Audio/SpeechInputParts.swift:104,187`,
  `Audio/AudioSpeechControls.swift:52`, `Components/SteeringPresetEditorSheet.swift:147`,
  `Components/AgentEditorSheet.swift:110`, `Browser/BrowserPaneView.swift:661` (3 steppers),
  `Server/TypeSafeBenchmarkView.swift:26`.
- Section headings missing `.accessibilityAddTraits(.isHeader)` (vocabulary
  heading views add it themselves; bare `Text`/`Label` do not):
  `Components/PluginDetailView.swift:56-206`, `Components/SkillDetailView` partial,
  `Components/McpTransportFieldsView.swift:22,121`, `Components/AgentModeHintsSectionView.swift:19`,
  `Components/AppearanceSettingsPaneView.swift:121`, `Components/AgentsCharacterView` n/a,
  `Components/Hooks/HookSourceDetailSheet.swift:188`, `Components/ThemeConfigCardView` ok,
  `Diagnostics/ServerChartsView? no`, `Server/ServerMenuDashboardView.swift:157`,
  `Server/ServerInspectorView.swift:10,53`, `Server/TypeSafeBenchmarkView.swift:24`,
  `Server/ServerChartsView.swift:22`, `Generation/ChatParitySheets.swift:369` (shared helper),
  `Generation/ChatSidebarGroupedProjectsView.swift:419`, `Generation/ProjectMcpSettingsSheet.swift:136-225`,
  `Generation/TodoChecklistViews.swift:152`, `Generation/ProjectChatSummaryView.swift:73`,
  `Generation/ProjectToolApprovalSheet.swift:35`, `Generation/ContextUsageRingView.swift:110`,
  `Images/ImageStylePicker.swift:126`, `Images/ImageGalleryCarousel.swift:24`,
  `Installation/ModelHardwareFitCardView.swift:34,96`, `Installation/ModelInstallView.swift:120`,
  `Installation/ModelSteeringCardView.swift:70`, `Installation/ImageModelManagerView.swift:136,152`,
  `Installation/ContextLadderView.swift:23`, `Installation/InstalledModelMemoryFitCardView.swift:16`,
  `Installation/InstalledModelFeatureBadgesView.swift:10`,
  `Installation/InstalledModelOrganizationCardView.swift:15`,
  `Installation/ModelTechnicalSpecsCardView.swift:26,79`,
  `Installation/InstalledModelTechnicalSpecsView.swift:12`,
  `Installation/InstalledModelDeveloperCommandsView.swift:33`,
  `Audio/VoiceReferenceControls.swift:16`, `Audio/Qwen3TTSInputsView.swift:12`,
  `Generation/PromptExamplesView.swift:12`, `App/MemorySettingsPaneView` partial.
- Missing `.isSelected` on selectable rows: `Generation/InteractiveMarkdownTableView.swift:107`,
  `Generation/WorktreeView.swift:78`, `Generation/WorktreeTimelineView.swift:73`,
  `Generation/WorktreeCommitListView.swift:85`, `Generation/ChatSearchResultRowView.swift:13`,
  `Generation/ToolApprovalDropdown.swift:117`, `Generation/ProjectPickerView.swift:85`,
  `Components/PluginMarketplaceSheet.swift:109`.
- Hover-only actions not exposed via `.accessibilityAction`:
  `Generation/ChatSidebarChatRowView.swift:68` (Archive, System Prompt, Split
  View missing from the mirrors), `Generation/ChatSidebarGroupedProjectsView.swift:220`,
  `Files/FilesSectionView.swift:238` (Go to chat/Reveal/Detach),
  `Server/ServerConsoleView.swift:232` (expand row is tap-only, no keyboard path).
- `.accessibilityElement(children: .combine)` swallowing row buttons:
  `Generation/ChatSidebarView.swift:421` (Restore/Delete folded),
  `Generation/SubagentLiveCardView.swift:105` (expand/stop/dismiss folded),
  `Generation/BackgroundShellsStripView.swift:89` (Kill folded).
- Icon-only `Menu` labels with `.help` but no `.accessibilityLabel`:
  `Images/ImageGalleryCarousel.swift:31`, `Images/ImageModelDownloadOffer.swift:62`
  (the correct spelling exists at `ImageGalleryCard.swift:66`).
- Toast/status changes that appear silently where `showToast` (which
  announces) should be used: `Server/ImageAPIPaneView.swift:122`,
  `Server/AudioAPIPaneView.swift:218`, `Server/TypeSafePlaygroundView.swift:117`,
  `Server/ServerConsoleView.swift:218`, `Generation/ChatGitSheets` errors.

## Priority 8 - Disabled controls with no "why"

The chat Send button is the documented reference; these disable without an
explanation (representative ~40 sites): `App/GeneralSettingsPaneView.swift:108,350`,
`App/MemorySettingsPaneView.swift:271`, `Components/ProfilesSettingsPaneView.swift:266,300`,
`Components/SkillEditorSheet.swift:151`, `Components/AgentEditorSheet.swift:186`,
`Components/HfAuthTokenCardView.swift:229`, `Components/Hooks/HookEditorSheet.swift:224`,
`Components/ModelInstallToastView.swift:56`, `Audio/AudioModelPickerView.swift:82,356,365`,
`Audio/AudioWorkspaceView.swift:259-380`, `Audio/AudioPromptLibraryView.swift:25`,
`Audio/SpeechToTextAnyAudioPane.swift:57`, `Audio/AudioSearchResultsView.swift:66`,
`Audio/VoiceReferenceControls.swift:22,54`, `Audio/AudioModelDownloadControls.swift:63`,
`Audio/AudioLibraryList.swift:90`, `Audio/AudioLibraryFilterBar.swift:149`,
`Generation/ChatSidebarView.swift:313`, `Generation/ProjectSettingsSheet.swift:413`,
`Generation/PromptComposerControls.swift:87`, `Generation/SplitChatPanes.swift:79`,
`Generation/ChatShareButton.swift:23`, `Generation/ChatGitSheets` ok,
`Server/TypeSafeBenchmarkView.swift:35`, `Server/ServerLoadedModelsView.swift:32`,
`Server/ImageAPIPaneView` Generate, `Installation/ModelProbeSheet` n/a,
`Installation/ImageModelManagerView.swift:294`, `Installation/AudioInstalledModelDetailView.swift:129`,
`Installation/AudioDiscoverCatalogView.swift:59`, `Installation/ModelHubFilterBarView.swift:112`,
`Files/FilesSectionView.swift:84`, `Components/GenerateControl.swift:48`,
`Components/EmbeddingModelPicker.swift:93`, `Components/ModelLibraryRowView` n/a,
`Components/HfMirrorEndpointRowView.swift:51`, `Components/DownloadManagerView.swift:173`,
`Components/McpImportSheet.swift:156`, `Components/AgentContentImportSheet.swift:156`,
`App/SoulSettingsPaneView` n/a.

## Priority 9 - Theme/ internals (the vocabulary itself)

- `Theme/AppControls.swift:364-371` - `AppChip` uses `Color.primary` /
  `Color.secondary` and hard-coded opacities with no `isHighContrast` branch
  (AppPillPicker does this correctly and is the in-file reference); its
  `cornerRadius: 8` also drifts from `AppMetrics.controlRadius` (7) despite
  DESIGN.md mapping chips to it.
- `Theme/AppControls.swift:308` - `AppEmptyState` message cap `maxWidth: 440`
  is not multiplied by `AppMetrics.textScale(theme)`.
- `Theme/TurboSparkMotion.swift:200,231` - TSIdlingSparkView reads bare
  `TurboSparkTheme.accentColor` (wrong mode under `.preferredColorScheme`);
  lines 213,245 use inline `easeInOut` durations (1.7s/2.3s) - add a named
  slow-breathe curve; line 9's doc comment says "four curves" but there are
  seven.
- `Theme/ThemeCodePreviewView.swift:68,88,103,143` and
  `Theme/ThemeTypographyPreviewView.swift:108,116,62-71` - ambient-resolved
  `Color(nsColor: .separatorColor)` / `.secondary` / `Color.primary.opacity`
  over cards that resolve their theme explicitly; one of the side-by-side
  Light/Dark cards always disagrees with the ambient appearance.
- `Theme/AppWaveform.swift:69` - the ripple loop is a `TimelineView(.animation)`
  rather than the documented phaseAnimator pattern (informational; honors
  pause and Reduce Motion).
- Clean vocabulary files: AppProgressBar, AppAttention, AppPillPicker,
  AppPreferenceOptions, `Components/SettingsPathText` (fulfills its contract),
  `Components/ExtensionScopePicker`.

## Also worth fixing (cross-cutting)

- `Generation/ToolCallCardView.swift` is 1186 lines (3x the ~400 guidance);
  split the preview builders into sibling files. Also 726+ lines:
  `Browser/BrowserPaneView.swift`, `Components/AgentContentImportSheet.swift` (603).
- Empty-state title drift: "No Models Installed" (ModelManagerView) vs
  "No models installed" (ModelHubView) - two catalog keys for one phrase.
- `Generation/ConversationLayout.readingWidth` and `ContextLadderView` widths
  are not textScale-multiplied.
- Non-view code feeding the UI with English (root cause outside the view
  files, surfaced through them): `ModelInstallGate` reason strings,
  `MusicMemoryAdvisor.tierSummary`, `AppSteeringPolicy.Compatibility.summary`,
  `McpApprovalSummary` labels, `LoadingPhrases`, `WorktreeFileChange.Status.color`,
  `ModelFitPresentation.color`, `ModelFamilyVisuals` palette, `AppToolCategory.label`,
  `WorktreeTabMode.title`, `AudioWorkspacePage` (unregistered enum).

## Suggested fix order

1. Whole-file localization bypasses (Priority 1) - pure user-visible English
   in 21 languages; also add the missing keys found in otherwise-bundled
   files and register the three enums in the parity test.
2. Confirmation dialogs on destructive actions (Priority 2 table) - one shared
   dialog pattern; then the quiet-`minus.circle` sweep for rebuildable
   removals.
3. `App/ProfileUnlockView` scene theming - the lock screen is shown to every
   locked user and currently renders the light fallback theme.
4. Static `TurboSparkTheme` accessor sweep (Priority 3.1) - mechanical
   (`theme.accent` / `.appAccent` / `.appSurface`); lowers the ratchet
   baseline in one coordinated change.
5. System-palette status colors sweep with `AppAttention.color` plus
   theme-resolved success/error tokens; unify model-row selection on
   `theme.selection`.
6. Vocabulary adoption sweeps (Priority 4): `AppProgressBar`, `AppEmptyState`,
   `AppIconButton`, `.appMenuAction()`, `.appCardSurface`, `SettingsPathText`,
   `AppChip` in MusicPromptGeneratorView; extract the shared mascot and the
   shared micro meter.
7. Motion sweep (Priority 5) - TSMotion curves, `.tsPulse`, kill the
   `repeatForever` sites.
8. textScale/layout sweep (Priority 6) - breakpoints, row widths, sheet
   frames.
9. Sheet focus + keyboard shortcut sweep and `.accessibilityValue` /
   `.isHeader` / `.isSelected` / hover-action mirrors (Priority 7).
10. Theme/ internals (Priority 9) - AppChip contrast/radius, TSIdlingSparkView,
    preview-view ambient colors, AppEmptyState width scale.

Ratchet rule while doing 4-6: `DesignTokenRatchetTests` counts these
spellings outside `Theme/`; fix sites and lower their baselines in the same
change, never raise one.

Verification after each batch of fixes (DESIGN.md section 9):

```sh
make compile-strings
(cd swift/TurboSparkApp && swift build -Xbuild-tools-swiftc -suppress-warnings)
(cd swift/TurboSparkApp && swift test --filter 'LocalizationParityTests|LocalizationTests|LocalizationAndAccessibilityTests|FontPropagationTests|KeyboardShortcutCatalogTests|UIAuditFixesTests|SettingsWindowLayoutTests')
python3 scripts/generate-settings-catalog.py --check
make swift-test-app
```

plus the in-app checklist (text sizes, narrow window, themes, keyboard,
VoiceOver, Reduce Motion/Transparency/Contrast, EN/DE/RTL) on every touched
screen.

---

# Per-file findings

Findings below are verbatim from the per-batch reviews, grouped by batch.
Paths are relative to the repository root. Severity labels are the reviewing
agent's; see the calibration note above.

### Batch batch01_App

Audit against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md (read in full). Section references below are DESIGN.md sections. Cross-checked facts: `SettingsTab.memory.title == "Memory"` (App/SettingsTab.swift:15), `AppThemeColor` is a struct conforming to `ShapeStyle` (Theme/AppThemeColor.swift:18), and `TurboSparkApp.swift` applies `.appThemed()` only to `AppSettingsView` (line 225) and the MenuBarExtra (line 261), never to the main Window branch that shows `ProfileUnlockView` when the vault is locked.

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/GeneralSettingsPaneView.swift
- [LOW] line 108 - Section 2 action hierarchy ("A disabled control says why, in its tooltip or in adjacent text") - "Automatically check for updates" toggle is disabled when `!SparkleUpdateController.isBundledApp` with no tooltip or adjacent text explaining why (running unpackaged) - add `.help(Text("Updates are checked only in the installed app.", bundle: .module))` or a caption. (possible)
- [LOW] line 350 - Section 2 action hierarchy - "Use it even when the first turn is slow" toggle is disabled when native tool calling is off; the caption below describes the feature but never states the dependency on the toggle above - add a tooltip or extend the caption with "Requires native tool calling". (possible)
- Otherwise compliant: purpose line (line 23), `.formStyle(.grouped)` + `.scrollContentBackground(.hidden)` + `.background(.appPage)` (lines 52-55), one computed property per Section, `theme.ui`/`theme.code` fonts, all strings bundled or verbatim numbers, min/ideal/max field widths (lines 314, 381), shared option rows for Language and Text Size. 424 lines, not far beyond ~400.

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift
- [MEDIUM] lines 112-116 - Section 5 Motion ("Use the TSMotion curves... No inline durations"; bare `.animation(_:value:)` instead of `.tsAnimation`) - five `.animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value:)` calls with an inline duration; Reduce Motion is honored by hand but the curve is not a TSMotion curve - replace with `.tsAnimation(TSMotion.pane, value:)` (one call per value), which carries the curve and the Reduce Motion handling.
- Otherwise compliant: `.appThemed()` on the whole scene content (line 233), `.appBorder` hairlines, `HSplitView` adjustable split (line 408), min/ideal/max frames, transitions gated on Reduce Motion (lines 351, 373), `AlternateShortcutBridge` mounted at window root per Section 7.

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/EngineSettingsPaneView.swift
- [LOW] line 335 - Section 4 Color ("In a ternary that mixes two styles, wrap each side in AnyShapeStyle") - `.foregroundStyle(enabled ? AppThemeColor.appText : AppThemeColor.appSecondary)` is a ternary of two styles without `AnyShapeStyle`; both sides share the same `AppThemeColor` type so it compiles, but the rule asks for the wrapped form - `enabled ? AnyShapeStyle(AppThemeColor.appText) : AnyShapeStyle(AppThemeColor.appSecondary)`. (possible)
- Otherwise compliant: one property per Section with the type-checker rationale documented, `.labelsHidden()` + real label closure on every row control, `.accessibilityValue` on sliders (lines 60, 86), min/ideal/max widths everywhere including the shared `EngineNumberFieldStyle`, `Text(verbatim:)` for numeric readouts, locale-aware readout shared with VoiceOver (line 326), disabled sliders paired with their enable toggle in the same row.

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/MemorySettingsPaneView.swift
- [HIGH] line 206 - Section 2 action hierarchy ("A destructive role with no confirmation dialog is always a defect") - the "Forget" button is `Button(role: .destructive)` and calls `MemoryLedgerStore.shared.forget(claim.id)` directly; there is no `confirmationDialog` anywhere in the file, and line 115 states forgetting removes this profile's memory records and future exports, so this is real data loss - add a `.confirmationDialog` with localized title, message, and destructive button, keyed on a `@State` holding the claim to forget.
- [MEDIUM] line 106 - Section 4 Color ("Never use system colors on a themed surface"; the theme decides color) - `Text(verbatim: managerError).foregroundStyle(.red)` draws system red that ignores custom themes - use a themed attention treatment (the Section 2 `.appAttention` + `AppAttentionNote` vocabulary) or a theme-resolved destructive style.
- [MEDIUM] line 246 - Section 3 window structure ("A pane never repeats its own title"; the shell already renders it) - the first Section header is `Text("Memory", bundle: .module)` and the settings shell header for this tab is also "Memory" (AppSettingsView.swift line 95 with `SettingsTab.memory = "Memory"`) - rename the section to what it contains (e.g. "Model Memory") or drop the header and lead with the Section 3 one-line purpose line, which this pane also lacks.
- [MEDIUM] lines 108-111 - Section 3 empty states ("Every list and workspace has an AppEmptyState that says what goes here and offers the action that fills it") - the filtered claims list's empty state is a bare `Text("No memories match this view.")` - use `AppEmptyState` with the filling action (approve pending memories, clear the search, or switch scope/status).
- [LOW] lines 224-227 - Section 3 empty states - "No reflections yet." is a bare Text line rather than `AppEmptyState` - wrap in `AppEmptyState(systemImage:title:message:)`.
- [LOW] line 271 - Section 2 action hierarchy ("A disabled control says why") - "Suggest memories from idle conversations" is disabled when `!model.memoryEnabled` with no tooltip or adjacent text naming the dependency; the caption below covers usage syntax only - add `.help` stating it needs memory enabled. (possible)
- [LOW] line 23 - Section 3 window structure ("It starts with a one-line purpose (.themedFont(.small), .appSecondary)...") - the pane opens directly into the "Memory" section with no one-line purpose line, unlike General/Engine/Keyboard Shortcuts - add one.

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/AppSettingsView.swift
- [MEDIUM] line 86 - Section 4 Surfaces and effects ("Surfaces go through the theme so Reduce Transparency and Increase Contrast apply: surfaceOpacity forces opaque structural surfaces") - the settings sidebar background is `.appPage.opacity(0.85)`, a hard-coded translucency that bypasses `surfaceOpacity`, so the structural sidebar stays see-through under Reduce Transparency - use `.background(.appPage)` and let the theme supply any softening, or route the opacity through the theme's surface opacity.
- [LOW] line 85 - Section 3 headers and rows ("a list column with min, ideal, and max widths. Never a fixed list width") - the navigation sidebar is `.frame(width: 252)` fixed with no flexibility; tab titles only survive Extra Large text by wrapping inside the fixed column - switch to `frame(minWidth:idealWidth:maxWidth:)` or an `HSplitView` column. (possible: the window minimum in `minimumSize` is derived from this 252, so the value is load-bearing)
- Otherwise compliant: shell renders the pane title once with `.isHeader` (line 97), themed fonts throughout, `.settingsControl`/`Text(LocalizedStringKey(_, bundle: .module))` for catalog-backed strings, sidebar rows expose `.accessibilityLabel`, `.isSelected`, `.help`, and the gradient overlay is theme-resolved (lines 126-133).

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutsSettingsPaneView.swift
- [MEDIUM] line 75 - Section 4 Color (`DesignTokenRatchetTests` counts hard-coded `.appBorder.opacity(...)` borders outside `Theme/` and "counts may only go down"; also Increase Contrast "raises borders") - the keycap outline is `.stroke(.appBorder.opacity(0.3), lineWidth: 1)`, a hard-coded low-opacity border that gets fainter under Increase Contrast - use plain `.appBorder` and let the theme raise it.
- Otherwise compliant: purpose line (line 13), grouped Form with hidden scroll background on `.appPage`, rows from `KeyboardShortcutCatalog` (one source of truth), `ViewThatFits` fallback so keys drop under the label, `.accessibilityElement(children: .ignore)` + combined spoken label (lines 56-57), catalog-key section titles and row labels.

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/ProfileUnlockView.swift
- [HIGH] line 9 (body) - Section 4 Color ("Every scene injects the theme... or it renders the light fallback theme in English") - this view is the locked-state root of BOTH scenes (TurboSparkApp.swift lines 153 and 227) and neither branch applies `.appThemed()`: the main Window branch sets only `preferredColorScheme`/`locale`/`layoutDirection` (the theme normally comes from RootView's own `.appThemed()`, which is not mounted while locked), and in the Settings scene `.appThemed()` is attached to `AppSettingsView` only. The lock screen therefore renders `ResolvedAppTheme.fallback` (light) under every dark and custom theme - apply `.appThemed()` to the scene Group in both branches, or to this view's body.
- [MEDIUM] line 13 - Section 4 Color (`.secondary` is an explicitly forbidden SwiftUI spelling on a themed surface) - lock glyph uses `.foregroundStyle(.secondary)` - use `.appSecondary`.
- [MEDIUM] line 62 - Section 4 Color - same: the recovery-passphrase note uses `.foregroundStyle(.secondary)` - use `.appSecondary`.
- [MEDIUM] line 56 - Section 4 Color (the theme decides color; system colors ignore the palette) - the unlock error message uses `.foregroundStyle(.red)` - use a theme-resolved destructive/attention style so custom themes and Increase Contrast apply.
- [MEDIUM] line 68 - Section 4 Color (system colors on a themed surface) - the whole lock screen paints `.background(.background)`, the system hierarchical background, which ignores the palette entirely (compounding the missing `.appThemed()` above) - use `.background(.appPage)` or `.appSurface`.
- [LOW] line 29 - Section 5 Keyboard ("a sheet focuses its first field") - the passphrase `SecureField` never takes initial focus on appear; only the quick-unlock path is auto-attempted - add `@FocusState` and focus the field on appear when quick unlock is unavailable. (possible: this is a window-covering lock screen rather than a sheet)
- Compliant otherwise: `.appAction(.primary)` + `.keyboardShortcut(.defaultAction)` on Unlock (lines 39-40), `Label`-based Touch ID action with bundled text, `Text(verbatim:)` for the stored label (line 24).

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/MemoryEditSheet.swift
- clean. Title carries `.isHeader` (line 23), the TextEditor uses the vocabulary `.appEditorSurface(minHeight:)` (line 26) with an `.accessibilityLabel`, Cancel/Save bind `.cancelAction`/`.defaultAction` (lines 32, 37), `MemoryClaimText` returns bundled `Text` per value with `Text(verbatim:)` only for stored open-ended kinds, and `confidence` uses `AppStrings.localized`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/AlternateShortcutBridge.swift
- clean. This is the mechanism DESIGN.md Section 7 itself prescribes for alternate chords ("Alternate chords are hidden buttons in AlternateShortcutBridge that mirror the menu item's disabled state"); the verbatim labels are intentional and the container is `.opacity(0)`, zero-sized, and `.accessibilityHidden(true)` (lines 50-52), and each button's disabled state mirrors the menu command it shadows.

#### swift/TurboSparkApp/Sources/TurboSparkApp/App/SoulSettingsPaneView.swift
- [LOW] line 8 - Section 3 window structure ("It starts with a one-line purpose (.themedFont(.small), .appSecondary)...") - the pane opens straight into the "SOUL.md" section with no one-line purpose line, unlike the other panes - add one.
- Otherwise compliant: no title repetition (headers "SOUL.md" and "Personality" differ from the shell's "Soul"; `Text(verbatim: "SOUL.md")` is correct for a file name), `.formStyle(.grouped)` + `.scrollContentBackground(.hidden)` + `.background(.appPage)` + `SettingsLayout.formInsets`, and the Personality section is the same shared component the Engine pane uses, which is the Section 3 sanctioned reuse for a second location.

---

### Batch batch02_Audio

Audit of 9 SwiftUI files in swift/TurboSparkApp/Sources/TurboSparkApp/Audio against DESIGN.md.
Catalog lookups were used to confirm which verbatim strings are absent from Localizable.xcstrings.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioModelPickerView.swift
- [MEDIUM] line 67-74 - Section 3 "Headers and rows" / section 2 vocabulary - Pane header built from a bare HStack (Label + Spacer + AppIconButton) instead of AppAdaptiveHeader; the "Audio model" title can truncate under German or Extra Large text, which is the exact defect AppAdaptiveHeader exists for - Replace with AppAdaptiveHeader(leading:actions:) (compact action set for Refresh).
- [MEDIUM] line 321 - Section 4 "Color" - `.foregroundStyle(decision.isBlocked ? Color.red : Color.orange)` uses system colors on a themed surface and mixes two styles in a ternary without AnyShapeStyle - Use themed status styles (or `.appAccent`/`.appSecondary` semantics) and wrap each side in AnyShapeStyle; the octagon/triangle glyphs already carry the state, so color need not be system red/orange.
- [MEDIUM] lines 82, 356, 365 - Section 2 action hierarchy "A disabled control says why" - The model Picker (disabled while `isBusy || hasRecordingActivity`) and the "Use downloaded model" / "Choose local model folder" buttons give no tooltip or adjacent text for why they are off - Add `.help(Text("...", bundle: .module))` or adjacent small text stating that recording is active or a job is running.
- [LOW] (possible) lines 394, 405 - Section 2 destructive guidance - `Button(role: .destructive)` for Cancel download has no confirmationDialog; canceling a download discards no user data, so per the rule it is either a quiet action or needs a dialog. The same convention exists in ModelInstallToastView.swift line 59, so fix belongs in the shared treatment, not this file alone.
- [LOW] lines 350-352 - Section 2 vocabulary "A filesystem path in a row" - Local-model folder shown as `Text(folder.lastPathComponent)` with manual truncation and `.help(folder.path)` instead of SettingsPathText (code font, selectable, middle truncation) - Use SettingsPathText or note why a bare name is intended here.
- [LOW] file length - Section 3 "Keep view files near 400 lines" - 483 lines - Split downloadProgress/installManagement into a sibling file.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioWorkspaceView.swift
- [MEDIUM] lines 32, 35 - Section 3 "A workspace that switches between one and two columns" - Two-column breakpoint is raw `geometry.size.width >= 940` and the side panel is a fixed `.frame(width: 300)`; neither is multiplied by `AppMetrics.textScale(theme)`, so Extra Large text squeezes both columns instead of falling back to one - Read the theme, scale the breakpoint and the column width like VoiceCloneWorkspaceView (the cited reference, which uses AppMetrics.textScale(theme)).
- [MEDIUM] lines 75-81 - Section 2 vocabulary "A pane or workspace header" - Workspace header hand-rolls ViewThatFits(HStack + Spacer / VStack) instead of AppAdaptiveHeader(leading:actions:compactActions:); it reflows but does not keep the primary action visible and fold the rest into a labeled menu - Replace with AppAdaptiveHeader.
- [MEDIUM] lines 115-127 - Section 3 "Empty states" / section 2 vocabulary - Hand-rolled empty state (waveform glyph, "No outputs yet", dashed-stroke RoundedRectangle(cornerRadius: 12)) instead of AppEmptyState(systemImage:title:message:){actions}, and it offers no action that fills it - Use AppEmptyState with an action (e.g. import or record).
- [MEDIUM] lines 302-306 - Section 3 "Text in layouts" (Gotcha 18) / section 2 vocabulary - Script TextEditor gets `.frame(minHeight: 180)` plus manual padding and border instead of the measured-height editor the rule names for Audio (AudioMeasuredTextEditor) or `.appEditorSurface(minHeight:)` - Use AudioMeasuredTextEditor or TextEditor(...).appEditorSurface(minHeight:).
- [MEDIUM] lines 149, 159-160, 178-179, 229-230 - Section 2 vocabulary "A card" / section 4 "Corners are continuous and use AppMetrics radii" - Cards hand-rolled as `.padding(...).background(.appSurface, in: RoundedRectangle(cornerRadius: 10/12))` + `.stroke(.appBorder)` instead of `.appCardSurface(padding:)`, with hard radii and no `.continuous` - Use .appCardSurface(padding:) (or AudioLibraryView's shared card helper).
- [MEDIUM] (possible) line 388 - Section 2 action hierarchy "When all that is missing is input the user can still give, do not leave the primary action dead" - Run button is `.disabled(!controller.canRun)`; AudioWorkspaceController.canRun (line 264) returns false for empty script/caption/clips (user-fixable input), not only for app-state blockers - Keep it pressable for input-missing cases and surface what is missing via .appAttention per the Voice Cloning reference.
- [LOW] lines 130, 301 - Section 5 "Titles and section headings carry .accessibilityAddTraits(.isHeader)" - "Saved audio" (line 130) and "Script" (line 301) headings lack the header trait - Add .accessibilityAddTraits(.isHeader).
- [LOW] lines 259, 279, 374, 380 - Section 2 "A disabled control says why" - Record/Import/Presets/Save draft disabled while busy with no tooltip; the status strip shows progress but not on the control - Add .help or accept a grouped convention; lowest priority of the disabled-without-why sites.
- [LOW] line 126 - Section 4 corners - Dashed overlay uses `RoundedRectangle(cornerRadius: 12)` without `style: .continuous` and a non-AppMetrics radius - Use AppMetrics.cardRadius with .continuous.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/RecordingAccessSheet.swift
- [LOW] lines 219-223, 331-335 - Section 2 vocabulary "A card" - Permission rows and the audio-use card hand-roll the exact AppCardSurfaceModifier recipe (padding + .appSurface background + .appBorder stroke) instead of `.appCardSurface(padding:)` - Use .appCardSurface(padding:) so future treatment changes land once.
- Otherwise clean: recordingAccessGate used correctly, refresh-only onAppear (no prompt), Allow buttons are the only request path, glyph + text status rows, Not now + fallback paths, .defaultAction/.cancelAction bound, disabled primary explained by footerNote, sliders n/a, AppSectionHeader with header trait.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/MusicPromptGeneratorView.swift
- [MEDIUM] line 56 - Section 3 "No window has a fixed frame" / layouts adapt - Sheet is `.frame(width: 640, height: 720)`, fixed and unscaled, while sibling sheets (RecordingAccessSheet, AudioCustomVocabularySheet) use min/ideal/max times textScale - Use `.frame(minWidth:idealWidth:maxWidth:minHeight:idealHeight:)` scaled by AppMetrics.textScale(theme).
- [MEDIUM] line 70 - Section 6 localization (bundled spellings table) - Placeholder `Text(verbatim: "a late-night city drive")` is an English UI string that never resolves the catalog (key absent from Localizable.xcstrings, verified) - `Text("a late-night city drive", bundle: .module)`.
- [MEDIUM] lines 324-336 - Section 2 vocabulary "A composer-style toggle chip" / "a second local copy is how the drift starts" - Local `chip(_:selected:disabled:action:)` (Capsule backgrounds, accent tint 0.18, border overlays) re-implements AppChip and is used ~15 times across both tabs - Use AppChip or move this selection-chip variant into the vocabulary.
- [LOW] line 87 - Section 3 "No fixed .frame(width:) on a picker, search field, or label in a header or form row" - BPM TextField is `.frame(width: 90)` - Use minWidth/idealWidth/maxWidth.
- [LOW] lines 161-162, 226-227 - Section 4 corners - Preview box and layout buttons use `RoundedRectangle(cornerRadius: 8)` stroke .appBorder without .continuous and outside the vocabulary surfaces - Use AppMetrics.controlRadius with .continuous via a shared surface modifier.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioMeetingNotesView.swift
- [HIGH] lines 71, 105, 127, 135, 146, 149, 152, 166, 170, 179, 183, 193, 205, 242, 273, 275, 286, 303 - Section 6 "Never pass a bare literal to a key-only API" - Every user-facing string is `Text(verbatim:)` on an English literal ("Transcript preserved: ground truth is immutable.", "No chat models installed", "Generate Notes", "Re-generate", "Copied!", "Export MD", "Jump to discussion:", "No Meeting Notes Yet", the whole empty-state paragraph, .help tooltips, "Generated with %@", the "Save Notes"/"Edit Notes" ternary, and more). None of these keys exist in Localizable.xcstrings (verified), so all 21 non-English languages render English - Convert to `Text("...", bundle: .module)` / `AppStrings.localized` and add the keys in all 21 languages.
- [HIGH] line 76 - Section 6 interpolation rule - `Text(verbatim: "\(rev.segments.count) segments (\(audioTime(...)))")` embeds the English word "segments" and an Int in a verbatim string - Localize with a %lld key and render the time via Text(verbatim:) concat.
- [HIGH] line 269 - Section 4 "Type: never write .font(.caption) or a point size" - Empty-state glyph uses `.font(.system(size: 40))` - Use a themedFont role or `.themedFont(fitting:)` per the glyph rule.
- [MEDIUM] lines 118, 288 - Section 6 "A String that reaches the UI without a Text uses AppStrings.localized" - Recommendation badges from AudioMeetingNotesAdvisor.recommendationBadge ("Recommended: Fast, 32k context", "Recommended: High speed") are English source strings rendered with Text(verbatim:) - Return catalog keys from the advisor and render through Text(LocalizedStringKey, bundle: .module).
- [MEDIUM] lines 218-225 - Section 3 TextEditor rule (Gotcha 18) / section 4 - Notes editor is `TextEditor.frame(minHeight: 280)` with a manual .appBorder overlay and no `.scrollContentBackground(.hidden)`, so the system text background draws on the themed surface and the height is not measured - Use AudioMeasuredTextEditor / .appEditorSurface(minHeight:).
- [MEDIUM] lines 266-314 - Section 2 vocabulary "An empty list or workspace" - Hand-rolled empty state (icon, title, paragraph, pill, primary button) instead of AppEmptyState(systemImage:title:message:){actions} - Use AppEmptyState; also add .accessibilityAddTraits(.isHeader) to its title.
- [MEDIUM] (possible) line 309 - Section 2 glows (".appPrimaryGlow under the one primary action while it can run") - `.appPrimaryGlow(true)` is a constant, so the glow runs while the button is disabled (no model, recording active) - Pass whether the action can run, as AudioRecordBar line 115 does.
- [LOW] lines 88-99 - Section 3 "A header never truncates" - headerControls is a bare HStack with Spacer (menu + buttons + primary) with no ViewThatFits/AppAdaptiveHeader reflow; can squeeze under German and Extra Large - Reflow with ViewThatFits or AppAdaptiveHeader.
- [LOW] lines 157, 258, 310 - Section 2 "A disabled control says why" - Summarize and time-jump buttons disabled with no tooltip explaining (no model installed, no clips) - Add .help with the reason.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioControlsView.swift
- [MEDIUM] line 188 - Section 6 "Ternaries hide from the gates: write cond ? Text : Text, never a ternary inside one Text" - `Text(controller.isPaused ? "Paused" : "Recording", bundle: .module)` puts the ternary inside the Text initializer, hiding both keys from the localization gates - `controller.isPaused ? Text("Paused", bundle: .module) : Text("Recording", bundle: .module)` (as done correctly at line 29).
- [MEDIUM] line 126 - Section 3 "No fixed .frame(width:) ... nor on a glyph sized with .themedFont (it grows with Text Size)" - AudioActionLabelStyle pins the icon to `.frame(width: 18)` while the icon inherits the button's stepped themed label font, so Extra Large text can clip the glyph - Use a minWidth or scale the frame by AppMetrics.textScale(theme).
- [MEDIUM] lines 187-191 - Section 2 "A live indicator (recording, active stage) | .tsPulse(active:)" - AudioRecordingBadge is the app-chrome live recording indicator but its record.circle/pause glyph has no .tsPulse(active:), unlike the VoiceReferenceCard recorder dot - Apply .tsPulse(active: controller.isRecording && !controller.isPaused) to the glyph.
- [LOW] line 149 - Section 2 "A disabled control says why" - Delete disabled when !controller.canDelete(item) with only a "Delete" tooltip - Add the reason to .help.
- Otherwise strong: slider with real label + .accessibilityValue + adjustable action, AppIconButton everywhere, destructive delete with confirmation and localized message, AppProgressBar with min/ideal/max width, AudioNameSheet resolves its String title through LocalizedStringKey + bundle with both keys present in the catalog (verified), sheet binds .defaultAction/.cancelAction and focuses its field.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceReferenceCard.swift
- [MEDIUM] line 138 - Section 4 "Never use system colors on a themed surface" - Recording dot is `Circle().fill(Color.red)`; the .tsPulse is right but the color ignores the theme - Use a themed red/accent (theme-resolved or a semantic status style).
- [MEDIUM] line 209 - Section 4 "Color" - Invalid-reference warning glyph uses `.foregroundStyle(.orange)`, a system color on a themed surface - Use a themed warning style; the glyph + text pairing itself is correct.
- [MEDIUM] line 221 - Section 3 "No fixed frame on a glyph sized with .themedFont" - Source icon is `.themedFont(.base)` inside `.frame(width: 32, height: 32)`, which clips at Extra Large - Use minWidth/minHeight or scale the badge by textScale.
- [LOW] (possible) line 106 - Section 4 "Glows are accent shadows from the vocabulary only" - Drop-target halo is a hand-rolled `.shadow(color: theme.accent.opacity(0.45), radius: 8)` outside .appFocusGlow/.appPrimaryGlow - Add a drop-target halo to the vocabulary or reuse an existing glow modifier.
- [LOW] (possible) line 157 - Section 4 "Surfaces go through the theme so Reduce Transparency applies" - Levels inset uses `.background(.appPage.opacity(0.6))`, a hard-coded translucency that stays see-through under Reduce Transparency - Use an opaque themed surface or route through surfaceOpacity.
- [LOW] (possible) lines 150-154 - Section 2 "AppProgressBar ... animated: false while paused" - Recording progress omits `animated:`, so the bar keeps animating if the capture is paused - Pass `animated: !(controller.isRecording && controller.isPaused)` or the equivalent.
- Otherwise a reference file: recordingAccessGate with refresh-then-guard, AppAttentionNote + appAttention ping, sliders with labelsHidden + .accessibilityValue, quiet neutral "Remove" (xmark, not red trash), tsAnimation/tsPulse, AnyShapeStyle ternaries at lines 98 and 104 done correctly.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioRecordBar.swift
- [MEDIUM] line 81 - Section 2 vocabulary ("A live input level ... AppLevelWaveform"; "Progress ... AppProgressBar") - Panel layout draws the live input level with a bare `ProgressView(value: level)` instead of AppProgressBar or the AudioLiveLevels waveform used by the capsule layout (line 102) and VoiceReferenceCard - Use AppProgressBar(value:label:size:) or AudioLiveLevels; the .accessibilityValue here is good and must be preserved.
- Otherwise clean: this file is the disabled-reason reference (startDisabledReason shown adjacent and in .help at lines 86-88 and 131, and gating the glow at line 115), gate usage and declined-source handling are correct, ternary-of-Texts shape correct, no bare animations.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioCustomVocabularySheet.swift
- [HIGH] lines 65, 74, 84, 89, 92, 93, 99, 102, 103, 111, 114, 115, 126, 143, 175, 176, 237, 250 - Section 6 "Never pass a bare literal to a key-only API" - All UI copy is `Text(verbatim:)` on English literals ("Custom Vocabulary & Jargon", the purpose sentence, "Add New Term", "Canonical Term", "Category (Optional)", "Sounds Like / Misrecognitions (comma separated)", "Registered Terms", "Filter terms...", the "Remove term from vocabulary" tooltip, "Done", plus prompts). Keys absent from Localizable.xcstrings (verified), so every non-English language renders English; even "Done" is rendered verbatim so the catalog entry is bypassed - Convert to Text("...", bundle: .module) and add the keys in all 21 languages.
- [HIGH] line 69 - Section 6 interpolation rule - `Text(verbatim: "\(terms.count) terms")` hardcodes English "terms" with an interpolated Int - Use a %lld localized key.
- [HIGH] line 152 - Section 4 "Type: never ... a point size" - Empty-state glyph uses `.font(.system(size: 28))` - Use a themedFont role.
- [MEDIUM] line 154 - Section 6 "Ternaries hide from the gates" - `Text(verbatim: searchQuery.isEmpty ? "No vocabulary terms registered yet" : "No terms matching '\(searchQuery)'")` is a ternary inside one Text with interpolation, all verbatim English - Two bundled Texts, with the match case as a %@ key.
- [MEDIUM] lines 171-194 - Section 2 vocabulary "Search | AppSearchField(text:prompt:)" - Hand-rolled search field (magnifyingglass icon, plain TextField, clear button, capsule background) duplicating AppSearchField - Use AppSearchField(text:prompt:).
- [MEDIUM] lines 230-236 - Section 5 "Icon-only actions use AppIconButton" / section 2 - Trash button is icon-only via .buttonStyle(.plain) with only a verbatim .help and no .accessibilityLabel, so VoiceOver gets no reliable name - Use AppIconButton(symbol, label: Text("Remove term...", bundle: .module)).
- [MEDIUM] line 106 - Section 3 "No fixed .frame(width:) ... in a form row" - Category TextField is `.frame(width: 140 * scale)`; scaled but still a fixed width, which the rule forbids in favor of minWidth/idealWidth/maxWidth - Switch to minWidth/idealWidth/maxWidth.
- [LOW] lines 149-160 - Section 2 vocabulary - Hand-rolled empty state (icon + message, no action) instead of AppEmptyState - Use AppEmptyState; it can also carry the Add Term action.
- [LOW] (possible) line 233 - Section 2 destructive guidance - Trash glyph for removing a rebuildable vocabulary entry; the rule prefers a quiet neutral glyph (minus.circle) for removals that delete nothing irreplaceable - Consider minus.circle via AppIconButton.
- [LOW] - Section 5 keyboard - Sheet never focuses its first field (no @FocusState/onAppear focus), unlike AudioNameSheet - Focus the Canonical Term field on appear.
- Positive: sheet frame uses min/ideal/max scaled by AppMetrics.textScale(theme) (line 53), appCardSurface used for rows and cards.

### Cross-file summary
- Two files (AudioMeetingNotesView.swift, AudioCustomVocabularySheet.swift) bypass the string catalog wholesale with Text(verbatim:) English literals; they are the batch's only HIGH-severity localization defects, plus hard point-size fonts in both.
- Recording-permission handling is exemplary across the batch: every capture path goes through .recordingAccessGate, no onAppear/toggle prompts, and only row-level Allow buttons request (RecordingAccessSheet.swift, VoiceReferenceCard.swift, AudioRecordBar.swift).
- Recurring vocabulary drift: hand-rolled cards duplicating .appCardSurface (AudioWorkspaceView, RecordingAccessSheet, MusicPromptGeneratorView), hand-rolled empty states instead of AppEmptyState (AudioWorkspaceView, AudioMeetingNotesView, AudioCustomVocabularySheet), and a local chip/search-field instead of AppChip/AppSearchField (MusicPromptGeneratorView, AudioCustomVocabularySheet).
- Recurring theme-blind colors: Color.red/Color.orange/.orange (AudioModelPickerView line 321, VoiceReferenceCard lines 138 and 209).
- No bare withAnimation/.animation/.repeatForever anywhere in the batch; motion goes through tsAnimation/tsPulse.

---

### Batch batch03_Audio

Audit of 14 Audio workspace SwiftUI files against DESIGN.md (sections 2-8). Read-only audit; no repository files changed.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioSpeakerNamingSheet.swift
- [HIGH] line 156 - Localization (section 6) - `Text(verbatim: "You (Microphone)")` is fixed English user-facing copy marked verbatim; the key is absent from Localizable.xcstrings, so it renders English under all 21 languages - use `Text("You (Microphone)", bundle: .module)` and add the key with translations.
- [HIGH] line 169 - Localization (section 6) - `.help(Text(verbatim: "Use suggested speaker name."))` is a fixed English sentence marked verbatim; key missing from the catalog - use `Text("Use suggested speaker name.", bundle: .module)`.
- [LOW] line 59 - Layout, text-scale rule (section 2/3) - sheet frame scales `idealWidth`/`maxWidth`/`idealHeight` by `scale` but leaves `minWidth: 460` and `minHeight: 360` unscaled while its name fields use `160 * scale` (possible) - scale the minimums the same way.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioWorkspaceInputs.swift
- [HIGH] lines 136-144 - Theming (section 4) - music validation note draws `.foregroundStyle(.yellow)` on the glyph and `.background(Color.yellow.opacity(0.12))` plus `.cornerRadius(8)`: system colors on a themed surface and a bespoke warning card duplicating the `AppAttentionNote`/`AppAttention.color` vocabulary; ignores custom themes (Gotcha 62 class) - use `AppAttentionNote` and the vocabulary attention color; radius from `AppMetrics`.
- [MEDIUM] lines 63-70 and 88-94 - Headers and rows (section 3) - "Music description" and "Lyrics" section headings are a bare `HStack { Text; Spacer; Button/Text }`, the exact shape DESIGN.md says truncates the title under German and Extra Large text - use `AppSectionHeader` (or fold the action into an `AppAdaptiveHeader`-style row).
- [MEDIUM] lines 71-84 and 95-108 - Vocabulary (section 2) - both music `TextEditor`s are hand-rolled: fixed `.frame(height: 110)` / `.frame(height: 100)`, manual placeholder overlay, and `RoundedRectangle(cornerRadius: 8).stroke(.appBorder)` instead of `TextEditor(...).appEditorSurface(minHeight:)` (or the module's own `AudioMeasuredTextEditor`); hard radius 8 is not `AppMetrics.controlRadius` - adopt the editor vocabulary.
- [MEDIUM] line 117 - Localization (section 6) - `Text(advice.tierSummary)` renders a String built in `MusicMemoryAdvisor.swift` lines 88-96 with hard-coded English `String(format:)` ("%.0f GB Unified Memory - Full context supported...") that never passes through `AppStrings.localized`, so it reaches the UI unlocalized - localize the format strings in the advisor (display site flagged here; root cause is in MusicMemoryAdvisor.swift).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/SpeechToTextModelPanel.swift
- clean

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioLibraryView.swift
- [HIGH] lines 16-18 - Layout, two-column breakpoint rule (section 3) - the library switches between `HSplitView` and stacked at `geometry.size.width > 850` with column widths `minWidth: 260, idealWidth: 310, maxWidth: 400`, none scaled by `AppMetrics.textScale(theme)`; the struct does not even read `\.appTheme`. Extra Large text squeezes both columns instead of falling back to one - scale the breakpoint and column widths (reference: `SpeechToTextSplit` in SpeechToTextWorkspaceView.swift lines 118-128).
- [HIGH] lines 164, 166, 173 - Localization (section 6) - detail tab picker rows use `Text(verbatim: "Transcript")`, `Text(verbatim: "Notes & Summary")`, and `Text(verbatim: "Detail View")`; fixed English marked verbatim, and "Notes & Summary"/"Detail View" keys are absent from the catalog entirely - use `Text(..., bundle: .module)` (tags are independent of the label text, so this is a drop-in change).
- [MEDIUM] lines 53-58 - Empty states (section 3) - the library's "No recordings or takes found" empty state is a hand-rolled `VStack` of two `Text`s, not `AppEmptyState`, and offers no fill action - use `AppEmptyState` with the Import audio action.
- [LOW] lines 78-83 - Empty states (section 3, possible) - the detail pane's "Choose a recording or take" placeholder is also hand-rolled rather than `AppEmptyState`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCloneScriptComposer.swift
- [MEDIUM] lines 76, 83 - Theming (section 4) - over-limit counter uses `Image(systemName:).foregroundStyle(.orange)` and `AnyShapeStyle(.orange)`: a system color and a second local copy of the attention color; the vocabulary already defines it once as `AppAttention.color` (Theme/AppAttention.swift line 12) - use `AppAttention.color`.
- [MEDIUM] line 231 - Theming (section 4) - `AudioMeasuredTextEditor` placeholder uses `.foregroundStyle(.tertiary)`, a spellings the rule and `DesignTokenRatchetTests` explicitly count as theme-blind - use `.appSecondary`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/SpeechInputParts.swift
- [MEDIUM] lines 104-106, 110-112 - Accessibility (section 5) - the Intelligibility and Similarity sliders carry `.accessibilityLabel` but no `.accessibilityValue`; their formatted values sit in adjacent `Text`s that are separate VoiceOver elements - add `.accessibilityValue` with the formatted value (same for the Temperature slider).
- [MEDIUM] line 187 - Accessibility (section 5) - Temperature slider in `Qwen3SpeechGenerationSettings` has `.accessibilityLabel` but no `.accessibilityValue` - add it.
- [MEDIUM] lines 170-172 - Vocabulary (section 2) - `Qwen3ReferenceTranscriptEditor` hand-rolls its editor: `TextEditor(...).frame(height: 90)` fixed height plus `RoundedRectangle(cornerRadius: 8).stroke(.appBorder)` instead of `.appEditorSurface(minHeight:)`/`AudioMeasuredTextEditor`, and no focus glow on a ring-less editor; radius 8 is not `AppMetrics.controlRadius` - adopt the editor vocabulary.
- [LOW] line 61 - Accessibility (section 2, possible) - when `onNeedsReference == nil` the consent toggle disables until a reference exists and the adjacent helper text explains consent, not why the box is disabled - add the reason to the helper text or `.help` for the Advanced-page placement.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioTranscriptEditorView.swift
- [HIGH] lines 97, 100 - Localization (section 6) - `Text(verbatim: "Vocabulary...")` and `.help(Text(verbatim: "Manage custom vocabulary and jargon"))` are fixed English marked verbatim; both keys are missing from the catalog - use `Text(..., bundle: .module)`.
- [MEDIUM] lines 44-47 - Vocabulary (section 2) - each transcript segment's `TextEditor` is styled by hand: `.frame(minHeight: 72)` plus `.padding(6).overlay { RoundedRectangle(cornerRadius: 6).stroke(.appBorder) }` instead of `.appEditorSurface(minHeight:)`; hard radius 6 is not an `AppMetrics` radius - adopt the editor vocabulary.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/SpeechToTextWorkspaceView.swift
- [HIGH] lines 86, 91 - Localization (section 6) - `Text(verbatim: "Vocabulary")` and `.help(Text(verbatim: "Custom vocabulary & jargon"))` are fixed English marked verbatim; both keys missing from the catalog, so the header button stays English in every language - use `Text(..., bundle: .module)`.
- (The rest of the file is the pattern to copy: `AppAdaptiveHeader`, `AppPillPicker`, `tsAnimation(TSMotion.pane)`, and `SpeechToTextSplit` scales its breakpoint and columns.)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCloneWorkspaceView.swift
- clean (this is the reference implementation for the scaled two-column breakpoint and column widths).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioLibraryFilterBar.swift
- [LOW] line 116 - Vocabulary (section 2, possible) - `facetMenu` styles its menus with `.menuStyle(.borderlessButton).fixedSize()` instead of `.appMenuAction(size:)`, a second menu treatment - consider the shared menu action.
- [LOW] line 149 - Accessibility / disabled-control rule (section 2, possible) - the Add tag field disables at `maxTags` with no explanation of why - add a `.help` or adjacent text stating the tag limit.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceReferencePickerSheet.swift
- [MEDIUM] line 107 - Layout, fixed-frame rule (section 3) - the row glyph is drawn with `.themedFont(.base)` but pinned in `.frame(width: 28, height: 28)`; DESIGN.md forbids a fixed frame on a glyph sized with `.themedFont` because the font grows with Text Size while the circle does not - scale the frame by `AppMetrics.textScale(theme)` or size it from the font.
- [LOW] line 44 - Layout, text-scale rule (section 2, possible) - the sheet's `minWidth: 460, idealWidth: 560, minHeight: 420, idealHeight: 580` are not scaled by text scale (compare AudioSpeakerNamingSheet.swift line 59, which scales its ideal/max).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCloneTakesList.swift
- [LOW] lines 12-16 - Headers (section 3, possible) - `AppSectionHeader` is constrained inside an `HStack` with a trailing count text; under long translations the shared width can squeeze the header - let the count wrap into the header's subtitle or give it `minLength` room.
- [LOW] lines 85-91 - Vocabulary (section 2, possible) - the per-take play control is an icon-only `Button` with `.labelStyle(.iconOnly)` plus a duplicated `.help`, instead of `AppIconButton` (which unifies tooltip and VoiceOver name); its `.frame(width: 18)` is also unscaled.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCloneProgressCard.swift
- [MEDIUM] line 52 - Surfaces and effects (section 4, possible) - the card casts a bespoke accent glow via `.shadow(color: theme.isHighContrast ? .clear : theme.accent.opacity(0.18), radius: 14, ...)`; DESIGN.md says glows are accent shadows from the vocabulary only (`.appFocusGlow`, `.appPrimaryGlow`). It is at least high-contrast-aware - either justify it as a documented vocabulary addition or drop it.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCloneStepsStrip.swift
- [HIGH] line 130 - Theming (section 4) - the to-do step capsule background uses `Color.primary.opacity(hovered ? 0.08 : 0.04)`, a spelling DESIGN.md explicitly forbids (ignores a custom theme's text colors; counted by `DesignTokenRatchetTests`) - derive it from `theme.secondaryText.opacity(...)` or a themed surface style.
- [MEDIUM] lines 121, 129 - Theming (section 4) - the attention state uses `.orange` for the tint and `Color.orange.opacity(...)` for the background: system colors and second copies of the attention color that `AppAttention.color` already defines - use `AppAttention.color`.
- [LOW] line 29 - Layout, text-scale rule (section 2, possible) - the stacked fallback `LazyVGrid` uses `GridItem(.adaptive(minimum: 150))`, a text-holding width not multiplied by `AppMetrics.textScale(theme)`, so Extra Large step capsules get cramped.

### Summary
- HIGH: 8 findings across 6 files (5 localization: verbatim English missing from the catalog; 2 theming: system colors / `Color.primary.opacity`; 1 layout: unscaled two-column breakpoint in AudioLibraryView).
- MEDIUM: 12 findings (editor vocabulary drift, bare HStack headers, missing slider accessibilityValues, system warning colors, unlocalized tierSummary, bespoke glow).
- LOW: 10 findings, most marked possible.
- Clean files: SpeechToTextModelPanel.swift, VoiceCloneWorkspaceView.swift (the reference implementation).

---

### Batch batch04_Audio

Audit of 14 Audio SwiftUI files against DESIGN.md. Read-only audit; no repository files changed.
Notes that apply across the batch:

- Localization is clean throughout. Every suspect string was checked against
  `swift/TurboSparkApp/Localization/Localizable.xcstrings` and all keys exist
  (including the enum-backed and ternary-held values "Collection", "Rename",
  "Save Prompt", the Draft/Recording/Processing/Completed/Interrupted/Failed
  status words, and all interpolated format keys such as "Downloading %@",
  "Play from %@", "Download live text model (%@)"). `Text(verbatim:)` is used
  only for user content, numbers, durations, and file paths, which is correct.
- `AppThemeColor.appAccent/.appSecondary` (AudioModelStatus.swift line 48) is
  environment-resolved (`Theme/AppThemeColor.swift` resolve(in:)), so it is
  theme-correct, and `.appPage.opacity(0.6)` routes through
  `theme.surfaceOpacity`, so it stays opaque under Reduce Transparency.
- `RoundedRectangle(cornerRadius:)` defaults to `style: .continuous`, so the
  flags below cover only the radius values not coming from `AppMetrics`
  (controlRadius 7, cardRadius 10, composerRadius 20).
- No file exceeds the ~400 line guideline (largest is 131 lines).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCloneEngineCards.swift
- [LOW] line 62 - Section 4, surfaces and effects - The "consent given" stroke is a hard-coded `.appAccent.opacity(0.35)` overlay that ignores Increase Contrast, where borders should strengthen; status is also carried by the shield glyph, so this is decorative - Use full `.appBorder`/`.appAccent` or the theme's `selection` color, or accept and note it (possible)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioModelStatus.swift
- clean

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioLibraryList.swift
- [MEDIUM] lines 94-96 - Section 2, control vocabulary - Row card is hand-rolled (`padding(12)` + `.appSurface` background + `.overlay` stroke with a selected/unselected ternary) instead of `.appCardSurface(padding: 12)`; duplicates the card treatment that AudioSearchResultsView re-copies - Use `.appCardSurface(padding: 12)` and overlay the selection stroke only
- [LOW] lines 95-96 - Section 4, corners - Magic `cornerRadius: 10` instead of `AppMetrics.cardRadius` (same value today; drifts silently if the token changes) - Use `AppMetrics.cardRadius`
- [LOW] line 90 - Section 2, action hierarchy ("A disabled control says why") - The row actions Menu is disabled for `hasRecordingActivity || isBusy` with no tooltip or adjacent text explaining why - Add `.help(Text("Unavailable while recording or generating.", bundle: .module))` or equivalent
- [LOW] lines 37, 39-40 - Localization, fixed English in a variable - `AudioNameSheet(title: editingCollection ? "Collection" : "Rename", ...)` passes English held in a String; safe today because AudioNameSheet resolves `Text(LocalizedStringKey(title), bundle: .module)` and both keys exist in the catalog, but it bypasses the gates - No change required; noted so future call sites keep every value a catalog key

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/SpeechToTextAnyAudioPane.swift
- [LOW] line 57 - Section 2, action hierarchy - Import button disabled for `isBusy || hasRecordingActivity` with no tooltip saying why (the Transcribe button on line 106 shows the intended pattern) - Add a `.help` with the blocked reason
- (Otherwise clean: AppEmptyState with action, appAction/appPrimaryGlow, themedFont, appCardSurface all used correctly)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioLiveTextView.swift
- [MEDIUM] line 86 - Section 2, control vocabulary ("Progress, determinate or not") - The live-text model download is shown as an indeterminate `ProgressView().controlSize(.small)` spinner although `download.fractionDone` (lines 71-73, 84) is available; the shared `AudioModelDownloadControls.progress` renders the same state with `AppProgressBar` - Replace the spinner + text with `AppProgressBar(value: download.fractionDone, label: Text("Downloading the live text model", bundle: .module))` or embed `AudioModelDownloadControls`
- [LOW] line 44 - Section 5, motion - `withAnimation(TSMotion.select)` relies on a hand-rolled reduce-motion check instead of `TSMotion.animate`, which exists exactly for action closures without environment scope; one of only three such sites in the app - Use `TSMotion.animate(TSMotion.select) { proxy.scrollTo(Self.end, anchor: .bottom) }`
- [LOW] lines 39-40 - Section 4, corners - `RoundedRectangle(cornerRadius: 8)` is a magic radius matching no AppMetrics token - Use `AppMetrics.controlRadius` (7)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceReferenceWaveformView.swift
- clean

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioModelDownloadControls.swift
- [LOW] line 91 - Section 2, action hierarchy ("A destructive role with no confirmation dialog is always a defect") - Cancel Download uses `Button(role: .destructive)` with `.appAction(.destructive)` and no confirmationDialog; cancelling discards partial bytes but is restartable, so it may not be destructive at all. The comment claims parity with the install toast, and the identical treatment exists app-wide (`Components/ModelInstallToastView.swift:59`, `Installation/ModelDetailPaneView.swift:319`, `Audio/AudioModelPickerView.swift:394,405`), so it reads as deliberate - Either drop the destructive role for a neutral quiet cancel, or add the confirmation dialog; if the pattern is intended, the rule needs a stated exception (possible)
- [LOW] line 63 - Section 2, action hierarchy - Download button disabled for `!canQueueModelDownload || isBusy` with no tooltip saying why - Add `.help` with the blocked reason

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioPromptLibraryView.swift
- [MEDIUM] line 72 - Section 1 principles / Section 3 layout ("Layouts adapt instead of clipping", "No window has a fixed frame") - The sheet is pinned with `.frame(width: 600, height: 540)`; the fixed height cannot grow under Extra Large text or German, squeezing header, error line, and footer - Use `minWidth`/`maxWidth`/`idealHeight` (AudioNameSheet at AudioControlsView.swift:314 uses `minWidth: 340` and is the local reference)
- [MEDIUM] lines 31-34 - Section 3, empty states - "No saved prompts" is a hand-rolled bare Text while the action that fills the list (Save Prompt) exists in the header; every list should use `AppEmptyState(systemImage:title:message:)` offering that action - Replace with AppEmptyState and pass the Save Prompt action
- [LOW] lines 58-59 - Section 2, control vocabulary - Prompt card is `padding(14)` + `.appSurface` background instead of `.appCardSurface(padding: 14)` (radius already uses AppMetrics.cardRadius with .continuous) - Use `.appCardSurface(padding: 14)`
- [LOW] line 25 - Section 2, action hierarchy - Save Prompt disabled via `!controller.canSavePrompt` with no explanation of why - Add a `.help` or adjacent text

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioSpeechControls.swift
- [MEDIUM] lines 52-53 - Section 5, accessibility ("Sliders and steppers expose their formatted value with .accessibilityValue") - The Speed Slider has `.accessibilityLabel` but no `.accessibilityValue`; the formatted value in the adjacent Text (line 54) is not programmatically tied to the slider - Add `.accessibilityValue(Text(controller.recipe.speed, format: .number.precision(.fractionLength(2))))`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioSearchResultsView.swift
- [MEDIUM] lines 74-76 - Section 2, control vocabulary - Result row card duplicates the AudioLibraryList hand-rolled treatment (`padding(12)` + `.appSurface` + `.overlay` stroke) instead of a shared `.appCardSurface` treatment - Use `.appCardSurface(padding: 12)` plus the selection stroke only
- [LOW] lines 75-76 - Section 4, corners - Magic `cornerRadius: 10` instead of `AppMetrics.cardRadius` - Use the token
- [LOW] line 25 - Section 3, empty states - "No matches" is a hand-rolled bare Text instead of `AppEmptyState(systemImage:title:message:)`; no action is required for a search miss, but the vocabulary component is the consistent form - Replace with AppEmptyState (possible)
- [LOW] line 20 - Section 2, control vocabulary ("A live indicator") - The live indexing stage is shown with a `ProgressView().controlSize(.mini)` spinner inside the Label icon rather than a `.tsPulse(active:)` indicator from the vocabulary - Consider `.tsPulse(active: search.isIndexing)` on the glyph or an AppProgressBar (possible)
- [LOW] line 66 - Section 2, action hierarchy - "Play from ..." quiet button disabled for `hasRecordingActivity || !hasPlayableAudio` with no tooltip saying why - Add `.help`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceReferenceControls.swift
- [MEDIUM] line 16 - Section 5 accessibility ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") and Section 2 (heading vocabulary) - The card heading "Reference voice" is a bare `Text` with `themedFont(.base, weight: .semibold)`; it is a section heading with no `.isHeader` trait, while the sibling engine cards in VoiceCloneEngineCards.swift use `AppSectionHeader`, which adds the trait itself - Use `AppSectionHeader(title: Text("Reference voice", bundle: .module))`
- [LOW] lines 22, 54 - Section 2, action hierarchy - Import (`.disabled(controller.importTask != nil)`) and Preview (`.disabled(!reference.isValid)`) are disabled with no tooltip saying why - Add `.help` with the blocked reason

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/SpeechToTextCapturePane.swift
- clean

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioSidebarPanel.swift
- clean

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/Qwen3TTSInputsView.swift
- [HIGH] lines 13-15 - Section 2, control vocabulary ("A multi-line editor: TextEditor(...).appEditorSurface(minHeight:)") and Section 3 ("TextEditor has no intrinsic height; measure with a hidden Text instead") - The style editor is a bare `TextEditor` pinned with a fixed `.frame(height: 80)` and a hand-rolled border (`.scrollContentBackground(.hidden).padding(8).overlay { RoundedRectangle(...).stroke(.appBorder) }`) instead of `.appEditorSurface(minHeight:)`; the fixed height clips or wastes space under Extra Large text, and `AudioMeasuredTextEditor` already exists in this same directory for measured sizing - Replace with `TextEditor(...).appEditorSurface(minHeight: 80)` (plus the existing `.accessibilityLabel`)
- [LOW] line 15 - Section 4, corners - Magic `cornerRadius: 8` on the hand-rolled border, matching no AppMetrics token - Use `AppMetrics.controlRadius` (moot once `.appEditorSurface` is adopted)
- [LOW] line 12 - Section 5 accessibility - "Voice description or style" is a bare Text group label with no `.accessibilityAddTraits(.isHeader)`; comparable group headings in VoiceCloneEngineCards use AppSectionHeader (possible) - Use `AppSectionHeader` or add the trait

---

### Batch batch06_Browser

Audit against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md (sections cited as S1-S9). Read-only audit; no repo files modified. Catalog keys were verified against swift/TurboSparkApp/Localization/Localizable.xcstrings.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Browser/BrowserPaneView.swift

- [HIGH] line 307 - S4 Color (theme-blind styles) - Element-picker highlight fill uses `Color.accentColor.opacity(0.12)`, which ignores a custom theme's accent - use `.appAccent`
- [HIGH] line 310 - S4 Color - Highlight stroke uses `Color.accentColor` on themed chrome - use `.appAccent`
- [HIGH] line 519 - S4 Color - Selected-tab background is the ternary `model.selectedTabID == tab.id ? Color.accentColor.opacity(0.18) : Color.clear`: theme-blind accent plus a style ternary without `AnyShapeStyle` - `AnyShapeStyle(.appAccent.opacity(0.18))` / `AnyShapeStyle(Color.clear)`
- [MEDIUM] line 417 - S4 Color / S1 status - Invalid-address error text uses `.foregroundStyle(.red)`; static system red ignores the theme palette - use `AppAttention.color` (status also keeps its message text, satisfying not-color-alone)
- [MEDIUM] lines 255-258 - S2 vocabulary (Progress) - Loading state is a hand-rolled `ProgressView(value:)` in a material capsule - use `AppProgressBar(value:)`
- [MEDIUM] lines 435-444 - S3 Empty states - The no-tab workspace is a bare centered Button; every workspace gets an `AppEmptyState(systemImage:title:message:)` that offers the filling action - wrap the New Tab action in `AppEmptyState`
- [MEDIUM] lines 482-489, 525-533, 539-605 - S2 vocabulary (icon-only actions) - All icon-only actions (new tab, per-tab close, back/forward/reload/stop, open in system browser, pick element, viewport) are plain `.borderless` Buttons with separately spelled `.help` and `.accessibilityLabel`; `AppIconButton` exists so the tooltip and the VoiceOver name cannot drift - convert to `AppIconButton(symbol, label: Text(..., bundle: .module))`. (Labels are present, so not HIGH.)
- [MEDIUM] lines 661-719 - S5 Accessibility (values) - Width/Height/Zoom Steppers expose their formatted value only as label text; no `.accessibilityValue` on any stepper - add `.accessibilityValue(Text(verbatim: ...))` per stepper
- [MEDIUM] lines 350-362 - S2 action hierarchy - Retry is the failure card's single primary action but a default-styled Button - `.appAction(.primary)` with the card-row size
- [LOW] lines 306, 309, 366, 520 - S4 corners - `RoundedRectangle(cornerRadius: 4/12/6)` values are not `AppMetrics` radii (controlRadius 7, cardRadius 10, composerRadius 20) - use vocabulary radii (RoundedRectangle already defaults to `style: .continuous`)
- [LOW] lines 627-637 - S2 vocabulary (actions) - Attach/Cancel in the element-picker bar are default-styled Buttons - `.appAction(.secondary)` / `.appAction(.quiet)`
- [LOW] line 1 (file) - S3 keep view files near 400 lines - File is 726 lines, near double the guidance - split tab strip, toolbar, and viewport popover into sibling files

Notes: `Text(tab.title)` (line 508), `Text(verbatim:)` of candidate role/name (618) and px/zoom values (676, 696, 716) are correct verbatim uses of page content, units, and numbers. `.regularMaterial`/`.bar` appear only on floating overlays and navigation chrome (258, 276, 366, 492, 641), which S4 permits. No animation APIs in the file. All checked keys exist in the catalog ("Loading", "Agent controlled", "Enter a valid HTTP or HTTPS address.", "Attach", "Pick Element", "Fixed CSS viewport", etc.).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Browser/ChromeBookmarkImportPreviewView.swift

- [HIGH] line 451 - S6 Localization - Issue key "The selected profile Bookmarks file exceeds the size limit." is NOT in Localizable.xcstrings (exact grep count 0); rendered via `Text(issueKey(issue), bundle: .module)` (lines 313, 361, 376, 384) it falls back to English under all 21 languages - add the key with translations in the same change
- [HIGH] line 453 - S6 Localization - Issue key "The selected profile Bookmarks file could not be parsed." is also missing from the catalog (grep count 0) - add the key with translations
- [MEDIUM] lines 290-295, 459-466 - S2 "a surface that still needs the user" - The "Select at least one item to continue." feedback after a failed Preview/Import press is plain text; the vocabulary is `.appAttention(_:ping:radius:)` with an `AppAttentionNote`, ping bumped on each attempt - apply the attention treatment to the surface that still waits
- [MEDIUM] line 293 - S4 Color - Feedback text uses `.foregroundStyle(.red)`, theme-blind on a themed surface - use `AppAttention.color`
- [MEDIUM] lines 361-362 - S4 Color - Per-profile issue text uses `.foregroundStyle(.red)` - use `AppAttention.color`
- [MEDIUM] line 286 - S2 vocabulary (Progress) - Indeterminate `ProgressView()` while scanning profiles - `AppProgressBar` (determinate or not)
- [MEDIUM] lines 306-307, 350-351, 358-359 - S5 Accessibility (headings) - "Choose profiles to preview", the "Preview" sub-heading, and the per-profile name heading lack `.accessibilityAddTraits(.isHeader)`, so VoiceOver heading navigation skips them - add the trait
- [LOW] lines 309-315 - S3 Empty states - The no-profiles case is hand-rolled secondary text instead of `AppEmptyState(systemImage:title:message:)` - use AppEmptyState
- [LOW] (possible) lines 434-466 - S6 "Components take Text, not a key" - countLabel/issueKey/feedbackKey funnel `LocalizedStringKey` through helpers that apply `bundle: .module` internally instead of taking `Text`; keys are literals at call sites so the gates partially see them, but the two missing keys above show exactly how this pattern drifts - take `Text` parameters applied at the call site

Notes: `Toggle(profile.displayLabel, ...)` (319) and `Text(row.title)` / `Text(detail)` (418-421) are Chrome profile names, bookmark titles, and origin URLs: dynamic content, verbatim is correct. `Text(count, format: .number)` (437) is a number, correct. All other checked keys exist ("Import Chrome bookmarks", "Choose profiles to preview", "Skipped items", "Imported", "Saved bookmarks exceed the 2 MiB settings limit.", "Select at least one item to continue.", the remaining issue keys).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Browser/BrowserSettingsPaneView.swift

- [LOW] lines 48, 88 - S1 one source of truth - The same long explanation sentence literal appears twice (first-enable alert message and the automation section text); the copies will drift - share one constant feeding both bundled `Text`s
- Otherwise clean: grouped Form with `.scrollContentBackground(.hidden)` and `.background(.appPage)` (26-35), one computed property per Section, no repeated pane title, `.labelsHidden()` field style keeps the real label for VoiceOver (210), min/ideal/max frame instead of fixed width (212), verbatim unit glyphs with `.accessibilityHidden(true)` (197-204), `.appAction(.secondary)` on the import button (141), sheet gets `.appThemed()` (53). All checked keys exist.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Browser/BrowserViewportController.swift

- clean. As expected it is mostly controller logic and an NSView host; it contains no UI strings, no colors, no fonts, and no animation calls.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Browser/BrowserDialogApprovalView.swift

- [MEDIUM] (possible) lines 58-62 - S5 Keyboard ("a sheet focuses its first field") - The prompt TextField in this modal overlay has no `.focused` binding and never takes initial focus, so keyboard users must tab into it - focus the field when the dialog appears
- [LOW] lines 66-80 - S2 action hierarchy - Dismiss/Accept are default-styled Buttons; Accept is the surface's primary action - `.appAction(.primary)` / `.appAction(.secondary)` (the `.cancelAction`/`.defaultAction` shortcuts already present are correct)
- [LOW] line 85 - S4 corners - `RoundedRectangle(cornerRadius: 14)` is not an AppMetrics radius (controlRadius 7, cardRadius 10, composerRadius 20) - use a vocabulary radius
- [LOW] (possible) lines 100-106 - S6 "fixed English held in a variable" - `titleKey` is a computed property returning `LocalizedStringKey` literals; all three values ("Page alert", "Page confirmation", "Page prompt") are verified catalog keys, but the pattern is invisible to the source gates (the import view's identical pattern shipped two missing keys) - consider registering it in `testEnumBackedDisplayStringsAreCatalogKeys`

Notes: `Text(request.message)` (53) is web page dialog content: verbatim correct. `Text("Message from \(...)")` (49) interpolates a String; the "Message from %@" key exists in the catalog. `Color.black.opacity(0.28)` scrim (39) sits over web page content, not a themed app surface, so it is not flagged; `.shadow(radius: 24)` (86) is modal elevation, not an accent glow. `.accessibilityAddTraits(.isModal)` (88) is correct.

### Summary

- 4 HIGH: three theme-blind `Color.accentColor` sites in BrowserPaneView, two missing catalog keys in ChromeBookmarkImportPreviewView (issue keys for oversized and unparseable Bookmarks files render English under all 21 languages).
- 11 MEDIUM: ProgressView instead of AppProgressBar (x2), hand-rolled empty state, plain icon-only buttons instead of AppIconButton, steppers without accessibilityValue, plain primary actions, theme-blind `.red` (x3), missing `.appAttention` feedback surface, missing header traits, and one possible focus defect in the dialog approval overlay.
- LOW findings: non-AppMetrics corner radii, default-styled secondary actions, a duplicated long literal, helper-localized key pattern, and one over-length file (726 lines).

No withAnimation/.animation/.repeatForever violations in this batch (no animation calls at all). No Text(verbatim:) on fixed English UI copy. No destructive-role buttons anywhere in the batch, so no confirmation-dialog defects.

---

### Batch batch07_Chrome

Audit of the app-chrome files against DESIGN.md (sections 2-8). Localization
keys were verified against `swift/TurboSparkApp/Localization/Localizable.xcstrings`
(all keys used by these files, including every interpolated format key, exist
with all 21 languages). Ratchet counts referenced below are from
`Tests/TurboSparkAppTests/DesignTokenRatchetTests.swift`; a ratchet-counted
site is still a DESIGN.md section 4 violation (counts may only go down).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/ModelLoaderControl.swift
- [MEDIUM] line 84 - DESIGN.md 4, TurboSparkTheme static accessors - `.background(TurboSparkTheme.surfaceColor, in: Capsule())` reads the bare static accessor (resolved via `NSApp.effectiveAppearance`) inside a view body; draws the wrong mode under `.preferredColorScheme` (Gotcha 36) and is counted by DesignTokenRatchetTests - use the resolved theme (`.appSurface`) or the mode-explicit form.
- [MEDIUM] line 99 - DESIGN.md 4, TurboSparkTheme static accessors - `strokeColor` falls back to bare `TurboSparkTheme.hairlineColor` in a view body - use `TurboSparkTheme.hairlineColor(isDark: theme.isDark)`.
- [MEDIUM] line 182 - DESIGN.md 4, SwiftUI style spellings - `.foregroundStyle(.tertiary)` on the chooser chevron ignores a custom theme's text colors (ratchet-counted "system secondary style") - use `.appSecondary`.
- [MEDIUM] line 283 - DESIGN.md 4, SwiftUI style spellings - same `.foregroundStyle(.tertiary)` on the reasoning chevron - use `.appSecondary`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/StatusBarView.swift
- [MEDIUM] line 88 - DESIGN.md 5, motion - `TSMotion.animate(.easeInOut(duration: 0.15))` embeds an inline duration; section 5 says no inline durations, use the named TSMotion curves - use `TSMotion.hover`/`TSMotion.select`.
- [MEDIUM] line 145 - DESIGN.md 4, system colors - `MeterBar(fraction: tint: hint?.urgent == true ? .orange : theme.accent)` uses system `.orange` for the urgent context meter tint - use `AppAttention.color` for the warning tint.
- [MEDIUM] lines 253-256 - DESIGN.md 4, system colors - thermal tint switch returns system `.red` / `.orange` on the themed strip (color-alone rule is satisfied by the glyph plus text, but the colors ignore the palette) - resolve through the theme / `AppAttention.color`.
- [MEDIUM] line 308 - DESIGN.md 4, system colors - `.foregroundStyle(isCritical ? .red : .orange)` on the memory-pressure readout - use `AppAttention.color` (critical could get a theme-resolved error token).
- [LOW] lines 387-404 - DESIGN.md 2, shared vocabulary - private `MeterBar` hand-rolls a two-tone meter instead of `AppProgressBar` (which already has a `.compact` "status strip" size); an identical private copy exists in ChromeTelemetryView (`TSMeterBar`), and DESIGN.md says something needed twice joins the vocabulary - promote one meter into `Theme/` or adopt `AppProgressBar(size: .compact)`.
- [LOW] line 100 - DESIGN.md 4, surfaces - `RoundedRectangle(cornerRadius: 4)` hard-coded radius not from `AppMetrics` and without `style: .continuous` - use an `AppMetrics` radius with `.continuous`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/TopBarView.swift
- [MEDIUM] line 63 - DESIGN.md 3/2, width breakpoints - `if barWidth >= 760` gates the telemetry group on a raw point breakpoint; a width breakpoint is set in points times `AppMetrics.textScale(theme)` so Extra Large text does not squeeze the band - multiply the threshold by `AppMetrics.textScale(theme)`.
- [MEDIUM] line 240 - DESIGN.md 4, TurboSparkTheme static accessors - `gitPill` background `.background(TurboSparkTheme.surfaceColor, in: Capsule())` reads the bare static accessor in a view body (ratchet-counted) - use the resolved theme or mode-explicit form.
- [MEDIUM] line 311 - DESIGN.md 4, hard-coded opacity on secondary text - `theme.secondaryText.opacity(0.35)` for the disabled eject button hard-codes dimming below what the theme gives (and ignores Increase Contrast) - let the disabled state render, or use a theme-provided disabled treatment.
- [MEDIUM] lines 228-233 - DESIGN.md 4, system colors - `.foregroundStyle(.green)` / `.foregroundStyle(.red)` for git additions/deletions on themed chrome; system colors ignore the palette (same class as `.red`/`.orange` in the rule) - use theme-resolved semantic tokens.
- [LOW] lines 133, 154, 195, 292, 307 - DESIGN.md 3, fixed frames - `.frame(width: buttonSize, height: buttonSize)` pins square frames around glyphs sized with `theme.ui(.callout, ...)`, which grows with the in-app Text Size while `buttonSize` is a `@ScaledMetric` tied to system Dynamic Type only; no clipping today, but this is the pattern section 3 warns about - scale `buttonSize` by `AppMetrics.textScale(theme)` or let the frame grow.
- [LOW] line 241 - DESIGN.md 4, borders - `Capsule().stroke(.appText.opacity(0.08), lineWidth: 0.5)` hard-codes a border opacity Increase Contrast cannot raise - stroke `.appBorder`.
- [LOW] lines 160, 201 - DESIGN.md 4, surfaces - `RoundedRectangle(cornerRadius: 7, style: .continuous)` hard-codes the radius; 7 happens to equal `AppMetrics.controlRadius`, but the constant is the rule - use `AppMetrics.controlRadius`.
- [LOW] lines 311-314 - DESIGN.md 2, action hierarchy - the disabled unload button's tooltip is just "Unload Model"; a disabled control says why (the chat Send button is the reference) - extend `.help` to explain when it is disabled (e.g. no model loaded).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/NavigationRailView.swift
- [LOW] line 121 - DESIGN.md 4, surfaces - rail menu background uses `RoundedRectangle(cornerRadius: 8)` without `style: .continuous` (the sibling `railButton` at line 144 uses `.continuous`), and 8 is a hard-coded radius not from `AppMetrics` - add `.continuous` and use an `AppMetrics` radius.
- Other checks pass: mode-explicit `TurboSparkTheme.sidebarBackgroundColor(isDark:reduceTransparency:)` / `railBackgroundColor(...)` (lines 66-69), `.isSelected` on rail items (lines 129, 170), TSMotion for hover (lines 172, 199).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/ChromeTelemetryView.swift
- [MEDIUM] lines 92, 112 - DESIGN.md 4, SwiftUI style spellings - `.foregroundStyle(.tertiary)` on the "CPU" and "GPU wait" captions ignores a custom theme (ratchet-counted) - use `.appSecondary`.
- [MEDIUM] lines 131-132 - DESIGN.md 4, system colors - `memoryTint` returns system `.red` / `.orange` thresholds - use `AppAttention.color` / theme-resolved tokens.
- [MEDIUM] lines 174-190 - DESIGN.md 2, shared vocabulary - private `TSMeterBar` is a second local copy of StatusBarView's private `MeterBar` (the comment at lines 171-173 admits it); DESIGN.md: if the vocabulary lacks something you need twice, add it there - promote one shared micro meter or use `AppProgressBar(size: .compact)`.
- [LOW] line 35 - DESIGN.md 4, surfaces - `.background(.appText.opacity(0.04), in: RoundedRectangle(cornerRadius: 9, style: .continuous))` hard-codes a surface tint opacity and a radius (9) that is not an `AppMetrics` value (possible; intended as a quiet inset card) - use `.appElevated`/`.appSurface` and an `AppMetrics` radius.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/FanReadoutView.swift
- [MEDIUM] line 53 - DESIGN.md 4, system colors - `.foregroundStyle(isPinned ? Color.orange : theme.secondaryText)` on the fan glyph uses system `Color.orange` for the pinned/warning state - use `AppAttention.color`.
- [MEDIUM] line 61 - DESIGN.md 4, system colors - `.foregroundStyle(Color.orange)` on the "MAX" badge - use `AppAttention.color`.
- [MEDIUM] line 96 - DESIGN.md 4, system colors - `.foregroundStyle(fan.isHeld ? Color.orange : theme.secondaryText)` on the "Held" state - use `AppAttention.color`.
- [MEDIUM] line 152 - DESIGN.md 4, system colors - `.foregroundStyle(.red)` on the fan error text ignores the palette - use a theme-resolved error/attention color.
- [MEDIUM] lines 103, 115, 157 - DESIGN.md 4, SwiftUI style spellings - `.foregroundStyle(.tertiary)` in three readouts ignores a custom theme (ratchet-counted) - use `.appSecondary`.
- Other checks pass: `.appAction(.secondary, size: .small)` on the popover buttons (lines 124, 129), real Toggle label (line 144), correct ternary-outside-Text pattern (line 95), accessibility value on the readout (lines 68-70). Popover content without `.appThemed()` matches the codebase convention (only 1 of 7 popovers adds it), so not flagged.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/SidebarPanelList.swift
- [LOW] lines 74, 80 - DESIGN.md 4, surfaces - `RoundedRectangle(cornerRadius: 7, style: .continuous)` hard-codes the row radius instead of `AppMetrics.controlRadius` (same value today) - use the constant.
- Otherwise this file is the reference implementation the audit context calls part of the vocabulary: header trait (line 32), `.isSelected` (line 85), hover through `TSMotion.animate` (line 87), titles wrap instead of truncate (lines 65-66), group container with label (lines 122-123).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/ModelLoaderIndicators.swift
- [MEDIUM] lines 53-57 - DESIGN.md 5, motion - `ModelStatusDot`'s pulse hand-rolls a `phaseAnimator` with inline `.easeInOut(duration: 0.85)`; that is exactly `TSMotion.breathe`, and the vocabulary for a live indicator is `.tsPulse(active:)` - use `.tsPulse(active:)` or `TSMotion.breathe` (the modifier-only-while-live structure itself is correct).
- [MEDIUM] lines 103-107 - DESIGN.md 5, motion - `ModelLoadProgressBar`'s indeterminate sweep uses inline `.easeInOut(duration: 0.95)`, a near-copy of `TSMotion.sweep` - use the TSMotion curve; no inline durations.
- [LOW] lines 71-114 - DESIGN.md 2, shared vocabulary - hand-rolled hairline progress bar; the doc comment rules out `ProgressView` but not `AppProgressBar`, which already has `.compact` and an indeterminate animated sweep - adopt it, or if the 3pt hairline size is genuinely missing add it to `AppProgressBar` (it is now needed in three places: here, MeterBar, TSMeterBar).
- [LOW] line 86 - DESIGN.md 4, surfaces - track `.appText.opacity(0.12)` hard-codes an opacity the theme does not control - resolve through the shared meter once it exists.
- Reduce-motion handling is present and correct (lines 93-98 static half-bar; the pulse is gated off, line 40).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/ProfileMenuButton.swift
- clean. Uses the shared `AppearanceModeOptions()`, `TextSizeOptions()`, `LanguageOptions()` (lines 65-74, satisfying the section 7 display-preference rule for the sidebar profile menu), bundled Texts throughout, all keys present in 21 languages, avatar scales with `theme.sidebarScale`, mode-resolved `theme.selection`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/AppSidebarView.swift
- [LOW] line 42 - DESIGN.md 3, list widths (possible) - the panel column is pinned with `.frame(width: AppChromeLayout.sidebarPanelWidth)` rather than min/ideal/max; it is a deliberately fixed chrome band whose width feeds `MainWindowMinimumSize`/`leftChromeWidth` and rows wrap instead of truncate, so this reads as intended - flag only if the panel starts hosting a primary resizable list/detail split.
- Otherwise clean: mode-explicit sidebar background (lines 50-51), panel never repeats the section list (rail answers navigation), band structure matches section 3's four-chrome-band rule.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/GenerationPhaseIndicator.swift
- [MEDIUM] line 36 - DESIGN.md 4, TurboSparkTheme static accessors - `.background(TurboSparkTheme.surfaceColor, in: Capsule())` reads the bare static accessor in a view body (ratchet-counted) - use the resolved theme or mode-explicit form.
- [MEDIUM] lines 67-71 - DESIGN.md 5, motion - `PulsingIndicatorDot` uses a `phaseAnimator` with inline `.easeInOut(duration: 0.7)`, another near-copy of `TSMotion.breathe` - use `.tsPulse(active:)` / `TSMotion.breathe` (the live-state gating at line 21 is correct).
- [LOW] line 20 - DESIGN.md 2, shared vocabulary - `ProgressView().controlSize(.mini).scaleEffect(0.7)` for the loading state; progress belongs to `AppProgressBar` (or the sibling `ModelLoadProgressBar`), and the `scaleEffect` hack is the tell (possible; a native mini spinner may be defensible under "native first").

#### swift/TurboSparkApp/Sources/TurboSparkApp/Chrome/SidebarPresentation.swift
- [LOW] line 54 - DESIGN.md 4, surfaces (possible) - `.opacity(isEnabled ? 1 : 0.5)` hard-codes the disabled dimming in `SidebarActionButtonStyle`; it does not respond to `theme.isHighContrast` the way themed borders/secondary text must - consider a theme-resolved disabled treatment.
- [LOW] lines 49, 51, 53 - DESIGN.md 4, surfaces - `RoundedRectangle(cornerRadius: 7)` (fill, stroke, contentShape) hard-codes the radius instead of `AppMetrics.controlRadius`, and none pass `style: .continuous` - use the constant with `.continuous`.

### Cross-file observations

- Repeated defect class across this batch: bare `TurboSparkTheme.surfaceColor` / `hairlineColor` reads in view bodies (ModelLoaderControl 84/99, TopBarView 240, GenerationPhaseIndicator 36) and `.foregroundStyle(.tertiary)` (ModelLoaderControl 182/283, ChromeTelemetryView 92/112, FanReadoutView 103/115/157). All are ratchet-counted sites; fixing any of them requires lowering the baseline in the same change.
- Warning/status tints reach for system `.orange`/`.red` in every telemetry readout (StatusBarView 145/253-256/308, ChromeTelemetryView 131-132, FanReadoutView 53/61/96/152, TopBarView 228-233). `AppAttention.color` exists and is the designated spelling for warning tints; a theme-resolved critical/error token would close the rest.
- Three hand-rolled micro meters now exist (StatusBarView `MeterBar`, ChromeTelemetryView `TSMeterBar`, `ModelLoadProgressBar`) against `AppProgressBar` with its `.compact` size; this is the "needed twice, add to the vocabulary" case DESIGN.md 2 calls out.
- No localization defects: no bundleless literals, no `Text(verbatim:)` on translatable copy, no ternaries inside `Text`, no Double/CGFloat/Date interpolation into keys; every interpolated format key (including `%lld%% until auto-compact`, `Ready -- %@ ctx`, `Reasoning effort: %@. ...`) exists with all 21 languages.
- No file approaches the ~400 line guideline (largest is StatusBarView.swift at 420 lines including its private chart, borderline informational only).

---

### Batch batch08_Components

Audit against DESIGN.md (sections cited per line). Read-only audit; string-catalog keys for every flagged file were checked programmatically, and all bundled/interpolated keys used by these files exist with 21 translations, so no localization defects are reported. Note: `.appSurface.opacity(n)` was NOT flagged anywhere - `AppThemeColor.opacity` composes with `theme.surfaceOpacity` in `Theme/AppThemeColor.swift`, so it stays theme- and Reduce-Transparency-aware.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/AgentContentImportSheet.swift
- [HIGH] line 250 - Section 4 Color - "Already imported" pill background is `Color.secondary.opacity(0.15)`, a system style that ignores the custom theme (ratchet-gated spelling) - use a themed fill, e.g. `.appSecondary.opacity(0.15)` or `.appElevated`.
- [HIGH] line 217 - Section 4 Color - `.foregroundStyle(.tertiary)` on the group count (ratchet-gated `(.tertiary)`) - use `.appSecondary`.
- [HIGH] line 260 - Section 4 Color - `.foregroundStyle(.tertiary)` on the source-location text - use `.appSecondary`.
- [HIGH] line 280 - Section 4 Color - `.foregroundStyle(.tertiary)` on the "|" footer separator - use `.appSecondary` (or drop the divider).
- [HIGH] line 283 - Section 4 Color - `.foregroundStyle(.tertiary)` on "Servers are added disabled." - use `.appSecondary` (Increase Contrast raises `.appSecondary`, never `.tertiary`).
- [MEDIUM] line 156 - Section 2 Action hierarchy ("A disabled control says why") - the Target Scope picker disables when no project is open with no tooltip or adjacent reason (McpImportSheet's scope bar shows "Open a project to install into one." at its line 122) - add `.help(Text(..., bundle: .module))` or the same adjacent hint.
- [MEDIUM] line 170 - Section 2 vocabulary ("Progress, determinate or not | AppProgressBar"; AppProgressBar's doc says "Not ProgressView") - the pane-level loading state draws a bare `ProgressView()` - use `AppProgressBar(value: nil, label: ...)` (themed indeterminate sweep).
- [MEDIUM] lines 177-192 - Section 3 Empty states / Section 2 vocabulary - the "No importable content found" state is a hand-rolled VStack (Image + Text + Button) instead of `AppEmptyState(systemImage:title:message:){actions}` - rebuild, keeping the Choose Custom Folder action in the actions closure.
- [MEDIUM] line 258 - Section 2 vocabulary ("A filesystem path in a row | SettingsPathText") - `candidate.sourceLocation` (`~/.claude/skills/pdf`, config paths) is drawn as a plain `.themedFont(.tiny)` Text: not code font, not selectable, no middle truncation, no full-path `.help` - use `SettingsPathText(candidate.sourceLocation, size: .tiny)`.
- [MEDIUM] lines 288-294 - Section 5 Keyboard ("Sheets bind Save or Done to .keyboardShortcut(.defaultAction)") - the sheet's primary action "Import Selected" has no default-action shortcut (Close correctly has `.cancelAction` at line 128) - add `.keyboardShortcut(.defaultAction)` to the Import Selected button.
- [LOW] (possible) lines 112-133 - Section 3 Headers ("A header never truncates. Use AppAdaptiveHeader") - sheet header is a bare HStack(title + Spacer + Close); the texts do wrap freely today, but AppAdaptiveHeader is the vocabulary for title-plus-action chrome.
- [LOW] line 188 - Section 2 vocabulary ("An action | .appAction") - "Choose Custom Folder..." uses `.buttonStyle(.bordered)` instead of `.appAction(.secondary)`.
- [LOW] (possible) lines 223-225, 275-277 - Section 2 vocabulary - "Select All"/"Deselect All" quiet actions are `.buttonStyle(.plain)` + hand-set `.themedFont` instead of `.appAction(.quiet, size: .small)`.
- [LOW] file length - Section 3 ("Keep view files near 400 lines") - informational: 603 lines; the nonisolated scan helpers (lines 337-502) could move to a sibling file.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ProfilesSettingsPaneView.swift
- [HIGH] line 175 - Section 4 Color - "Current" pill background is `Color.accentColor.opacity(0.15)` (ratchet-gated) - use `.appAccent.opacity(0.15)`.
- [HIGH] line 176 - Section 4 Color - `.foregroundStyle(Color.accentColor)` ignores a custom theme's accent (ratchet-gated) - use `.appAccent`.
- [MEDIUM] lines 32-44 - Section 3 Window structure ("A pane ... starts with a one-line purpose (.themedFont(.small), .appSecondary)") - the pane opens directly into the "Profiles on this Mac" Section with no one-line purpose lead, unlike Files & Permissions and Models & Storage - add one above the first Section.
- [MEDIUM] lines 266-268 - Section 2 Action hierarchy ("A disabled control says why") - the permanently disabled "Switch to Export" button explains itself nowhere in the UI; the reason (inactive profiles cannot export) lives only in a code comment at lines 249-251 - add `.help(...)` or adjacent text.
- [LOW] lines 311-325, 364-379, 395-398 - Section 2 Action hierarchy ("Actions inside rows, cards, and inspectors use .small") - Change Passphrase / Disable Protection / Lock Now / the Restore buttons default to `.regular` while the same pane's row Switch button uses `.appAction(.secondary, size: .small)` (line 196) - use `.small` for in-row actions consistently.
- [LOW] (possible) lines 200-219 - Section 2 vocabulary ("An icon-only action | AppIconButton"; overflow | `.appMenuAction()`) - the row's ellipsis "More actions" menu is hand-styled with `.menuStyle(.borderlessButton)` + `.menuIndicator(.hidden)` + `.fixedSize()`; consider `.appMenuAction()` so header and row overflow menus share one treatment.
- [LOW] lines 300-306 - Section 2 Action hierarchy - the "Touch ID or Mac login" toggle disables when quick unlock is unavailable with no explanation of why - add `.help` or adjacent text.
- Clean otherwise: grouped Form with `.scrollContentBackground(.hidden)` + `.background(.appPage)` (lines 40-43), one property per Section, both destructive doors confirm via localized confirmationDialogs, row text combined for VoiceOver with real labels, Switch disabled-with-help.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ModelMoveSheet.swift
- [MEDIUM] lines 163-166 - Section 2 vocabulary ("Progress, determinate or not | AppProgressBar"; AppProgressBar's doc explicitly rejects ProgressView) - the move progress is a bare `ProgressView(value: fraction)` / `ProgressView().progressViewStyle(.linear)`, untintable with the theme accent on macOS 14 - use `AppProgressBar(value: fraction, label: ..., size:)` and `AppProgressBar(value: nil, ...)`.
- [MEDIUM] line 113 - same rule - the "Measuring..." row uses a small `ProgressView()` spinner - `AppProgressBar(value: nil, size: .compact)` is the themed indeterminate form.
- [MEDIUM] line 290 - Section 4 Color - refusal tint is `AnyShapeStyle(Color.orange)`, a system color no custom theme controls - use `AnyShapeStyle(AppAttention.color)` (the app's warning hue token).
- [LOW] (possible) line 206 - Section 4 Color - failure tint `AnyShapeStyle(Color.red)`; it matches the app's destructive convention (Theme's own `.appAction(.destructive)` uses `.red`) but is still a system spelling - adopt a themed destructive token here if one is added.
- Clean otherwise: the most disciplined file in the batch - every glyph pairs with text (status never color alone), keyboard shortcuts are deliberate and documented (no cancel shortcut while a move runs), `Text(LocalizedStringKey(...), bundle: .module)` for the enum-held phase titles per Section 6, verbatim only for drive-derived strings, AppMetrics radii with `.continuous`, `.accessibilityValue` on the combined progress element, `.accessibilityAddTraits(.isHeader)` on the title.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/PluginMarketplaceSheet.swift
- [HIGH] line 112 - Section 4 Color - marketplace row title is `name == selectedMarketplaceName ? Color.accentColor : Color.primary`: both system styles ignore a custom theme, and the ternary mixes styles without AnyShapeStyle (ratchet-gated) - `name == selectedMarketplaceName ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appText)`.
- [HIGH] line 268 - Section 4 Color - "No plugin entries loaded." uses `.foregroundStyle(.tertiary)` (ratchet-gated) - `.appSecondary`.
- [MEDIUM] line 58 - Section 3 ("List and detail splits are adjustable ... Never a fixed list width") - the marketplace source column is `.frame(width: 250)` - use `HSplitView` or a minWidth/idealWidth/maxWidth column so long source names and Extra Large text reflow.
- [MEDIUM] lines 234-244 - Section 2 Action hierarchy ("A destructive role with no confirmation dialog is always a defect"; removing a rebuildable entry "is a quiet action", not destructive) - "Remove" is `Button(role: .destructive)` with no confirmationDialog, and the sheet's own copy (line 46) says "Removing a source keeps installed items" - either drop the role for a quiet neutral action or add a localized confirmationDialog.
- [MEDIUM] (possible) lines 310-313 - Section 2 Action hierarchy - "Uninstall" is `role: .destructive` with no confirmationDialog; it removes installed files (rebuildable by reinstalling) - either add a localized confirmationDialog or make it a quiet action.
- [MEDIUM] line 36 - Section 5 Keyboard ("Sheets bind Save or Done to .keyboardShortcut(.defaultAction), Cancel to .cancelAction") - the sheet's only chrome button "Done" carries no keyboard shortcut at all; Return does nothing - bind `.keyboardShortcut(.cancelAction)` (this sheet has no other claim on it).
- [MEDIUM] lines 253-258 - Section 4 Color / vocabulary - the load-error banner tints the whole Label `.foregroundStyle(.orange)` (system orange) - use `AppAttention.color`, and consider the AppAttentionNote pattern for the glyph-plus-text error row.
- [MEDIUM] lines 265-270 - Section 3 Empty states ("Every list and workspace has an AppEmptyState") - the empty entries state is a bare Text, no icon, no action - use `AppEmptyState(systemImage:title:message:)` offering Refresh.
- [MEDIUM] lines 294-295 - Section 4 Color - "non-strict" pill: `Capsule().fill(Color.orange.opacity(0.15))` + `.foregroundStyle(.orange)` - system orange ignores the theme - use `AppAttention.color` for both.
- [MEDIUM] (possible) line 308 - Section 4 Color - the "Installed" confirmation uses `.foregroundStyle(.green)`, a system color no custom theme controls; no success token is named in DESIGN.md - adopt/define a themed success spelling (or `.appAccent`).
- [MEDIUM] lines 322-324 - Section 2 vocabulary ("A card | .appCardSurface(padding:)") + Section 4 radii - entry rows are hand-rolled cards: `RoundedRectangle(cornerRadius: 8)` filled `.appSurface` (cardRadius is 10) - use `.appCardSurface(padding: 10)`.
- [LOW] lines 98, 134, 222-228 - Section 5 Accessibility ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") - the "Sources", "Add marketplace" and selected-marketplace column headings carry no header trait.
- [LOW] line 109 - Section 5 Accessibility ("Selected rows ... carry .isSelected") - the selected marketplace row conveys selection by color only - add `.accessibilityAddTraits(.isSelected)` when selected.
- [LOW] lines 170, 231-233, 311-316 - Section 2 vocabulary ("An action | .appAction") - Add / Refresh / Install / Uninstall / Remove are unstyled native buttons while the sibling "Manage sources" (line 50) uses `.appAction(.secondary, size: .small)` - one treatment for the same sheet.
- [LOW] (possible) line 115 - Section 2 vocabulary - in-row busy spinner `ProgressView().controlSize(.small)` - `AppProgressBar(value: nil, size: .compact)` is the themed form.
- [LOW] (possible) lines 32-39 - Section 3 Headers - sheet header is a bare HStack + Spacer + Done, not `AppAdaptiveHeader`; safe today only because the title is short.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/McpServerEditorSheet.swift
- [HIGH] line 230 - Section 4 Color - the discovered-tools box background is `Color.secondary.opacity(0.06)` (ratchet-gated) - use `.appSurface` or `.appElevated`.
- [MEDIUM] lines 197-204 - Section 4 Color - connection-test result: `.foregroundStyle(testIsError ? .red : .green)` over `.background(testIsError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))` - system red/green tints on a themed surface (the message text itself carries the status, which is right) - use `AppAttention.color` for the failure tint and a themed spelling for success.
- [MEDIUM] lines 247-252 - Section 2 vocabulary ("An action | .appAction") - "Save Server" uses `.buttonStyle(.borderedProminent)` instead of `.appAction(.primary)` (the `.defaultAction` binding itself is correct).
- [MEDIUM] (possible) line 243 - Section 4 Color - validation message drawn `.foregroundStyle(.red)`; system red matches the destructive convention but no custom theme can retint it - use the themed destructive/error treatment.
- [LOW] line 76 - Section 3 ("No window has a fixed frame"; sibling sheets use minWidth/minHeight) - the editor is `.frame(width: 540, height: 640)`; the ScrollView absorbs Extra Large text but the frame cannot grow for long translations - prefer `minWidth`/`minHeight` like AgentContentImportSheet line 104.
- [LOW] lines 106, 135, 164, 179 - Section 5 Accessibility - the sheet's section headings ("Server Identity", "Transport Type", "Execution Settings", "Connection & Discovered Tools") carry no `.accessibilityAddTraits(.isHeader)`.
- [LOW] line 193 - Section 2 vocabulary - "Test Connection" uses `.buttonStyle(.bordered)` instead of `.appAction(.secondary)`.
- [LOW] (possible) lines 186-188 - Section 2 vocabulary / Section 3 - in-button busy spinner `ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)` (also a fixed 16pt frame) - `AppProgressBar(value: nil, size: .compact)` is the themed form if it fits the bezel.
- [LOW] lines 204, 231 - Section 4 ("Corners are continuous and use AppMetrics radii") - hard-coded `cornerRadius: 6` clips (controlRadius is 7).
- [LOW] lines 77, 113 - Section 5 Keyboard ("a sheet focuses its first field") - no `@FocusState`; the Server Name field is not focused on appear (other sheets in the app, e.g. AudioSpeakerNamingSheet, do focus).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/McpImportSheet.swift
- [HIGH] line 176 - Section 4 Color - owner pill background `Color.secondary.opacity(0.12)` (ratchet-gated) - themed fill.
- [HIGH] line 265 - Section 4 Color - entry card border stroked with `Color.secondary.opacity(0.15)` (ratchet-gated) - stroke `.appBorder` as the theme gives it.
- [MEDIUM] lines 261-265 - Section 2 vocabulary ("A card | .appCardSurface(padding:)") + Section 4 radii - entry rows are hand-rolled cards with `cornerRadius: 8` background/overlay - use `.appCardSurface(padding: 10)` (cardRadius 10, `.appBorder` hairline included).
- [MEDIUM] line 146 - Section 2 vocabulary - "Fetch Catalog" uses `.buttonStyle(.borderedProminent)` instead of `.appAction(.primary)` (shortcut binding is right).
- [MEDIUM] lines 285-297 - Section 4 Color / vocabulary - the fetch-error row tints the glyph `.foregroundStyle(.orange)` and the box `Color.orange.opacity(0.1)` - system orange on a themed surface; use `AppAttention.color` / the AppAttentionNote pattern.
- [MEDIUM] (possible) line 257 - Section 4 Color - per-entry error text `.foregroundStyle(.red)` - themed destructive/error treatment.
- [MEDIUM] (possible) line 275 - Section 4 Color - "Added" confirmation `.foregroundStyle(.green)` - system green ignores custom themes.
- [LOW] line 86 - Section 3 - `.frame(width: 620, height: 660)` fixed sheet frame - prefer minWidth/minHeight (sibling AgentContentImportSheet does).
- [LOW] line 67 - Section 2 vocabulary - "Manage sources" is an unstyled Button; PluginMarketplaceSheet styles the same action `.appAction(.secondary, size: .small)` - one shared treatment (the sheet-chrome comment says the two sheets are siblings).
- [LOW] line 280 - Section 2 vocabulary - "Install" uses `.buttonStyle(.bordered)` instead of `.appAction(.secondary)`.
- [LOW] (possible) line 141 - Section 2 vocabulary - in-button spinner `ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)`.
- [LOW] - Section 5 Keyboard - the source form's first field is not focused when the sheet appears.
- Clean otherwise: the scope bar explains its disabled state with adjacent text (lines 121-125, the reference pattern), the command is shown selectable in code font before Install (lines 241-244), header Close uses `.appAction(.quiet)`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ProfileSettingsSheets.swift
- [MEDIUM] (possible) line 122 - Section 4 Color - vault error text `.foregroundStyle(.red)` - system red; use the themed destructive/error spelling.
- [MEDIUM] (possible) line 292 - Section 4 Color - export error text `.foregroundStyle(.red)` - same.
- [MEDIUM] lines 212-215 - Section 5 Keyboard ("a sheet focuses its first field") - ProfileRenameSheet's single "Profile name" TextField is never focused on appear - `@FocusState` + focus on appear (the app's sheet-focus convention exists elsewhere, e.g. AudioSpeakerNamingSheet).
- [LOW] (possible) lines 90-134 - Section 5 Keyboard - ProfileSecuritySheet does not focus its first passphrase field on appear.
- [LOW] (possible) lines 248-251 - Section 2 "One source of truth" / Section 6 settings search - the sheet title reads "Export a Plaintext ZIP" while its `.settingsControl` registers the search entry as "What to Include in This Backup", so search names a different screen than the one that opens (the sibling sheet at line 209-211 keeps them identical).
- [LOW] lines 273, 277 - Section 2 vocabulary - "Select All"/"Clear All" use `.buttonStyle(.link)` while AgentContentImportSheet draws the same pair with `.buttonStyle(.plain)` and the vocabulary spells quiet actions `.appAction(.quiet)` - three spellings of one control; one should win.
- Clean otherwise: `ProfileSheetParts.buttons` binds Cancel to `.cancelAction` and confirm to `.defaultAction` for every sheet (lines 61-67), destructive mode passes both `role:` and `.appAction(.destructive)` and is gated by a passphrase step, all strings are bundled catalog keys (verified, 21 languages), `.frame(minWidth:idealWidth:)` not fixed frames.

### Batch summary
- 30 findings: 8 HIGH (all Section 4 system-color/ratchet-gated spellings), 17 MEDIUM, 15 LOW/informational entries (counts include per-line splits above).
- Recurring themes in this batch: `.tertiary`/`Color.secondary` accents and pill fills (AgentContentImportSheet, McpImportSheet, McpServerEditorSheet, PluginMarketplaceSheet), system orange/green/red status tints where `AppAttention.color` or a themed token belongs, hand-rolled cards/empty states/progress instead of `appCardSurface`/`AppEmptyState`/`AppProgressBar`, unstyled or `.bordered`/`.borderedProminent` buttons instead of `.appAction`, destructive roles without confirmationDialog (PluginMarketplaceSheet), missing sheet default-action shortcuts and first-field focus.
- No localization defects: every bundled literal and `AppStrings.localized` format key used by these files ("Import Selected (%lld)", "Delete \"%@\"", "Servers added: %lld. Skipped (name already in use): %lld.", "Connected. Tools discovered: %lld.", etc.) exists in `swift/TurboSparkApp/Localization/Localizable.xcstrings` with 21 translations.
- No motion violations: none of the seven files animates.

---

### Batch batch09_Components

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/AgentsSettingsPaneView.swift
- [LOW] line 115 - Section 4 Surfaces and effects - the override banner hard-codes `.appSurface.opacity(0.5)`, multiplying a dimmer alpha on top of the theme surface so Reduce Transparency / Increase Contrast cannot restore it - (possible) use `.appSurface` unmodified or a vocabulary callout surface
- [LOW] line 142 - Section 4 Surfaces and effects - same pattern on the project instructions banner: `.background(.appSurface.opacity(0.3))` - (possible) same fix

Otherwise compliant: extension header layout (scope picker, search, spacer, overflow, primary), one-line purpose via `ExtensionPaneIntro`, `AppEmptyState`, delete goes through a localized alert with destructive role + Cancel, all strings bundled, `.settingsControl` literals intact.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/PluginSettingsPaneView.swift
- [MEDIUM] line 186 - Section 4 Color (theme-blind status tint) - plugin load diagnostics use `.foregroundStyle(.orange)`, a system palette color that ignores custom themes - use `AppAttention.color` (the shared warning hue) or a theme-derived warning style; the triangle glyph is already paired, so only the color spelling is wrong

Otherwise compliant: extension header layout, `AppEmptyState` empty state, uninstall goes through a `confirmationDialog` with destructive roles, `scopeName` resolves scope labels via `AppStrings.localized`, list toggle named by plugin and `labelsHidden()`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SkillsSettingsPaneView.swift
- [HIGH] line 228 - Section 4 Color / Gotcha 36 - row glyph reads bare static `TurboSparkTheme.accentColor` inside the view body (draws the wrong mode under `.preferredColorScheme`; ratchet-counted) and the ternary's other arm is system `Color.secondary` - use `.appAccent` (or `theme.accent`) and `.appSecondary`, wrapping the ternary sides in `AnyShapeStyle` if the style types mix
- [MEDIUM] line 234 - Section 4 Color - enabled/disabled skill name uses `Color.primary` / `Color.secondary` system spellings in a ternary, ignoring custom themes (ratchet-counted) - use `.appText` / `.appSecondary`
- [MEDIUM] line 114 - Section 4 Color - detail placeholder icon uses `.foregroundStyle(.tertiary)`, an enumerated forbidden spelling (ratchet-counted) - use `.appSecondary` or `.appSurface`
- [MEDIUM] line 257 - Section 4 Color - row origin label uses `.foregroundStyle(.tertiary)` - same fix
- [MEDIUM] lines 302-303 - Section 4 Color - "Shadowed" badge tints are system `Color.purple.opacity(0.15)` background and `Color.purple` foreground on a themed surface - derive the hue from the theme or add it to the vocabulary
- [LOW] line 29 - Section 3 One source of truth - `scopeFilter` state drives `filteredSkills` (line 61) but no control binds to it: the All/User/Project filter is unreachable from the UI (dead state or a missing filter picker; the Agents pane has one) - (possible) bind it in the header filters or remove the state and `SkillScopeFilter`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/HfAuthTokenCardView.swift
- [HIGH] lines 233-237 - Section 2 Action hierarchy (destructive) - "Remove Token" is `Button(role: .destructive)` with `.appAction(.destructive)` and no `confirmationDialog`; the stored token is masked and unrecoverable from the app once cleared, which is data loss ("a destructive role with no confirmation dialog is always a defect") - add a localized confirmation dialog (title, message, button), or drop the destructive role if the removal is deemed non-destructive
- [MEDIUM] lines 62-63, 163, 168, 185, 195, 200, 206 - Section 4 Color - status tints are system palette colors on themed surfaces: `Color.green.opacity(0.15)` / `.green` ("Configured", valid), `.red` (invalid), `.orange` (rate limited) - use `AppAttention.color` for the warning tints and resolve success/danger through the theme or extend the vocabulary; glyph+text pairing is already present so severity is the color spelling only
- [MEDIUM] line 152 - Section 2 vocabulary (progress) - validation status row uses `ProgressView` instead of `AppProgressBar`, which the vocabulary defines for determinate and indeterminate progress - use the `AppProgressBar` indeterminate form
- [LOW] line 229 - Section 2 Action hierarchy - Save Token is disabled (empty / not dirty / validating) with no tooltip or adjacent text saying why - add `.help(Text(..., bundle: .module))` explaining the disabled state

Otherwise compliant: shared `modelsSettingsCard()`, `theme.ui(...)` fonts throughout, verbatim `hf_...` prompt is a token prefix (correctly verbatim), `AppIconButton` with labels for reveal/clear, toasts via `AppStrings.localized`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/AppearancePreferencesCardView.swift
- [MEDIUM] lines 37, 287 - Section 4 Surfaces and effects (ratchet: hard-coded border opacity) - `.stroke(.appBorder.opacity(0.4), lineWidth: 1)` on the card and on the size stepper defeats Increase Contrast - stroke `.appBorder` as the theme gives it (see `modelsSettingsCard()`)
- [MEDIUM] line 33 - Section 3 (a pane's card treatment is defined once) - the card treatment (`.appSurface.opacity(0.7)` fill + `AppMetrics.cardRadius` + 0.4 border stroke) is re-implemented inline here and again in ThemeConfigCardView (its line 75) instead of one shared helper - extract one shared card modifier, as Models & Storage does with `modelsSettingsCard()`
- [LOW] lines 302, 314-320 - Section 4 Type (size by role, never by number) - font-picker rows pass hard point `size: 13` to `AppFontDescriptor`, so the previews ignore Text Size - (possible) intentional typeface previews; size them from the theme (e.g. `theme.ui(.small)`) or document the exception
- [LOW] line 16 - Section 5 Accessibility (headings) - the "Preferences" card title lacks `.accessibilityAddTraits(.isHeader)`, unlike other card/section titles in these panes - add the trait

Otherwise compliant: `preferenceRow(title:` keeps string literals at every call site, `adaptivePicker` uses min/ideal/max-free `ViewThatFits`, `sizeStepper` exposes `.accessibilityValue` and hides the duplicate number, `TextSizeOptions()` used for Text Size.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ModelsSettingsPaneView.swift
- [MEDIUM] line 176 - Section 4 Color - LM Studio presence status uses `Color.green` / `Color.orange` system spellings in the ternary - resolve through the theme or `AppAttention.color` for the warning arm; glyph+text pairing is present so only the color spelling is wrong
- [LOW] lines 134-135 - Section 3 Headers and rows - path-row label gets `.frame(minWidth: 52).fixedSize()`, where the bare `.fixedSize()` prevents wrapping in a form row; min/ideal/max is the compliant form - (possible) drop `.fixedSize()` and keep `minWidth` (labels are single words today, so low risk)

Otherwise exemplary: one-line purpose (line 44) without repeating the shell title, shared `modelsSettingsCard()` used by every section including the card in HfAuthTokenCardView, `AppAdaptiveHeader` carrying the section toggle, `minWidth` (not width) on the scaled glyph (line 197), `SettingsPathText` for paths, quiet secondary actions, localization clean.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SkillImportSheet.swift
- [MEDIUM] line 176 - Section 3 Empty states - the fetched-marketplace skill `List` has no empty state: a manifest with zero skills renders a blank list - add an `AppEmptyState` for the no-skills case
- [MEDIUM] line 211 - Section 2 vocabulary (actions) - Install row buttons use `.buttonStyle(.bordered)` while the same sheet's Fetch and Manage sources use `.appAction`; a second action style is the drift DESIGN.md bans - use `.appAction(.secondary, size: .small)` or `.quiet`
- [MEDIUM] line 192 - Section 4 Color - category pill background is system `Color.blue.opacity(0.12)`, theme-blind on a themed surface - derive from the theme or extend the vocabulary
- [MEDIUM] line 135 - Section 4 Color - error icon uses `.foregroundStyle(.red)` system spelling (the message text beside it uses `.appSecondary`) - use a theme-derived danger style or the AppAttention treatment
- [LOW] lines 115, 206 - Section 2 vocabulary (progress) - Fetch/Install buttons swap their label for `ProgressView` instead of `AppProgressBar`'s indeterminate form - (possible) use `AppProgressBar`
- [LOW] line 70 - Section 2 vocabulary (actions) - the header Close button has no `.appAction` treatment while every other action in the sheet has one - give it `.appAction(.quiet, size: .small)` for consistency (it already binds `.cancelAction`)
- [LOW] line 113 - Section 5 Keyboard - the sheet's primary action (Fetch) carries no `.keyboardShortcut(.defaultAction)`, so Return does not trigger it - bind `.defaultAction` to Fetch (Close already binds `.cancelAction`)
- [LOW] line 106 - Section 5 Keyboard - no `@FocusState` or initial focus anywhere in the sheet: the first field (the marketplace input) is never focused on present - focus the field when the sheet appears
- [LOW] lines 97, 125 - Section 4 Surfaces and effects - bar fills hard-code `.appSurface.opacity(0.4)` / `.appSurface.opacity(0.2)`, bypassing the theme's surface opacity under Reduce Transparency - (possible) use the surface style unmodified
- [LOW] lines 218-229 - Section 3 Empty states - the initial "Enter a GitHub repository..." placeholder is a hand-rolled icon+text block instead of `AppEmptyState(systemImage:title:message:)` - (possible) reuse `AppEmptyState`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/PermissionsSettingsPaneView.swift
- [HIGH] line 99 - Section 4 Color / Gotcha 36 - folder icon reads bare static `TurboSparkTheme.accentColor` inside the body (wrong mode under `.preferredColorScheme`; ratchet-counted) and the ternary's other arm is system `.secondary` (also ratchet-counted) - use `.appAccent` (or `theme.accent`) and `.appSecondary`
- [HIGH] lines 251-259 - Section 2 Action hierarchy (destructive) - removing an authorized workspace folder uses `AppIconButton("trash", role: .destructive)` additionally forced `.foregroundStyle(.red)` with no confirmation dialog; folder authorization is rebuildable, and DESIGN.md says exactly this case is "a quiet action with a neutral glyph (`minus.circle`), not a red trash can", and "a destructive role with no confirmation dialog is always a defect" - drop the destructive role and use a neutral glyph, or add a confirmation dialog
- [MEDIUM] line 100 - Section 3 Headers and rows (no fixed frame on a themedFont glyph) - `.frame(width: 24)` fixes the width of a glyph sized `.themedFont(.title3)`, so it clips at Extra Large text - use `.frame(minWidth: 24)` as ModelsSettingsPaneView line 197 does
- [MEDIUM] lines 204-228 - Section 2 vocabulary (empty states) - the custom-folders empty state is a hand-rolled dashed card (icon + two texts) instead of `AppEmptyState(systemImage:title:message:)` - reuse `AppEmptyState`
- [LOW] line 115 - Section 3 Text in layouts - the folder purpose sentence is `.lineLimit(1)` with no tooltip carrying the full text - add `.help` with the full sentence or let it wrap
- [LOW] line 222 - Section 4 Surfaces and effects - empty-state fill hard-codes `.appSurface.opacity(0.5)` - (possible) use the surface unmodified

Otherwise compliant: purpose line with the moved search anchor, per-row status shown as glyph + word + color, `AppIconButton` labels everywhere, `Grant Access` gated behind a user press, sections split into computed properties, `SettingsPathText` for paths. (`settingsControl("Add Folder\u{2026}", ...)` verified: the catalog generator unescapes `\u{...}` and the "Add Folder..." key exists, so no finding.)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ThemeConfigCardView.swift
- [HIGH] line 119 - Section 4 Color (enumerated forbidden spelling) - the preset menu chip background is `Color.accentColor.opacity(0.15)`, which ignores a custom theme's accent (ratchet-counted) - use `.appAccent.opacity(...)` or `theme.accent`
- [MEDIUM] lines 79, 135 - Section 4 Surfaces and effects (ratchet: hard-coded border opacity) - the card and the preset chip stroke `.appBorder.opacity(0.4)`, defeating Increase Contrast - stroke `.appBorder` as the theme gives it
- [MEDIUM] line 75 - Section 3 (card treatment defined once) - `.background(.appSurface.opacity(0.7))` + `AppMetrics.cardRadius` + 0.4 border stroke re-implements the AppearancePreferencesCardView treatment inline and hard-dims the surface - share one card helper across both Appearance cards (see AppearancePreferencesCardView line 33 finding)
- [LOW] line 120 - Section 4 Surfaces and effects (radii) - the "Aa" chip clips with hard `cornerRadius: 4` instead of an `AppMetrics` radius - use `AppMetrics.controlRadius` or the smallest documented vocabulary radius
- [LOW] line 164 - Section 4 Color - accent swatch fallback `Color(hex:) ?? .primary` resolves to system `Color.primary`, theme-blind - (possible, rarely hit) fall back to `.appAccent` / `theme.accent`
- [LOW] line 138 - Section 2 vocabulary (menus) - the preset menu uses `.menuStyle(.borderlessButton)` with a hand-drawn chip label rather than the vocabulary's `.appMenuAction()` treatment - (possible) migrate to `.appMenuAction()` or add the labeled menu pill to the vocabulary and document it here

Otherwise compliant: `spokenName` prefixes mode for VoiceOver, slider exposes `.accessibilityValue` with a locale-aware percent, ternaries are split `Text(..., bundle: .module)` on each side, preset names verbatim, no bare animations.

---

### Batch batch10_Components

Audit against DESIGN.md (2026-10-11 rules). Read-only audit; findings only, no repo files changed.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SoulSettingsSection.swift
- [HIGH] line 109 - section 2 action hierarchy ("A destructive role with no confirmation dialog is always a defect") - "Delete Selected" is `Button(role: .destructive)` calling `deleteSelected()` (line 193) which removes a saved soul (user-authored content) with no confirmationDialog anywhere in the file - either add a confirmationDialog with localized title/message/button (as SkillsSettingsPaneView and ModelLibrariesSectionView do), or if removal is considered rebuildable, drop the destructive role and red styling.
- [MEDIUM] lines 277-282 - section 3 forms - `SoulNameSheet`'s Form has `.formStyle(.grouped)` but is missing `.scrollContentBackground(.hidden)` and `.background(.appPage)`, so the grouped Form paints system AppKit chrome that ignores the theme - add both modifiers (compare SafetySettingsPaneView lines 40-43).
- [MEDIUM] lines 264-299 - section 5 keyboard ("a sheet focuses its first field") - `SoulNameSheet` never focuses its Name TextField; no `@FocusState`/`defaultFocus` - focus the Name field on appear.
- [MEDIUM] line 298 - section 3 ("No window has a fixed frame"; layouts adapt instead of clipping) - `.frame(width: 360, height: 190)` fixes the sheet size; German strings and Extra Large text can clip inside 190pt - use `minWidth`/`minHeight` (compare AgentEditorSheet line 194).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ModelLibrariesSectionView.swift
- [MEDIUM] lines 178, 180 - section 4 color ("The SwiftUI spellings count too") - message banner glyphs use `.foregroundStyle(.red)` / `.foregroundStyle(.green)` system colors that ignore custom themes - use `AppAttention.color` (or a themed status style) for the tints; the glyph+text pairing itself is correct.
- [MEDIUM] lines 196-199 - section 4 color + DesignTokenRatchetTests (hard-coded `.opacity` borders) - the banner stroke uses `Color.red.opacity(0.55)` / `Color.green.opacity(0.55)`, both a system color and a hard-coded border opacity - derive from `AppAttention.color` or a themed border token.
- [MEDIUM] line 260 - section 4 color - `MovedStoreNoticeView` warning glyph uses `.foregroundStyle(.orange)` system color - use `AppAttention.color`.
- [LOW] (possible) lines 174-203 - section 2 vocabulary ("A card: `.appCardSurface(padding:)`") - the message banner is a hand-rolled surface (`.padding(10)` + `.appSurface` + custom stroke); a second local copy of the card/banner treatment - consider a shared banner surface in the vocabulary.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/McpSettingsPaneView.swift
- [MEDIUM] line 215 - section 4 color (`.secondary`/`Color.primary.opacity` spellings ignore a custom theme) - summary pill background is `Color.secondary.opacity(0.08)` - use `.appElevated`/`.appSurface` (or `AnyShapeStyle(.appBorder)` tint). Header layout, search, empty state, and AppStrings usage are otherwise clean.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SkillDetailView.swift
- [MEDIUM] lines 73-79 - section 2 vocabulary (`.appCardSurface(padding:)`) + section 4 radii - headerCard is a hand-rolled card (`.padding(16)` + `.appSurface` + hard-coded `cornerRadius: 8` + manual `.appBorder` stroke) instead of `.appCardSurface(padding:)` with `AppMetrics.cardRadius`.
- [MEDIUM] lines 169-171 - section 4 color - path-trigger chips use `Color.purple.opacity(0.12)` background and `.foregroundStyle(.purple)` system colors that ignore the theme - use themed/attention styles; also hard-coded `cornerRadius: 4` instead of `AppMetrics.controlRadius` for chips (LOW).
- [MEDIUM] line 195 - section 4 surfaces ("Do not hard-code a lower opacity... than the theme gives"; Reduce Transparency) - reference-file rows use `.appSurface.opacity(0.6)`, a hard-coded dimmed themed surface - use `.appSurface` or `.appElevated`.
- [MEDIUM] line 242 - section 4 color (`.tertiary` ignores a custom theme) - meta item titles use `.foregroundStyle(.tertiary)` - use `.appSecondary`.
- [MEDIUM] line 237 - section 3 ("No fixed `.frame(width:)` ... nor on a glyph sized with `.themedFont`") - the meta icon is `.themedFont(.small)` inside `.frame(width: 16)`, which clips the glyph at Extra Large text - drop the fixed width or scale it by `AppMetrics.textScale(theme)`.
- [MEDIUM] lines 204-208 - section 3 headers ("A header never truncates. Use AppAdaptiveHeader... a bare HStack with a Spacer truncates") - the "Instruction Body (SKILL.md)" heading plus Copy action is a bare `HStack { Text... Spacer(); Button }` - use `AppAdaptiveHeader` for the heading+action row.
- [MEDIUM] lines 268-275 - section 4 color - `SkillScopeBadge` uses `Color.orange.opacity(0.15)`/`Color.blue.opacity(0.15)` backgrounds and `.foregroundStyle(... Color.orange : Color.blue)` system colors - use themed or `AppAttention.color` tints.
- [LOW] (possible) line 68 - section 6 localization (enum-backed display strings) - `skill.manifest.context.rawValue.capitalized` renders fixed English from the model layer through `Text(String)` with no catalog lookup (unlike `SkillScopeBadge` line 263, which does it correctly) - map it to catalog keys like the badge does.
- Note: the detail's Delete button (line 147) is covered - `onDelete` sets `skillToDelete`/`isConfirmingDelete` and SkillsSettingsPaneView.swift lines 139-155 present the confirmationDialog.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/PluginDetailView.swift
- [MEDIUM] line 40 - section 2 vocabulary ("A filesystem path in a row: SettingsPathText") - the plugin Location row renders `plugin.directoryURL.path` through `PluginLabeledRow` as plain `.themedFont(.small)` text: no code font, no middle truncation, no full-path `.help` - use `SettingsPathText(path, size:)` (line 263's Text does get `.textSelection`, but not the rest).
- [MEDIUM] lines 90-111 - section 4 color - `originBadge` assigns `.blue`/`.green`/`.orange`/`.purple` system colors for both capsule fill (`color.opacity(0.15)`) and foreground, ignoring custom themes - use themed styles or `AppAttention.color` per origin.
- [MEDIUM] lines 56, 117, 138, 206 - section 5 ("Titles and section headings carry `.accessibilityAddTraits(.isHeader)`") - the "Not applied by this client", "Contributions", "Options" and "Project override" headings lack the header trait (SkillDetailView adds it on every equivalent heading) - add `.accessibilityAddTraits(.isHeader)` to each.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/AgentEditorSheet.swift
- [MEDIUM] lines 36-196 - section 5 keyboard ("a sheet focuses its first field") - the sheet never focuses the Agent Name field on appear (no `@FocusState`) - focus the first field when shown.
- [MEDIUM] lines 82-88 - section 3 ("a control directly under a section header that names it ... hides it with `.labelsHidden()`") - the Scope Picker keeps a visible label "Scope" right under the visible "Scope" heading, so the word renders twice; the picker also carries no `.labelsHidden()` (every other labeled control in this sheet does) - add `.labelsHidden()`, keeping the real label for VoiceOver.
- [LOW] (possible) lines 110-113 - section 5 ("Sliders and steppers expose their formatted value with `.accessibilityValue`") - the Max Turns Stepper carries its value only inside the label; add `.accessibilityValue` for robustness.
- [LOW] (possible) lines 186-188 - section 2 ("A disabled control says why") - Save is disabled by `!canSave` with no tooltip or adjacent sentence saying which required field is missing - add a `.help` or helper line.
- [LOW] lines 8, 223-221 - section 8 ("A view type whose helpers call each other needs an explicit `@MainActor` on the TYPE") - the struct has no `@MainActor` while `saveAction()` calls `parseToolList(_:)` (every other component in this batch carries it with the Gotcha 45 comment) - add `@MainActor` to the type.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ModelLibraryRowView.swift
- [MEDIUM] lines 222-231 (with State/ModelLibraryRowModel.swift lines 36, 47-49) - section 2 ("Removing an entry from a list the user can rebuild ... deletes nothing: it is a quiet action with a neutral glyph (minus.circle), not a red trash can") - Remove Library renders `Button(role: .destructive)` + `.appAction(.destructive)` with the `trash` glyph, yet the confirmation's own message says the files are not changed and the folder can be re-added - make remove a quiet/secondary action with `minus.circle` (same for `abandonConversion`, whose dialog says the models stay and the folder can be re-added). The confirmation dialogs themselves (parent file) are present and localized.
- [MEDIUM] line 133 - section 4 color - connected status glyph uses `.foregroundStyle(.green)` system color - use a themed/attention status tint (status is already glyph+text, so only the spelling is wrong).
- [MEDIUM] lines 46, 193 - section 4 color - slow-link and speed-test-failure warnings pass `tint: .orange` system color into `noteLine` - use `AppAttention.color`.
- [MEDIUM] line 176 - section 2 vocabulary ("Progress, determinate or not: AppProgressBar") - the running speed test shows a bare `ProgressView()` - use `AppProgressBar` (animated) as the vocabulary requires.
- [LOW] (possible) lines 65-67 - section 2 vocabulary - the row draws its own card surface (`.padding(10)` + `.appSurface` + `AppMetrics.controlRadius` clip) instead of `.appCardSurface(padding:)`; intentional row-in-card look, but it is a second local copy of the treatment.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SafetySettingsPaneView.swift
- [HIGH] lines 227-229 (with line 147) - section 2 ("A destructive role with no confirmation dialog is always a defect") - `SteeringPresetRow`'s Delete is `Button(role: .destructive)` wired straight to `model.deleteSteeringPreset(preset.id)` with no confirmationDialog anywhere; a preset holds user configuration (mode, scale, layers, path) that is not trivially rebuildable - add a confirmationDialog, or drop the destructive role if the removal is meant to be quiet.
- [MEDIUM] line 229 - section 4 color - a manual `.foregroundStyle(.red)` is layered on `.appAction(.quiet)` to fake the destructive look; system red ignores custom themes (and `.appAction(.destructive)` already renders the correct themed red) - use `.appAction(.destructive, size: .small)` or drop the tint.
- [MEDIUM] line 236 - section 4 color - compatibility Label uses `.foregroundStyle(compatibility.allowsEnabling ? Color.secondary : Color.red)` - both are theme-blind system spellings - use `.appSecondary` and `AppAttention.color` (wrap mixed ternary sides in `AnyShapeStyle`).
- [MEDIUM] lines 131-134 - section 3 empty states ("Every list ... has an AppEmptyState that says what goes here and offers the action that fills it") - the Directions section's empty state is a bare `Text("No directions registered.")`; it names neither what goes here nor the filling action in user terms (the Add button sits below as a separate element) - use `AppEmptyState(systemImage:title:message:)` with a "what goes here" message and the add action.
- Positive: Form styling (lines 40-43), one property per section, purpose line, and the disabled-toggle explanation (line 75) all follow DESIGN.md.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/CronJobsSettingsPaneView.swift
- [HIGH] line 127 - section 4 ("TurboSparkTheme's static accessors are for non-view code only", Gotcha 36) - the clock icon reads `TurboSparkTheme.accentColor` (a static resolving `NSApp` appearance) inside the view body, plus `Color.secondary` - it draws the wrong mode under `.preferredColorScheme`; use `.appAccent` / `.appSecondary` shape styles.
- [MEDIUM] lines 134, 145, 201 - section 4 color - `.foregroundStyle(.tertiary)` on the cron expression, "next run" caption and empty-history note ignores custom themes - use `.appSecondary`.
- [MEDIUM] lines 168-176 - section 5 ("Icon-only actions use AppIconButton") - the delete control is a hand-rolled `Image(systemName: "trash")` in `.buttonStyle(.plain)` with manually duplicated `.help` and `.accessibilityLabel`, which can drift - use `AppIconButton` so the label is both tooltip and VoiceOver name.
- [MEDIUM] line 172 - section 4 color - the trash glyph uses `Color.red.opacity(0.75)`, a hard-coded system color and opacity - use `.appAction(.destructive)` styling or a themed tint.
- [MEDIUM] lines 189-190, 208 - section 4 color - run-summary and run-status ternaries use `Color.orange`/`Color.green`/`Color.secondary` system spellings - use `AppAttention.color` and `.appSecondary` (wrap mixed sides in `AnyShapeStyle`).
- [MEDIUM] lines 226, 232 - section 4 color (`Color.primary.opacity` ignores a custom theme) - job rows and the expanded history draw on `Color.primary.opacity(0.04)` / `0.03` - use `.appSurface`/`.appElevated`.
- [MEDIUM] lines 224-233 - section 2 vocabulary + radii - job cards are hand-rolled (padding + fill + hard-coded `cornerRadius: 8`/`10`) instead of `.appCardSurface(padding:)` with `AppMetrics` radii.
- [LOW] lines 41-53 - section 2 ("Destructive actions ... go through a confirmationDialog") - deletion is confirmed via `.alert` rather than `.confirmationDialog`; the confirmation exists and is localized, but the named API is confirmationDialog - align with the other panes (possible, letter-of-rule).
- Positive: the empty state (lines 77-85) is clean - it says what a scheduled task is in user terms, offers "Schedule a Task...", and no longer names `CronCreate` in UI copy (only in code comments, which is fine); `AppStrings` status strings format Ints/strings correctly (lines 113-117).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/LocalModelDefaultsSectionView.swift
- clean

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/LibraryOfferBanner.swift
- [MEDIUM] line 146 - section 4 color - the failure line tints its glyph with `Color.orange` (system color, though at least correctly wrapped in `AnyShapeStyle`) - use `AppAttention.color`; the glyph+sentence pairing itself follows the status rule.
- [MEDIUM] line 171 - section 2 vocabulary ("Progress, determinate or not: AppProgressBar") - the adding-in-flight row shows a bare `ProgressView()` - use `AppProgressBar`.
- Positive: disabled Add/Not-now buttons explain why via `DisabledReason` (lines 198-213, 219-229), actions reflow with ViewThatFits, and all drawn strings come from the model layer.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SteeringPresetEditorSheet.swift
- [MEDIUM] lines 41-46 - section 3 forms - the Form has `.formStyle(.grouped)` but is missing `.scrollContentBackground(.hidden)` and `.background(.appPage)`, so the grouped form paints theme-blind AppKit chrome (compare SafetySettingsPaneView lines 40-43).
- [MEDIUM] line 50 - section 3 ("No window has a fixed frame"; layouts adapt instead of clipping) - `.frame(width: 560, height: 620)` fixes the sheet size; German/Extra Large text can clip - use `minWidth`/`minHeight`.
- [MEDIUM] lines 37-52 - section 5 keyboard ("a sheet focuses its first field") - the Name field is never focused on appear (no `@FocusState`) - focus the first field.
- [MEDIUM] line 121 - section 4 color - the read-error Label uses `.foregroundStyle(.red)` system red - use `AppAttention.color` or a themed error style.
- [MEDIUM] line 131 - section 4 color - compatibility Label uses `.foregroundStyle(compatibility.allowsEnabling ? Color.secondary : Color.red)` - theme-blind spellings (same defect as SafetySettingsPaneView line 236) - use `.appSecondary` / `AppAttention.color`.
- [MEDIUM] line 167 - section 4 color - the high-strength warning uses `.foregroundStyle(.orange)` system color - use `AppAttention.color` (glyph+text pairing is correct).
- [MEDIUM] lines 147-151 - section 5 ("Sliders and steppers expose their formatted value with `.accessibilityValue`") + section 3 ("No fixed `.frame(width:)` on ... label ... in a form row") - the Strength Slider has an `.accessibilityLabel` but no `.accessibilityValue` (LocalModelDefaultsSectionView lines 113-119 is the correct reference), and the value Text is pinned with `.frame(width: 48)` - add `.accessibilityValue` and use `minWidth`/scale the frame by `AppMetrics.textScale(theme)`.
- [LOW] (possible) line 194 - section 2 ("When all that is missing is input the user can still give ... do not leave the primary action dead") - Save is disabled while the vector path is empty, though that is exactly input the user can still give; DESIGN.md prefers keeping it pressable with `.appAttention`/focus pointing at the missing field.

### Summary
- 3 HIGH: missing destructive confirmation (SoulSettingsSection delete, SafetySettingsPaneView preset delete), and `TurboSparkTheme.accentColor` static read in CronJobsSettingsPaneView's body (wrong color mode under `.preferredColorScheme`).
- The most common defect class is theme-blind system color spellings (`.red/.green/.orange/.purple/.blue`, `.tertiary`, `Color.secondary`, `Color.primary.opacity`) on themed surfaces - 12 files carry at least one; `AppAttention.color` and the `.app*` styles are the fixes.
- Vocabulary drift repeats: bare `ProgressView()` where `AppProgressBar` is required (ModelLibraryRowView, LibraryOfferBanner), hand-rolled cards (`SkillDetailView`, `CronJobsSettingsPaneView`, `ModelLibraryRowView`), one non-`AppIconButton` icon-only button (CronJobsSettingsPaneView), and a filesystem path not using `SettingsPathText` (PluginDetailView).
- Sheets: three of four sheets never focus their first field (SoulNameSheet, AgentEditorSheet, SteeringPresetEditorSheet) and two pin fixed frames (SoulNameSheet 360x190, SteeringPresetEditorSheet 560x620).
- Checked and clean: Scheduled Tasks empty state no longer names `CronCreate` in UI copy; Skills delete is confirmed by the pane's dialog; McpSettingsPaneView's header/empty state match the extension-pane layout; LocalModelDefaultsSectionView is fully clean.

---

### Batch batch11_Components

Audit of 14 SwiftUI files in `swift/TurboSparkApp/Sources/TurboSparkApp/Components/`
against `/Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md`. Cross-checks run:
catalog keys for flagged strings verified present in
`swift/TurboSparkApp/Localization/Localizable.xcstrings`; `ProfileSheetParts.buttons`
verified to bind `.defaultAction`/`.cancelAction`; MCP server delete verified to have no
confirmation at any layer (`AppModel+Mcp.swift:64`); `AppMetrics` radii confirmed
(controlRadius 7, cardRadius 10, composerRadius 20). No file exceeds the ~400-line guideline.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SkillEditorSheet.swift
- [MEDIUM] line 62 - DESIGN.md 5 Accessibility ("a sheet focuses its first field") - The sheet never focuses its first field: no `@FocusState` anywhere, so the Name `TextField` starts unfocused - Add a `@FocusState` bound to the name field and focus it `onAppear`.
- [MEDIUM] line 151 - DESIGN.md 2 Action hierarchy ("A disabled control says why") - Save Skill is disabled while the name is empty with no tooltip or adjacent text saying why; per the same section, input the user can still give should ideally keep the action pressable with `.appAttention` (Voice Cloning reference) - Add a `.help`/hint naming the missing name, or keep it pressable and flag the empty name field with `.appAttention` on first attempt.
- [MEDIUM] line 171 - DESIGN.md 6 Localization ("a seeded chat draft uses AppStrings.localized") - The seeded markdown draft (fixed English UI copy reaching the UI through the TextEditor binding) is a bare literal, not `AppStrings.localized` (possible; it is skill content, but it ships as visible copy in every language) - Move to `AppStrings.localized` with a catalog key translated in the same change.
- [LOW] line 31 - DESIGN.md 3 Layout ("A header never truncates. Use AppAdaptiveHeader") - Sheet header is a bare `HStack` + `Spacer`; a long translated Cancel label or long skill name ("Edit Skill: ...") compresses the title instead of wrapping (possible) - Consider `AppAdaptiveHeader` or stack title above actions.
- [LOW] line 77 - DESIGN.md 2 Action hierarchy (disabled control says why) - Scope segmented picker silently disabled when no project path is captured; no explanation of why Project Scope is unavailable (possible) - Add a `.help` or caption explaining project scope needs an open project.
- Correct here: Save bound to `.defaultAction`, Cancel to `.cancelAction`; `TextEditor` uses `.appEditorSurface(minHeight:)`; `Text(verbatim:)` prompts on lines 62/96/110 are code/identifier examples (correct verbatim); "Edit Skill: %@" key verified in catalog.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/CustomModelFoldersSectionView.swift
- [MEDIUM] lines 76-89 - DESIGN.md 3 Empty states / 2 vocabulary (`AppEmptyState(systemImage:title:message:){actions}`) - Hand-rolled empty state (icon + text centered in `HStack`/`Spacer`) instead of `AppEmptyState`, and it offers no action; the rule says the empty state "offers the action that fills it" (Add Folder lives only in the header) - Replace with `AppEmptyState(systemImage:title:message:) { Add Folder button }`.
- Correct here: `.settingsControl` keeps a literal first arg (line 58); header uses `AppAdaptiveHeader`; path row delegated to `ScanFolderRowView`; NSOpenPanel strings via `AppStrings.localized`; localized-key interpolations are String-only.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/DownloadManagerView.swift
- [MEDIUM] line 32 - DESIGN.md 2 vocabulary ("Progress, determinate or not | AppProgressBar") - `ProgressView().controlSize(.mini)` for the active-download indicator instead of `AppProgressBar` - Use `AppProgressBar` at its small size.
- [MEDIUM] lines 60-70 - DESIGN.md 3 Layout (headers; bare `HStack` + `Spacer` truncates under German / Extra Large) - Downloads panel header (title + Clear action) is a bare `HStack` with `Spacer` instead of `AppAdaptiveHeader` (possible; small overlay panel, but it is a titled header with an action) - Use `AppAdaptiveHeader`.
- [MEDIUM] lines 72-76 - DESIGN.md 3 Empty states - "No downloads yet" is a bare `Text` instead of `AppEmptyState` saying what goes here and offering the action - Use `AppEmptyState`.
- [LOW] lines 110-115 - DESIGN.md 4 ("Corners ... use AppMetrics radii") - Panel corner radius hard-coded 16, not an `AppMetrics` radius (controlRadius 7 / cardRadius 10 / composerRadius 20); the drop shadow itself is acceptable floating-chrome treatment - Use an `AppMetrics` radius.
- [LOW] lines 15-45 - DESIGN.md 2 vocabulary (local variants vs shared treatment) - Bespoke drawn capsule disclosure button (hand padding/background/stroke) instead of a vocabulary control (possible; a corner overlay chip is not currently in the vocabulary) - Add it to the vocabulary or reuse a chip treatment.
- [LOW] line 173 - DESIGN.md 2 Action hierarchy (disabled control says why) - Retry is disabled via `!model.canRetryModelDownload(download)` with no tooltip explaining why (possible) - Add a `.help` stating the block reason.
- Correct here: `historyFooter` LocalizedStringKey keys verified in catalog; `Text(_, format:)` for the date (line 155) is the correct date handling; Double formatted to String before interpolation (line 137 pattern is in GenerateControl); themed colors and `.themedFont` throughout.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/GenerateControl.swift
- [MEDIUM] lines 56, 90 - DESIGN.md 3 Layout / 4 Type ("nor on a glyph sized with .themedFont (it grows with Text Size)") - Fixed `.frame(width: circularButtonSize, height: circularButtonSize)` around glyphs sized with `.themedFont(.callout)`/`.themedFont(.tiny)`; the frame is `@ScaledMetric`, which follows Dynamic Type but not the app's Text Size scale, so Extra Large text can overflow the circle - Derive the size from the glyph (`themedFont(fitting:)` with `size * 0.45`) or multiply by `AppMetrics.textScale(theme)`.
- [MEDIUM] lines 63-67 - DESIGN.md 4 Surfaces ("Glows are accent shadows from the vocabulary only: ... .appPrimaryGlow under the one primary action") - Hand-rolled accent glow (`.shadow(color: ready ? theme.accent.opacity(0.45) : .clear, radius: 8, y: 2)`) on the composer's one primary action instead of `.appPrimaryGlow` (possible; the comment documents intent, but it is a vocabulary duplicate) - Use `.appPrimaryGlow`.
- [LOW] line 48 - DESIGN.md 4 ("Do not hard-code a lower opacity ... for secondary text than the theme gives") - `theme.secondaryText.opacity(0.5)` dims the disabled ink below the theme's secondary text (possible; comment documents the disabled state on purpose) - Prefer a theme-provided disabled treatment if one exists.
- [LOW] line 60 - DESIGN.md 4 Color (theme decides color) - `Color.white.opacity(0.2)` literal edge highlight on the themed accent fill; a custom-theme accent can wash it out (possible) - Derive the edge from `theme.accent.contrastForeground` or a theme token.
- Correct here: ternary fills wrapped in `AnyShapeStyle` (lines 46-47); looping sweep is a `phaseAnimator` inside `TSMotionContext` honoring Reduce Motion (lines 180-190); `.tsAnimation`/`TSMotion.animate` only; disabled Send explains itself via `.help` (line 76) - this is the reference pattern; Double formatted to String before interpolation (line 137); Int interpolations use verified `%lld` keys.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/AgentDetailView.swift
- [MEDIUM] line 113 - DESIGN.md 4 Color (`.foregroundStyle(.red)` is a theme-blind spelling; "Red is the destructive convention") - Disallowed Tools value is styled `.foregroundStyle(.red)`; this is informational content, not destructive, and it ignores a custom theme's palette - Use `.appSecondary`/`.appText`, or `AppAttention.color` if a warning tint is wanted, paired with the text it already has.
- [MEDIUM] lines 124-133 - DESIGN.md 2 vocabulary ("A filesystem path in a row | SettingsPathText(path, size:)") - File Location row hand-rolls the path display (`Text(path)` + lineLimit/truncation/help) and uses the UI font instead of the code font - Use `SettingsPathText(path, size:)`.
- [MEDIUM] lines 137-139, 152-155 - DESIGN.md 2 vocabulary ("A card | .appCardSurface(padding:)") - Hand-rolled card fills (`RoundedRectangle(cornerRadius: 8).fill(.appSurface)`) for Configuration and System Instructions instead of `.appCardSurface(padding:)` (possible) - Use `.appCardSurface(padding:)` for both.
- [LOW] lines 139, 155, 169 - DESIGN.md 4 ("Corners ... use AppMetrics radii") - Hard-coded radii 8/8/6 instead of `AppMetrics` - Use `AppMetrics.controlRadius`/`cardRadius`.
- [LOW] lines 46, 188-195 - DESIGN.md 4 Color (theme decides color) - `AgentBadge` tints are literal `.blue`/`.purple`/`.green`/`.orange`/`.indigo`, fixed palette colors that ignore a custom theme; text is present so this is not color-alone status (possible) - Consider theme-derived badge colors or document the exception.
- Correct here: destructive Delete (line 75) is confirmed by the parent pane's alert (`AgentsSettingsPaneView.swift` "Delete Agent" alert with destructive role and message) - dialog present, fine; headings carry `.accessibilityAddTraits(.isHeader)` (lines 42, 89, 146); enum-backed badge strings go through `Text(LocalizedStringKey(...), bundle: .module)`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ExtensionPaneHeader.swift
- [LOW] line 175 - DESIGN.md 4 Surfaces ("surfaceOpacity forces opaque structural surfaces" under Reduce Transparency) - List column background `.appSurface.opacity(0.5)` hard-codes half opacity on a themed surface, bypassing the theme's transparency handling (possible) - Use `.appSurface` directly or a theme-provided subdued surface.
- Otherwise clean: this file IS the shared vocabulary (AppAdaptiveHeader, AppEmptyState, AppIconButton, `.appMenuAction`, `SettingsLayout` insets, min/ideal/max list widths).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/AppearanceSettingsPaneView.swift
- [HIGH] line 102 - DESIGN.md 4 Color ("DesignTokenRatchetTests counts ... hard-coded `.appBorder.opacity(...)` borders"; counts may only go down) - Reset card border is `.appBorder.opacity(0.4)` - Use `.appBorder` (the theme already raises borders under Increase Contrast).
- [MEDIUM] lines 176-177 - DESIGN.md 4 Color (`.primary`/`.secondary` spellings ignore a custom theme; ternary sides need `AnyShapeStyle` when mixing styles) - Mode-card title uses `foregroundStyle(isSelected ? .primary : .secondary)` - Use `.appText` / `.appSecondary`.
- [MEDIUM] line 60 - DESIGN.md 4 Surfaces (do not hard-code lower opacity than the theme gives) - Pane background `.appPage.opacity(0.6)` dims the themed page below its value - Use `.appPage`.
- [MEDIUM] line 98 - DESIGN.md 4 Surfaces - Reset card background `.appSurface.opacity(0.7)` below the theme's surface - Use `.appSurface`.
- [MEDIUM] line 173 - DESIGN.md 4 Surfaces (glows from the vocabulary only) - Selected mode card carries a hand-rolled glow (`.shadow(color: isSelected ? accent.opacity(0.2) : Color.black.opacity(0.04) ...)`) instead of `.appFocusGlow`/`.appPrimaryGlow`-style vocabulary treatment (possible; thumbnail previews are informational, the selection glow is not) - Use the vocabulary glow or a plain `.appBorder` selection ring (the ring at line 168 already exists).
- [MEDIUM] lines 121-124 - DESIGN.md 5 Accessibility ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") - "Appearance mode" section heading has `.settingsControl` but no `.isHeader` trait - Add `.accessibilityAddTraits(.isHeader)`.
- [MEDIUM] lines 27-58 - DESIGN.md 3 Window structure ("It starts with a one-line purpose (.themedFont(.small), .appSecondary ...)") - The Appearance pane opens directly with the mode cards; no one-line purpose under the shell-rendered title (possible; the rule names Files & Permissions and Models & Storage as the reference) - Add a one-line purpose in `.themedFont(.small)` + `.appSecondary`.
- [LOW] lines 160-172 - DESIGN.md 4 ("Corners ... use AppMetrics radii") - `cornerRadius: 10` literals (three sites) instead of `AppMetrics.cardRadius` (same value, but the constant is the source of truth) - Use `AppMetrics.cardRadius`.
- Correct here: destructive Reset uses `Button(role: .destructive)` plus a `confirmationDialog` with localized title/message/button (lines 104-115), with the reset description shared between card and dialog; `ThemeConfigCardView(title:)` String args are documented catalog keys (verified in ThemeConfigCardView.swift); mode labels are enum-backed through `Text(LocalizedStringKey(mode.label), bundle: .module)`; mode cards carry `.isSelected` (line 184); `manager.activeAccentColor(isDark:)` is the mode-explicit form.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/EmbeddingModelPicker.swift
- [MEDIUM] line 127 - DESIGN.md 4 Color (system spellings count too; "AppAttention.color for warning/attention tints") - Probe warning label uses `.foregroundStyle(.orange)` - Use `AppAttention.color` or a themed attention tint (glyph + text are present, so only the color spelling is wrong).
- [MEDIUM] line 172 - DESIGN.md 4 Color - EmbeddingStatusLabel "caveat" case uses `.foregroundStyle(.orange)` - Same fix as above.
- [LOW] line 93 - DESIGN.md 2 Action hierarchy (disabled control says why) - Download button disabled when `canQueueModelDownload` is false with no tooltip explaining why (possible; caller-controlled) - Add a `.help` naming the block reason.
- Correct here: field prompt/label localized with `bundle: .module`; profile rows combined into one accessibility element (line 71); "Selected" state carries `.isSelected` (line 103); status is glyph + word (Differentiate Without Color safe); `Text(verbatim:)` correctly used for model ids, byte counts, reasons.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/McpServerCardView.swift
- [HIGH] line 26 - DESIGN.md 4 Color ("TurboSparkTheme's static accessors are for non-view code only ... never the bare property (TurboSparkTheme.accentColor)"; system spellings count too) - `foregroundStyle(server.isEnabled ? TurboSparkTheme.accentColor : Color.secondary)` inside a view body: the static accessor draws the wrong mode under `.preferredColorScheme`, and `Color.secondary` ignores a custom theme - Use `.appAccent` / `.appSecondary` (or `theme.accent` via the environment).
- [HIGH] line 40 - DESIGN.md 4 Color (never system colors on a themed surface) - Tools-count chip background is `Color.secondary.opacity(0.12)`, a system color on a themed card - Use `.appSurface`/`.appElevated` or a themed fill.
- [HIGH] lines 117-121 - DESIGN.md 2 Action hierarchy ("A destructive role with no confirmation dialog is always a defect") - Menu item `Button(role: .destructive)` "Delete Server" calls `model.deleteGlobalMcpServer(id:)` directly; verified the sole call site and `AppModel+Mcp.swift:64` deletes immediately with no dialog at any layer - Add a localized `confirmationDialog` (title, message, destructive button) before deleting.
- [MEDIUM] lines 57-62 - DESIGN.md 4 Color (system spellings; "Status is never color alone: pair it with a glyph or text") - Test result uses `.foregroundStyle(result.isError ? .red : .green)` with `Color.red.opacity(0.1)`/`Color.green.opacity(0.1)` fills; system colors on a themed card, and no glyph distinguishes the state beyond color - Use themed attention/error styles plus a glyph, and theme-derived fills.
- [MEDIUM] lines 77-97 - DESIGN.md 2 vocabulary ("An icon-only action | AppIconButton(symbol, label: Text(...))") - Test and Edit icon-only buttons are hand-rolled `Button` + `Image` with separate `.help` and `.accessibilityLabel`, so tooltip and VoiceOver name can drift - Use `AppIconButton`.
- [MEDIUM] lines 122-128 - DESIGN.md 2 vocabulary ("Overflow ... Menu { ... }.appMenuAction()"; 3 Layout: no fixed `.frame(width:)` on labels) - Overflow menu is hand-rolled (`.menuStyle(.borderlessButton)`, `.menuIndicator(.hidden)`, fixed `.frame(width: 18)`) - Use `Menu { ... }.appMenuAction()` with a labeled or AppIconButton label.
- [MEDIUM] lines 78-81 - DESIGN.md 2 vocabulary ("Progress, determinate or not | AppProgressBar") - `ProgressView().scaleEffect(0.5)` for the testing indicator - Use `AppProgressBar` at its small size.
- [LOW] lines 62, 181 - DESIGN.md 4 ("Corners ... use AppMetrics radii") - `cornerRadius: 6` literals - Use `AppMetrics.controlRadius`.
- [LOW] line 180 - DESIGN.md 4 Surfaces - `.appSurface.opacity(0.6)` disclosure background below the theme's value (possible) - Use `.appSurface`.
- Correct here: row card uses `AppMetrics.cardRadius` + `.appBorder` stroke (lines 70-72); Enable toggle keeps a real localized label and hides it with `.labelsHidden()` (lines 103-105); per-server `accessibilityLabel`s are localized.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/McpTransportFieldsView.swift
- [MEDIUM] lines 34, 38, 72, 146-148 - DESIGN.md 3 Text in layouts ("TextEditor has no intrinsic height ... Measure with a hidden Text") / 2 vocabulary (`.appEditorSurface(minHeight:)`) - Four multi-line editors (Arguments, Environment Variables, Environment Passthrough, Headers) use fixed `.frame(height: 60/50/60)`; the fixed height truncates under Extra Large text and long pasted values - Use `.appEditorSurface(minHeight:)` or the hidden-Text measured pattern.
- [MEDIUM] line 138 - DESIGN.md 4 Color (`.foregroundStyle(.red)` is a theme-blind spelling) - "Not a valid URL." validation text uses system `.red` - Use an `AppAttention`/themed error treatment (text is present, so not color-alone).
- [MEDIUM] lines 165-178 - DESIGN.md 2 vocabulary ("A surface that still needs the user | .appAttention(_:ping:radius:) with an AppAttentionNote") / 4 Color - The remote-transport warning is hand-rolled with `.foregroundStyle(.orange)` and `Color.orange.opacity(0.1)` fill - Rebuild as `.appAttention` + `AppAttentionNote` with the themed attention tint.
- [MEDIUM] lines 22-23, 121-122 - DESIGN.md 5 Accessibility ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") - "Subprocess Configuration" and "SSE Endpoint Configuration" headings are plain `Text` with no `.isHeader` trait (and are not `AppSectionHeader`) - Add the trait or use `AppSectionHeader`.
- [LOW] line 57-58 - DESIGN.md 2 vocabulary (`.appAction(.secondary / .quiet ...)`) - "Choose..." button uses `.buttonStyle(.bordered)` instead of the shared action treatment - Use `.appAction(.secondary, size: .small)`.
- Correct here: all prompts/labels/help localized with `bundle: .module`; NSOpenPanel strings via `AppStrings.localized`; `Text(verbatim:)` only for the example URL (correct); error surfaced beside the field rather than silently.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ThemeGalleryView.swift
- [HIGH] lines 111-116 - DESIGN.md 2 Action hierarchy ("Destructive actions ... go through a confirmationDialog"; "A destructive role with no confirmation dialog is always a defect") - "Delete saved theme" (`role: .destructive`, `.appAction(.destructive)`) calls `manager.deleteTheme(id:)` immediately; a saved theme is lost (the `.help` mitigates but does not confirm) - Add a localized `confirmationDialog` (title, message, destructive button).
- [MEDIUM] line 73 - DESIGN.md 4 Color (`.foregroundStyle(.red)` theme-blind spelling; status not color alone) - Import/save error text uses system `.red` with no glyph - Use a themed error/attention treatment paired with a glyph.
- [LOW] lines 17-20 - DESIGN.md 3 Layout ("A header never truncates. Use AppAdaptiveHeader") - Gallery heading row is a bare `HStack` + `Spacer` with title and trailing status text; a long translated status or title can compress the other (possible) - Use `AppAdaptiveHeader` or stack them.
- [LOW] line 42 - DESIGN.md 4 ("Corners ... use AppMetrics radii") - Thumbnail `cornerRadius: 6` literal - Use `AppMetrics.controlRadius`.
- [LOW] lines 169-171 - DESIGN.md 4 Color - Thumbnail fallbacks `.blue`/`.primary`/`.white` literals when hex parsing fails (possible; thumbnails are informational previews of the theme's own colors, so the fallback path is the only concern) - Prefer a theme token fallback.
- Correct here: gallery items carry `.isSelected` (line 60) as required for the theme gallery; checkmark is `accessibilityHidden` while the trait carries selection; contrast warning pairs glyph + text (lines 67-71); `.settingsControl` literals kept; `AppStrings.localized` for the seeded copy name (line 98).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/WelcomeCharacterView.swift
- clean
- (Verified: `.tsEntrance`, `TSMotion.animate`, `TSMotionContext`-style motion only; accent-derived gradient wash and starfield are the documented single-gradient exception; `theme.ui(..., systemDesign: .serif)` is the sanctioned way to pin a family; greeting strings are per-language data rendered via `Text(String)` correctly, fallback via `AppStrings.localized`.)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ProfileBackupSheets.swift
- [MEDIUM] line 55 - DESIGN.md 4 Color (`.foregroundStyle(.red)` is a theme-blind spelling) - Error text uses system `.red` - Use a themed error/attention treatment (error text is already present, so not color-alone).
- [MEDIUM] lines 29-67, 129-158 - DESIGN.md 5 Accessibility ("a sheet focuses its first field") - Neither backup sheet focuses its first field (no `@FocusState`) - Focus the first field (profile name or password) on appear.
- [LOW] line 148 - DESIGN.md 2 Action hierarchy (disabled control says why) - Import confirm disabled while the name is empty; no text beyond the field's own prompt says why (possible) - Add a hint or keep pressable with `.appAttention`.
- Correct here: shared `ProfileSheetParts.buttons` verified to bind `.keyboardShortcut(.cancelAction)` and `.keyboardShortcut(.defaultAction)` (`ProfileSettingsSheets.swift:63/66`); date and version are formatted to Strings before interpolation into the localized key (lines 160-166, the sanctioned pattern); `restoreName` seeded via `AppStrings.localized` (line 20).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/PermissionsInfoSectionsView.swift
- [HIGH] line 123 - DESIGN.md 4 Color ("DesignTokenRatchetTests counts ... hard-coded `.appBorder.opacity(...)` borders"; counts may only go down) - Command Classifier Veto card border is `.appBorder.opacity(0.4)` - Use `.appBorder`.
- [MEDIUM] lines 99-112 - DESIGN.md 3 Layout ("This includes a section header whose only action is a toggle: a bare HStack with a Spacer truncates the title under German and Extra Large text") - "Command Classifier Veto" header is exactly that case: bare `HStack` + `Spacer` + Toggle - Use `AppAdaptiveHeader` (or stack the toggle under the title).
- [MEDIUM] lines 100-101 - DESIGN.md 5 Accessibility ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") - The heading `Label` has no `.isHeader` trait - Add `.accessibilityAddTraits(.isHeader)`.
- [LOW] lines 148, 152 - DESIGN.md 4 Surfaces (do not hard-code opacity below the theme) - Safety callout wash `.appAccent.opacity(0.08)` and stroke `.appAccent.opacity(0.25)` hard-code the tint strengths (possible) - Use a theme-provided attention surface if one exists.
- [LOW] lines 139-140 - DESIGN.md 5 Accessibility - Callout title "TurboSpark Safety & Sandbox Guarantee" could carry `.isHeader` (possible; it is a callout title, not a section heading) - Add the trait if VoiceOver heading navigation should reach it.
- Correct here: `AppSectionHeader` used for the links section (line 16); link cards use `AppMetrics.cardRadius`, `.appSurface`, `.appBorder`, `.appAccent`, `.isLink` trait, and hidden decorative glyphs; settings search literal kept.

### Cross-file notes
- Recurring theme-blind color spellings to sweep together (each lowers the ratchet baseline): `.foregroundStyle(.red)` (AgentDetailView 113, McpTransportFieldsView 138, ThemeGalleryView 73, ProfileBackupSheets 55), `.orange` (EmbeddingModelPicker 127/172, McpTransportFieldsView 169/176), `.primary/.secondary` ternary (AppearanceSettingsPaneView 177), `Color.secondary` fills (McpServerCardView 26/40), system `.red/.green` fills (McpServerCardView 59-61).
- Two ratchet-counted `.appBorder.opacity(...)` borders: AppearanceSettingsPaneView 102 and PermissionsInfoSectionsView 123.
- Two destructive actions missing confirmationDialog: McpServerCardView 117-121 (verified no dialog at any layer) and ThemeGalleryView 111-116.
- Missing first-field sheet focus: SkillEditorSheet and both ProfileBackupSheets sheets.

---

### Batch batch12_Components

Audit against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md. All 14 files read in full.
Verification notes: `AppThemeColor` is an environment-resolved ShapeStyle (Theme/AppThemeColor.swift), so `AppThemeColor.appSecondary` in a view body is NOT a static-theme-read defect. `AppAttention.color` is defined as `Color.orange`. `ScanFolderRowModel` builds every string with `AppStrings.localized` at construction, so the row view's `Text(verbatim:)` on model fields is correct. `TSMotion.breathe` (0.85s) and `snap` are the named looping curves. All quoted catalog keys ("Cancelling", "Paused", "Loading model", "Downloading", "Installing", "%@ of %@", "Failed to set mirror endpoint: %@", etc.) verified present in Localizable.xcstrings. DesignTokenRatchetTests counts bare `TurboSparkTheme.\w+(Color|Background)`, `.appBorder.opacity(`, `Color.accentColor`, and `Color.secondary|.primary.opacity|(.secondary)|(.tertiary)` with a baseline that may only fall.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ScanFolderRowView.swift
- [LOW] line 111 - DESIGN 4 (AppAttention.color is the warning-hue spelling) - warning glyph tinted with raw `.foregroundStyle(.orange)`; the sanctioned spelling for the warning hue is `AppAttention.color` (same value, but the vocabulary spelling is what the audit and ratchet track) - use `AppAttention.color`.
- Compliant (per the batch brief): the remove action is the quiet neutral `AppIconButton("minus.circle", ...)` with `.appAction(.quiet)` (lines 67-74), not a red trash can; the disabled Reveal states its reason via `.help` (line 139); offline rows are one accessibility element with actions (lines 29-44); `Text(verbatim:)` on model fields is correct because ScanFolderRowModel localizes with `AppStrings.localized`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/InstallDestinationPicker.swift
- clean. (`AppThemeColor.appSecondary` at line 129 resolves from the environment, not a static read; `.fixedSize()` on the Picker is content sizing, not a fixed width; `Text(verbatim: choice.menuTitle)` is drive-supplied text per the doc comment.)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ModelProviderDiscoverySheet.swift
- [MEDIUM] line 25 - DESIGN 2 vocabulary (progress: AppProgressBar, determinate or not) - indeterminate `ProgressView()` instead of the vocabulary control - use `AppProgressBar(value: nil, label: Text("Scanning known locations...", bundle: .module))`.
- [LOW] line 76 - DESIGN 4 (system colors on a themed surface) - status text tinted with a raw `.green : .orange` ternary; both hues ignore custom themes. The hues match the AppToast status convention, but they are spelled inline in a view (possible) - reference a shared status tint (e.g. `AppToast.Style.tintColor`) or keep the secondary style, since the words already carry the status.
- [LOW] line 29 - DESIGN 3 (every list has an AppEmptyState that says what goes here and offers the filling action) - bare `Text("No known provider folders were found.")` instead of `AppEmptyState` - use AppEmptyState with, e.g., a pointer to the Add Folder action.
- [LOW] line 58 - DESIGN 3 (no fixed frame; layouts adapt) - `.frame(width: 620)` fixed sheet width; a long translation plus Extra Large text crowds it (possible) - prefer min/ideal/max width.
- [LOW] line 88 - DESIGN 4 (radii from AppMetrics) - `RoundedRectangle(cornerRadius: 7)` hard-coded - use `AppMetrics.controlRadius`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/TaskProgressIndicatorView.swift
- [HIGH] lines 97-101 - DESIGN 5 ("Never a repeatForever started from withAnimation"; no inline durations) - bare `withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true))` for the bob. The manual reduce-motion guard at line 96 mitigates Reduce Motion, but the banned pattern and the inline duration remain, and looping should be a phaseAnimator applied only while live - replace with a `.tsPulse(active:)`-style phaseAnimator modifier or `TSMotion.breathe`.
- [MEDIUM] lines 47-51 - DESIGN 5 (TSMotion curves; looping = phaseAnimator while the state is live) - `TSMotion.animate(.easeInOut(duration: 0.9).repeatForever(autoreverses: true))` honors Reduce Motion via the wrapper but still carries an inline duration and a repeatForever animation instead of a named looping curve - use a TSMotion looping curve inside a phaseAnimator modifier gated on `isRunning`.
- [LOW] line 89 - DESIGN 4 (system colors) - fallback flame `.foregroundStyle(Color.orange)`; decorative fallback glyph, but a theme-blind spelling in a view (possible) - `AppAttention.color` or a themed accent if the character is meant to follow the theme.
- [LOW] line 29 - DESIGN 4 (themedFont(fitting:) is for layout-derived values, never a constant) - `.themedFont(fitting: max(11, size * 0.8))` embeds a constant 11pt floor; the multiplier form is fine, the constant is not (possible) - drop the floor or make it a reviewed AppFontStep case.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ModelInstallToastView.swift
- [HIGH] lines 59-62 - DESIGN 2 action hierarchy ("A destructive role with no confirmation dialog is always a defect") - `Button(role: .destructive)` Cancel for the active download with no confirmationDialog anywhere; sibling sites (ModelDetailPaneView:317-320 uses `role: .cancel` + `.appAction(.secondary)`, CatalogSheet:88-92 and ModelInstallCards:176-182 use a non-destructive secondary Cancel) treat the same action as non-destructive. Cancelling loses nothing the user made - drop the destructive role (and `.appAction(.destructive)`) to match the peers, or add the localized confirmationDialog.
- [MEDIUM] lines 114-124 - DESIGN 2/4 (card vocabulary; radii from AppMetrics; second local copies start drift) - hand-rolled toast chrome: `RoundedRectangle(cornerRadius: 14)` + shadow + `.appBorder` stroke, where 14 is not an AppMetrics radius, and the identical chrome is duplicated in ToastOverlayView:77-85 - extract a shared toast surface with an AppMetrics radius.
- [LOW] line 56 - DESIGN 2 (a disabled control says why) - Pause disabled when `!model.canPauseModelDownload` with no tooltip or adjacent explanation (possible) - add `.help` stating why pausing is unavailable.
- [LOW] lines 9-17 - DESIGN 6 (fixed English held in a variable needs the catalog-key test registration) - `title` is a `LocalizedStringKey` computed property built from literals; all five keys exist in the catalog (verified), but this property is not covered by `testEnumBackedDisplayStringsAreCatalogKeys` (possible) - register it or inline `Text("...", bundle: .module)` at the use sites.
- Compliant: posts an accessibility announcement on appear (lines 125-129); uses AppProgressBar everywhere (69, 91, 94); `Text(verbatim:)` only on MetricFormat numbers.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SystemPromptSettingsSection.swift
- [HIGH] lines 29-32 - DESIGN 2 (destructive actions use role .destructive AND a confirmationDialog with localized title, message, and button) - "Delete Selected" (`role: .destructive`) deletes the user-authored system prompt immediately; no confirmationDialog in the section (peer settings panes: SkillsSettingsPaneView:139, ProfilesSettingsPaneView:60/81, ModelLibrariesSectionView:101/125 all confirm) - add a localized confirmationDialog.
- [LOW] line 125 - DESIGN 2 (a disabled control says why) - Save Prompt disabled when name or instructions are empty, with no explanation (possible; empty required fields are fairly self-evident) - add `.help` or adjacent hint.
- [LOW] line 129 - DESIGN 3 (no fixed frame; layouts adapt instead of clipping) - editor sheet `.frame(width: 540, height: 360)`; a fixed height can clip the grouped form and the 160pt-min editor under Extra Large text or German (possible) - size with min/ideal/max.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/PersonalitySettingsSection.swift
- [HIGH] lines 27-30 - DESIGN 2 (destructive actions go through a confirmationDialog) - "Remove Selected" (`role: .destructive`) deletes the user-authored personality immediately, no confirmationDialog - add a localized confirmationDialog (or, if removal is meant to be quiet, drop the destructive role; a personality is not rebuildable, so the dialog is the right fix).
- [MEDIUM] lines 15-20 - DESIGN 3 (a control directly under a section header that names it keeps its real label for VoiceOver and hides it with .labelsHidden()) - the Picker sits directly under the "Personality" section header and repeats a visible "Personality" label - keep `label:` (it feeds VoiceOver) and add `.labelsHidden()` (the pattern InstallDestinationPicker uses at lines 98-102).
- [LOW] line 105 - DESIGN 2 (a disabled control says why) - Save disabled on empty fields, no explanation (possible).
- [LOW] line 109 - DESIGN 3 - sheet `.frame(width: 500, height: 330)` fixed size, same clipping risk under long translations / Extra Large (possible).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/AgentModeHintsSectionView.swift
- [HIGH] line 59 - DESIGN 4 (DesignTokenRatchetTests: hard-coded `.appBorder.opacity(...)` borders; counts may only go down) - `.stroke(.appBorder.opacity(0.4), lineWidth: 1)` hard-codes a border below the theme's opacity, so Increase Contrast cannot raise it - stroke `.appBorder` as the theme gives it and lower the ratchet baseline in the same change.
- [MEDIUM] line 76 - DESIGN 4 (theme-blind text styles; `.tertiary` counts too) - `.foregroundStyle(.tertiary)` on the "one sentence per line" hint ignores a custom theme's text colors (counted by the ratchet's system-secondary rule) - use `.appSecondary`/`.appText`.
- [MEDIUM] line 100 - DESIGN 4 - same `.foregroundStyle(.tertiary)` on the placeholder overlay - same fix.
- [MEDIUM] lines 56-59 - DESIGN 2/4 (card vocabulary; radii from AppMetrics) - hand-rolled card (padding + `.background(.appPage)` + clip + stroke) with hard-coded `cornerRadius: 10` instead of `.appCardSurface(padding:)` and `AppMetrics.cardRadius` - adopt the vocabulary card.
- [MEDIUM] line 19 - DESIGN 2/5 (headings carry `.accessibilityAddTraits(.isHeader)`; the vocabulary heading views add it themselves, a bare Label does not) - bespoke `Label` heading "Agent Mode Classifier Hints" carries no header trait, so VoiceOver heading navigation skips it - use `AppSectionHeader` or add `.accessibilityAddTraits(.isHeader)`.
- [MEDIUM] line 91 - DESIGN 3/2 (TextEditor height: measure, do not cap; vocabulary is `.appEditorSurface(minHeight:)`) - `.frame(minHeight: 44, maxHeight: 110)` sits on top of `.appEditorSurface()`, capping every hint editor at 110pt so long hint lists scroll inside a fixed box (possible; the cap may be deliberate to keep four editors compact) - let `appEditorSurface(minHeight:)` size it, or measure the content.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/McpCatalogSourceFormView.swift
- [MEDIUM] lines 81-84 - DESIGN 3 (TextEditor has no intrinsic height; do not fix it with a frame) - sparse-paths editor gets `.frame(height: 50)` instead of `.appEditorSurface(minHeight:)` - use `appEditorSurface(minHeight:)`.
- [LOW] lines 20-24 and 37 - DESIGN 1 (one source of truth) - the segmented Picker renders a visible "Source" label (line 24) directly above the field caption "Source" (line 37): two visible "Source" labels in one block (possible; depends on pane composition) - hide the Picker's label (it duplicates the caption) or rename one.
- [LOW] lines 48-49 - DESIGN 2 (action vocabulary) - `Button { chooseFolder() }.buttonStyle(.bordered)` instead of `.appAction(.secondary, size: .small)`; native but off-vocabulary (possible) - align with the other actions in the form.
- Compliant: verbatim placeholders are example identifiers/paths (lines 42, 68, correctly commented); real labels with `.labelsHidden()`; NSOpenPanel copy via `AppStrings.localized`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/HfMirrorEndpointRowView.swift
- [LOW] line 51 - DESIGN 2 (a disabled control says why) - Save Mirror disabled while the input equals the saved endpoint, with no tooltip explaining the disabled state (possible; "unchanged" is fairly self-evident) - add `.help`.
- Compliant: `.settingsControl("Mirror Endpoint ($HF_ENDPOINT)", pane: .models, timing: .action)` keeps a literal first argument (line 20); field uses `AppMetrics.controlRadius` and a full-opacity `.appBorder` stroke; toasts go through `AppStrings.localized` with `%@` format keys that exist in the catalog.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ToastOverlayView.swift
- [MEDIUM] line 34 - DESIGN 5 (bare withAnimation instead of TSMotion.animate/.tsAnimation; no inline durations) - `withAnimation(reduceMotion ? .none : .easeOut(duration: 0.2))` resolves Reduce Motion by hand and hard-codes 0.2s - use `TSMotion.animate(TSMotion.<curve>)` or `.tsAnimation(_:value:)`.
- [MEDIUM] line 60 - DESIGN 5 - same pattern with inline `.easeOut(duration: 0.15)` in the dismiss action - `TSMotion.animate(TSMotion.hover)`.
- [MEDIUM] lines 77-85 - DESIGN 2/4 (card vocabulary; radii from AppMetrics; second local copies start drift) - hand-rolled toast card, `RoundedRectangle(cornerRadius: 14)` + shadow + stroke, character-identical to ModelInstallToastView:116-124 - extract one shared toast surface with an AppMetrics radius.
- [MEDIUM] lines 65-68 - DESIGN 3 (no fixed frame on a glyph sized with .themedFont; it grows with Text Size) - the xmark is `.themedFont(.tiny)` but framed `.frame(width: dismissButtonSize, height: dismissButtonSize)` where `dismissButtonSize` is an @ScaledMetric (tracks system Dynamic Type, not the in-app Text Size), so the glyph can clip at Extra Large (possible) - size the frame from the font (e.g. `themedFont(fitting:)` companion) or let the glyph size the button.
- [LOW] lines 59-73 - DESIGN 2 (icon-only actions use AppIconButton so tooltip and VoiceOver name cannot drift) - hand-built dismiss button with `.buttonStyle(.plain)`; it does carry `.help` and `.accessibilityLabel`, but the pair is maintained by hand (possible) - use `AppIconButton`.
- Compliant: severity is spoken first in the combined accessibility label (lines 86-98); `Text(verbatim: toast.message)` is correct (caller owns localization); width uses maxWidth 420, not a fixed width.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/AgentsCharacterView.swift
- [HIGH] line 52 - DESIGN 4 (TurboSparkTheme's static accessors are for non-view code only; Gotcha 36: a static read draws the wrong mode under .preferredColorScheme; ratchet-counted) - `TurboSparkTheme.accentColor.opacity(0.28)` inside the glow's RadialGradient - use the environment theme (`@Environment(\.appTheme)`, `theme.accent`) or the mode-explicit form.
- [HIGH] line 81 - DESIGN 4 - same bare static read, `.foregroundStyle(TurboSparkTheme.accentColor)` on the fallback glyph - same fix.
- [MEDIUM] lines 65 and 95 - DESIGN 5 (use the TSMotion curves; no inline durations; breathe/snap are the looping curves) - the phaseAnimator loops (correct shape, correct reduce-motion branch) but hard-codes `.easeInOut(duration: 1.8)` and `.easeInOut(duration: 2.4)` - add a named slow idle curve to TSMotion (per DESIGN: a value the scale lacks is a new named case) and reference it.
- [LOW] lines 40-99 - DESIGN 2 (a thing you need twice joins the vocabulary; a second local copy is how drift starts) - this idle-character body is one of three identical copies (Agents/Memory/Soul) differing only in asset name and fallback glyph - extract a shared idling-character view parameterized by asset and fallback symbol.
- Compliant: `.accessibilityHidden(true)` on the decorative mascot (line 46); reduce-motion branches render a static glow and image.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/MemoryCharacterView.swift
- [HIGH] line 52 - DESIGN 4 (static TurboSparkTheme accessor read in a view body; Gotcha 36; ratchet-counted) - `TurboSparkTheme.accentColor.opacity(0.28)` - use the resolved environment theme.
- [HIGH] line 81 - DESIGN 4 - `.foregroundStyle(TurboSparkTheme.accentColor)` - same fix.
- [MEDIUM] lines 65 and 95 - DESIGN 5 (inline durations instead of TSMotion curves) - `.easeInOut(duration: 1.8)` / `.easeInOut(duration: 2.4)` in the phaseAnimators - use a named TSMotion idle curve.
- [LOW] lines 40-99 - DESIGN 2 - second identical copy of the idle-character animation (with AgentsCharacterView, SoulCharacterView) - extract the shared view.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SoulCharacterView.swift
- [HIGH] line 52 - DESIGN 4 (static TurboSparkTheme accessor read in a view body; Gotcha 36; ratchet-counted) - `TurboSparkTheme.accentColor.opacity(0.28)` - use the resolved environment theme.
- [HIGH] line 81 - DESIGN 4 - `.foregroundStyle(TurboSparkTheme.accentColor)` - same fix.
- [MEDIUM] lines 65 and 95 - DESIGN 5 (inline durations instead of TSMotion curves) - `.easeInOut(duration: 1.8)` / `.easeInOut(duration: 2.4)` in the phaseAnimators - use a named TSMotion idle curve.
- [LOW] lines 40-99 - DESIGN 2 - third identical copy of the idle-character animation - extract the shared view.

### Batch summary
- HIGH (7 findings across 6 files): destructive role without confirmationDialog in SystemPromptSettingsSection:29, PersonalitySettingsSection:27, ModelInstallToastView:59; `.appBorder.opacity(0.4)` in AgentModeHintsSectionView:59; bare `TurboSparkTheme.accentColor` static reads in all three mascot views (lines 52/81 each); bare `withAnimation(...repeatForever)` in TaskProgressFlameIcon:97.
- MEDIUM (14): theme-blind `.tertiary` (AgentModeHints x2), hand-rolled toast/card chrome with non-AppMetrics radii and duplication (ModelInstallToast, ToastOverlay, AgentModeHints), TextEditor fixed heights (McpCatalog:83, AgentModeHints:91), repeated visible Picker label under its section header (Personality:15), ProgressView (ModelProviderDiscovery:25), missing heading trait (AgentModeHints:19), bare withAnimation inline durations (ToastOverlay:34/60), TSMotion.animate with repeatForever/inline duration (TaskProgressIndicator:47), inline mascot loop durations (x3 files).
- LOW (17): raw status hues (ScanFolder:111, ModelProviderDiscovery:76, TaskProgress:89), non-AppMetrics radii and fixed sheet frames, bare empty-state text, disabled controls without a why, icon-only dismiss not using AppIconButton, fixed frame on a themedFont glyph (ToastOverlay:67), unregistered LocalizedStringKey property, duplicated Source labels, off-vocabulary bordered button, fitting-font constant floor, and the three-way mascot code duplication.
- No file exceeds the ~400-line guideline (largest: 142 lines). No localization defects found: every visible string is bundled, verbatim is used only for model-localized strings, paths, identifiers, and numbers, and all keys quoted above exist in the catalog.

---

### Batch batch13_misc

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/MarketplaceSourcesView.swift
- [HIGH] line 32 - Section 2 action hierarchy: "A destructive role with no confirmation dialog is always a defect" and "Removing an entry from a list the user can rebuild ... is a quiet action with a neutral glyph, not a red trash can" - `Button(role: .destructive)` "Remove source" removes a marketplace source, which the pane's own copy (line 24) says keeps installed items, and inherited removals are restorable (line 46) - it deletes nothing the user cannot rebuild - Make it a quiet action (`Button` without role, neutral `minus.circle` styling); if the team insists it is destructive, it needs a localized confirmationDialog.
- [MEDIUM] line 17-21 - Section 2/3 headers - Sheet header built from bare `HStack` + `Spacer` (title Text + Done) instead of `AppAdaptiveHeader`; long German titles at Extra Large text have no adaptive fallback (possible: it is a sheet title row, not a pane header) - Use `AppAdaptiveHeader(leading:actions:)`.
- [MEDIUM] line 66 - Section 4 color - `.foregroundStyle(.red)` on the error text is a system color that ignores a custom theme's palette (a counted DesignTokenRatchet spelling) - Use a theme/vocabulary error or attention style (e.g. `AppAttention.color` with glyph, or an `.app*` style); status already pairs with text, good.
- [LOW] line 64 - Section 2 vocabulary - "Add source" is a plain default `Button` while the same action in FetchedSourceSaveBar (line 37) is `.appAction(.secondary, size: .small)`; the row "Remove source"/"Restore inherited source" buttons are likewise unstyled natives - two treatments for the same extension-source actions (possible) - Use `.appAction` consistently for the form action and row actions.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/GhostGlyph.swift
- [MEDIUM] line 54 - Section 4 color - Default parameter `color: Color = .primary` hands every default call site a theme-blind color: `Color.primary` ignores a custom theme's text color, which is exactly the spellings DESIGN.md section 4 bans (possible: call sites that pass `theme.accent`/`theme.secondaryText`, like GhostChatButton, are fine) - Drop the default or default to a resolved style; document that callers pass a theme-resolved color.
- The fixed `width: size, height: size` frames are inherent to an NSImage-masked glyph (documented in the header) and do not involve `.themedFont`, so not flagged.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/FetchedSourceSaveBar.swift
- [MEDIUM] line 42 - Section 4 color - `.foregroundStyle(.orange)` hard-codes the attention tint; the vocabulary accessor is `AppAttention.color` (which the checklist names for warning/attention tints) and a raw system orange ignores custom themes - Use `AppAttention.color` (or an `.app*` attention style).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ExtensionScopePicker.swift
- clean

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ErrorBanner.swift
- [MEDIUM] line 43 - Section 4 color/surfaces - `Capsule().stroke(.red.opacity(0.55), lineWidth: 1)` is a theme-blind system red with a hand-tuned opacity border; it ignores custom themes and Increase Contrast (which is supposed to raise borders), and hard-coded `.appBorder.opacity`-style dimming is a counted ratchet spelling - Use `.appBorder`/an app error style for the stroke and let the theme own opacity/contrast.
- [MEDIUM] line 12 - Section 4 color - `.foregroundStyle(.red)` on the warning glyph is a raw system color on a themed `.appSurface` capsule (counted spelling) - Use the vocabulary attention/error style; the triangle glyph already satisfies status-not-color-alone.
- [MEDIUM] line 23-34 - Section 2 vocabulary - Icon-only dismiss action hand-rolled: `Button` + `.buttonStyle(.plain)` + `Label` `.labelStyle(.iconOnly)` + separate `.help` and `.accessibilityLabel` carrying the same string - the drift `AppIconButton` exists to prevent (its label is both tooltip and VoiceOver name) - Use `AppIconButton("xmark", label: Text("Dismiss error", bundle: .module))`.
- [MEDIUM] line 29 - Section 3 text in layouts - `.frame(width: 28, height: 28)` fixed on a glyph sized with `.themedFont(.small)`; the glyph grows with Text Size while the frame cannot, clipping at Extra Large - Size the hit target from the glyph or use min/ideal/max, or use AppIconButton's standard metrics.
- Localization, combined accessibility element, announcements (lines 18-19, 46-55) are correct.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/ExcludedScanPathsSectionView.swift
- [LOW] line 22-25 - Section 3 empty states - Empty branch is a bare `Text` paragraph instead of `AppEmptyState(systemImage:title:message:)` (possible: inline-in-card scale may be intended, but AppEmptyState is the one vocabulary for empty lists) - Use `AppEmptyState` with the eye.slash glyph and the existing copy.
- Everything else follows the rules: Label header with `.isHeader` (line 19), `SettingsPathText` (line 32), `.appAction(.secondary, size: .small)` restore with `.help` + hint, `AppMetrics.controlRadius` with `.continuous` (line 43), shared `modelsSettingsCard()` (line 47).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/SettingsPathText.swift
- clean - fulfills the documented contract: `.themedCode` (code font), `.textSelection(.enabled)`, `.lineLimit(1)` + `.truncationMode(.middle)`, full path in `.help`, `Text(verbatim:)` because a path is data.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/StreamRecoveryEventNoticeView.swift
- [MEDIUM] line 19 - Section 4 color - Ternary `presentation.isTerminal ? Color.orange : Color.secondary` uses two theme-blind system colors (raw orange for attention instead of `AppAttention.color`; `.secondary` ignores a custom theme's secondary text) - Use `AppAttention.color` for the terminal case and `.appSecondary` otherwise; the companion glyph already satisfies status-not-color-alone.
- [MEDIUM] line 28 - Section 4 surfaces - `.background(.quaternary, in: ...)` is a system hierarchical fill that ignores a custom theme palette, and `cornerRadius: 10` is a hard-coded literal without `style: .continuous` instead of `AppMetrics.cardRadius` - Use an `.appSurface`/`.appElevated` fill with `AppMetrics.cardRadius, style: .continuous`.
- [LOW] line 23 - Section 4 color - `.foregroundStyle(.primary)` is a banned SwiftUI spelling for themed text (use `.appText` / the resolved theme text color).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/CompactionDividerView.swift
- [MEDIUM] line 27 - Section 4 color - `.background(Color.primary.opacity(0.03))` is the explicitly banned `Color.primary.opacity` spelling; the tint ignores a custom theme's text color (and surfaceOpacity/contrast rules) - Use `.appSurface`/`.appElevated` or a resolved theme color for the summary block background.
- [LOW] line 28 - Section 4 corners - `RoundedRectangle(cornerRadius: 8, style: .continuous)` hard-codes 8, which is neither `AppMetrics.controlRadius` (7) nor `AppMetrics.cardRadius` (10) - Use the matching AppMetrics radius.
- Theme-resolved font/color (`theme.code(.small)`, `theme.metadataForeground`), the single/multi ternary of two `Text`s, and the accessibility label/hint are all correct.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/GhostChatButton.swift
- [LOW] line 17-38 - Section 2 vocabulary - Icon-only action hand-rolled with `Button` + `.buttonStyle(.plain)` + separate `.help` and `.accessibilityLabel` set to the same ternary (lines 36-37), duplicating the pairing `AppIconButton` guarantees; GhostGlyph is not an SF Symbol so `AppIconButton` cannot take it directly (possible: argues for extending the vocabulary rather than this copy) - Either add a custom-image variant to `AppIconButton` per DESIGN.md's "add it to the vocabulary" rule, or keep but note the duplication.
- Destructive end-of-ghost flow is the reference pattern done right: `role: .destructive` inside a localized `confirmationDialog` with cancel and message (lines 39-49); `.isSelected` trait on the active state (line 38) is correct.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/BackgroundAgentsStripView.swift
- clean - thin container over `SubagentLiveCardView`; localization-free by construction (IDs and toasts only), no styling decisions made here.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatShareButton.swift
- [MEDIUM] line 20-22 - Section 2 vocabulary - The labeled menu is hand-rolled as `.menuStyle(.borderlessButton)` + `.fixedSize()` + `.themedFont(.small)`, a parallel local spell of the documented treatment `Menu { ... }.appMenuAction()` (which is `menuStyle(.button)` + bordered + control size + label font + fixedSize) - Replace with `.appMenuAction()`.
- [LOW] line 23-26 - Section 2 action hierarchy ("A disabled control says why") - The button is also disabled by `!model.hasOutputTranscript`, but the `.help` ternary only explains the ghost case; when a chat simply has no output transcript the tooltip says "Save a local copy of this conversation" with no hint of why the control is dead - Add a third branch (or extend the help) for the no-transcript case.
- Ternary of two `Text`s outside the interpolation, bundled literals throughout, and the per-chat options reset are correct.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ConversationLayout.swift
- [LOW] line 9 - Section 2 vocabulary (radii) - `static let cardRadius: CGFloat = 10` re-declares `AppMetrics.cardRadius` (10) as a second spelling of the same token; a future radius change to AppMetrics will not reach consumers of this constant - Delete it and use `AppMetrics.cardRadius` at call sites.
- [LOW] line 5, 13-16 - Section 2 vocabulary ("A text-holding frame or a width breakpoint multiply by AppMetrics.textScale(theme)") - `conversationColumn()` applies `maxWidth: ConversationLayout.readingWidth` (760) with no `textScale` multiplication, so at Extra Large text the measure does not widen (possible: as a max the column still wraps rather than clips, but the shared measure is text-holding) - Multiply by `AppMetrics.textScale(theme)` (needs theme access in the modifier) or document why the measure is exempt.
- `.appBorder` divider with `.accessibilityHidden(true)` is correct.

### Summary
13 files audited: 3 clean (ExtensionScopePicker, SettingsPathText, BackgroundAgentsStripView), 10 with findings. Highest-priority item is the `role: .destructive` "Remove source" button in MarketplaceSourcesView (destructive role with no confirmation dialog on a rebuildable action). The recurring themes are raw system colors (`.red`, `.orange`, `.secondary`, `.primary`, `Color.primary.opacity`, `.quaternary`) and hand-rolled spellings of vocabulary controls (AppIconButton, appMenuAction, AppEmptyState, AppMetrics radii). No file exceeds the ~400-line guidance (largest is 95 lines). No Text(verbatim:) on fixed English copy, no bare-literal localization misses, no animation violations, and no TurboSparkTheme static accessors in view bodies were found in this batch.

---

### Batch batch14_Components_Hooks

Audit of the Hooks extension-style settings pane and its sheets against
DESIGN.md (repo root /Users/whit3rabbit/Documents/GitHub/turbospark).
Read-only audit; no repository files modified.

Verification notes used while auditing: `.appAction` (Theme/AppControls.swift)
adds button style, control size, and label font only, never a keyboard
shortcut, so missing `.keyboardShortcut(.defaultAction)` findings are real.
`AppAttention.color` exists as the centralized attention tint.
`AppMetrics.controlRadius` is 7 and `cardRadius` is 10. A token census of all
four files found only canonical spellings (.appAction, .appSurface, .appElevated,
.appBorder, .appPage, .appText, .appSecondary, .appEditorSurface, .themedFont,
.themedCode). All interpolated localization keys used by these files were
verified present in Localizable.xcstrings. No confirmationDialog or .alert
exists anywhere in the folder.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/Hooks/HookSourceDetailSheet.swift
- [HIGH] line 392 - Section 2 action hierarchy - destructive Delete uses `Button(role: .destructive)` with no confirmationDialog anywhere in the file; DESIGN.md: "A destructive role with no confirmation dialog is always a defect" and a custom hook definition is not trivially rebuildable - add a localized confirmationDialog (title, message, button) before `hookStore.deleteCustomHook(id:)`
- [HIGH] lines 138, 161, 170, 174, 202, 205, 209, 267, 270, 325 - Section 4 color - system `.orange` tints, text, backgrounds, and strokes on themed surfaces ignore custom themes: `.foregroundStyle(.orange)` at 138/202/205/267, `.tint(.orange)` at 161/325, `Color.orange.opacity(0.12)` backgrounds at 170/270, `Color.orange.opacity(0.15)` at 209, stroke `Color.orange.opacity(0.35)` at 174 - use `AppAttention.color` for attention tints and the `.appAttention` surface vocabulary for the banner
- [HIGH] line 247 - Section 4 color - `Color.accentColor.opacity(0.15)` chip background behind the matcher ignores the custom theme accent (explicitly named theme-blind spelling in DESIGN.md) - use `.appAccent`
- [MEDIUM] lines 219, 223, 257, 364, 400 - Section 4 surfaces - hard-coded `.appSurface.opacity(0.6)` (219), `.appBorder.opacity(0.35)` (223), `.appBorder.opacity(0.3)` (257), `.appElevated.opacity(0.7)` (364), `.appSurface.opacity(0.3)` (400) bypass the theme's surface and border treatment under Reduce Transparency / Increase Contrast; `.appBorder.opacity(...)` is a counted DesignTokenRatchetTests spelling outside Theme/ - use plain `.appSurface` / `.appElevated` / `.appBorder`
- [MEDIUM] lines 135-176 - Section 2 vocabulary - review banner hand-rolled from orange fill, stroke, and cornerRadius 8 instead of `.appAttention` with an `AppAttentionNote`; the unreviewed-hooks banner is exactly the "surface that still needs the user" case the vocabulary covers
- [MEDIUM] lines 84-93 - Section 2 vocabulary - hand-rolled empty state (icon plus text in a VStack) instead of `AppEmptyState(systemImage:title:message:)`
- [MEDIUM] lines 214-224, 346-366 - Section 2 vocabulary - hand-rolled cards (background + clipShape + overlay stroke, radii 6 and 8) instead of `.appCardSurface(padding:)`; radii not from `AppMetrics` (controlRadius 7, cardRadius 10)
- [MEDIUM] line 56 - Section 3 layout - fixed `.frame(width: 24, height: 24)` on a glyph sized with `.themedFont(.callout)`; the glyph grows with Text Size and the fixed frame clips it - drop the fixed frame
- [MEDIUM] line 295 - Section 5 motion - inline duration in `TSMotion.animate(.easeInOut(duration: 0.18))`; use a named TSMotion curve (for example `TSMotion.select`) - no inline durations
- [MEDIUM] lines 50-64 - Section 2 vocabulary - icon-only close button hand-rolled (xmark glyph, circle background, separate `.help` and `.accessibilityLabel`) instead of `AppIconButton`, which keeps tooltip and VoiceOver name from drifting
- [MEDIUM] lines 159-161, 317-325 - Section 2 action hierarchy - Trust actions use bespoke `.buttonStyle(.borderedProminent)` + `.tint(.orange)` styling instead of `.appAction`; the attention tint should come from `AppAttention.color`
- [MEDIUM] line 188 - Section 5 accessibility - event section headings (`Text(event.displayName)` at 188) lack `.accessibilityAddTraits(.isHeader)`; headings do not navigate in VoiceOver
- [MEDIUM] lines 42-48 - Section 2 vocabulary - header "Options" button is a bare `.controlSize(.small)` Button while sibling actions in the same file use `.appAction` (390, 395); inconsistent action treatment
- [LOW] lines 25-69 - Section 3 headers - sheet header built from a bare HStack + Spacer rather than `AppAdaptiveHeader` (possible; sheets may have their own convention, but this folder now has three hand-rolled close-button variants, which is the drift the vocabulary rule warns about)

Localization in this file is clean: all fixed copy uses `Text(..., bundle: .module)`, dynamic hook names and config values render verbatim correctly, the expand/collapse ternary at 308-310 is written as two bundled Texts, and the Toggle at 330-337 keeps a real label hidden with `.labelsHidden()`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/Hooks/HooksSettingsPaneView.swift
- [HIGH] lines 139, 143, 147, 151, 220, 223, 227, 246, 250, 287, 290, 294 - Section 4 color - pervasive system `.orange` on themed surfaces: review status capsule (139, 143 text; 147 background; 151 stroke), diagnostics banner (220, 223, 227 text; 246 background; 250 stroke), group badge (287, 290 text; 294 background) - use `AppAttention.color` for the tint and `.appAttention` + `AppAttentionNote` for the banners
- [MEDIUM] lines 256-313 - Section 2 vocabulary - source group cards hand-rolled as `Button` labels (`.appSurface.opacity(0.6)` background, `.appBorder.opacity(0.35)` stroke, cornerRadius 8) instead of `.appCardSurface(padding:)`; the `.appBorder.opacity` spelling is a counted ratchet site (line 308)
- [MEDIUM] lines 212-252 - Section 2 vocabulary - diagnostics banner hand-rolled (orange background, stroke, radius 8) instead of `.appAttention` + `AppAttentionNote`
- [MEDIUM] line 265 - Section 3 layout - fixed `.frame(width: 28, height: 28)` on a glyph sized with `.themedFont(.base)`; the icon grows with Text Size and clips inside the tile
- [LOW] lines 163-174, 178-189, 193-204 - Section 2 vocabulary - section headings built from `Text` instead of `AppSectionHeader(title:subtitle:)` (LOW because they do carry `.settingsControl` anchors and `.accessibilityAddTraits(.isHeader)`)

Otherwise a strong extension-style pane: header uses `ExtensionPaneHeader` with `AppSearchField`, overflow menu, and `.appAction(.primary)` Add Hook in the shared order; empty states use `AppEmptyState` and `ExtensionNoMatchesState`; the purpose line anchors settings search with a string literal.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/Hooks/HookEditorSheet.swift
- [HIGH] lines 221-227 - Section 5 keyboard - Save Hook (`.appAction(.primary)`) has no `.keyboardShortcut(.defaultAction)`; `.appAction` adds no shortcut, and the sheet's Cancel already binds `.cancelAction` at line 217, so Return does not submit while Esc cancels - bind Save to `.keyboardShortcut(.defaultAction)`
- [HIGH] lines 37-248 - Section 5 keyboard - no `@FocusState` anywhere; the sheet never focuses its first field (Hook Name) on appear - focus the first field when the sheet opens
- [MEDIUM] lines 224-227 - Section 2 action hierarchy - Save is disabled while the command script is empty, which is input the user can still give; DESIGN.md says keep the primary action pressable and let the press show what is missing (`.appAttention` on the unfinished field, Voice Cloning reference) - at minimum add a `.help` explaining why it is disabled
- [MEDIUM] line 204 - Section 2 disabled controls - the async toggle is disabled for blocking events with the reason only in a code comment; a disabled control says why in its tooltip or adjacent text - add `.help`
- [MEDIUM] lines 55-64 - Section 2 vocabulary - icon-only close hand-rolled (`xmark.circle.fill` with separate `.help` and `.accessibilityLabel`) instead of `AppIconButton`
- [LOW] line 179 - Section 3 layout - fixed `.frame(width: 80)` on the Timeout field in a form row (possible; the rule names pickers, search fields, and labels, but a fixed width does not adapt at Extra Large) - prefer maxWidth/ideal
- [LOW] lines 40-68 - Section 3 headers - sheet header from bare HStack + Spacer (possible; see HookSourceDetailSheet note)

Good vocabulary use here: the command editor uses `TextEditor(...).appEditorSurface(minHeight: 90)` with `.accessibilityLabel`, `t.title` renders through `Text(LocalizedStringKey(...), bundle: .module)`, verbatim prompts are technical matcher syntax (correct), the persisted hook name stays unlocalized (correct), and every field keeps a real label hidden with `.labelsHidden()`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Components/Hooks/HookOptionsConfigSheet.swift
- [HIGH] lines 71-74 - Section 5 keyboard - Done (`.appAction(.primary)`) lacks `.keyboardShortcut(.defaultAction)`; a sheet binds Done to the default action - add it
- [HIGH] line 94 - Section 4 color - `.foregroundStyle(.red)` on the required-option marker is a system color on a themed surface and ignores custom themes - use a theme or `AppAttention` color
- [MEDIUM] lines 15-84 - Section 5 keyboard - no `@FocusState`; the sheet does not focus its first field on appear - focus the first option field
- [MEDIUM] lines 191-197 - Section 2 vocabulary - option rows hand-rolled as cards (`.appSurface.opacity(0.6)` background, `.appBorder.opacity(0.3)` stroke, cornerRadius 8) instead of `.appCardSurface(padding:)`; the `.appBorder.opacity` spelling is a counted ratchet site
- [MEDIUM] lines 44-54 - Section 2 vocabulary - hand-rolled empty state (icon plus text) instead of `AppEmptyState(systemImage:title:message:)`
- [MEDIUM] lines 27-36 - Section 2 vocabulary - icon-only close hand-rolled (`xmark.circle.fill` with separate `.help` and `.accessibilityLabel`) instead of `AppIconButton`
- [LOW] lines 18-40 - Section 3 headers - sheet header from bare HStack + Spacer (possible; see HookSourceDetailSheet note)

Good here: the env var name and declared default value use `Text(verbatim:)` correctly, sensitive options use `SecureField`, `valuePrompt` returns two bundled Texts rather than a ternary inside one Text, and fields keep real hidden labels.

### Cross-file observations
- Three different hand-rolled close buttons in one folder (xmark in circle background at HookSourceDetailSheet 50-64, xmark.circle.fill at HookEditorSheet 55-64 and HookOptionsConfigSheet 27-36). The vocabulary rule ("if the vocabulary lacks something you need twice, add it") points at one shared sheet-header or AppIconButton-based close.
- None of the three sheets binds its primary action to `.keyboardShortcut(.defaultAction)`; only cancel/close paths are bound.
- All four files sit near or below the 400-line guidance (404, 347, 285, 230), so no split is required.

---

### Batch batch15_Diagnostics

Audit against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md. All localization
keys referenced by these files (including "FROM TEMPLATE", "Custom…", "beyond", "trained",
"max fit", and every interpolated format key) verified present in
swift/TurboSparkApp/Localization/Localizable.xcstrings. All files are under the ~400-line
guideline.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/GenerationSamplingSection.swift
- [HIGH] lines 208-217 - DESIGN.md section 2, action hierarchy ("A destructive role with no confirmation dialog is always a defect") - "Remove Override" is `Button(role: .destructive)` with no `confirmationDialog`. Either the action is not destructive (it reverts the chat to app defaults, the sibling of "Use Default") and should drop the role for `.appAction(.secondary)`, or it needs a localized confirmationDialog.
- [HIGH] lines 273-279 - DESIGN.md section 2, action hierarchy - `AppIconButton("trash", ..., role: .destructive)` deletes a saved sampling preset permanently with no `confirmationDialog`; this is real data loss, the exact case the rule targets.
- [MEDIUM] lines 286-289 - DESIGN.md section 2, action hierarchy - context-menu `Button(role: .destructive)` "Delete Preset" performs the same unconfirmed deletion; one confirmationDialog fix should cover both entry points.
- [MEDIUM] lines 91, 124, 143 - DESIGN.md section 3, "No fixed .frame(width:) ... in a header or form row" - `.frame(width: 36, alignment: .trailing)` on the Temperature / Top-P / Penalty value readouts, which grow with Text Size (inherited body font); "2.00" can clip at Extra Large. Scale by `AppMetrics.textScale(theme)` or use `minWidth`.
- [MEDIUM] line 160 - DESIGN.md section 3, same rule - seed `TextField` fixed `.frame(width: 100)` in a form row; the field does not grow with Text Size.
- [LOW] line 182 - DESIGN.md section 2, "A disabled control says why" - the whole section `.disabled(model.isRunning)` with no tooltip or adjacent text explaining that knobs are dead while a generation runs (the reasoning rows below do carry their own explanations; the sampling knobs do not).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/InspectorEssentialsSection.swift
- [MEDIUM] line 211 - DESIGN.md section 5, motion ("No inline durations") - `.tsAnimation(.smooth(duration: 0.2), value: isCustom)` invents an inline spring; TSMotion.select is the documented curve for "disclosure open/close". Use `.tsAnimation(TSMotion.select, value: isCustom)`.
- [MEDIUM] line 233 - DESIGN.md section 3, fixed width on a form-row value - `.frame(width: 55, alignment: .trailing)` on the custom-size token readout; "131,072" clips at Extra Large text. Scale by `AppMetrics.textScale(theme)` or use `minWidth`.
- [LOW] line 29 - DESIGN.md section 2, "A disabled control says why" - the section is disabled while `model.isRunning || model.isInstallingModel` with no tooltip or adjacent text saying which condition holds.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/ContextSelectionPopover.swift
- [HIGH] line 32 - DESIGN.md section 4 ("Color.primary.opacity ignores a custom theme") - the collapsed button's background is `Color.primary.opacity(0.05)`. Use `.appSurface` or another theme-resolved style.
- [HIGH] line 133 - DESIGN.md section 4, Gotcha 36 ("TurboSparkTheme's static accessors are for non-view code only") - `.background(TurboSparkTheme.surfaceColor)` reads the bare static accessor inside the popover body, so it draws the wrong mode under `.preferredColorScheme`. Use `.background(.appSurface)`.
- [MEDIUM] line 27 - DESIGN.md section 4 - `.foregroundStyle(.tertiary)` on the chevron is theme-blind; use `.appSecondary`.
- [MEDIUM] lines 243, 255 - DESIGN.md section 4 - `.foregroundStyle(Color.primary)` on the row's primary text and the checkmark ignores a custom theme's text color; use `.appText`.
- [LOW] lines 262-263 - DESIGN.md section 2, radii row - hard-coded `cornerRadius: 6` (background RoundedRectangle and contentShape) instead of `AppMetrics.controlRadius` (7); the sibling `ContextSelectionButton` in this same file uses `AppMetrics.controlRadius`.
- [LOW] line 267 - (possible) hard-coded disabled dimming `.opacity(disabled ? 0.45 : 1)` stacks on the native `.disabled` dim; a theme-owned disabled treatment would keep this consistent under Increase Contrast.
- [LOW] line 132 - (possible, informational) fixed `.frame(width: 300)` on the popover; secondary lines wrap via `.fixedSize(horizontal: false, vertical: true)` so it mostly holds, but Extra Large text squeezes the rung rows.
- [LOW] lines 56-57, 70 - (possible, cross-file note) `presentation.color` comes from `ModelFitPresentation.color` (Installation/ModelFitPresentation.swift, outside this batch), which returns system `.green/.blue/.orange/.red/.secondary`; the dot and fit-label tint in this file are therefore theme-blind too. Status is correctly paired with a text label, so the color-alone rule is met.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/UsageDashboardSection.swift
- [HIGH] line 181 - DESIGN.md section 4 ("Color.primary.opacity ignores a custom theme") - `heatColor` returns `Color.primary.opacity(0.05)` for empty heatmap cells; theme-blind fill on the themed surface.
- [HIGH] line 183 - DESIGN.md section 4, Gotcha 36 - `TurboSparkTheme.accentColor` is the bare static accessor, read in `heatColor` which is called from the view body; draws the wrong mode under `.preferredColorScheme`. Pass the environment `theme.accent` into `heatColor` instead.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/RunnerDiagnosticsSection.swift
- [MEDIUM] line 62 - DESIGN.md section 4 - `groupLabel` uses `.foregroundStyle(.tertiary)`, which ignores a custom theme's text colors; use `.appSecondary`.
- [MEDIUM] line 46 - DESIGN.md section 4 - the "No runs yet" empty-state text uses `.foregroundStyle(.tertiary)`; use `.appSecondary`.
- Clean otherwise: labels are `Text(_, bundle: .module)`, readouts are `Text(verbatim:)` numbers/units as DESIGN.md section 6 prescribes, `groupLabel` carries `.accessibilityHeading(.h3)`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/InspectorView.swift
- [LOW] lines 39-49 - (possible) DESIGN.md section 2/3, headers - the inspector header is a bare `HStack + Spacer` rather than `AppAdaptiveHeader`; it has no actions so the reflow behavior may not apply, but the title `Text` can truncate under German + narrow inspector with no fallback. It does correctly carry `.accessibilityElement(children: .combine)` and `.accessibilityAddTraits(.isHeader)`.
- Clean otherwise: mode-explicit `TurboSparkTheme.barBackgroundColor(isDark: theme.isDark)` (exactly the section 4 prescription), `.appBorder` hairline, `.background(.appPage)`, Form with `.formStyle(.grouped)` and `.scrollContentBackground(.hidden)`, `.themedFont` throughout, one section per struct.

---

### Batch batch17_Generation

Audited against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md (sections 2-8). Catalog presence was verified in swift/TurboSparkApp/Localization/Localizable.xcstrings; vocabulary controls confirmed in Sources/Theme/AppControls.swift and AppProgressBar.swift.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ToolCallCardView.swift
- [HIGH] line 489 (with 497-515, 344) - Localization (section 6) - `Text(statusLabel)` renders fixed English status copy ("Needs Approval", "Running...", "Classifier Approved", "Completed", "Denied", "Failed") through the non-localizing `Text(String)` init; the catalog has "Needs Approval" but not "Running..." or "Classifier Approved", and even cataloged strings are bypassed by the String init. The same English string feeds `.accessibilityValue` at line 344. - Return localized text per case (`Text(..., bundle: .module)` from a helper, or `AppStrings.localized` for the String form) and rebuild the accessibility value from it.
- [HIGH] line 470 - Localization (section 6, ternary rule) - `Text(risk.level == .high ? "High-Risk Operation Detected" : "Security Notice")` is a ternary inside one `Text` AND a bare literal to the key-only API (resolves against Bundle.main, empty in a SwiftPM build). Both keys exist in the catalog but are unreachable here. - `risk.level == .high ? Text("High-Risk Operation Detected", bundle: .module) : Text("Security Notice", bundle: .module)`.
- [HIGH] lines 348-356 - Localization (section 6) + empty-state naming rule - `efficiencyOutcome` builds English UI copy from literals ("Validation: ...", "Archived output available through recall_tool_output", "Verified local reduction", "Completed-todo boundary compacted"), joins them, and renders via `Text(String)`; none are catalog keys and one names the internal tool id `recall_tool_output` in user-facing copy. - Localized format keys (`%@`) at the call site with `bundle: .module`; rephrase the archived-output line in user terms.
- [HIGH] line 1095 - Theming (section 4, Gotcha 36) - `.tint(isHighRisk ? Color.orange : TurboSparkTheme.accentColor)` reads the static `TurboSparkTheme.accentColor` accessor inside a view body (it keys off `NSApp.effectiveAppearance`, not `.preferredColorScheme`), and mixes a system color into the ternary. - Use the environment theme (`theme.accent`) and AppAttention.color for the warning tint.
- [MEDIUM] line 278 - Motion (section 5) - bare `withAnimation(.easeInOut(duration: 0.18))` with an inline duration; animates even under Reduce Motion. - `.tsAnimation(TSMotion.select, value: isExpanded)` (or `TSMotion.animate` with a named curve).
- [MEDIUM] lines 74-82 - Theming (section 4) - `borderColor` paints the card stroke with `Color.red.opacity(0.7)`, `Color.orange.opacity(0.6)`, `Color.red.opacity(0.4)` system colors on a themed surface. - AppAttention.color for the approval/error attention states, `.appBorder` otherwise.
- [MEDIUM] lines 517-537 - Theming (section 4) - `statusBackground`/`statusForeground` use `Color.orange/.blue/.green/.secondary/.red` with hard-coded `.opacity(0.15)`; ignores custom themes and isHighContrast. - AppAttention.color tints; the paired glyphs already satisfy the status-not-color-alone rule.
- [MEDIUM] lines 362-389 - Theming (section 4) - status icon foregrounds use `Color.red`, `Color.orange`, `Color.green`, `Color.secondary`. - AppAttention.color or theme-resolved styles.
- [MEDIUM] lines 459-460, 467-472, 483 - Theming (section 4) - risk badge and risk warning box use `Color.red`/`Color.orange` foregrounds, `Color.red.opacity(0.15)`/`Color.orange.opacity(0.12)` capsule fills and a `.opacity(0.08)` box background. - AppAttention.color with AppAttentionNote-style treatment.
- [MEDIUM] lines 547, 595, 637, 691, 695, 701, 719, 757, 788, 808, 818, 858, 898, 933, 979, 1030 - Theming (section 4) - `Color.primary.opacity(0.03)` panel washes and `.primary.opacity(0.9)` code text across all the preview builders; `Color.primary.opacity` ignores a custom theme's text color. - `.appElevated` / theme-resolved foreground.
- [MEDIUM] lines 11-15 - Theming (section 4) - `diffAdditionColor`/`diffDeletionColor` fall back to `Color.primary`/`Color.secondary` when diff markers are off. - theme-resolved styles (`theme.foreground` / `.appSecondary`).
- [MEDIUM] lines 333, 379, 408, 422, 433, 445, 605, 616, 676-677, 709, 852, 893, 953-954, 1045, 1054 - Theming (section 4) - remaining theme-blind spellings: `.tertiary` chevron, `.secondary` strikethrough colors, `.orange` findings icon, `Color.secondary.opacity(0.12)` capsules, `Color.purple` background badge, `Color.indigo` badge, `Color.red` stop icon, `Color.orange` classifier notice. - `.appSecondary`/`.appAccent`/AppAttention.color as appropriate.
- [MEDIUM] line 741 - Localization (section 6) - `Text(verbatim: "\(lineCount) lines")` hard-codes English "lines" as verbatim; it is translatable copy. - `Text("\(lineCount) lines", bundle: .module)` (Int interpolates as `%lld`).
- [MEDIUM] line 848 - Localization (section 6) - `Text(verbatim: "Lines \(start)-\(end)")` hard-codes English "Lines". - `Text("Lines \(start)-\(end)", bundle: .module)`.
- [LOW] lines 484, 547, 595, 637, 695, 701, 719, 757, 791, 811, 818, 858, 898, 933, 979, 1025, 1055 - Theming (section 4, radii) - inline `cornerRadius: 6` / `cornerRadius: 8` instead of AppMetrics radii. - `AppMetrics.controlRadius` / `AppMetrics.cardRadius`.
- [LOW] whole file - Layout (section 3, Gotcha 15) - 1186 lines, far beyond the ~400-line guidance. - Split the preview builders (agent/file/web/mcp/question previews) into sibling files.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectSettingsSheet.swift
- [HIGH] lines 158, 179, 223, 258, 296, 327, 376 - Theming (section 4, Gotcha 36) - `TurboSparkTheme.surfaceColor` static accessor read inside view bodies for every field, note card, and section background; draws the wrong mode under `.preferredColorScheme`. - Read the environment theme (`theme.surface`) or use the `.appSurface` shape style.
- [HIGH] lines 366-371 - Action hierarchy (section 2) - `Button(role: .destructive)` "Remove Index" has no confirmationDialog; DESIGN says a destructive role with no confirmation dialog is always a defect. - Add a localized confirmationDialog (or, since the index is rebuildable via Reindex, drop the destructive role for a quiet action).
- [MEDIUM] lines 405-414 (also 100, 106) - Accessibility, Keyboard (section 5) - the sheet footer Cancel and Save bind no `.keyboardShortcut(.cancelAction)` / `.keyboardShortcut(.defaultAction)`, and the nested Skills/Plugins sheets' Done buttons lack `.defaultAction`. - Add the shortcuts; the Delete Project dialog (lines 390-402) shows the intended pattern for role buttons.
- [MEDIUM] lines 413-414 - Action hierarchy (section 2) - Save is disabled when the name is empty or `model.isRunning` / `submitting` / `pendingToolCall` with no tooltip or adjacent text saying why. - Add `.help` or adjacent text; the chat Send button is the reference.
- [MEDIUM] lines 337-341 - Vocabulary (section 2) - `ProgressView()` for indexing progress instead of `AppProgressBar`. - `AppProgressBar` (determinate value when stats allow).
- [MEDIUM] line 90 (possible) - Layout (section 3) - fixed `.frame(width: 580, height: showingAdvancedSettings ? 660 : 390)`: hard-coded height that does not scale with Text Size; under Extra Large text the header/footer crowd the fixed budget. - minWidth/idealWidth/maxWidth or an ideal height, or derive height from content.
- [LOW] lines 92-94, 158-163, 179-184 - Theming (section 4, radii) - `cornerRadius: 16` sheet chrome and `cornerRadius: 8` fields are not AppMetrics radii. - `AppMetrics.cardRadius` / `AppMetrics.controlRadius`.
- [LOW] lines 152-163, 173-184 - Vocabulary (section 2) - the same hand-rolled plain TextField + background + border overlay treatment appears twice; a treatment needed twice belongs in the vocabulary per section 2. - Extract a shared field surface (or adopt the existing vocabulary field treatment).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ArtifactPanelView.swift
- [MEDIUM] lines 121-211 - Headers (section 3) - pane header is a bare `HStack` + `Spacer` with about seven trailing controls instead of `AppAdaptiveHeader`; the title uses `lineLimit(1)` + `.truncationMode(.middle)` with no tooltip carrying the full title, so it truncates under German/Extra Large with no recovery. - Use `AppAdaptiveHeader(leading:actions:compactActions:)`; give the title `.help(title)` if it truncates.
- [MEDIUM] lines 216-232 - Vocabulary (section 2) - bespoke `headerButton(_:help:label:action:)` duplicates `AppIconButton`; `.help` and `.accessibilityLabel` are set separately so they can drift, which AppIconButton binds together. - Use `AppIconButton`.
- [MEDIUM] lines 224-225, 261-262, 412-413 - Layout (section 3) - fixed `.frame(width: 22, height: 22)` / `.frame(width: 16, height: 16)` on glyphs sized with `.themedFont`, which grows with Text Size. - Drop the fixed frames or scale them by `AppMetrics.textScale(theme)`.
- [MEDIUM] lines 229, 240, 272, 379 - Theming (section 4) - `Color.secondary` foregrounds, the `Color.primary.opacity(0.03)` banner wash, and `.quaternary` unavailable icon ignore custom themes. - `.appSecondary`, `.appElevated`/theme surface, theme-resolved dim.
- [MEDIUM] lines 468, 471 - Motion (section 5) - `TSMotion.animate(.easeInOut(duration: 0.15))` / `.easeOut(duration: 0.15)` pass inline durations instead of named TSMotion curves. - `TSMotion.animate(TSMotion.snap/select)` or the fitting named curve.
- [MEDIUM] line 426 - Layout (section 3) - the maximized preview sheet is a fixed `.frame(width: 1100, height: 800)`; no window-like surface has a fixed frame and it can exceed smaller displays. - Size to the presenting container/visible frame with min/ideal/max.
- [LOW] lines 375-395 (possible) - Empty states (section 3) - `unavailableView` is a hand-rolled nothing-to-show state; `AppEmptyState(systemImage:title:message:) { actions }` is the vocabulary for a workspace with no content. - Consider AppEmptyState so the treatment matches other empty workspaces.
- [LOW] line 158 - Action hierarchy (section 2) - Copy source disabled while `sourceText == nil` with no explanation why. - Add `.help` naming the reason.

Note: lines 46 and 427 use `TurboSparkTheme.barBackgroundColor(isDark: theme.isDark)`, the correct mode-explicit form; not flagged.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarView.swift
- [HIGH] lines 338-357 - Empty states (section 3) - the chat list's empty state is hand-rolled (icon + two Texts) instead of `AppEmptyState`, and it offers no filling action in user terms (no New chat affordance); DESIGN requires every list's empty state to say what goes here AND offer the action that fills it. - `AppEmptyState(systemImage:title:message:) { New chat button }`, with the search/no-match variants as the message.
- [MEDIUM] lines 142-177 - Vocabulary (section 2) - hand-rolled search field (magnifyingglass Image + plain TextField + clear button) duplicates `AppSearchField(text:prompt:)`. - Use `AppSearchField`.
- [MEDIUM] lines 421-448 - Accessibility (section 5) - archived rows apply `.accessibilityElement(children: .combine)` over an HStack that contains the Restore button and the Delete AppIconButton; combining folds those controls into one element so they are no longer individually reachable by VoiceOver. - Combine only the icon and title (wrap them separately), leaving the two buttons as their own elements.
- [MEDIUM] line 105 - Motion (section 5) - `TSMotion.animate(.easeInOut(duration: 0.12))` uses an inline duration instead of a named TSMotion curve. - `TSMotion.animate(TSMotion.select)`.
- [MEDIUM] lines 313-315 - Action hierarchy (section 2) - New chat is disabled while `model.isRunning` with no explanation (the tooltip only names the shortcut). - Extend `.help` or add adjacent text saying why it is disabled; the chat Send button is the reference.
- [LOW] lines 100-137 (possible) - Vocabulary (section 2) - bespoke drawn tab selector (full-width pills with hand-mixed selected states) for switching between two modes, near-duplicating the `AppPillPicker` vocabulary. - `AppPillPicker` or document why the sidebar tabs stay bespoke.
- [LOW] lines 120, 132, 285, 293 (possible) - Theming (section 4) - hard-coded `theme.foreground.opacity(0.03/0.04/0.06/0.12)` washes bypass `surfaceOpacity` and `isHighContrast` handling for surfaces. - Theme surface tokens (`.appElevated` / `theme.surface`).
- [LOW] lines 121-123, 132-135, 170-173, 293 - Theming (section 4, radii) - inline `cornerRadius: 6` / `cornerRadius: 8` instead of AppMetrics radii. - `AppMetrics.controlRadius`.

Positive observations (no action): ChatSidebarView's Delete Project / Delete Chat destructive flows stage localized alerts with cancel buttons (DESIGN-conformant); the rename and delete alerts, filter placeholder ternaries, and reindex-button label ternary all use the correct two-Text spelling; sliders expose `.accessibilityValue`; section headers carry `.accessibilityAddTraits(.isHeader)`; ArtifactPanelView uses the mode-explicit `barBackgroundColor(isDark:)` form.

---

### Batch batch18_Generation

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/WorktreeView.swift
- [HIGH] line 87 - Localization (fixed English held in a variable, section 6) - `Text(tab.title)` renders `WorktreeTabMode.title` ("Changes"/"Timeline"/"Worktrees", State/WorktreeModel.swift:121-125) as verbatim `Text(String)`; none of those values are catalog keys (verified absent from Localizable.xcstrings), so the tab bar is English under all 21 languages - render `Text(LocalizedStringKey(tab.title), bundle: .module)` and register the enum in `testEnumBackedDisplayStringsAreCatalogKeys`, or make `title` a bundled `Text` like `SlashCommandMenuGroup.title`.
- [HIGH] lines 92, 94 - Theming (section 4, TurboSparkTheme static accessors) - tab buttons read `TurboSparkTheme.accentColor` (and `.opacity(0.12)`) inside the view body and mix it with system `.secondary`; `NSApp.effectiveAppearance` statics draw the wrong mode under `.preferredColorScheme` and ignore custom themes - use `theme.accent` / `.appAccent` and `.appSecondary`.
- [HIGH] line 160 - Theming (section 4) - `TurboSparkTheme.accentColor` static read for the search-toggle tint - use `theme.accent`.
- [HIGH] line 188 - Theming (section 4) - `TurboSparkTheme.accentColor` static read for the split-mode toggle tint - use `theme.accent`.
- [HIGH] line 470 - Theming (section 4) - `.tint(TurboSparkTheme.accentColor)` static read on the Initialize button - use `theme.accent`.
- [HIGH] lines 173-176 - Motion (section 5) - `.animation(worktree.isRefreshing ? .linear(duration: 0.8).repeatForever(autoreverses: false) : .default, value:)` spins the refresh glyph with a `repeatForever` started from `.animation`; it never stops with the state and ignores Reduce Motion - replace with `.tsPulse(active: worktree.isRefreshing)` on the glyph.
- [HIGH] line 191 - Localization (bundled spellings, section 6) - `.help(worktree.isExpandedSplitMode ? "Collapse split view" : "Expand into two-column diff view")` uses the Bundle.main `String` overload, and both keys are absent from the catalog (verified), so the tooltip is English under every language - `.help(isExpandedSplitMode ? Text("Collapse split view", bundle: .module) : Text("Expand into two-column diff view", bundle: .module))` and add both keys.
- [HIGH] line 242 - Localization (AppStrings.localized, section 6) - `model.showToast(result.error ?? "Failed to switch branch.", style: .error)` sends a raw English literal to the UI and the key is not in the catalog (verified) - `AppStrings.localized("Failed to switch branch.")` with the key added.
- [HIGH] line 318 - Localization (ternary inside Text; missing key) - `Text("\(count) \(count == 1 ? "file" : "files")", bundle: .module)` hides the plural choice inside one Text; the interpolated key "%lld %@" is not in the catalog (verified), so it falls back to English under all languages - use two bundled Texts or a proper plural format key instead of interpolating the ternary result.
- [MEDIUM] lines 80, 113, 153, 182 - Motion (section 5) - bare `withAnimation(.easeInOut(duration: 0.12/0.15))` with inline durations; animates even with Reduce Motion on - `TSMotion.animate(TSMotion.pane/select)`.
- [MEDIUM] lines 104, 148, 324 - Theming (section 4) - `Color.primary.opacity(0.02)` / `(0.04)` strip and capsule backgrounds ignore custom themes - `.appSurface` or a theme-resolved fill.
- [MEDIUM] lines 138, 143 - Theming (section 4) - `.foregroundStyle(.green)` / `(.red)` for the +/- counts are system colors on a themed surface (the +/- glyphs carry the meaning, but the hues ignore the theme) - theme-resolved diff add/delete colors.
- [MEDIUM] line 120 - Layout (section 3) - fixed `.frame(width: 22, height: 22)` on a glyph sized with `.themedFont(.small)` clips at Extra Large text - drop the fixed frame or scale by `AppMetrics.textScale(theme)`.
- [MEDIUM] lines 295, 316, 386, 393, 399 - Theming (section 4) - `.foregroundStyle(.tertiary)` theme-blind spellings - `theme.secondaryText` / `.appSecondary`.
- [MEDIUM] lines 288, 304-310 - Localization (section 6) - `comparisonLabel` composes fixed English " -> working tree" into a `String` shown via `Text(comparisonLabel)` (verbatim) - a localized format key with the branch/hash interpolated.
- [MEDIUM] lines 336, 348 - Layout (section 3) - fixed list column widths `.frame(width: 270)` and `.frame(width: 320)` in the split layouts; "Never a fixed list width" - min/ideal/max frames scaled by `AppMetrics.textScale(theme)`.
- [MEDIUM] lines 363, 416 - Vocabulary (section 2) - `ProgressView` for the loading-diff state instead of `AppProgressBar` - `AppProgressBar(animated:)` with the label.
- [MEDIUM] line 390 (390-401) - Vocabulary (section 2) - hand-rolled empty side pane (glyph + two Texts) instead of `AppEmptyState` - `AppEmptyState(systemImage:title:message:)`.
- [MEDIUM] line 441 (441-473) - Vocabulary (section 2) - hand-rolled non-Git empty card instead of `AppEmptyState` with the Initialize action.
- [MEDIUM] lines 78-99 - Accessibility (section 5) - custom tab buttons convey selection only via color/weight; tabs carry `.isSelected` - `.accessibilityAddTraits(worktree.activeTab == tab ? [.isSelected] : [])`.
- [MEDIUM] line 469 - Vocabulary (section 2) - `.buttonStyle(.borderedProminent)` bespoke primary action instead of `.appAction(.primary)`.
- [LOW] line 445 - Surfaces (section 4) - hard-coded `.appAccent.opacity(0.8)` dimming on the glyph - let the theme decide.
- [LOW] informational - file is 475 lines, past the ~400 guidance (section 3); split subviews into siblings.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/MessageRowView.swift
- [HIGH] line 94 - Theming (section 4, DesignTokenRatchet-counted) - hard-coded `.appBorder.opacity(0.4)` stroke on the artifact chip; bypasses the theme's border weight (Increase Contrast) and raises the ratchet count - `.appBorder` unmodified or a vocabulary surface.
- [MEDIUM] lines 90, 219 - Theming (section 4) - `Color.primary.opacity(0.04)` chip and shell-output backgrounds - `.appSurface`.
- [MEDIUM] line 155 - Theming (section 4) - `Color.primary.opacity(0.08)` bubble stroke - `.appBorder`.
- [MEDIUM] lines 152, 226, 320 - Surfaces (section 4) - `.appSurface.opacity(0.85)` hard-coded dimming bypasses `surfaceOpacity` (Reduce Transparency / Increase Contrast) - use `.appSurface` as the theme gives it.
- [MEDIUM] lines 154-158, 220, 228-231, 322-325 - Radii (section 4) - bubble `cornerRadius: 16` is not an AppMetrics radius (controlRadius 7, cardRadius 10, composerRadius 20) - use `AppMetrics.composerRadius` for the chat bubbles.
- [MEDIUM] line 122 - Theming (section 4) - `.foregroundStyle(.secondary)` on the missing-image label - `.appSecondary`.
- [MEDIUM] line 85 - Theming (section 4) - `.foregroundStyle(.tertiary)` on the chip's trailing glyph - `theme.secondaryText`.
- [MEDIUM] line 419 - Motion (section 5) - `TSMotion.animate(.easeInOut(duration: 0.15))` inline duration instead of a TSMotion curve - `TSMotion.animate(TSMotion.hover)`.
- [LOW] lines 91, 93 - Radii (section 4) - `cornerRadius: 6` literal - `AppMetrics.controlRadius`.
- [LOW] informational - file is 466 lines, past the ~400 guidance (section 3).
- Note: hover actions here are correctly mirrored with named `.accessibilityAction`s and rows use `.accessibilityElement(children: .contain)` (section 5 reference behavior). Lines 114-121 verbatim counts and the ANSI palette at 279-299 are correct (fixed palette by design, verbatim numbers).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarGroupedProjectsView.swift
- [MEDIUM] line 308 - Surfaces (section 4) - `theme.foreground.opacity(0.88)` hard-coded text dimming for unselected rows, lower than the theme's secondary text - `theme.secondaryText` / `.appSecondary`.
- [MEDIUM] lines 419-426 - Accessibility (section 5) - "Other Tasks" section heading is a bare HStack of Text with no `.accessibilityAddTraits(.isHeader)` (and not `AppSectionHeader`) - add the header trait.
- [MEDIUM] lines 441-460 - Vocabulary (section 2) - hand-rolled empty state (glyph + Text + button) instead of `AppEmptyState(systemImage:title:message:) { actions }`.
- [LOW] line 361 - Surfaces (section 4) - hover fill `theme.foreground.opacity(0.04)` hard-coded instead of a theme-provided hover token.
- [LOW] line 396 - Surfaces (section 4) - `theme.secondaryText.opacity(0.35)` dimming below what the theme gives for the idle dot.
- [LOW] lines 42, 321, 352 - Action hierarchy (section 2) - controls disabled while a run is active with no tooltip or adjacent text saying why - `.help` explaining the disabled state.
- [LOW] lines 220-277 (possible) - Accessibility (section 5) - the project header's hover-revealed New Task button and actions menu are not mirrored in a `.contextMenu` or `.accessibilityAction` the way the task rows (line 368) are.
- [LOW] informational - file is 461 lines, past the ~400 guidance (section 3).
- Note: deletion flows here are correct - destructive menu items set state and the alert at lines 63-80 is the confirmationDialog with localized title/button; selection traits and accessibilityLabels on rows are present.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatParitySheets.swift
- [HIGH] line 231 - Theming (section 4, TurboSparkTheme static accessors) - rewind radio rows read `TurboSparkTheme.accentColor` in the view body (mixed with system `Color.secondary`) - `theme.accent` / `.appAccent` and `.appSecondary`.
- [MEDIUM] line 81 - Theming (section 4) - `Color.green` / `Color.secondary.opacity(0.5)` status dot; the status Text at line 105 carries the meaning so it is not color-alone, but the hues ignore the theme - theme-resolved colors.
- [MEDIUM] lines 94-98 - Action hierarchy (section 2) - `Button(role: .destructive)` Stop on a background agent has no confirmationDialog, and line 98 hand-tints it `Color.red.opacity(0.8)`; either stopping is not destructive (quiet action, no red) or the dialog is missing - pick one; `.appAction(.destructive)` already conveys the color.
- [MEDIUM] lines 127-131 - Action hierarchy (section 2) - same defect for the Kill shell button (role `.destructive`, no dialog, `Color.red.opacity(0.8)` tint at line 131).
- [MEDIUM] lines 107, 124, 240, 352 - Theming (section 4) - `.foregroundStyle(.tertiary)` theme-blind spellings - `theme.secondaryText` / `.appSecondary`.
- [MEDIUM] line 287 - Vocabulary (section 2) - `ProgressView` for the recap loading state instead of `AppProgressBar(animated:)` with the label.
- [MEDIUM] lines 369-374 - Accessibility (section 5) - the shared `sectionHeader` helper never adds `.accessibilityAddTraits(.isHeader)`, so no sheet section heading navigates as a header - add the trait in the helper (covers `statusGroup` too).
- [LOW] line 366 - Layout (section 3) - fixed `.frame(width: 520, height: 460)` on the shared sheet; content scrolls, but the fixed height squeezes at Extra Large text (possible) - scale by `AppMetrics.textScale(theme)` or use min/max.
- [LOW] lines 257-259 (possible) - Action hierarchy (section 2) - Rewind is `role: .destructive` without a `confirmationDialog`; the sheet is itself an explicit confirm surface, but DESIGN's rule wants the dialog for destructive roles.
- [LOW] lines 376-382 - Vocabulary (section 2) - `emptyText` helper instead of `AppEmptyState` (text-only empties; possible).
- Note: line 307's `isCopied ? Text(...) : Text(...)` is the correct ternary form; `approvalLabelText` is a bundled `Text` so lines 181/347 compose correctly; the 520-wide `statusRow` value column uses middle truncation and text selection.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/WorktreeDiffView.swift
- [HIGH] line 199 - Localization (missing key) - `Text("\(count) unmodified lines", bundle: .module)`; the interpolated key "%lld unmodified lines" is absent from the catalog (verified: zero matches), so it renders English under all languages - add the key with all 21 translations.
- [HIGH] line 220 - Theming (section 4, TurboSparkTheme static accessors) - `TurboSparkTheme.accentColor` (and `.opacity(0.06)`) read inside `lineRow` for hunk-header coloring - `theme.accent`.
- [MEDIUM] lines 213-224 - Diff colors (checklist; section 4) - system `.green`/`.red` and `Color.green.opacity(0.09)`/`Color.red.opacity(0.09)`/`.secondary`; the +/- text carries the meaning so it is not color-alone, but the hues are theme-blind - theme-resolved diff tokens.
- [MEDIUM] line 95 - Diff colors (possible) - `.foregroundStyle(file.status.color)`; `WorktreeFileChange.Status.color` (State/WorktreeModel.swift:28-36) returns system `.orange/.green/.red/.blue/.secondary` - theme-resolved status colors (status glyph present, so not color-alone).
- [MEDIUM] lines 116, 121 - Theming (section 4) - `.foregroundStyle(.green)`/`(.red)` on the verbatim +/- counts (the verbatim usage itself is correct) - theme-resolved tokens.
- [MEDIUM] line 140 - Theming (section 4) - `copied ? .green : .secondary` system-color ternary on the copy button - theme-resolved.
- [MEDIUM] lines 83, 157 - Surfaces (section 4) - `.appSurface.opacity(0.3)` and `.appPage.opacity(0.8)` hard-coded dimming - use the surface styles as given.
- [MEDIUM] lines 87, 126, 206 - Theming (section 4) - `Color.primary.opacity(0.08)` stroke and `Color.primary.opacity(0.04)`/`(0.03)` fills - `.appBorder` / theme fill.
- [MEDIUM] lines 102-106 - Vocabulary (section 2, filesystem paths) - `directoryPath` is middle-truncated but not selectable and has no `.help` with the full path; the vocabulary row is `SettingsPathText(path, size:)` (code font, selectable, middle truncation, help) - use `SettingsPathText`.
- [MEDIUM] lines 129-143, 145-153 - Vocabulary/Accessibility (section 2, 5) - icon-only copy and close buttons are bare `Button`s, not `AppIconButton`, and carry only `.help` (no VoiceOver name) - `AppIconButton(symbol, label: Text(...))`.
- [MEDIUM] line 232 - Layout (section 3, possible) - fixed `.frame(width: 54, alignment: .trailing)` on the line-number gutter set in `theme.code(.small)`, which grows with Text Size; 5-digit line numbers can clip - scale by `AppMetrics.textScale(theme)` or use `minWidth`.
- [LOW] line 84 - Radii (section 4) - `cornerRadius: 8` literal not from AppMetrics - `AppMetrics.controlRadius`/`cardRadius`.
- Note: code font here correctly uses `theme.code(...)` (themed code role) per the diff-view rule; line 199's Int interpolation specifier is right apart from the missing key.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerPlusMenu.swift
- [MEDIUM] line 49 - Theming (section 4) - `Color.primary.opacity(isHovered ? 0.08 : 0.04)` capsule background; a ternary of two system-color opacities on a themed surface - `.appSurface`/`.appElevated` or a theme-resolved hover token.
- [LOW] line 57 - Action hierarchy (section 2) - `.disabled(isRunning)` with a `.help` that names the shortcut but never says why it is disabled - extend the help or adjacent text.
- Note: otherwise clean. Group titles come from `SlashCommandMenuGroup.title`, which is a bundled `Text` (verified in SlashCommandMenuGroup.swift:20-30); menu rows all use `Text(..., bundle: .module)`; the model-prompt insertion strings at lines 383-385 are correctly NOT localized (model prompts); Toggles at 298-312 give VoiceOver the on/off state; `Image(systemName: "")` at lines 258/269 is a runtime oddity, not a DESIGN rule breach.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectMcpSettingsSheet.swift
- [MEDIUM] lines 181, 239 - Theming (section 4) - `Color.secondary.opacity(0.04)` / `(0.06)` empty-state and rule-strip backgrounds; system `Color.secondary` ignores custom themes - `.appSurface` or a theme fill.
- [MEDIUM] line 257 - Theming (section 4) - `isAllow ? Color.green : Color.red` system hues (glyphs present so status is not color-alone, but they ignore the theme) - theme-resolved colors.
- [MEDIUM] lines 262-266 - Action hierarchy (section 2) - `AppIconButton("trash", ..., role: .destructive)` removes a permission rule with no `confirmationDialog`; DESIGN says removing an entry the user can rebuild is a quiet action with a neutral glyph (minus.circle), and "A destructive role with no confirmation dialog is always a defect" - quiet role + minus.circle, or add the dialog.
- [MEDIUM] line 270 - Surfaces (section 4) - `.appSurface.opacity(0.8)` hard-coded dimming bypasses `surfaceOpacity` - `.appSurface` unmodified.
- [MEDIUM] lines 136, 157, 225 - Accessibility/Vocabulary (sections 2, 5) - section headings ("Inherited user servers", "Configured Project Servers (...)", "Permission Rules") are bare Texts without `.accessibilityAddTraits(.isHeader)` and not `AppSectionHeader` - use `AppSectionHeader` or add the trait.
- [MEDIUM] lines 173-182 - Vocabulary (section 2) - hand-rolled "No project MCP servers configured." empty block instead of `AppEmptyState`.
- [LOW] lines 182, 240, 271 - Radii (section 4) - `RoundedRectangle(cornerRadius: 8/6)` literals - `AppMetrics.controlRadius`/`cardRadius`.
- [LOW] line 126 - Keyboard (section 5) - the sheet's header Close button lacks `.keyboardShortcut(.cancelAction)` while Done at 289 has `.defaultAction` - add `.cancelAction`.
- Note: Done/Cancel keyboard binding, `AppStrings.localized` toast strings with Int formatting at 340/344, the `%@`/`%lld` interpolations at 118/157 (keys verified present), and the destructive project deletion living behind the caller's alert are all correct.

---

### Batch batch19_Generation

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerView.swift
- [MEDIUM] line 31 - DESIGN 4 Color - import error line uses `.foregroundStyle(.red)`, a theme-blind system color on the themed composer surface - use the AppAttention vocabulary (`AppAttention.color`) or a theme-resolved attention color, and keep the error as text (it already is)
- [MEDIUM] line 47 - DESIGN 4 Color - `GhostGlyph(size: 11, color: .secondary)` passes `Color.secondary`, which ignores a custom theme's text colors (ChatSidebarChatRowView line 187 shows the resolved form: `GhostGlyph(size: 13, color: theme.accent)`) - pass a theme-resolved color, e.g. `theme.secondaryText`
- [LOW] line 47 (possible) - DESIGN 4 Type - ghost glyph drawn at a fixed 11pt point size; the vendored `GhostGlyph` cannot inherit the ambient font so it does not grow with Text Size - consider scaling by `AppMetrics.textScale(theme)` at call sites
- [MEDIUM] line 315 - DESIGN 4 Color - `.foregroundStyle(.tertiary)` on the "Queued behind the running turn..." caption; `.tertiary` ignores a custom theme - use `.appSecondary`
- [LOW] lines 303-304 - DESIGN 4 Surfaces/radii - queued-pill background is a hand-rolled `RoundedRectangle(cornerRadius: 8)` (not `AppMetrics.controlRadius` 7) filled with `theme.composerBackground.opacity(0.6)`, a hard-coded dim that bypasses `surfaceOpacity`/`isHighContrast` - use an AppMetrics radius and a theme-provided fill

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/InteractiveMarkdownTableView.swift
- [HIGH] line 160 - DESIGN 4 Color - `.foregroundStyle(isCopied ? Color.accentColor : Color.secondary)`: both `Color.accentColor` and `Color.secondary` ignore a custom theme's accent/text colors, and the ternary mixes bare styles - `.foregroundStyle(isCopied ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))`
- [HIGH] line 189 - DESIGN 4 Color - header sort glyph reads `TurboSparkTheme.accentColor` (a static accessor, `TurboSparkTheme.swift` line 100) inside a view body, and ternary partner is `Color.secondary.opacity(0.5)`, theme-blind plus hard-coded dimming - use `theme.accent` from `@Environment(\.appTheme)` and `.appSecondary`
- [HIGH] line 213 - DESIGN 4 Color - selected row background `TurboSparkTheme.accentColor.opacity(0.12)`: static accessor read in a view body plus hard-coded 0.12 tint - use `theme.accent.opacity(...)` or `theme.selection`
- [MEDIUM] line 114 - DESIGN 4 Surfaces - `.background(.appElevated.opacity(0.5))` hard-codes a lower surface opacity than the theme gives, bypassing Reduce Transparency/Increase Contrast - use `.appElevated` directly
- [MEDIUM] lines 117-118 - DESIGN 4 Color - hard-coded `.appBorder.opacity(0.4)` stroke border (a spelling `DesignTokenRatchetTests` counts; baselines may only go down) - use `.appBorder` at the theme's weight
- [MEDIUM] lines 128, 133 - DESIGN 4 Color - `.foregroundStyle(.tertiary)` on the "Table" tag and the rows/columns count - use `.appSecondary`
- [MEDIUM] lines 107-108, 202-213 - DESIGN 5 Accessibility - row selection is `.onTapGesture` only: rows are not keyboard-reachable, carry no `.accessibilityElement(children: .combine)`, no `.isSelected` trait when selected, and selection changes are never announced - make rows selectable via keyboard and add `.accessibilityAddTraits(.isSelected)` on selected rows
- [LOW] line 110 - DESIGN 4 Surfaces - `Divider().opacity(0.5)` hard-coded dimming of a separator
- [LOW] lines 115, 117 - DESIGN 4 radii - cornerRadius 8 literals instead of `AppMetrics` radii (controlRadius 7 / cardRadius 10)
- [LOW] lines 139-141 (possible) - DESIGN 2 vocabulary - quiet "Clear" action styled `.buttonStyle(.plain)` + `.themedFont(.tiny)` instead of `.appAction(.quiet, size: .small)`
- [LOW] lines 162-163 (possible) - DESIGN 2 vocabulary - copy/export overflow menu styled with bespoke `.menuStyle(.borderlessButton)` + `.fixedSize()` instead of `.appMenuAction()`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectPermissionsSectionView.swift
- [HIGH] line 122 - DESIGN 4 Color - `.background(Color.primary.opacity(0.03), in: ...)` under the execution-mode description; `Color.primary.opacity` is an explicitly banned theme-blind spelling - use `.appElevated`/`.appSurface` or a resolved theme fill
- [HIGH] line 158 - DESIGN 3 Headers and rows - fixed `.frame(width: 175)` on the Forge Guardrails Picker in a form row (the exact defect class that clipped the Agents scope picker) - drop the fixed width or use minWidth/idealWidth
- [HIGH] line 264 - DESIGN 3 Headers and rows - fixed `.frame(width: 175)` on the per-category permission Picker in `permissionRow` - same fix
- [HIGH] lines 201-206 - DESIGN 2 Action hierarchy - `Button(role: .destructive)` "Remove" for revoking an allowlist origin with no confirmationDialog: per DESIGN "a destructive role with no confirmation dialog is always a defect", and removing a rebuildable list entry should instead be a quiet action with a neutral glyph (minus.circle) - drop the role or add the dialog
- [MEDIUM] lines 73-90 - DESIGN 3 Headers - section header "Tool Permissions & Risk Policy" + Presets menu built from bare `HStack { ... Spacer() }` instead of `AppAdaptiveHeader`; truncates under German and Extra Large text
- [MEDIUM] lines 126-135, 140-161, 165-238 - DESIGN 2 vocabulary - three hand-rolled cards (`padding(12)` + `.background(.appSurface, in: RoundedRectangle(cornerRadius: 8))`) instead of `.appCardSurface(padding:)`, with radius 8 instead of `AppMetrics.cardRadius` 10; the treatment is repeated three times in one file, which is how drift starts
- [MEDIUM] line 233 - DESIGN 4 Color - permission feedback text uses `.foregroundStyle(.red)`, theme-blind, and the status is color alone (no glyph) - use AppAttention treatment with a glyph
- [LOW] lines 94, 167, 189 - DESIGN 5 Accessibility - sub-headings "Execution Mode", "Browser access", "Allowed site origins" lack `.accessibilityAddTraits(.isHeader)` (the top title at line 74-76 has it)
- [LOW] line 86 (possible) - DESIGN 2 vocabulary - Presets menu uses bespoke `.menuStyle(.borderlessButton)` instead of `.appMenuAction()`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/WorktreeFileListView.swift
- [HIGH] lines 72-79 - DESIGN 6 Localization - ternaries INSIDE single `Text(..., bundle: .module)` calls for both the title and the message of the empty state; the four runtime keys ("No matching changed files", "No changes", "Try clearing the search filter.", "Files appear here when the selected comparison finds modifications.") are all ABSENT from `Localizable.xcstrings`, so this renders English under every language - split into `cond ? Text("A", bundle: .module) : Text("B", bundle: .module)` and add all four keys in 21 languages
- [HIGH] lines 101, 233 - DESIGN 4 Color - file name uses `.foregroundStyle(isSelected ? TurboSparkTheme.accentColor : .primary)`: static theme accessor in a view body plus theme-blind `.primary` - use `theme.accent` and `.appText`
- [HIGH] lines 122, 247 - DESIGN 4 Color - selected-row background `TurboSparkTheme.accentColor.opacity(0.12)`: static accessor in a view body plus hard-coded tint - use `theme.accent`/`theme.selection`
- [HIGH] line 182 - DESIGN 5 Motion - `withAnimation(.easeInOut(duration: 0.12))` on directory expand/collapse: bare withAnimation with an inline duration ignores Reduce Motion - use `TSMotion.animate(TSMotion.select) { ... }`
- [HIGH] lines 194, 229 - DESIGN 3 Text in layouts - fixed `.frame(width: 12)` around the chevron/status glyphs that inherit `.themedFont` and grow with Text Size - use minWidth or let the glyph size itself
- [MEDIUM] lines 67-83 - DESIGN 2 vocabulary - hand-rolled empty state (icon + title + message VStack) instead of `AppEmptyState(systemImage:title:message:)`; the two-variant content fits the shared component's parameters
- [MEDIUM] lines 70, 81, 107 - DESIGN 4 Color - `.foregroundStyle(.tertiary)` on empty-state texts and the directory suffix - use `.appSecondary`
- [MEDIUM] lines 132, 137, 257, 262 - DESIGN 4 Color - stat pills use `.foregroundStyle(.green)`/`.red` system colors on themed rows (the +/- text carries the status, so not color-alone, but the colors ignore the theme) - use theme-resolved add/delete colors
- [MEDIUM] lines 89-125, 220-250 - DESIGN 5 Accessibility - selected file rows expose no `.isSelected` trait, and the status glyph (line 95-97, 226-228) is not `.accessibilityHidden(true)` so VoiceOver reads the SF Symbol name
- [LOW] line 198 - DESIGN 4 Surfaces - `.appAccent.opacity(0.85)` hard-coded dim on the accent folder glyph
- [LOW] lines 123, 217, 248 - DESIGN 4 radii - cornerRadius 6 literals instead of `AppMetrics.controlRadius`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatGitSheets.swift
- [HIGH] line 156 - DESIGN 6 Localization - `AppStrings.localized("Copied \(commit.shortHash).")`: the key "Copied %@." is ABSENT from `Localizable.xcstrings`, so the toast renders English in all 20 other languages - add the key in 21 languages
- [HIGH] line 164 - DESIGN 6 Localization - `.accessibilityLabel(Text("Copy SHA for \(commit.summary)", bundle: .module))`: key "Copy SHA for %@" ABSENT from the catalog - add it
- [HIGH] line 210 - DESIGN 6 Localization - `Text("Open pull request #\(pr.number)", bundle: .module)`: key "Open pull request #%lld" ABSENT from the catalog - add it
- [HIGH] line 240 - DESIGN 6 Localization - `Text("Pull request state \(stateText)", bundle: .module)` interpolates a `Text` into a localized key and the resulting "Pull request state %@" key is ABSENT from the catalog - keep the pieces apart: `.help(stateText)` and `.accessibilityLabel(Text("Pull request state", bundle: .module))` + `.accessibilityValue(stateText)`, or add the %@ key
- [HIGH] line 69 - DESIGN 3 Window structure - `.frame(width: 680, height: 520)` pins the sheet: a fixed frame that does not adapt to Extra Large text (rule: no window has a fixed frame) - use minWidth/idealWidth or an ideal-size-driven presentation
- [HIGH] line 115 - DESIGN 4 Color - `.background(Color.primary.opacity(0.03))` behind the diff block - use a resolved theme fill (`.appElevated`)
- [HIGH] line 125 - DESIGN 4 Color - `lineColor` returns `TurboSparkTheme.accentColor` (static accessor) for `@@` hunk lines in view code - use `theme.accent` (the view already holds `@Environment(\.appTheme)`)
- [MEDIUM] lines 82, 132, 175 - DESIGN 2 vocabulary - `ProgressView()` for loading states in all three tabs instead of `AppProgressBar` (indeterminate)
- [MEDIUM] lines 119-127 - DESIGN 4 Color - `lineColor` returns system `.secondary`, `.green`, `.red`, `.primary` for diff lines - route through theme-resolved equivalents
- [MEDIUM] lines 221-227 - DESIGN 4 Color - PR state colors are `.purple`/`.red`/`.green` system colors (state is not color-alone: distinct glyph + localized text, but colors ignore the theme)
- [MEDIUM] lines 243-257 - DESIGN 2 vocabulary / 4 Color - hand-rolled error banner with `.foregroundStyle(.orange)` and `Color.orange.opacity(0.08)` instead of `.appAttention` with an `AppAttentionNote` and `AppAttention.color`
- [MEDIUM] line 42 - DESIGN 4 Color - `.foregroundStyle(.tertiary)` applied over the shared `AppIconButton` close control, overriding its treatment with a theme-blind style
- [LOW] lines 116, 174 - DESIGN 4 radii - cornerRadius 8 literals instead of AppMetrics radii
- [LOW] lines 38-41 (possible) - DESIGN 5 Keyboard - the sheet's close `AppIconButton` carries no `.keyboardShortcut(.cancelAction)`; the rule binds a sheet's Cancel to it (Esc dismisses natively, but the visible control is unbound)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarChatRowView.swift
- [MEDIUM] lines 68-120 - DESIGN 5 Accessibility - the context menu's Archive (line 85-88), Chat System Prompt (77-80) and Open/Close Split View (93-96) actions are not mirrored as `.accessibilityAction` the way Pin, Rename, Duplicate and Delete are; VoiceOver users lose them at the row
- [LOW] lines 40, 47, 62 - DESIGN 4 radii - cornerRadius 9 literals (row fill, selection stroke, contentShape) instead of an `AppMetrics` radius
- [LOW] lines 205-250 (possible) - DESIGN 2 vocabulary - the hover "..." overflow is a bespoke `Menu` with `.menuStyle(.borderlessButton)`, `.menuIndicator(.hidden)`, `.fixedSize()` and a manual hover fade/scale instead of the AppIconButton/.appMenuAction treatment; mitigated by `.help` + `.accessibilityLabel` and the `.accessibilityAction` mirrors
- [LOW] line 172 - DESIGN 2 Action hierarchy - `selectButton` is disabled while a turn runs (other rows) with no why in its `.help` ("Switch to <title>" does not explain the disabled state)
- otherwise clean: this row is the accessibility reference (resolved theme colors, `.isSelected`, accessibilityValue, accessibilityActions, TSMotion), the destructive Delete correctly defers to the localized alert in ChatSidebarView.swift lines 86-96

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerEditor.swift
- [MEDIUM] line 90 - DESIGN 4 Color - "Ask anything" placeholder uses `.foregroundStyle(.tertiary)`, theme-blind - use `.appSecondary`
- [MEDIUM] line 191 - DESIGN 4 Color - expand/collapse glyph uses `.foregroundStyle(.tertiary)` - use `.appSecondary`
- [LOW] lines 183-203 (possible) - DESIGN 2 vocabulary - icon-only expand/collapse is a hand-rolled plain Button instead of `AppIconButton`; it does carry `.help` + `.accessibilityLabel` so the name/tooltip can only drift manually
- otherwise clean: this file is the measured-height TextEditor reference (hidden sizer, `@ScaledMetric` bounds, TSMotion, IME-safe Return handling)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/MessageActionBarView.swift
- [MEDIUM] line 226 - DESIGN 5 Motion - `TSMotion.animate(.easeInOut(duration: 0.15))`: inline duration instead of a named TSMotion curve (`TSMotion.press`/`.select`); Reduce Motion is honored by the wrapper, the duration is not from the set
- [MEDIUM] line 236 - DESIGN 5 Motion - same: `TSMotion.animate(.easeOut(duration: 0.15))` inline duration for the copy-reset
- [LOW] line 145 - DESIGN 4 Surfaces - `.opacity(isEnabled ? 1 : 0.4)` hard-codes the disabled dimming; the disabled state also has no why in the tooltip (the guard lives model-side per the comment on line 29-31)
- otherwise clean: correct `AnyShapeStyle` ternary (line 139), `AppIconButton` with catalog-keyed names/hints, textScale-scaled 24pt frame, TSMotion for hover, ShareLink with bundled Label

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSearchOverlayView.swift
- [HIGH] line 74 - DESIGN 4 Color - `.background(TurboSparkTheme.surfaceColor, in: .rect(cornerRadius: 12))`: `TurboSparkTheme.surfaceColor` is a static accessor (TurboSparkTheme.swift line 115) read inside a view body, so the card draws the wrong mode under `.preferredColorScheme` (Gotcha 36) - use `.appSurface` or the resolved environment theme
- [MEDIUM] lines 101, 156, 166 - DESIGN 4 Color - `.foregroundStyle(.tertiary)` on the clear button, the navigation hint, and the count - use `.appSecondary`
- [MEDIUM] line 28 - DESIGN 4 Color (possible) - scrim `Color.black.opacity(0.32)` is hard-coded; consider a theme-resolved scrim so Increase Contrast/Reduce Transparency are honored
- [LOW] line 73 - DESIGN 3 (possible) - fixed `.frame(width: 640, height: 480)` on the palette card; content scrolls so it survives, but the footer and field row squeeze at Extra Large
- [LOW] lines 74, 76 - DESIGN 4 radii - cornerRadius 12 literals instead of an AppMetrics radius
- [LOW] lines 95-105 (possible) - DESIGN 2 vocabulary - icon-only clear button hand-rolled instead of `AppIconButton` (has `.help` + `.accessibilityLabel`, so drift is manual only)
- otherwise clean: AppEmptyState for both empties, hidden shortcut buttons as the AlternateShortcutBridge pattern, `.accessibilityElement(children: .contain)` on the dialog, themed 0.5pt dividers

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatActivityLine.swift
- [HIGH] line 230 - DESIGN 4 Color - `ReasoningTraceView` background `Color.primary.opacity(0.03)` - explicitly banned theme-blind spelling; use `.appElevated` or a resolved theme fill
- [MEDIUM] lines 133, 140 - DESIGN 4 Color - attention tone uses `Color.orange` directly; `AppAttention.color` is the vocabulary for the attention tint (same value, but the vocabulary is the maintained source)
- [LOW] line 231 - DESIGN 4 radii - cornerRadius 8 literal instead of an AppMetrics radius
- otherwise clean: `.tsPulse(active:)` for the live mark, `AppProgressBar` with a textScale-scaled max, `TSMotion.animate` on the disclosure, verbatim only for machine readouts (documented), accessibilityValue/hint on the disclosure, `.accessibilityElement(children: .combine)` on the plain row

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/WorktreeTimelineView.swift
- [HIGH] lines 58, 86, 118, 194, 228 - DESIGN 4 Color - five reads of the static `TurboSparkTheme.accentColor` inside view code (timeline node, commit summary, selection fills x2, file name), several with hard-coded `.opacity(0.1/0.12)` tints - use `theme.accent` from the environment (and `theme.selection` for fills)
- [HIGH] lines 64, 96, 118, 177 - DESIGN 4 Color - `Color.primary.opacity(0.1)`, `(0.04)`, `(0.02)`, `(0.08)` for the rail, hash chip, row rest state and detail-card stroke: banned theme-blind spellings - use `.appBorder`/`.appElevated`/resolved fills
- [HIGH] line 74 - DESIGN 5 Motion - `withAnimation(.easeInOut(duration: 0.15))` on commit select/deselect: bare withAnimation + inline duration ignores Reduce Motion - use `TSMotion.animate(TSMotion.select)`
- [HIGH] line 146 - DESIGN 6 Localization - `Text(verbatim: "\(worktree.selectedCommitFiles.count) files")`: "files" is translatable copy hidden in verbatim; key "%lld files" is ABSENT from the catalog while a "Files: %lld" key exists - use `Text("Files: \(count)", bundle: .module)` or add the key in 21 languages
- [HIGH] line 190 - DESIGN 3 Text in layouts - fixed `.frame(width: 12)` around the status glyph that inherits `.themedFont` - use minWidth or intrinsic sizing
- [MEDIUM] lines 144, 154 - DESIGN 2 vocabulary - `ProgressView().controlSize(.mini)` twice instead of `AppProgressBar`
- [MEDIUM] lines 35-49 - DESIGN 2 vocabulary - hand-rolled empty state ("No Commit History Found") instead of `AppEmptyState(systemImage:title:message:)`
- [MEDIUM] lines 39, 46, 93, 105, 109, 148, 163, 200 - DESIGN 4 Color - `.foregroundStyle(.tertiary)` on eight secondary texts - use `.appSecondary`
- [MEDIUM] lines 212, 217 - DESIGN 4 Color - `.foregroundStyle(.green)`/`.red` stat colors ignore the theme
- [MEDIUM] lines 73-131 - DESIGN 5 Accessibility - the selected commit row carries no `.isSelected` trait and the selection state is invisible to VoiceOver (buttons are keyboard-reachable, the state is not announced)
- [LOW] line 173 - DESIGN 4 Surfaces - `.appSurface.opacity(0.6)` hard-coded dim on the detail card
- [LOW] lines 96, 119, 174, 229 - DESIGN 4 radii - cornerRadius 4/6 literals instead of AppMetrics radii

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatTranscriptView.swift
- [HIGH] lines 113, 126, 138 - DESIGN 5 Motion - three bare `withAnimation(.easeInOut(duration: 0.2))` calls (recovery-event scroll, question announcement scroll, turn navigation scroll): animate even under Reduce Motion and carry inline durations - use `TSMotion.animate(TSMotion.pane) { ... }`
- [HIGH] lines 122-125 - DESIGN 6 Localization - the VoiceOver announcement is built from bare English literals ("Assistant is asking a question. Please choose an answer." / "...asking \(count) questions..."); keys ABSENT from the catalog and never localized - build with `AppStrings.localized` and an Int format key (%lld)
- [HIGH] lines 142-148 - DESIGN 6 Localization - `accessibilityRotor("Messages")` takes a bundleless `LocalizedStringKey` that resolves against `Bundle.main` (empty in SwiftPM builds), so the existing "Messages" catalog key is never used; the entries' "User: "/"Assistant: " prefixes are fixed English with no catalog keys - use the Text overload (`Text("Messages", bundle: .module)`) and localize the prefixes
- [MEDIUM] line 195 - DESIGN 4 Color - compaction notice uses `.foregroundStyle(.secondary)`, theme-blind - use `.appSecondary`
- [MEDIUM] line 198 - DESIGN 4 Color - compaction notice background `.quaternary` (system hierarchy style) ignores custom themes - use `.appElevated`
- [MEDIUM] line 213 - DESIGN 3 Text in layouts (possible) - fixed `.frame(width: ConversationLayout.activityIconWidth)` (20pt constant, not scaled by `AppMetrics.textScale`) around the terminal glyph whose font comes from the container's `.themedFont(.small)`; ChatActivityLine line 168 shows the scaled pattern - scale the constant or use minWidth
- [LOW] line 198 - DESIGN 4 radii - cornerRadius 10 literal; value matches `AppMetrics.cardRadius` but bypasses the constant
- otherwise clean: the Messages rotor exists, `TurnActivityDisclosure` has `.accessibilityValue`, the compaction notice combines children, "Activity" and count use bundled Text + verbatim number

---

### Batch batch20_Generation

Audit against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md (sections cited as 2 vocabulary, 3 layout, 4 theming/type, 5 accessibility/motion, 6 localization). Verified: AppMetrics.controlRadius 7 / cardRadius 10 / composerRadius 20 (Theme/AppControls.swift:17-19, AppSurfaces.swift:13); AppAttention.color = Color.orange (Theme/AppAttention.swift:14); Localizable.xcstrings key presence checked for every localization flag below.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/InteractiveQuestionCardView.swift
- [HIGH] line 185 - 4 Theming - `TurboSparkTheme.accentColor` static accessor read inside the view body (draws the wrong mode under `.preferredColorScheme`), ternary also mixes with `Color.secondary` without `AnyShapeStyle` - resolve from the environment theme (`theme.accent`, `.appSecondary`), wrap the two sides in `AnyShapeStyle`
- [HIGH] line 200 - 4 Theming - selected/unselected row backgrounds use static `TurboSparkTheme.accentColor.opacity(0.10)` and `Color.primary.opacity(0.03)`: static theme read plus theme-blind `Color.primary.opacity` in a ternary - use theme-resolved styles (`.appAccent` / a theme surface fill) wrapped in `AnyShapeStyle`
- [MEDIUM] lines 66-76 - 2 Vocabulary - the card's primary action is a bespoke `.buttonStyle(.borderedProminent)` button; system accent tint is theme-blind and duplicates `.appAction(.primary, size: .small)` - use `.appAction(.primary, size: .small)`
- [MEDIUM] lines 53-57, 201 - 4 Surfaces - hand-rolled accent card: hard-coded `.appAccent.opacity(0.06)` fill and `.appAccent.opacity(0.35)` stroke, corner radii 8/6 not from AppMetrics and missing `style: .continuous` - use `.appCardSurface` treatment or AppMetrics radii with `.continuous`
- [LOW] lines 84, 130-136 - 2 Vocabulary - Skip and per-question submit are hand-rolled quiet actions (`.buttonStyle(.plain)` + inline `.foregroundStyle`) instead of `.appAction(.quiet, size: .small)` - move to the shared action style
- [LOW] line 59 - 2 Vocabulary (informational) - accessibility label text is fine, but the card is a hand-rolled composite rather than a vocabulary surface; acceptable as-is

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ToolResultOutputView.swift
- [HIGH] line 184 - 6 Localization - `Text(isError ? "ERROR" : "OUTPUT")`: a String ternary selects the verbatim `Text(String)` path, so these fixed English labels bypass the catalog (neither key exists in Localizable.xcstrings) - write `isError ? Text("ERROR", bundle: .module) : Text("OUTPUT", bundle: .module)`
- [HIGH] lines 38, 175-180 - 6 Localization - `hiddenNoticeText` builds "Show all (N earlier lines hidden)" in English as a `String` rendered via `Text(...)`; no such key exists in the catalog, so it renders English under all 21 languages - build the label in the view as `Text("Show all (\(count) earlier lines hidden)", bundle: .module)` (Int interpolates as %lld) or two `AppStrings.localized` keys
- [HIGH] line 202 - 6 Localization - `Text(isCopied ? "Copied" : "Copy Output")`: String ternary takes the verbatim path and "Copy Output" is absent from the catalog - ternary of two `Text(..., bundle: .module)` values
- [MEDIUM] lines 54, 60-64, 66-74, 77-82 - 4 Theming - error output surface is painted entirely with system colors: `Color.red` foreground, `Color.red.opacity(0.05/0.3/0.4)` background/stroke/bar, `Color.primary` base text; line 54 ternary mixes styles without `AnyShapeStyle`; `Color.primary.opacity(0.15)` for the normal bar ignores the theme border - resolve through the theme (`.appText`, `.appBorder`, theme-resolved danger/attention color) and wrap ternary sides in `AnyShapeStyle`
- [MEDIUM] lines 43, 146, 171, 186, 191, 205 - 4 Theming - theme-blind spellings: `Color.primary.opacity(0.06)` chip background, `Color.primary).opacity(0.5)` faint text, `.primary` ANSI default, `Color.secondary` header, `.tertiary` duration, `Color.green : Color.secondary` copy feedback - use `.appSecondary`/`.appText`/theme-resolved status colors
- [MEDIUM] lines 31, 218, 224 - 5 Motion - bare `withAnimation(.easeInOut(duration: 0.15))` with inline durations: ignores Reduce Motion and the no-inline-durations rule - use `TSMotion.animate(TSMotion.snap/select)` or `.tsAnimation(_:value:)`
- [LOW] lines 44, 65, 67 - 4 Surfaces - corner radii 4/6 hard-coded instead of AppMetrics radii; line 44 lacks `style: .continuous` - use AppMetrics.controlRadius and `.continuous`
- [LOW] lines 196-208 (possible) - 2 Vocabulary - bespoke icon+text copy button with `.buttonStyle(.plain)` duplicates the quiet-action treatment - candidate for `.appAction(.quiet, size: .small)` with the glyph inside the label

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatDialogSheets.swift
- [MEDIUM] lines 100, 204 - 4 Theming - stats/help group cards use `Color.primary.opacity(0.03)` backgrounds, a theme-blind spelling that ignores custom themes - use `.appElevated` or a theme-resolved fill
- [MEDIUM] lines 25, 146 - 4 Theming - close buttons tinted `.foregroundStyle(.tertiary)`, a theme-blind style - use `.appSecondary`
- [MEDIUM] lines 92-94, 197-198 - 5 Accessibility - group headings (`statsGroup`, `helpSection` titles) are plain `Text` without `.accessibilityAddTraits(.isHeader)`, so they do not navigate as headings - add the trait
- [MEDIUM] lines 85, 190 (possible) - 3 Layout - sheets pinned to fixed `.frame(width: 460, height: 480)` / `(width: 520, height: 560)` clip under long translations and Extra Large text ("layouts adapt instead of clipping") - prefer ideal/min-max sizing or a scrollable, resizable sheet
- [LOW] lines 101, 205 - 4 Surfaces - `RoundedRectangle(cornerRadius: 10)` literals instead of `AppMetrics.cardRadius` - reference the metric
- [LOW] lines 21-24, 142-145 (possible) - 2 Vocabulary - `AppIconButton` followed by `.buttonStyle(.plain)` may override its designed treatment - drop the extra style unless intentional

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ContextUsageRingView.swift
- [HIGH] line 97 - 4 Theming - popover background `TurboSparkTheme.surfaceColor` is a bare static accessor read in a view body (wrong mode under `.preferredColorScheme`) - use `.appSurface` or the environment theme's resolved surface color
- [HIGH] line 41 - 4 Theming - hover halo `Color.primary.opacity(isHovered ? 0.08 : 0)`: `Color.primary.opacity` is theme-blind and the ternary mixes styles without `AnyShapeStyle` - use a theme surface style wrapped in `AnyShapeStyle`
- [MEDIUM] lines 65, 127, 165 - 4 Theming - ring track, bar track, and free-space swatch use `Color.primary.opacity(0.15/0.08)` - use `.appBorder` / theme-resolved track color
- [MEDIUM] line 198 - 4 Theming - compacting status tinted `Color.orange`; the vocabulary token for the warning hue is `AppAttention.color` - use `AppAttention.color`
- [MEDIUM] lines 170, 204, 208, 214 - 4 Theming - four `.foregroundStyle(.tertiary)` uses are theme-blind - use `.appSecondary`
- [MEDIUM] line 72 - 5 Motion - `.tsAnimation(.linear(duration: 0.15), value: fraction)` inlines a duration instead of a named `TSMotion` curve - use e.g. `TSMotion.snap`/`select`
- [LOW] line 110 (possible) - 5 Accessibility - the popover's "Context window" title is a heading without `.accessibilityAddTraits(.isHeader)` - add the trait
- [LOW] line 96 (possible) - 3 Layout - fixed `.frame(width: 320)` popover can crowd at Extra Large text - consider a scaled width

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarProjectsSectionView.swift
- [MEDIUM] lines 21-42 (possible) - 3 Headers - the Projects section header is a bare `HStack` + `Spacer` with an action button, the exact shape DESIGN.md says truncates under German and Extra Large text; the short title mitigates it - use the AppAdaptiveHeader reflow pattern
- [LOW] lines 54-71, 184-188 (possible) - 2 Action hierarchy - Delete Project is confirmed by an `.alert` whose message is only the project name (no consequence sentence), while the codebase reference (OutputPaneView delete alert) states "This cannot be undone"; DESIGN specifies confirmationDialog with localized title, message, and button - keep the confirmation but give it a real message
- [LOW] lines 100, 207 - 4 Surfaces - `.rect(cornerRadius: 8)` hard-coded instead of `AppMetrics.controlRadius` (7) - reference the metric
- [LOW] lines 27-38, 143-155, 189-203 (possible) - 2 Vocabulary - icon-only Add/New-chat/ellipsis controls are hand-rolled `Button`/`Menu` instead of `AppIconButton`; each does carry `.help` + `.accessibilityLabel`, so the contract is met manually, but the label/tooltip can drift - consider AppIconButton or the existing sidebar row vocabulary

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectChatSummaryView.swift
- [MEDIUM] lines 105-106, 123, 145 - 2 Vocabulary - hand-rolled empty states (plain secondary `Text` via the `empty` helper) instead of `AppEmptyState(systemImage:title:message:)`; the keys exist in the catalog so text localizes, but the surface drifts from the vocabulary - use `AppEmptyState`
- [MEDIUM] lines 73-74, 170-179 - 5 Accessibility - panel title "Chat summary" and the collapsible section titles lack `.accessibilityAddTraits(.isHeader)` - add the trait to the title text inside the disclosure button
- [LOW] lines 82-84 - 4 Surfaces - `RoundedRectangle(cornerRadius: 18)` literal: not an AppMetrics radius (cardRadius 10) and missing `style: .continuous` - use `AppMetrics.cardRadius` with `.continuous`
- [LOW] line 108 (possible) - 6 Localization - `Text(TodoChecklistSummary.text(items))` renders the English summary string built in TodoChecklistViews.swift (see that file's HIGH finding) - fix at the source and compose bundled Texts here

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/TodoChecklistViews.swift
- [HIGH] lines 7-12 with 155 - 6 Localization - `TodoChecklistSummary.text` assembles "2/4 done (1 in progress)" in English and `header` renders it via `Text(String)`, which takes the verbatim path; the catalog has "%lld/%lld done" but no in-progress variant, so "(1 in progress)" is English under every language - return counts and compose `Text(..., bundle: .module)` segments, or add the missing key and format through it
- [HIGH] line 187 - 6 Localization - `Text(verbatim: "+\(display.foldedCompleted) completed")` embeds the English word "completed" in a verbatim Text; no key exists - `Text("+\(count) completed", bundle: .module)` (Int as %lld)
- [HIGH] line 136 - 4 Theming - panel background `Color(nsColor: .controlBackgroundColor).opacity(0.85)` is a system AppKit color on a themed surface, the exact Gotcha-62 defect class (neutral gray box on a custom theme) - use `.appSurface`/`.appElevated`
- [HIGH] line 149 - 4 Theming - `TurboSparkTheme.accentColor` static accessor read in the header (wrong mode under `.preferredColorScheme`) - use `theme.accent` / `.appAccent`
- [MEDIUM] line 58 - 4 Theming - `.foregroundStyle(item.isCompleted ? .secondary : .primary)` mixes theme-blind styles in a ternary without `AnyShapeStyle` - `.appSecondary` / `.appText` wrapped in `AnyShapeStyle`
- [MEDIUM] lines 75-76 - 4 Theming - "In Progress" pill uses `Color.orange` fill and text; the vocabulary warning hue is `AppAttention.color` - use `AppAttention.color`
- [MEDIUM] lines 86, 92, 96 - 4 Theming - status icons tinted `Color.green`, `Color.secondary`, `Color.secondary.opacity(0.7)` (hard-coded extra dimming) - theme-resolved equivalents
- [MEDIUM] lines 59, 141 - 4 Theming - `.strikethrough(color: .secondary)` and the `Color.primary.opacity(0.08)` card stroke are theme-blind - `.appBorder` and a theme-resolved strike color
- [MEDIUM] line 163 - 5 Motion - bare `withAnimation(.easeInOut(duration: 0.15))` with inline duration ignores Reduce Motion - `TSMotion.animate` with a named curve
- [MEDIUM] line 152 - 5 Accessibility - the panel heading "Tasks" lacks `.accessibilityAddTraits(.isHeader)` - add the trait
- [LOW] lines 137, 140 - 4 Surfaces - cornerRadius 10 literals instead of `AppMetrics.cardRadius` - reference the metric

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/SubagentLiveCardView.swift
- [HIGH] lines 99-102 - 4 Theming - card stroke reads static `TurboSparkTheme.accentColor.opacity(0.45)` and `TurboSparkTheme.hairlineColor` in the view body: static accessors are for non-view code (wrong mode under `.preferredColorScheme`), plus a hard-coded 0.45 accent border dimming - resolve from the environment theme and the theme's border token
- [MEDIUM] lines 76, 85 - 4 Theming - streamed text and final response use `.foregroundStyle(.primary.opacity(0.85))`: `Color.primary` ignores custom theme text colors and 0.85 is hard-coded dimming - use `.appText` / `.appSecondary`
- [MEDIUM] line 96 - 4 Surfaces - `.appSurface.opacity(0.7)` hard-codes a lower opacity than the theme's `surfaceOpacity` gives (Reduce Transparency escape) - use `.appSurface` unmodified
- [MEDIUM] lines 58, 151, 173-179 - 4 Theming - status colors are bare `Color.red`, `Color.green`, `Color.orange`, `Color.red.opacity(0.8)` system spellings - theme-resolved status colors (text always pairs with the glyph, so color-alone is satisfied)
- [MEDIUM] line 112 - 5 Motion - `TSMotion.animate(.easeInOut(duration: 0.12))` honors Reduce Motion but inlines a duration instead of a named curve - use `TSMotion.select`
- [MEDIUM] line 105 (possible) - 5 Accessibility - `.accessibilityElement(children: .combine)` on the whole card also swallows the expand, stop, and dismiss buttons inside it, flattening them out of the VoiceOver row - combine only the descriptive children or add `.accessibilityAction` for the controls
- [LOW] lines 146-154, 160-168 (possible) - 2 Vocabulary - icon-only stop/dismiss buttons hand-rolled instead of `AppIconButton` (help + accessibilityLabel present, so contract met manually)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/OutputPaneView.swift
- [MEDIUM] lines 25, 184 - 5 Motion - `TSMotion.animate(.easeOut(duration: 0.15))` and `.easeIn(duration: 0.15)` inline durations instead of named `TSMotion` curves - use `TSMotion.snap`/`select`
- [MEDIUM] line 81 (possible) - 2 Action hierarchy - "Clear chat history" context-menu action destroys the transcript with no destructive role and no confirmation dialog; DESIGN: "a destructive role with no confirmation dialog is always a defect" - add a localized confirmationDialog (or a destructive role plus one)
- Otherwise clean: localization is done right (bundled Texts, `AppStrings.localized` announcements with Int interpolation, localized delete alert with a real message), sheets are wired, disabled items are contextual.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptSafetyPills.swift
- [HIGH] lines 31, 129 - 4 Theming - pill icon tint `isEnabled ? TurboSparkTheme.accentColor : Color.secondary`: static theme accessor read in a view body plus theme-blind `Color.secondary` in a ternary without `AnyShapeStyle` - `theme.accent` / `.appSecondary` wrapped in `AnyShapeStyle`
- [HIGH] lines 35, 132 - 4 Theming - label tint `isEnabled ? Color.primary : Color.secondary`: both sides are theme-blind SwiftUI spellings that ignore custom theme text colors - `.appText` / `.appSecondary` wrapped in `AnyShapeStyle`
- [MEDIUM] lines 22-43, 122-145 - 2 Vocabulary - both pill controls hand-roll the same composer toggle-chip treatment (icon + label button + info button) instead of `AppChip`; two local copies of one treatment in one file is the drift pattern DESIGN.md names - extract a shared chip or adopt `AppChip`
- [MEDIUM] lines 64, 161 - 4 Theming - info buttons tinted `.foregroundStyle(.tertiary)`, theme-blind - `.appSecondary`
- [MEDIUM] line 140 - 4 Theming - pending-reload marker tinted `Color.orange`; vocabulary token is `AppAttention.color` - use it
- [LOW] lines 37, 134 (possible) - 3 Layout - `.fixedSize()` on pill labels prevents wrapping/truncation in crowded headers under long translations (German "Guardrails On" variants) - let the header reflow instead
- [LOW] lines 59-70, 156-167 (possible) - 2 Vocabulary - icon-only info buttons hand-rolled instead of `AppIconButton` (help + accessibilityLabel present)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/SplitChatPanes.swift
- [HIGH] line 62 - 4 Theming - status dot `isChatRunning ? TurboSparkTheme.accentColor : Color.primary.opacity(0.2)`: static theme accessor in a view body plus theme-blind `Color.primary.opacity` in a ternary without `AnyShapeStyle` - `theme.accent` / `.appBorder` wrapped in `AnyShapeStyle`
- [MEDIUM] lines 59-95 - 3 Headers - the pane header is a bare `HStack` + `Spacer` whose title is forced to `lineLimit(1)` + `.truncationMode(.tail)`; DESIGN: "a header never truncates. Use AppAdaptiveHeader" - adopt AppAdaptiveHeader (or its compact actions fold)
- [MEDIUM] lines 87, 122 - 4 Theming - `.foregroundStyle(.tertiary)` on the close icon and empty-state glyph, theme-blind - `.appSecondary`
- [MEDIUM] lines 118-127 - 2 Vocabulary - hand-rolled empty state (glyph + caption) instead of `AppEmptyState(systemImage:title:message:)` - use the vocabulary control
- [MEDIUM] lines 153-158 - 4 Surfaces - user bubble uses `.appSurface.opacity(0.85)` (hard-coded dimming below the theme surface), a `Color.primary.opacity(0.08)` stroke, and cornerRadius 12, which is not an AppMetrics radius - `.appSurface`, `.appBorder`, `AppMetrics.controlRadius`
- [LOW] lines 71-91 (possible) - 2 Vocabulary - promote/close icon-only buttons hand-rolled instead of `AppIconButton` (help + accessibilityLabel present)
- [LOW] line 79 (possible) - 2 Action hierarchy - the promote button is disabled while `model.isRunning` with no tooltip or adjacent text saying why ("A disabled control says why") - extend the `.help` with the reason
- [LOW] lines 61-64 (possible) - 5 Motion - the running indicator is a static dot; a live indicator should pulse via `.tsPulse(active:)` so its liveness is perceivable - wrap the dot

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/MessageEditingViews.swift
- [MEDIUM] lines 95-106 - 2 Vocabulary - the in-place edit editor is a hand-rolled composer: `TextEditor` with a fixed `.frame(minHeight: 56, maxHeight: 240)` instead of `.appEditorSurface(minHeight:)`, `.appSurface.opacity(0.85)` dimming, cornerRadius 16 (composerRadius is 20), and a hand-rolled `.appAccent.opacity(0.45)` focus edge - rebuild on `.appComposerSurface(focused:halo:)` / `.appEditorSurface(minHeight:)`
- [MEDIUM] lines 29-31, 48-50 - 3 Layout - fixed `.frame(width: 18, height: 18)` on chevron glyphs sized with `.themedFont(.tiny)`; the glyph grows with Text Size but the frame does not, clipping at Extra Large - size the frame from the font or drop the fixed frame
- [LOW] lines 117, 173 (possible) - 2 Action hierarchy - Save & rerun / Create branch buttons disabled on empty text with no tooltip explaining why (the chat Send button is the reference) - add a localized reason to `.help`
- [LOW] line 177 (possible) - 3 Layout - the branch sheet pins `.frame(width: 520)`; crowded at Extra Large text - prefer max-width sizing
- Otherwise a model citizen: `.appAction(.primary/.secondary)`, `.keyboardShortcut(.defaultAction/.cancelAction)`, `.appEditorSurface(minHeight: 120)` in BranchEditSheet, first field focused, localized keys throughout.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ToolPresentation.swift
- [LOW] line 177 - 4 Surfaces - fallback letter tile uses `RoundedRectangle(cornerRadius: 4)` without `.continuous` and a literal radius - add `style: .continuous` and reference an AppMetrics radius
- Otherwise clean: labels are catalog keys resolved via `AppStrings.localized(String.LocalizationValue(label))` with the in-app language, MCP-derived names correctly stay verbatim, icons are accessibility-hidden data, no theme or layout violations (fixed 16pt frames are on bundled raster artwork, not themedFont glyphs).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptContextPills.swift
- [HIGH] line 162 - 4 Theming - reasoning label tint `isThinkingActive ? Color.primary : Color.secondary`: theme-blind styles in a ternary without `AnyShapeStyle`, ignoring custom theme text colors - `.appText` / `.appSecondary` wrapped in `AnyShapeStyle`
- [MEDIUM] lines 113-114 - 4 Surfaces - project pill hard-codes `.appAccent.opacity(0.1)` fill and `.appAccent.opacity(0.25)` edge; hard-coded accent dimming bypasses the theme's surface/border treatment - derive from the theme's tokens
- [MEDIUM] lines 100, 105 - 4 Theming - git diff stats tinted bare `.green` / `.red`; the +/− sign carries the meaning (not color-alone), but the spellings ignore the theme palette - use theme-resolved diff colors
- [MEDIUM] lines 63, 166 - 4 Theming - chevrons tinted `.foregroundStyle(.tertiary)`, theme-blind - `.appSecondary`
- [LOW] line 115 (possible) - 3 Layout - `.fixedSize()` on the project pill prevents wrapping; a long project name or German tooltip neighbor can overflow the composer footer - allow truncation with the full name in `.help` (which exists)
- Menus are clean: bundled Section headers, catalog-key reasoning labels via `Text(LocalizedStringKey(...), bundle: .module)`, verbatim aliases and counts, accessibilityLabel/Value present.

---

### Batch batch21_Generation

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ToolApprovalDropdown.swift
- [HIGH] line 22 - 4 Theming (theme-blind styles) - `.foregroundStyle(Color.secondary)` on the mode glyph ignores the palette - use `.appSecondary`
- [HIGH] line 26 - 4 Theming - `.foregroundStyle(Color.primary.opacity(0.85))` hard-coded dimming on primary text - use `.appText`/`.appSecondary` (theme already provides the dimmed step)
- [HIGH] line 31 - 4 Theming - `.foregroundStyle(.tertiary)` on the chevron is theme-blind - use `.appSecondary`
- [HIGH] line 36-38 - 4 Theming (surfaces) - hover background `Color.primary.opacity(isHovered ? 0.08 : 0.04)` is a hand-drawn surface with hard-coded dimming and a ternary of two bare opacities - use `.appSurface`/`.appElevated` with the shared hover treatment
- [LOW] line 37 - 2 Radii - `cornerRadius: 8` is not `AppMetrics.controlRadius` (7)
- [HIGH] line 105 - 4 Theming (Gotcha 36) - `TurboSparkTheme.surfaceColor` static read inside the popover body draws the wrong mode under `.preferredColorScheme` - use `.appSurface` or the resolved `theme`
- [HIGH] line 121 - 4 Theming (Gotcha 36) - ternary `isSelected ? TurboSparkTheme.accentColor : Color.secondary` reads the static accent in a body and mixes theme-blind colors - use `.appAccent`/`.appSecondary` wrapped in `AnyShapeStyle`
- [MEDIUM] line 122 - 3 Layout (fixed frames) - `.frame(width: 20, height: 20)` on a glyph sized with `.themedFont(.callout, ...)`, which grows with Text Size - let it size naturally or scale with `AppMetrics.textScale`
- [HIGH] line 128 - 4 Theming - `.foregroundStyle(Color.primary)` on the mode title - use `.appText`
- [HIGH] line 142 - 4 Theming - `.foregroundStyle(Color.primary)` on the checkmark - use `.appText`
- [HIGH] line 149-150 - 4 Theming (surfaces) - `Color.primary.opacity(isHovered ? 0.08 : (isSelected ? 0.04 : 0))` hover/selection surface dimming - use the shared treatment
- [MEDIUM] line 117-155 - 5 Accessibility - selected option row never carries `.accessibilityAddTraits(.isSelected)` (contrast with ComposerSuggestionPopup which does) - add it
- [LOW] line 104 (possible) - 3 Layout - fixed `.frame(width: 320)` popover content does not scale with Text Size; the menu rows inside reflow, so consider `minWidth`/`idealWidth`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectMcpServerRowView.swift
- [HIGH] line 20 - 4 Theming (Gotcha 36) - ternary `server.isEnabled ? TurboSparkTheme.accentColor : Color.secondary` reads the static accent in a body - use `.appAccent`/`.appSecondary` in an `AnyShapeStyle` ternary
- [HIGH] line 31 - 4 Theming - source file chip background `Color.secondary.opacity(0.12)` - use `.appSurface`/`.appElevated`
- [MEDIUM] line 48-50 (possible) - 2 Vocabulary - `ProgressView().scaleEffect(0.5)` inline busy indicator instead of `AppProgressBar` - it is a spinner in a small button row; if the vocabulary lacks a tiny busy form, add one rather than a local `ProgressView`
- [HIGH] line 87-91 - 2 Action hierarchy - `Button(role: .destructive)` "Delete Server" with no `confirmationDialog` anywhere in the chain (`ProjectMcpSettingsSheet.swift` line 208 calls `model.deleteProjectMcpServer` directly, no dialog) - DESIGN.md: "A destructive role with no confirmation dialog is always a defect"; add a localized dialog or drop the role
- [MEDIUM] line 93-99 - 2 Vocabulary / 3 Layout - hand-rolled overflow `Menu` with `.menuStyle(.borderlessButton)`, hidden indicator and `.frame(width: 18)` instead of `.appMenuAction()` (or `AppIconButton` with a menu); the fixed width also fights Text Size
- [HIGH] line 107 - 4 Theming - `.foregroundStyle(toast.isError ? .red : .green)` theme-blind status colors; also ternary of two bare shape styles - use `AppAttention.color` / `.appAccent` with `AnyShapeStyle`
- [HIGH] line 109 - 4 Theming - `Color.red.opacity(0.1)` / `Color.green.opacity(0.1)` backgrounds ignore the palette - use the attention surface treatment
- [MEDIUM] line 104-111 - 4 Color (status never color alone) - test result is error vs success distinguished only by red/green tint; pair with a glyph (octagon / checkmark)
- [HIGH] line 149 - 4 Theming (surfaces) - `.appSurface.opacity(0.6)` hard-coded dimming of a themed surface (Reduce Transparency/Increase Contrast path bypassed)
- [LOW] lines 110, 150, 156 - 2 Radii - 6/6/8 not from `AppMetrics`
- [HIGH] line 157 - 4 Theming (borders) - `Color.secondary.opacity(0.15)` stroke is a hand-drawn border - use `.appBorder`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerControls.swift
- [HIGH] line 79 - 4 Theming (surfaces) - attach button hover background `Color.primary.opacity(isHovered ? 0.08 : 0.04)` - use the shared hover treatment
- [MEDIUM] line 87-92 - 2 Action hierarchy (disabled controls) - `.disabled(isRunning || isExtracting)` gives no "why" while `isRunning`; the tooltip still reads "Attach PDF, Word..." with no explanation (the `isExtracting` case is covered by the swapped accessibility label)
- [MEDIUM] lines 15, 77 - 3 Layout - fixed `.frame(width: iconButtonSize, height: iconButtonSize)` around glyphs sized with `.themedFont`/SF symbols; the constant comes from the composer and does not scale with Text Size
- [HIGH] line 146 - 4 Theming - `.foregroundStyle(Color.secondary)` on the mic glyph - use `.appSecondary`
- [MEDIUM] line 147 - 3 Layout - `.frame(width: 28, height: 28)` on a glyph sized with `.themedFont(.callout, ...)`
- [HIGH] line 149-151 - 4 Theming (surfaces) - `Color.primary.opacity(isHovered ? 0.08 : 0)` hover halo - shared treatment
- [LOW] line 13-21 - 2 Vocabulary - `PromptTipsButton` is an icon-only bare `Button` rather than `AppIconButton`; it does carry `.help` + `.accessibilityLabel` so the two cannot currently drift, minor drift risk only
- [LOW] line 47 (possible) - 3 Layout - fixed `.frame(width: 360)` popover content does not scale with Extra Large text

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/CollapsibleMessageContentView.swift
- [HIGH] line 88 - 5 Motion - `withAnimation(.easeInOut(duration: 0.2))` bare `withAnimation` with an inline duration; ignores Reduce Motion - use `TSMotion.animate(TSMotion.pane) { ... }`
- [HIGH] line 107 - 6 Localization - `.help(isExpanded ? "Collapse message" : "Expand full message")` passes bare `String` literals to the String overload, which resolves against `Bundle.main`; neither key exists in `Localizable.xcstrings`, so this renders English under every language - use `isExpanded ? Text("Collapse message", bundle: .module) : Text("Expand full message", bundle: .module)` and add both keys
- [HIGH] line 108 - 6 Localization - `.accessibilityLabel(isExpanded ? "Show more of message" : "Show less of message")` same defect; both keys absent from the catalog - use the `Text(_, bundle: .module)` ternary form

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectMcpDetectionSectionView.swift
- [MEDIUM] lines 18-41 - 3 Headers - section header built from bare `HStack` + `Spacer` with a trailing action; the title+subtitle truncates under German/Extra Large exactly as DESIGN.md describes - use `AppAdaptiveHeader`
- [MEDIUM] line 31 (possible) - 2 Vocabulary - `ProgressView().scaleEffect(0.6)` busy indicator inside the Scan button instead of an `AppProgressBar`/vocabulary spinner
- [HIGH] line 54 - 4 Theming - "no files detected" note background `Color.secondary.opacity(0.06)` - use `.appSurface`/`.appElevated`
- [HIGH] line 68 - 4 Theming - "assign a root directory" note background `Color.secondary.opacity(0.06)` - same
- [HIGH] line 114 - 4 Theming - "Imported" chip background `Color.secondary.opacity(0.12)` - same
- [HIGH] line 127 - 4 Theming (surfaces) - `.appSurface.opacity(0.8)` hard-coded dimming of the themed surface
- [MEDIUM] line 133 - 4 Theming (borders) - `.appAccent.opacity(0.3)` hard-coded border opacity on the card overlay
- [LOW] lines 55, 69, 128, 132 - 2 Radii - 8/6/6/8 not from `AppMetrics`
- [LOW] line 101 (possible) - 6 Localization - `Text(verbatim: server.approvalSummary.text)`: the argv/env content is user data (correct), but `McpApprovalSummary` (Tools/MCP/McpApprovalSummary.swift) embeds fixed English labels ("command:", "cwd:", "env (loader):", "url:") that ship untranslated; root cause is in the model file, out of this batch

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectToolApprovalSheet.swift
- [HIGH] lines 60, 67 - 6 Localization - `Text(verbatim: review.declaredCategory.label)` / `effectiveCategory.label`: `AppToolCategory.label` (Tools/Registry/AppToolTypes.swift) returns fixed English ("File & Codebase Reading", "Terminal & Shell Execution", ...) that is not a catalog key; verbatim guarantees untranslated UI copy - render via `Text(LocalizedStringKey(label), bundle: .module)` (or accept `Text` from the enum) and register the enum in `testEnumBackedDisplayStringsAreCatalogKeys`
- [HIGH] line 73 - 4 Theming - `.foregroundStyle(.orange)` theme-blind warning color - use `AppAttention.color`; the warning is text-only, add a triangle glyph so status survives color blindness
- [HIGH] line 47 - 4 Theming - pending-count capsule background `Color.secondary.opacity(0.12)`
- [HIGH] line 92 - 4 Theming - code block background `Color.secondary.opacity(0.08)`
- [MEDIUM] lines 96-99 - 2 Vocabulary (filesystem paths) - `review.sourcePath` rendered as a plain wrapped `Text` instead of `SettingsPathText` (code font, selectable, middle truncation, full path in `.help`)
- [MEDIUM] lines 113, 118 - 2 Vocabulary (action hierarchy) - `.buttonStyle(.bordered)` / `.borderedProminent` bespoke styling instead of `.appAction(.secondary)`/`.appAction(.primary)`; Approve/Reject are also inverted from the usual primary-on-right layout without the vocabulary's spacing - keep the Reject-is-default behavior but style through `.appAction`
- [LOW] line 93 - 2 Radii - `cornerRadius: 6` not from `AppMetrics` (line 105-106 correctly uses cardRadius 10 + `.appBorder`)
- [LOW] line 123 (possible) - 3 Layout - fixed `.frame(width: 560)` sheet body does not scale with Extra Large text
- [LOW] line 35 - 5 Accessibility - sheet title "Project Tool Approval" lacks `.accessibilityAddTraits(.isHeader)`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ToolGroupView.swift
- [HIGH] line 42, 130 - 6 Localization - `groupLabel` builds "\(count) tool \(count == 1 ? "call" : "calls")" as an English `String` with an in-string plural ternary and renders it `Text(verbatim:)` for both display and `accessibilityLabel`; keys "%lld tool calls" / "%lld tool call" absent from the catalog - move to `Text("\(count) tool calls", bundle: .module)` style (catalog plural or split keys)
- [HIGH] line 69 - 5 Motion - `withAnimation(.easeInOut(duration: 0.18))` bare withAnimation + inline duration - `TSMotion.animate`
- [HIGH] line 87 - 4 Theming - `.foregroundStyle(Color.red)` on the failure glyph - theme-blind; pair with `AppAttention.color`
- [HIGH] line 107 - 4 Theming - `.foregroundStyle(Color.orange)` on "approval required" - use `AppAttention.color` (and the glyph is absent here; the status is color-only text)
- [HIGH] line 114 - 4 Theming - `.foregroundStyle(.tertiary)` on the chevron - use `.appSecondary`
- [HIGH] line 131 - 6 Localization - `.accessibilityValue(isExpanded ? "Expanded" : "Collapsed")` bare `String` overload (does not consult the bundle); keys "Expanded"/"Collapsed" exist in the catalog but are never used here - use `Text("Expanded", bundle: .module)` ternary

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/WorktreeCommitListView.swift
- [HIGH] line 24 - 4 Theming (surfaces) - `.appSurface.opacity(0.4)` hard-coded dimming of a themed surface
- [HIGH] line 29 - 5 Motion - `withAnimation(.easeInOut(duration: 0.15))` bare + inline duration - `TSMotion.animate`
- [HIGH] lines 36, 48, 66, 104, 107, 115, 119 - 4 Theming - `.foregroundStyle(.tertiary)` (chevron, commit count, empty text, short hash, separators, date) theme-blind - use `.appSecondary`
- [MEDIUM] line 37 - 3 Layout - `.frame(width: 12)` on a chevron sized with `.themedFont(.tiny)` which grows with Text Size
- [HIGH] line 51 - 4 Theming - count capsule background `Color.primary.opacity(0.04)` - use `.appElevated`
- [HIGH] line 97 - 4 Theming (Gotcha 36) - ternary `isSelected ? TurboSparkTheme.accentColor : .primary` static accent read + system `.primary` - `.appAccent`/`.appText` in an `AnyShapeStyle` ternary
- [HIGH] line 129 - 4 Theming (Gotcha 36, surfaces) - `TurboSparkTheme.accentColor.opacity(0.12) : Color.clear` selected-row background: static accessor read in a body plus hard-coded dimming - use `.appAccent` (resolved) with the shared selection treatment
- [MEDIUM] lines 85-92 - 5 Accessibility - selectable commit rows never carry `.accessibilityAddTraits(.isSelected)` when `isSelected`
- [MEDIUM] lines 95-99 - 3 Text in layouts - `commit.summary` is `lineLimit(1)` truncated with no tooltip carrying the full value - add `.help`
- [LOW] line 130 - 2 Radii - `cornerRadius: 6` not from `AppMetrics`
- [LOW] lines 63-68 (possible) - 3 Empty states - "No recent commits found." is an inline `Text` rather than `AppEmptyState`; tolerable for a small drawer, but the vocabulary prefers the shared component

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ActiveStreamingRowView.swift
- clean (all fixed copy uses `Text(_, bundle: .module)` with verified keys; verbatim only on numbers and todo content; status resolution delegated to `ChatActivityLine`)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/WorktreeWorktreeListView.swift
- [HIGH] line 43 - 6 Localization - `Text(verbatim: "\(worktree.worktrees.count) worktrees")` is fixed English UI copy marked verbatim; key "%lld worktrees" is absent from the catalog - use `Text("\(worktree.worktrees.count) worktrees", bundle: .module)` and add the key
- [MEDIUM] lines 36-49 - 3 Headers - pane header built from bare `HStack` + `Spacer` instead of `AppAdaptiveHeader`; truncates under German/Extra Large
- [HIGH] line 49 - 4 Theming - header background `Color.primary.opacity(0.02)` - use `.appSurface`
- [MEDIUM] lines 52-66 - 2 Vocabulary (empty states) - hand-rolled empty state (icon + two Texts) instead of `AppEmptyState(systemImage:title:message:)`
- [HIGH] line 56 - 4 Theming - `.foregroundStyle(.tertiary)` on the empty-state glyph - use `.appSecondary`
- [HIGH] line 63 - 4 Theming - `.foregroundStyle(.tertiary)` on the empty-state message - use `.appSecondary`
- [LOW] line 82 (possible) - 4 Theming - `.foregroundStyle(.white)` on the `.appAccent` capsule: hard-coded white assumes a dark accent; a light custom accent fails contrast - prefer a theme-provided on-accent color or ensure the accent token guarantees contrast
- [HIGH] line 92 - 4 Theming - `.foregroundStyle(.tertiary)` on the worktree head - use `.appSecondary`
- [MEDIUM] lines 95-99 - 2 Vocabulary (filesystem paths) - `wt.path` rendered as plain truncated `Text` instead of `SettingsPathText(path, size:)` (missing selectable text and full path in `.help`)
- [HIGH] line 122 - 4 Theming - `wt.isCurrent ? TurboSparkTheme.accentColor.opacity(0.08) : Color(nsColor: .controlBackgroundColor).opacity(0.5)` - `Color(nsColor: .controlBackgroundColor)` is named explicitly in DESIGN.md as forbidden (Gotcha 62), plus the static accent read in a body and hard-coded dimming; use `.appSurface`/resolved `.appAccent` in an `AnyShapeStyle` ternary
- [HIGH] line 127 - 4 Theming - border ternary `TurboSparkTheme.accentColor.opacity(0.3) : Color.primary.opacity(0.08)` - static accessor + hand-drawn border opacity; use `.appBorder`/`.appAccent`
- [LOW] lines 123, 126 - 2 Radii - `cornerRadius: 6` not from `AppMetrics`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectMcpApprovalSheet.swift
- [HIGH] line 53 - 4 Theming - pending-count capsule background `Color.secondary.opacity(0.12)` - use `.appElevated`
- [MEDIUM] line 75 - 4 Theming - `.foregroundStyle(.red)` on the risky-env warning is theme-blind - use `AppAttention.color` (the "!" prefix means status is not color alone)
- [LOW] line 65 (possible) - 6 Localization - `Text(verbatim: summary.text)`: argv/env content is user data (correct), but `McpApprovalSummary` embeds fixed English labels ("command:", "env (loader):", "url:") that stay untranslated; root cause is in Tools/MCP/McpApprovalSummary.swift, out of this batch
- [LOW] lines 106-109 (possible) - 5 Keyboard - the primary "Approve" button carries no `.keyboardShortcut(.defaultAction)`; DESIGN.md binds sheet default actions through the shortcut system (may be deliberate, since this sheet should not approve on Return by accident - if so, document it)
- [LOW] line 113 (possible) - 3 Layout - fixed `.frame(width: 460)` sheet body does not scale with Extra Large text

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectPickerView.swift
- [HIGH] line 36 - 4 Theming (Gotcha 36) - `.background(TurboSparkTheme.surfaceColor, in: Capsule())` static theme read inside the body draws the wrong mode under `.preferredColorScheme` - use `.appSurface`
- [MEDIUM] lines 40, 116 - 2 Action hierarchy (disabled controls) - picker button and the whole popover are `.disabled(navigationBlocked)` (running/submitting/pending tool call) with no tooltip or adjacent text explaining why
- [MEDIUM] lines 85-87 - 5 Accessibility - the selected project row shows a checkmark image but never carries `.accessibilityAddTraits(.isSelected)`
- [LOW] line 114 (possible) - 3 Layout - fixed `.frame(width: 340)` popover containing a search field and rows does not scale with Extra Large text

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ComposerSuggestionPopup.swift
- [HIGH] line 39 - 4 Theming (borders) - `Color.primary.opacity(0.14)` hand-drawn border stroke - use `.appBorder`
- [HIGH] line 91 - 4 Theming (surfaces) - selected-row background `isSelected ? Color.primary.opacity(0.09) : Color.clear` - use the shared selection treatment (e.g. `.appElevated`/`.appAccent` in an `AnyShapeStyle` ternary)
- [MEDIUM] line 35 - 4 Surfaces - `.background(.regularMaterial)` system material inside the composer: the popup sits inside the themed composer box (per the doc comment), so a system material ignores the custom palette - prefer `.appElevated`; if the material is intentional, it must be availability-gated per the blur rule
- (Accessibility is otherwise exemplary here: `.isSelected` traits, combined elements, bundled hint texts)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ToolCodeCellView.swift
- [HIGH] line 46 - 6 Localization - `Text(label.uppercased())`: labels are fixed English from ToolCallCardView ("command", "edits", "request", "patch" - none are catalog keys) and `Text(String)` never localizes; also `uppercased()` is the wrong place to case-map localized copy - accept/produce a localized `Text` for the label or make the labels catalog keys rendered with `bundle: .module`
- [HIGH] line 58 - 6 Localization - `Text(isCopied ? "Copied" : "Copy")` is a ternary inside one `Text` with bare literals: a key-only API resolving against `Bundle.main`, invisible to the localization gates (keys "Copied"/"Copy" exist but are never used here) - write `isCopied ? Text("Copied", bundle: .module) : Text("Copy", bundle: .module)`
- [HIGH] line 61 - 4 Theming - `.foregroundStyle(isCopied ? Color.green : Color.secondary)` theme-blind colors in a ternary - use `.appAccent`/`.appSecondary` in an `AnyShapeStyle` ternary
- [HIGH] line 76 - 4 Theming - `.foregroundStyle(Color.secondary)` on the Download button - use `.appSecondary`
- [HIGH] line 28 - 4 Theming (surfaces) - `.appElevated.opacity(0.5)` hard-coded dimming of a themed surface (bypasses Reduce Transparency/Increase Contrast)
- [HIGH] line 32 - 4 Theming (borders) - `.appBorder.opacity(0.25)` hard-coded border opacity (counted by DesignTokenRatchetTests outside Theme/)
- [HIGH] line 38 - 4 Theming - left accent bar `Color.primary.opacity(0.15)` - use `.appBorder` or `.appAccent`
- [HIGH] lines 90, 96 - 5 Motion - `withAnimation(.easeInOut(duration: 0.15))` bare withAnimation with inline durations (copy feedback) - use `.tsAnimation(TSMotion.press, value: isCopied)` or `TSMotion.animate`
- [LOW] lines 29-31 - 2 Radii - `cornerRadius: 6` not from `AppMetrics`

### Cross-file notes
- Theme-blind spellings (`Color.primary.opacity`, `Color.secondary`, `.tertiary`, `.red/.green/.orange`, `Color(nsColor:)`) and hard-coded `.opacity` borders/surfaces dominate this batch; most sit in hand-drawn hover/selection/chip treatments that should be one shared vocabulary treatment (they are also the exact shapes DesignTokenRatchetTests counts).
- Two destructive actions lack confirmation dialogs in their whole chain: ProjectMcpServerRowView "Delete Server" (only one found in this batch; the sheet calls the model directly).
- Four unlocalized-English defects confirmed against `Localizable.xcstrings`: CollapsibleMessageContentView help/labels (keys absent), ToolGroupView group label + accessibilityValue, WorktreeWorktreeListView "worktrees" verbatim, ProjectToolApprovalSheet/AppToolCategory labels; ToolCodeCellView "Copied"/"Copy" ternary and its English labels.
- Motion: five files use bare `withAnimation` with inline durations (CollapsibleMessageContentView, ToolGroupView, WorktreeCommitListView, ToolCodeCellView x2) - all ignore Reduce Motion.
- All files are well under the 400-line guideline; no file-size findings.

---

### Batch batch22_Generation

Audit against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md. Read-only audit; no repository files changed.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/BrowserToolCardView.swift
- clean

(Verified: the verbatim duration text is a formatted number + unit, correct per section 6; `theme.code(.tiny)` is the theme-resolved form; the unused `Outcome` English raw values are not rendered anywhere in Generation, so no display defect is cited.)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/GoalBannerView.swift
- [HIGH] line 100 - DESIGN.md 4 Color - `Color(nsColor: .controlBackgroundColor).opacity(0.85)` system color plus hard-coded dimming on the banner surface; ignores the palette and the theme's surface opacity (Gotcha 62) - use `.appSurface` or `.appElevated`
- [HIGH] line 105 - DESIGN.md 4 Color - `Color.primary.opacity(0.08)` stroke; theme-blind plus hard-coded border dimming - use `.appBorder`
- [MEDIUM] lines 101, 104 - DESIGN.md 2 Radii - hard-coded `cornerRadius: 10` twice instead of `AppMetrics.cardRadius` (same value, but the constant is the source of truth)
- [MEDIUM] line 60 - DESIGN.md 2 vocabulary - `ProgressView().controlSize(.mini)` for the live "Evaluating" state instead of `AppProgressBar` (vocabulary table routes all progress, determinate or not, through `AppProgressBar(value: nil, ...)`)
- [LOW] line 42 - DESIGN.md 2 action hierarchy - bespoke `.buttonStyle(.link)` quiet action instead of `.appAction(.quiet, size: .small)` (possible)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/StreamingStatusFooterView.swift
- [HIGH] line 32 - DESIGN.md 6 Localization - `Text(verbatim: "\(model.liveTokenCount) tokens")` hard-codes the English word "tokens" outside the catalog; the `%lld tokens` key already exists in Localizable.xcstrings but is unused here - write `Text("\(model.liveTokenCount) tokens", bundle: .module)` (Int interpolates as `%lld`)
- [HIGH] lines 41, 45 - DESIGN.md 4 Color - `.foregroundStyle(Color.orange)` on the pending-confirmation and Esc-armed status; theme-blind attention color ignores custom themes - use `AppAttention.color`
- [HIGH] line 73 - DESIGN.md 4 Color - `Color.secondary.opacity(0.4)` dot separator; system color plus hard-coded dimming - use `.appSecondary`
- [HIGH] line 47 (source lines 92, 95, 104) - DESIGN.md 6 Localization - `currentStatusText` returns bare English strings ("Thinking...", "Running tools...", "Idle") that reach `Text(String)`, which never resolves the catalog, so they render in English under all 21 languages; all three keys exist in the catalog. The `LoadingPhrases.phrase` value (line 102) is also an English-only literal list - produce these with `AppStrings.localized("Thinking...")` and friends, and route LoadingPhrases through the catalog
- Good: elapsed/token text is duration + number (verbatim is correct per section 6); bundled Text elsewhere.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerFooter.swift
- [HIGH] line 92 - DESIGN.md 4 Color - `.foregroundStyle(.tertiary)` system spelling on the clear-prompt glyph; ignores a custom theme's text colors - use a themed dimmer (`.appSecondary`)
- [MEDIUM] lines 86-99 - DESIGN.md 2 vocabulary - icon-only clear Button hand-rolled instead of `AppIconButton(symbol:label:)`; naming is present (`.help` + `.accessibilityLabel`) but this duplicates the vocabulary control that keeps tooltip and VoiceOver name from drifting
- [MEDIUM] line 93 - DESIGN.md 3 Layout - fixed `.frame(width: iconButtonSize, height: iconButtonSize)` square around a glyph sized with `.themedFont(.callout)`; `@ScaledMetric` follows Dynamic Type but not the in-app Text Size scale, so the glyph can outgrow the frame at Extra Large (possible)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptExamplesView.swift
- [HIGH] lines 32, 68 - DESIGN.md 4 Color - `.separator.opacity(0.5)` and `.separator.opacity(0.4)` strokes; system separator color plus hard-coded dimming ignore custom themes (the exact nsColor: .separatorColor class) - use `.appBorder`
- [MEDIUM] lines 28, 31 - DESIGN.md 2 Radii - hard-coded `cornerRadius: 18` on the outer card instead of an `AppMetrics` radius (cardRadius 10 / composerRadius 20)
- [MEDIUM] lines 64, 67 - DESIGN.md 2 Radii - hard-coded `cornerRadius: 14` on the preset cards instead of `AppMetrics.cardRadius`
- [MEDIUM] line 12 - DESIGN.md 5 Accessibility - the "Try an example" section heading lacks `.accessibilityAddTraits(.isHeader)`, so VoiceOver heading navigation skips it
- Correct per guidance: `.buttonStyle(.plain)` on the selectable preset cards; `Text(verbatim:)` for preset title/prompt (bundled data sent to the model, comment documents it); `Button(preset.title)` inside Menu content.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSystemPromptSheet.swift
- [HIGH] lines 32-39 (rendered at line 53) - DESIGN.md 6 Localization - `effectiveSourceDescription` builds fixed English UI copy as a String ("This chat sends the prompt below.", "This chat sends no system prompt. No app-wide default is set.", "Empty, so this chat inherits the app-wide default (N characters, set in Settings > Engine).") that reaches `Text(String)` unlocalized in every language - build each variant with `AppStrings.localized` and interpolate the character count as `%lld`
- [HIGH] line 52 - DESIGN.md 4 Color - `Color.secondary.opacity(0.25)` editor stroke; system color plus hard-coded dimming - use `.appBorder`
- [MEDIUM] lines 47-52 - DESIGN.md 2 vocabulary / 3 Layout - hand-rolled TextEditor surface (overlay stroke, hard-coded cornerRadius 6, fixed-height frame) instead of `TextEditor(...).appEditorSurface(minHeight: 220)` with `AppMetrics.controlRadius`
- [LOW] line 84 - DESIGN.md 2 action hierarchy - "Use Default" disabled when the text is empty with no tooltip or adjacent text saying why (a disabled control states its reason)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptAttachmentsView.swift
- [HIGH] lines 43-44 - DESIGN.md 4 Color - ternary `TurboSparkTheme.accentColor : Color.secondary`; the bare static accessor resolves through `NSApp.effectiveAppearance` (Gotcha 36) and does not follow `.preferredColorScheme` or custom themes, plus the system color, and the two styles are mixed without `AnyShapeStyle` - use `AnyShapeStyle(.appAccent)` / `AnyShapeStyle(.appSecondary)`
- [HIGH] line 83 - DESIGN.md 4 Color - ternary `TurboSparkTheme.accentColor.opacity(0.12) : Color.primary.opacity(0.05)` chip background; bare static read, system color, hard-coded dimming, no `AnyShapeStyle` - use `.appAccent` / a themed elevated surface, wrapped in `AnyShapeStyle`
- [HIGH] line 88 - DESIGN.md 4 Color - ternary `TurboSparkTheme.accentColor.opacity(0.4) : TurboSparkTheme.hairlineColor` border; two bare static reads plus hard-coded border opacity, no `AnyShapeStyle` - use `.appAccent` / `.appBorder`
- [HIGH] line 72 - DESIGN.md 4 Color - `.foregroundStyle(.tertiary)` on the remove glyph; theme-blind system spelling - use a themed dimmer
- [MEDIUM] lines 65-76 - DESIGN.md 2 vocabulary - icon-only remove Button hand-rolled instead of `AppIconButton` (a11y label present)
- [MEDIUM] line 68 - DESIGN.md 3 Layout - fixed `.frame(width: 14, height: 14)` around a glyph sized with `.themedFont(.micro, weight: .bold)`; the glyph grows with Text Size, the frame does not
- [LOW] lines 84, 86 - DESIGN.md 2 Radii - hard-coded `cornerRadius: 8` instead of `AppMetrics.controlRadius` (7)
- [LOW] line 73 - DESIGN.md 2 action hierarchy - remove button disabled while `isRunning` and its `.help` states only the action, not why it is disabled

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/BackgroundShellsStripView.swift
- [HIGH] lines 70-79 - DESIGN.md 2 action hierarchy - `Button(role: .destructive)` Kill with no `confirmationDialog`; DESIGN.md: "A destructive role with no confirmation dialog is always a defect" - add a localized confirmation dialog (title, message, button) or downgrade the role
- [HIGH] line 77 - DESIGN.md 4 Color - `Color.red.opacity(0.8)` hand-rolled destructive tint; because the button uses `.buttonStyle(.plain)`, the destructive role's red never applies and this theme-blind red with hard-coded opacity replaces it - use `.appAction(.destructive, size: .small)`
- [MEDIUM] line 83 - DESIGN.md 4 Surfaces - `.appSurface.opacity(0.7)` hard-coded dimming on a themed structural surface; bypasses the theme's `surfaceOpacity` under Reduce Transparency and `isHighContrast`
- [MEDIUM] line 86 - DESIGN.md 4 Surfaces - `.appAccent.opacity(0.45)` hard-coded border dimming lower than the theme gives
- [MEDIUM] line 89 - DESIGN.md 5 Accessibility - `.accessibilityElement(children: .combine)` folds the Kill button into the row element, so the destructive action may no longer be individually reachable by VoiceOver; expose Kill via `.accessibilityAction` like the chat row reference (possible)
- [LOW] lines 85, 88 - DESIGN.md 2 Radii - hard-coded `cornerRadius: 10` twice instead of `AppMetrics.cardRadius`

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ConversationPaneView.swift
- [HIGH] line 50 - DESIGN.md 4 Color - `.foregroundStyle(.orange)` on the NoModelInstalledBanner icon; theme-blind system spelling - use `AppAttention.color`
- [HIGH] line 85 - DESIGN.md 4 Color - `Color.orange.opacity(0.08)` banner background; system color plus hard-coded dimming - use the AppAttention treatment
- [HIGH] line 88 - DESIGN.md 4 Color - `Color.orange.opacity(0.35)` banner stroke; same class - use `AppAttention` / `AppAttentionNote` styling
- [MEDIUM] lines 84-89 - DESIGN.md 2 vocabulary - hand-rolled warning banner (icon + orange fill + stroke) duplicating the `.appAttention` / `AppAttentionNote` vocabulary instead of reusing it (possible; the banner is a persistent state, so confirm the shared component fits)
- Good: native bordered buttons with one prominent primary action; the disabled Send's reason is documented; `ConversationLayout.cardRadius` is a shared constant.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/BrowserPermissionApprovalView.swift
- [HIGH] lines 56, 60 - DESIGN.md 4 Color - `.foregroundStyle(.red)` on both refusal messages; theme-blind system spelling - use `AppAttention.color` (the message text is present, so status is not color alone)
- [MEDIUM] line 66 - DESIGN.md 4 Surfaces - `.appElevated.opacity(0.45)` hard-coded dimming on a themed surface; below what the theme gives for Reduce Transparency / Increase Contrast - use `.appElevated`
- [MEDIUM] line 22 - DESIGN.md 3 Layout (Text in layouts) - `.lineLimit(1)` on the row truncates the origin URL with no tooltip carrying the full value; the middle of a selectable-but-truncated origin cannot be selected - add `.help` with the full origin (possible)
- [LOW] line 66 - DESIGN.md 2 Radii - hard-coded `cornerRadius: 8` instead of `AppMetrics` radii

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/NewChatSuggestionBanner.swift
- [HIGH] line 32 - DESIGN.md 4 Color - `.foregroundStyle(.orange)` icon; theme-blind system spelling - use `AppAttention.color`
- [HIGH] line 61 - DESIGN.md 4 Color - `Color.orange.opacity(0.08)` background; system color plus hard-coded dimming - AppAttention treatment
- [HIGH] line 64 - DESIGN.md 4 Color - `Color.orange.opacity(0.35)` stroke; same class
- [LOW] lines 63, 66 - DESIGN.md 2 Radii - hard-coded `cornerRadius: 10` twice instead of `AppMetrics.cardRadius`
- [LOW] line 55 - DESIGN.md 2 action hierarchy - "New Chat" disabled while running; `.help` states the action but not why it is disabled (possible)
- Good: uses `.appAction(.quiet)` and `.appAction(.primary)` from the vocabulary.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectRulesSectionView.swift
- [HIGH] line 51 - DESIGN.md 3 Layout - fixed `.frame(width: 180)` on a Picker in a form row; this is the exact defect class DESIGN.md cites ("a fixed width ... clipped the Agents scope picker") and it truncates longer Conflict Preference translations - drop the fixed width or use minWidth/idealWidth/maxWidth
- [MEDIUM] line 63 - DESIGN.md 2 vocabulary / 3 Layout - `TextEditor` with fixed `.frame(height: 100)` beside `.appEditorSurface()` called without its parameter; pass `.appEditorSurface(minHeight: 100)` and remove the fixed-height frame so Extra Large text is not pinned to 100pt
- Good: `.accessibilityAddTraits(.isHeader)` on the title, `.appAction(.quiet, size: .small)` on the row action, real picker label plus accessibilityLabel.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSearchResultRowView.swift
- [HIGH] line 16 - DESIGN.md 4 Color - ternary `TurboSparkTheme.accentColor : Color.secondary` on the bubble icon; bare static accessor (NSApp.effectiveAppearance-based, Gotcha 36), system color, and mixed styles without `AnyShapeStyle` - `AnyShapeStyle(.appAccent)` / `AnyShapeStyle(.appSecondary)`
- [HIGH] line 59 - DESIGN.md 4 Color - `TurboSparkTheme.accentColor` read inside `snippetText` (bare static in a view body); matched tokens render with the appearance-manager color, not the app theme - use the resolved theme accent or `.appAccent`
- [HIGH] line 43 - DESIGN.md 4 Color - `.foregroundStyle(.tertiary)` on the date bucket; theme-blind system spelling - use a themed dimmer
- [MEDIUM] line 13 - DESIGN.md 5 Accessibility - the row never carries `.accessibilityAddTraits(.isSelected)` when `isSelected` is true, so VoiceOver cannot announce the selection the visual highlight shows (possible; a parent List selection could also supply it)
- [MEDIUM] line 13 - DESIGN.md 5 Accessibility - the row is not one accessibility element (no `.accessibilityElement(children: .combine)`); title, snippet, match count and date read as separate items ("A row that reads as one item is one element")
- Correct per rules: `theme.ui(...)` fonts throughout; bundled keys for matches and date buckets.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProfileMemoryCaptureSheet.swift
- [HIGH] lines 26, 35 - DESIGN.md 4 Color - `.foregroundStyle(.secondary)` on the purpose text and the summarizing status; theme-blind system spelling in two places - use `.appSecondary`
- [HIGH] line 30 - DESIGN.md 4 Color - `.quaternary` editor stroke; system style that ignores custom themes - use `.appBorder`
- [MEDIUM] lines 27-30 - DESIGN.md 2 vocabulary - hand-rolled TextEditor surface (overlay stroke, hard-coded cornerRadius 8, bare min-frame) instead of `TextEditor(...).appEditorSurface(minHeight: 260)`
- [MEDIUM] line 33 - DESIGN.md 2 vocabulary - bare `ProgressView()` while summarizing instead of `AppProgressBar(value: nil, ...)` (vocabulary routes all progress, determinate or not)
- [MEDIUM] lines 38-47 - DESIGN.md 5 Keyboard - this sheet binds neither `.keyboardShortcut(.cancelAction)` on Cancel nor `.keyboardShortcut(.defaultAction)` on Save ("Sheets bind Save or Done to .defaultAction and Cancel to .cancelAction")
- Good: Save disabled while summarizing is explained by adjacent status text; minWidth forms used.

### Cross-file notes
- Recurring classes in this batch: attention/warning tints hand-rolled from `Color.orange`/`.orange`/`.red`/`Color.red` instead of `AppAttention.color` (6 files); hard-coded opacity on themed surfaces/borders (`.appSurface.opacity`, `.appElevated.opacity`, `.appAccent.opacity`) (3 files); bare `TurboSparkTheme.accentColor` / `hairlineColor` static reads inside view bodies (2 files, 4 sites); English status strings reaching `Text(String)` unlocalized (StreamingStatusFooterView, ChatSystemPromptSheet).
- None of the audited files exceed ~400 lines, so no length findings.
- Fixing `.separator` / `Color.primary.opacity` / `.tertiary` / `.secondary` sites also lowers the DesignTokenRatchetTests counts; per DESIGN.md section 4 those baselines must go down in the same change, never up.

---

### Batch batch24_Humanize

#### swift/TurboSparkApp/Sources/TurboSparkApp/Humanize/HumanizeView.swift
- [HIGH] line 336 - DESIGN.md 4 Theming (system colors) - checks status chip tints are bare system colors: `let tint: Color = clean ? .green : .orange`, then `.foregroundStyle(tint)` (line 343) and `tint.opacity(0.12)` background (line 346); the orange attention side ignores custom themes (the AppAttention orange exists precisely for this). The glyph keeps this off the color-alone rule, but the tint should be themed. - orange side: `AppAttention.color`; green has no themed token (possible follow-up for the vocabulary).
- [MEDIUM] lines 235-241 - DESIGN.md 2 vocabulary (empty states) - hand-rolled empty state for the Rewrite card (VStack + `Image(systemName: "text.alignleft")` + `Text("Your rewrite will appear here.", ...)`) instead of `AppEmptyState(systemImage:title:message:)`, and it offers no action that fills it. - replace with `AppEmptyState(systemImage: "text.alignleft", title: Text("Your rewrite will appear here.", bundle: .module))`; the run button can be offered as its action.
- [MEDIUM] line 209 - DESIGN.md 4 Theming (attention tint) - over-limit token count uses `AnyShapeStyle(Color.orange)`; system orange ignores custom themes. The `AnyShapeStyle` wrapping itself is correct. - use `AnyShapeStyle(AppAttention.color)`.
- [MEDIUM] lines 389-390 - DESIGN.md 4 Theming (attention tint) - "words to avoid" highlights in `highlightedRewrite` use `Color.orange.opacity(0.22)` background and `.orange` dash underline in the AttributedString. - use `AppAttention.color` (it is a concrete Color, so it works in AttributedString attributes) and keep `theme.accent` for the diff side as already done.
- [MEDIUM] lines 279-284 - DESIGN.md 2 action hierarchy (destructive) - the Clear Rewrite chip permanently destroys the rewrite with no confirmationDialog, while the pane's own doc comment (lines 131-136) says the rewrite "is the one thing a clear cannot give back" and Clear Both asks first whenever a rewrite exists. Either Clear Rewrite asks like Clear Both, or it is not destructive and the comment's rationale is wrong. - add a confirmationDialog matching the Clear Both one (lines 118-124).
- [LOW] line 346 - DESIGN.md 2 vocabulary (radii) - `RoundedRectangle(cornerRadius: 8, style: .continuous)` is a bespoke radius; AppMetrics defines controlRadius 7, cardRadius 10, composerRadius 20. - use `AppMetrics.controlRadius`.
- [LOW] lines 361-362 (possible) - DESIGN.md 4 Theming - deleted-diff marks in `markedDraft` use `Color.red.opacity(0.85)` and a `.red` strikethrough; red-as-deletion matches the destructive convention and is paired with strikethrough, but the spelling is theme-blind. - consider a theme-resolved red token or document the exception.
- [LOW] lines 419-422 (possible) - DESIGN.md 4 Surfaces and effects (glows) - the page backdrop is a hand-rolled accent `RadialGradient` wash outside the `.appFocusGlow`/`.appPrimaryGlow` vocabulary; the comment says it mirrors the welcome hero's wash, so if that pattern is now shared by three panes it should be a named vocabulary surface.
- [LOW] informational - file is 435 lines; DESIGN.md 3 says keep view files near 400 lines, split by subview (e.g. the checks chip and panel title pieces).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Humanize/HumanizeChrome.swift
- [HIGH] line 94 - DESIGN.md 4 Theming (Gotcha 36) - `TurboSparkTheme.surfaceColor` bare static accessor read inside `HumanizeModelBar` body (`...in: Capsule()`); the static resolves from `NSApp.effectiveAppearance` and draws the wrong mode under `.preferredColorScheme`. Counted by DesignTokenRatchetTests ("bare static TurboSparkTheme accessor"). - use the `.appSurface` shape style or the resolved theme's fill.
- [HIGH] line 97 - DESIGN.md 4 Theming (Gotcha 36) - same defect in the capsule stroke ternary: `... : TurboSparkTheme.hairlineColor` (static read in a view body, ratchet-counted). - use `.appBorder` or `theme.border` (the resolved form this file already uses at line 237).
- [HIGH] line 203 - DESIGN.md 4 Theming (system colors) - `HumanizeBarButton` hover fill falls back to `Color.primary.opacity(0.06)` (ratchet-counted "system secondary style"); ignores a custom theme's text color. Also a hard-coded `tint?.opacity(0.12)` wash. - use a themed secondary fill (`.appSecondary`-based) for the hover.
- [HIGH] line 229 - DESIGN.md 4 Theming (system colors) - idle run-button ink is `Color.secondary.opacity(0.5)` (ratchet-counted); draws theme-blind gray on the themed surface. - use `.appSecondary` (via `AnyShapeStyle`) for the not-ready ink.
- [HIGH] line 234 - DESIGN.md 4 Theming (system colors) - idle button fill is `AnyShapeStyle(Color.primary.opacity(0.08))` (ratchet-counted). - use `.appSurface`/`.appSecondary` so custom themes reach it.
- [HIGH] line 307 - DESIGN.md 4 Theming (system colors) - skeleton placeholder bars use `Color.primary.opacity(0.07)` (ratchet-counted). - use a themed secondary/border-derived fill.
- [MEDIUM] line 290 - DESIGN.md 5 Motion ("No inline durations") - RunRipple's phaseAnimator expands with `.easeOut(duration: 1.8)`, an inline duration and a curve outside the TSMotion set; the collapse side correctly uses `TSMotion.snap`. - use a TSMotion loop curve (breathe/sweep/snap) or add a named ripple curve to TSMotion.
- [LOW] line 97 - DESIGN.md 4 Surfaces and effects (hard-coded border opacity) - the ready-state capsule edge is hard-coded `theme.accent.opacity(0.32)` and, unlike the HumanizeCard dividers (lines 47, 49), is not raised under Increase Contrast. - resolve through the theme or gate on `theme.isHighContrast`.
- [LOW] lines 47, 49 (possible) - DESIGN.md 4 Surfaces and effects - `Divider().opacity(theme.isHighContrast ? 1 : 0.6)` hard-codes a 0.6 seam dimming; it is contrast-gated and documented, so mild, but it is a hand-tuned opacity rather than a theme value.
- [LOW] lines 252-258 (possible) - DESIGN.md 5 Motion - the busy spinner is a hand-rolled `TimelineView(.animation)` loop with an inline 1.1 s period; it is reduce-motion gated and torn down with the view (per its comment), but it sits beside TSMotion's phaseAnimator looping pattern rather than in it.
- [LOW] line 266 (possible) - DESIGN.md 3 Layout (fixed frames on growing glyphs) - `.frame(width: 50, height: 50)` pins the run button while its glyph grows with Text Size (`theme.ui(busy ? .callout : .large, ...)` line 262); this section already has `HumanizeMetrics.textScale` for exactly this and scales every other frame. - scale the button size by `HumanizeMetrics.textScale(theme)`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Humanize/HumanizeStatus.swift
- [MEDIUM] line 32 - DESIGN.md 2 action hierarchy (primary action) - the run button `.disabled(!ready)` goes dead when only the draft is missing: `HumanizeController.canRun` (HumanizeController.swift:124) is `isInstalled && hasDraft && !isBusy && currentPlan?.fits != false`, so an empty draft or an over-long draft disables the pane's one primary action. DESIGN: when all that is missing is input the user can still give, keep the primary action pressable and let the press show what is missing via `.appAttention` on the unfinished card (Voice Cloning's Generate speech is the reference; a missing model is a legitimate app-state disable). Related: the button's `.help` (line 37) names the action but never says why it is disabled. - keep it pressable for user-fixable input and flag the draft card with `.appAttention` on press; disable only for app-state blockers, with a why.
- [MEDIUM] lines 132, 135, 145, 155 - DESIGN.md 4 Theming (attention tints) - `HumanizeFeedback` labels use bare system tints: `.foregroundStyle(.red)` (error, line 132), `.foregroundStyle(.orange)` (long-draft plan, line 135), `.foregroundStyle(.orange)` (truncation, line 145), `.foregroundStyle(.orange)` (warnings loop, line 155). Each pairs with a glyph, so status is not color alone, but the spellings ignore custom themes. - use `AppAttention.color`; errors may keep red if a themed destructive/error token is added.
- [LOW] line 33 (possible) - DESIGN.md 7 Menus, commands, and shortcuts - the run button binds pane-local chords `Cmd-Return` / `Cmd-.` (KeyboardShortcutCatalog.swift lists those keys only as Chat's "Generate Response" and "Cancel Generation", lines 97-98; Audio's pane-bound chord got its own catalog section, Humanize's did not), so the Keyboard Shortcuts pane never tells a Humanize user about their pane's chords. - add a Humanize row or section to `KeyboardShortcutCatalog` like the Audio one.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Humanize/HumanizeAdvancedPanel.swift
- [HIGH] line 63 - DESIGN.md 6 Localization (Text(verbatim:) on fixed English UI copy) - the rule-field placeholders render `Text(verbatim: example)` where `example` is fixed English copy passed at lines 32, 35, and 38: "Skills Manager", "delve", "utilize -> use". None of the three is a key in `swift/TurboSparkApp/Localization/Localizable.xcstrings` (verified), so all 21 languages show English. - make `example` a `LocalizedStringKey`/`Text` like the `title`/`detail` parameters already are, add the three keys with translations, and run `make compile-strings`.
- [MEDIUM] lines 67-84 - DESIGN.md 2 vocabulary (multi-line editor) and DESIGN.md 3 (TextEditor height) - `ruleField` hand-rolls an editor surface: `TextEditor` with a fixed `.frame(height: (76 * textScale)...)` (line 75), `.background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius...))` (line 76) and a bespoke focus stroke overlay (lines 77-83) - a local copy of `.appEditorSurface(minHeight:)` plus `.appFocusGlow(_:halo:)`. The comment's contrast argument (solid accent edge over 3:1) is a vocabulary gap to fix in `Theme/`, not per-site. - move the solid-accent focus edge into the shared editor surface (or extend `.appFocusGlow`) and use `.appEditorSurface(minHeight:)` here.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Humanize/HumanizeHeader.swift
- clean

---

### Batch batch25_Images

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImagesSectionView.swift
- [HIGH] lines 107-113 - Section 2 action hierarchy ("A destructive role with no confirmation dialog is always a defect") - "Move to Trash" uses `Button(role: .destructive)` with no `confirmationDialog` anywhere in the file or the Images folder (verified by grep), and `AppModel.trashGeneratedImages` (State/AppModel+ImageGallery.swift) trashes files immediately with no internal confirm. Saved user images are destroyed on one click - Add a `.confirmationDialog` with localized title, message, and button before calling `trashGeneratedImages(ids:)`.
- [MEDIUM] lines 263-266 - Section 2 action hierarchy - destructive "Remove" (`discardUnsavedImage`) uses `Button(role: .destructive)` with no `confirmationDialog`; the unsaved result is lost on one click - Add a localized confirmation dialog, or if discarding an unsaved preview is judged non-destructive, drop the destructive role.
- [LOW] lines 228-231 - Section 4 surfaces ("Corners are continuous and use AppMetrics radii") - generation progress overlay uses `RoundedRectangle(cornerRadius: 12)` twice, an off-vocabulary radius with no `.continuous` - Use `AppMetrics.cardRadius` (10) with `style: .continuous`.
- [LOW] (possible) lines 258 - Section 5 accessibility ("images/galleries need accessible names") - the unsaved-result `Image(nsImage:)` carries no accessibility label; VoiceOver reads it as just "image" - Add `.accessibilityLabel` from the prompt or title.
- [LOW] (possible) lines 83-94 - Section 2 vocabulary ("A pane or workspace header | AppAdaptiveHeader") - the pane header is a bare `VStack` of title + count instead of `AppAdaptiveHeader` (it has no actions, so low risk, but it is off the shared header vocabulary) - Consider `AppAdaptiveHeader(leading:)` for consistency.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImageStylePicker.swift
- [MEDIUM] lines 206-207 - Section 4 color ("In a ternary that mixes two styles, wrap each side in AnyShapeStyle") - `.foregroundStyle(isSelected ? .appAccent : .appSecondary)` mixes two styles in a bare ternary; the codebase pattern wraps each side (e.g. Server/ServerMenuDashboardRows.swift:106) - Wrap each side in `AnyShapeStyle(.appAccent)` / `AnyShapeStyle(.appSecondary)`.
- [MEDIUM] lines 211-212 - Section 4 color, same rule - `.foregroundStyle(isSelected ? .appAccent : .appText)` bare ternary mixing two styles - Same fix: `AnyShapeStyle` on each side.
- [MEDIUM] lines 233-234 - Section 4 color, same rule - `.foregroundStyle(starred ? .appAccent : .appSecondary)` bare ternary mixing two styles - Same fix: `AnyShapeStyle` on each side.
- [MEDIUM] lines 171-181 - Section 2 vocabulary ("An empty list or workspace | AppEmptyState") - `ImageStyleBrowserSheet.emptyState` is a hand-rolled VStack (glyph + secondary text) instead of `AppEmptyState(systemImage:title:message:)` - Replace with `AppEmptyState`.
- [MEDIUM] lines 124-145 - Section 3 headers ("Use AppAdaptiveHeader"; audit checklist: bare HStack+Spacer header) - the sheet header is a bare `HStack { title, Spacer, search, picker }` that can wrap/compress under German and Extra Large text - Use `AppAdaptiveHeader(leading:actions:)`.
- [MEDIUM] line 126-127 - Section 5 accessibility ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") - the sheet title `Text("Style", bundle: .module)` has no `.isHeader` trait - Add `.accessibilityAddTraits(.isHeader)`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImageModelControls.swift
- [LOW] (possible) line 82 - Section 2 action hierarchy ("A disabled control says why, in its tooltip or in adjacent text") - the whole control row is disabled via `.disabled(model.isRunning)` with no tooltip or adjacent text explaining why; during image generation the gallery progress overlay provides context, but other `isRunning` states do not - Add a `.help` or adjacent line stating the model controls are busy while a run is active.
- clean otherwise: uses `.appMenuAction`, `AppMetrics`-free but consistent layout, themed fonts, bundled Text everywhere, verbatim only for model data.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImageComposerView.swift
- clean. Uses `.appEditorSurface(minHeight:)`, `AppMetrics.cardRadius` with `.continuous`, `AppChip`, `TSMotion.animate(TSMotion.pane)`, `.appAction` hierarchy with Generate as the one primary, bundled Text, verbatim only for model data/counts, and adjacent guidance text for the disabled Generate cases (ghost chat, missing model).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImageGalleryCard.swift
- [HIGH] lines 125-127 - Section 2 action hierarchy ("A destructive role with no confirmation dialog is always a defect") - "Move to Trash" in `ImageGalleryActions` (used by both the card menu and `.contextMenu`) is `Button(role: .destructive)` with no `confirmationDialog`; the file is trashed immediately - Add a localized confirmation dialog before `trashGeneratedImages(ids:)` (share one dialog with ImagesSectionView rather than a second local copy).
- [LOW] lines 39, 42 - Section 4 surfaces ("Corners are continuous and use AppMetrics radii") - card thumbnail uses `RoundedRectangle(cornerRadius: 12)` twice, off-vocabulary radius, no `.continuous` - Use `AppMetrics.cardRadius` with `style: .continuous`.
- [LOW] (possible) lines 69-70 - Section 3 text in layouts ("truncate with the full value in a tooltip") - the prompt under the card is truncated via `.lineLimit(compact ? 1 : 2)` with no `.help` carrying the full text (the open button's accessibilityValue has it, but sighted mouse users get nothing) - Add `.help(Text(verbatim: artifact.imageRequest?.prompt ?? artifact.title))`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImageGalleryCarousel.swift
- [MEDIUM] lines 23-41 - Section 3 headers (audit checklist: bare HStack+Spacer header) - the sheet header is `HStack { title, Spacer, menu, Done }` instead of `AppAdaptiveHeader(leading:actions:compactActions:)`; the title can wrap and the actions cannot reflow under long translations and Extra Large text - Use `AppAdaptiveHeader`.
- [MEDIUM] line 24 - Section 5 accessibility ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") - the title `Text("Preview", bundle: .module)` has no `.isHeader` trait - Add `.accessibilityAddTraits(.isHeader)`.
- [MEDIUM] (possible) lines 31-36 - Section 2 vocabulary ("An icon-only action | AppIconButton") and Section 5 ("every control has a name") - the actions `Menu` label is icon-only via `.labelStyle(.iconOnly)` and sets only `.help`, no `.accessibilityLabel`; the equivalent menu in ImageGalleryCard.swift:66 sets `.accessibilityLabel(Text("Image actions", bundle: .module))`, so the two drift - Add the same `.accessibilityLabel`, or use `AppIconButton`-style labeling.
- [LOW] line 46 - Section 4 surfaces - preview `RoundedRectangle(cornerRadius: 12)`, off-vocabulary radius, no `.continuous` - Use `AppMetrics.cardRadius` with `style: .continuous`.
- [LOW] (possible) lines 43-45 - Section 5 accessibility ("images/galleries need accessible names") - the full-size preview `Image(nsImage:)` has no accessibility label or description - Add `.accessibilityLabel` from the prompt or title.
- [LOW] (possible) lines 52-53 - Section 3 text in layouts - the prompt is `.lineLimit(4)` truncated; it is `.textSelection(.enabled)` but selection cannot reach text past the limit and there is no tooltip with the full value - Add `.help` with the full prompt or let it wrap.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImageGenerationSettings.swift
- [LOW] line 68 - Section 3 headers and rows ("No fixed .frame(width:) on a picker, search field, or label in a header or form row"; use minWidth/idealWidth/maxWidth) - the Seed `TextField` is pinned with `.frame(width: 130)`, so it neither grows with Extra Large text nor shows a long seed value - Use `.frame(minWidth: 130, maxWidth: 220, alignment: .leading)` or let it take available width.
- clean otherwise: real labels above pickers with `.labelsHidden()`, `.frame(minWidth:)` (allowed) on the size picker, `ViewThatFits` fallback row, bundled Text, verbatim numeric picker rows.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImageModelDownloadOffer.swift
- [MEDIUM] (possible) lines 62-67 - Section 2 vocabulary ("An icon-only action | AppIconButton") and Section 5 - the "More download options" `Menu` label is icon-only via `.labelStyle(.iconOnly)` with `.help` but no `.accessibilityLabel`; ImageGalleryCard.swift:66 shows the established spelling - Add `.accessibilityLabel(Text("More download options", bundle: .module))`.
- clean otherwise: `AppMetrics.cardRadius` with `.continuous`, `.appSurface`/`.appBorder`, bundled Text, verbatim only for model data, per-item disabled states mirror installability.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Images/ImagesSidebarPanel.swift
- clean. Uses the shared `SidebarModePanel(title:groupLabel:selection:)` vocabulary; `ImagesWorkspaceMode.title` is rendered through `Text(LocalizedStringKey(title), bundle: .module)` and both values ("Create", "Organize") verified present in `swift/TurboSparkApp/Localization/Localizable.xcstrings`.

### Cross-file notes
- The destructive "Move to Trash" defect appears in two files (ImagesSectionView.swift:107, ImageGalleryCard.swift:125 via ImageGalleryActions, which also feeds the card context menu and the carousel actions menu); one shared confirmation dialog should cover all entry points.
- The icon-only Menu label accessibility drift appears in three places (ImageGalleryCarousel.swift:31, ImageModelDownloadOffer.swift:62, versus the correct spelling at ImageGalleryCard.swift:62-66).
- Off-vocabulary `cornerRadius: 12` appears in ImagesSectionView.swift:228, ImageGalleryCard.swift:39/42, and ImageGalleryCarousel.swift:46; AppMetrics defines controlRadius 7 and cardRadius 10 (Theme/AppControls.swift:17-19).
- No violations found for: system colors, TurboSparkTheme static accessors in view bodies, hard-coded `.appBorder.opacity`, bare font APIs or point sizes, bare `withAnimation`/`.animation`, `repeatForever`, missing `.isSelected` on selected gallery items (ImageGalleryCard.swift:51 has it), or bundleless string literals. All files are under the ~400-line guideline (largest: ImagesSectionView.swift at 277 lines).

---

### Batch batch26_Installation

Audit of 9 SwiftUI files in `swift/TurboSparkApp/Sources/TurboSparkApp/Installation/` against DESIGN.md (all files read in full; catalog-key absences verified against `swift/TurboSparkApp/Localization/Localizable.xcstrings`; `ModelInstallGate`/`ModelProbeGating` and `ModelFamilyVisuals` checked only to confirm source-of-reason findings).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelProbeSheet.swift
- [MEDIUM] line 104, line 253 - Section 2 vocabulary (Progress) - inline `ProgressView().controlSize(.small)` spinners instead of the vocabulary control; the indeterminate form exists (`AppProgressBar(value: nil, ...)` is used elsewhere in this batch). - Use `AppProgressBar` or an existing small live indicator from the vocabulary.
- [MEDIUM] line 115, line 135 - Section 4 Color - probe error text uses `.foregroundStyle(.red)` and the error surface uses `Color.red.opacity(0.1)` background; theme-blind system colors on a themed surface. - Use `AppAttention` styles / theme spellings (`.appSecondary`, AppAttention.color) for the error surface.
- [MEDIUM] line 121 - Section 4 Color - key hint glyph `.foregroundStyle(.orange)`, a system color; warning/attention tints go through `AppAttention.color`. - Replace with `AppAttention.color`.
- [MEDIUM] line 135 - Section 4 Surfaces - hard-coded `cornerRadius: 8` on a non-continuous `RoundedRectangle`; radii come from `AppMetrics` with `style: .continuous`. - Use `AppMetrics.controlRadius`/`cardRadius` and `.continuous`.
- [MEDIUM] line 199 - Section 4 Color - info glyph `.foregroundStyle(.blue)`, theme-blind system color. - Use `.appAccent` or AppAttention styling.
- [MEDIUM] line 268, line 274 - Section 4 Color - caution texts `.foregroundStyle(.orange)`, system color. - `AppAttention.color`.
- [MEDIUM] line 316 - Section 4 Color - ternary `decision.isBlocked ? Color.red : Color.orange` mixes system colors with no `AnyShapeStyle`; ignores a custom theme entirely. - Use `AppAttention.color` (or wrap theme styles in `AnyShapeStyle`).
- [MEDIUM] line 314 - Section 6 Localization (possible) - `Label(reason, systemImage:)` renders a raw String. `ModelProbeGating` funnels into `ModelInstallGate.decide`, where several branches are raw English literals (e.g. "This fits with under ten percent to spare...") while others use `AppStrings.localized`, so the reason can reach this UI unlocalized. Defect site is `ModelInstallGate.swift` (outside this batch); verify when fixing there.
- [LOW] lines 80-96 - Section 2/3 Headers (possible) - sheet header is a bare `HStack` + `Spacer` + Done button; a long German title has nothing that reflows it. `AppAdaptiveHeader` is the header vocabulary for pane/sheet headers. - Consider `AppAdaptiveHeader(leading:actions:)`.
- [LOW] file is 445 lines - Section 3 "Keep view files near 400 lines" - informational only; split by subview if it grows.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelDetailPaneView.swift
- [HIGH] line 242 - Section 4 (DesignTokenRatchetTests) - hard-coded `.appBorder.opacity(0.6)` capsule stroke on capability tags; the ratchet counts `.appBorder.opacity(...)` borders outside `Theme/` and baselines may only go down. - Use `.appBorder` unmodified (theme already raises borders under Increase Contrast).
- [HIGH] line 398 - Section 4 Color - download percent `.foregroundStyle(Color.accentColor)`; `Color.accentColor` ignores a custom theme's accent. - Use `.appAccent`.
- [MEDIUM] lines 200-206 (consumed at line 153) - Section 4 Color - `statusTint` returns `.green`, `.orange`, `.secondary` system colors for the status text. - Resolve from theme/AppAttention; status is already paired with the seal glyph, so keep the glyph.
- [MEDIUM] line 182 - Section 4 Color - caveat seal `.foregroundStyle(.orange)` system color. - `AppAttention.color`.
- [MEDIUM] line 302 - Section 4 Color - ternary `installDecision.isBlocked ? Color.red : Color.orange` without `AnyShapeStyle`, system colors. - `AppAttention.color`.
- [MEDIUM] lines 136, 149, 259 - Section 4 Color - bullet separators use `.foregroundStyle(.tertiary)`; banned theme-blind spelling. - Use `.appSecondary` or a theme-resolved style.
- [LOW] line 124 - Section 4 Surfaces - CATALOG badge uses `.appAccent.opacity(0.16)` background and `cornerRadius: 4`; hard-coded opacity bypasses Increase Contrast and the radius is not an `AppMetrics` value. - Theme opacity + AppMetrics radius (or a badge vocabulary entry if a third copy appears).
- [LOW] lines 231-232 - Section 4 Surfaces - family tag uses `visuals.accentColor.opacity(0.15)` background and `.opacity(0.3)` stroke; the tint itself is a system-palette color resolved in `ModelFamilyVisuals.swift` (`.purple`, `.cyan`, ..., even `Color.accentColor`), so these chips ignore custom themes. - Resolve the family tint through the theme.
- [LOW] lines 110-160 - Section 3 Headers (possible) - hero header is a bare `HStack` + `Spacer` + button; under German/Extra Large the name/status row compresses instead of reflowing. - Consider `AppAdaptiveHeader`.
- Context (not a defect in this file): lines 296-303 and 378 render `installDecision.reason`, whose raw-English branches live in `ModelInstallGate.swift` (see ModelProbeSheet entry).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ImageModelManagerView.swift
- [HIGH] line 112 - Section 4 Color - ternary `model.hasInstalledMLXImageModel ? Color.green : Color.secondary` with no `AnyShapeStyle`; both sides are theme-blind system spellings. - Wrap theme styles in `AnyShapeStyle` / use `.appAccent` + `.appSecondary`.
- [HIGH] line 225 - Section 4 Color - `.background(TurboSparkTheme.surfaceColor, ...)` reads a bare `TurboSparkTheme` static accessor inside a view body; `NSApp.effectiveAppearance` does not follow `.preferredColorScheme`, so this draws the wrong mode (Gotcha 36). - Use the environment theme or the `.appSurface` shape style.
- [HIGH] line 298 - Section 4 Color - `.background(Color.primary.opacity(0.035), ...)` on curated rows; `Color.primary.opacity` is an explicitly banned spelling. - Use `.appSurface` / `.appElevated`.
- [HIGH] line 236 - Section 4 Color - ternary `installed == nil ? Color.accentColor : Color.green` with no `AnyShapeStyle`. - `.appAccent` / theme-resolved success color wrapped in `AnyShapeStyle`.
- [MEDIUM] line 366 - Section 4 Color - `Color.primary.opacity(0.035)` row background repeated in `ImageModelRecommendationSheet`; same banned spelling as line 298. - `.appSurface` / `.appElevated`.
- [MEDIUM] lines 76-132 (with 127-131) - Section 2 vocabulary - `recommendationCard` hand-rolls a card (padding 16 + accent-tinted background + stroke + radius 12) instead of `.appCardSurface(padding:)`; the same treatment is duplicated in `ModelHubView.recommendedHeaderCard` - a second local copy is how drift starts. - Add one shared treatment or use `.appCardSurface`.
- [MEDIUM] lines 127-131 - Section 4 Surfaces - `.appAccent.opacity(0.08)` background and `.opacity(0.2)` stroke; hard-coded opacities bypass `surfaceOpacity`/`isHighContrast`. - Theme-resolved surface and border.
- [MEDIUM] lines 263-267 - Section 6 Localization - `Text(verbatim: "Qwen Research License: research and evaluation use only, no commercial use. Built with Qwen.")` is fixed English UI copy; the key is absent from the catalog (verified). - `Text("...", bundle: .module)` with all 21 translations.
- [MEDIUM] line 253 - Section 4 Color - "Install-tested" tag `.foregroundStyle(.green)`, system color. - Theme-resolved success tint.
- [MEDIUM] line 266 - Section 4 Color - license note `.foregroundStyle(.orange)`, system color. - `AppAttention.color`.
- [MEDIUM] line 294 - Section 2 Action hierarchy - Download button `.disabled(!isTested || !model.canInstallImageModel(alias:))` with no tooltip or adjacent text saying why. - Add `.help(...)` or adjacent reason text (peer sheets in this batch do this).
- [MEDIUM] lines 136-137, 152-153 - Section 5 Accessibility - section headings "Installed image models" / "All curated variants" lack `.accessibilityAddTraits(.isHeader)`. - Add the trait.
- [MEDIUM] line 381 - Section 3 Layout - `ImageModelRecommendationSheet` body ends with `.frame(width: 560)`, a FIXED width sheet frame; sheets resize above a minimum. - Use `minWidth`/`idealWidth`/`maxWidth`.
- [LOW] lines 138-141 - Section 3 Empty states - inline "Nothing is installed..." text instead of `AppEmptyState`. - Use `AppEmptyState` with an action.
- [LOW] lines 205-218 - Section 2 Action hierarchy - unmanaged Remove (files stay on disk, dialog says so) is `.appAction(.destructive)`; per the rule a rebuildable-list removal is a quiet action with a neutral glyph, not destructive red. The managed Delete case is correct.
- [LOW] lines 164, 237 - Section 3 (possible) - `.frame(width: 22)` on row glyphs; they do not scale with Text Size. - Use `minWidth` or size from the font.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelHubView.swift
- [MEDIUM] line 146 - Section 4 Color - "Optimized for Your Mac" uses `.foregroundStyle(.primary)`; `Color.primary` ignores custom theme text colors. - `.appText`.
- [MEDIUM] line 156 - Section 4 Color - RAM text `.foregroundStyle(.tertiary)`; banned theme-blind spelling. - `.appSecondary`.
- [MEDIUM] lines 138-167 - Section 2/4 - `recommendedHeaderCard` hand-rolls the same accent-tinted card (`.appAccent.opacity(0.08)` background, `.opacity(0.2)` stroke, radius 8) as `ImageModelManagerView.recommendationCard`; duplicate local treatment with hard-coded opacities that bypass Increase Contrast. - Share one card treatment / `.appCardSurface`.
- [MEDIUM] line 237 - Section 4 Color - failure glyph `.foregroundStyle(.orange)`, system color. - `AppAttention.color`.
- [LOW] lines 219, 223 - Section 1 One source of truth - empty-state titles drift from `ModelManagerView`: "No models installed" vs "No Models Installed", "No matching models" vs "No Matching Models"; both spellings exist as separate catalog keys (verified). - Reuse one key per phrase.
- Clean otherwise: `AppAdaptiveHeader` used, master list has min/ideal/max widths, `AppEmptyState` used for all three list states and the detail placeholder.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelInstallView.swift
- [HIGH] line 68 - Section 4 Color - LM Studio banner glyph `.foregroundStyle(Color.accentColor)`; ignores a custom theme's accent. - `.appAccent`.
- [HIGH] lines 91-92 - Section 4 Color - banner background `Color.accentColor.opacity(0.1)` and stroke `Color.accentColor.opacity(0.3)`; theme-blind accent with hard-coded opacities. - `.appAccent` with theme-resolved opacity, or `.appAttention`.
- [MEDIUM] line 157 - Section 4 Color - failure glyph `.foregroundStyle(.orange)`, system color. - `AppAttention.color`.
- [MEDIUM] line 175 - Section 4 Color - empty-state glyph `.foregroundStyle(.quaternary)`; theme-blind hierarchical style (same class as `.tertiary`). - `.appSecondary`.
- [MEDIUM] lines 172-184 - Section 3 Empty states - hand-rolled empty state (glyph + text + button) instead of `AppEmptyState`. - Use `AppEmptyState(systemImage:title:message:) { actions }`.
- [MEDIUM] lines 120-121 - Section 5 Accessibility - "Recommended Models for Your Mac" heading lacks `.accessibilityAddTraits(.isHeader)` (the welcome title at lines 103-105 has it). - Add the trait.
- [LOW] line 141 - Section 3 Layout (possible) - `.fixedSize()` on the segmented filter picker in a header row; under German/Extra Large the neighboring title absorbs the squeeze instead of the row reflowing. - Drop `.fixedSize()` or let the row wrap.
- Clean otherwise: `AppProgressBar` for loading (lines 297-304), `InstallDestinationPicker`, `.appAction` everywhere, welcome title carries `.isHeader`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/AudioInstalledModelDetailView.swift
- [MEDIUM] lines 87-93 - Section 2 Action hierarchy ("Destructive means data is lost") - for `entry.group == .local` the Remove button is `.appAction(.destructive)` with a trash glyph even though nothing is deleted (its own dialog says "Its folder stays where it is."); a rebuildable-list removal is a quiet action with a neutral glyph (`minus.circle`), not a red trash can. The real delete case (installed group) is correct, including role + confirmationDialog. - Split the two cases: quiet neutral Remove for local, destructive for delete.
- [MEDIUM] lines 197-198 (row builder lines 206-214) - Section 2 vocabulary (filesystem path) - the Location value is a filesystem path rendered with `themedCode` + middle truncation instead of `SettingsPathText(path, size:)`, and the truncated row value carries no `.help` with the full path (rule: truncate only with the full value in a tooltip). Same for the Revision row. - Use `SettingsPathText`, or add `.help` with the full value.
- [LOW] line 129 - Section 2 Action hierarchy - Adopt button `.disabled(controller.isManagingInstall || controller.isBusy)` with no tooltip or adjacent reason; the sibling Unload/Load buttons in the same switch all carry `.help`. - Add a `.help` explaining why.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelFeatureBadgeView.swift
- [HIGH] line 24 (used at line 262) - Section 4 Color - default `tintColor: Color = .secondary`; `.secondary` is an explicitly banned theme-blind spelling, and `source(.turboSpark)` ships with it. - Default to `.appSecondary`-resolved color or require callers to pass a theme tint.
- [MEDIUM] lines 158, 169, 180, 191, 200, 218, 226, 238, 249, 270, 278, 287-289 - Section 4 Color ("The theme decides color") - badge presets hard-code a system palette (`.indigo`, `.blue`, `.cyan`, `.orange`, `.teal`, `.purple`, `.mint`) plus a custom RGB `emeraldAccent`; none resolve through the theme. (Possible mitigation: categorical badge tints are quasi-data, but they ignore custom themes entirely.) - Route the tints through the theme or add documented token colors.
- [MEDIUM] line 55 - Section 4 Type - `.themedFont(fitting: iconSize, weight: .semibold)` with constant sizes 8/10/11 and no frame dictating the size; `fitting:` is only for glyphs whose size a frame dictates (`size * 0.45`), never a constant. - Use a fixed `AppFontStep` role or give the icon a real frame to fit.
- [MEDIUM] lines 63-70 with 133-147 - Section 4 Surfaces - hard-coded background opacities (0.12/0.15/0.18) and border opacities (0.25/0.35/0.45); "do not hard-code a lower opacity for borders ... than the theme gives", and these never raise under Increase Contrast. - Let the theme resolve surface/border opacity.
- [LOW] lines 125-131 - Section 4 Surfaces - corner radii 4/6/8 are not `AppMetrics` radii. - Use `AppMetrics.controlRadius` or a documented badge radius.
- Clean otherwise: `Text(LocalizedStringKey(...), bundle: .module)` for title/tooltip is the documented variable-string pattern, `.help` + `.accessibilityLabel` agree, `.accessibilityElement(children: .combine)` present.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelManagerView.swift
- [MEDIUM] line 150 - Section 4 Color - group count uses `.foregroundStyle(.tertiary)`; banned theme-blind spelling. - `.appSecondary`.
- [LOW] lines 191, 214 - Section 1 One source of truth - empty-state titles "No Models Installed" / "No Matching Models" drift from `ModelHubView`'s "No models installed" / "No matching models"; both spellings exist as separate catalog keys (verified). - Reuse one key per phrase.
- Clean otherwise: `AppEmptyState` for all three empty conditions with correct actions, group headers carry `.accessibilityAddTraits(.isHeader)`, master list uses min/ideal/max widths, divider uses `AppChromeLayout.dividerWidth`, verbatim count on line 148 is correct, no bare static theme reads.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/SteeringVectorImportSheet.swift
- [HIGH] line 63 - Section 6 Localization - `Text(verbatim: "Add steering for " + installedModel.alias)` is fixed English UI copy; no "Add steering for %@" key exists in the catalog (verified). - `Text("Add steering for \(installedModel.alias)", bundle: .module)` with all 21 translations.
- [HIGH] lines 43-44 (rendered at line 191) - Section 6 Localization - `validationError` returns fixed English strings ("Strength must be a finite number.", "Strength must be between -2 and 2.") that reach the UI through `Label(validationError, ...)` (String overload, no localization); neither key exists in the catalog (verified). - Build with `AppStrings.localized("...")`.
- [HIGH] lines 146-163 with 177 and 181 - Section 6 Localization - `sourceLink` renders titles with `Text(verbatim: title)` and passes `hint` to `.accessibilityHint` as a raw String; "Ready-made GGUF vectors", "Vector generator", "Research axes (.pt)", "All community sources" and all four hints have no catalog keys (verified absent). - Take `Text` parameters (or bundled keys) so both the visible label and the VoiceOver hint localize.
- [MEDIUM] line 217 - Section 2 vocabulary - primary action uses `.buttonStyle(.borderedProminent)` instead of `.appAction(.primary)`; bespoke primary styling duplicating the vocabulary. - `.appAction(.primary)`.
- [MEDIUM] line 88, line 188, line 192 - Section 4 Color - `.foregroundStyle(.red)` (unsupported-family notice, error label) and `.foregroundStyle(.orange)` (validation notice); theme-blind system colors. - `AppAttention` styles.
- [MEDIUM] lines 60-77 - Section 3 Headers (possible) - sheet header is a bare `HStack` + `Spacer` + Done; same truncation risk as `ModelProbeSheet`'s header. - `AppAdaptiveHeader(leading:actions:)`.
- [LOW] lines 71-74 - Section 2 vocabulary - header Done button has no `.appAction` treatment, unlike the Done button in `ModelProbeSheet` (`.appAction(.secondary)`); sibling sheets drift. - `.appAction(.secondary)`.
- [LOW] line 206 - Section 2 vocabulary - `ProgressView().controlSize(.small)` in-button spinner instead of the AppProgressBar vocabulary. - Use the vocabulary control or a shared small live indicator.
- [LOW] line 127 - Section 3 Layout (possible) - `.frame(width: 90)` on the strength TextField in a form row. - Prefer `minWidth`/`maxWidth` so Extra Large text and longer values do not clip.
- [LOW] line 7 - Section 8 Platform constraints (possible) - the struct lacks the explicit `@MainActor` on the type that every sibling view in this folder carries for the macOS 14 SDK rule (only `body` is isolated). - Add `@MainActor` to the struct.

### Cross-file notes
- `ModelInstallGate.swift` (outside this batch) builds `ModelInstallDecision` reasons from a mix of `AppStrings.localized` sentences and raw English literals; the raw branches render unlocalized through `ModelProbeSheet` and `ModelDetailPaneView`. Worth a follow-up batch.
- `ModelFamilyVisuals.swift` (outside this batch) hard-codes system palette tints (`.purple`, `.cyan`, ..., `Color.accentColor`) that leak into `ModelDetailPaneView` chips.

---

### Batch batch27_Installation

Severity: HIGH = rule break with user-visible effect under real conditions (unlocalized fixed copy, unconfirmed destructive action). MEDIUM = concrete design-rule violation (ratchet-counted or checklist-listed theme-blind spellings, hard-coded opacity, vocabulary drift, missing a11y traits). LOW = minor or informational; "(possible)" = judgment call noted.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/AudioInstalledModelsView.swift
- [MEDIUM] line 142 - DESIGN.md 3 (Headers and rows: no fixed frame on a glyph sized with themedFont) - `.frame(width: 30, height: 30)` around a glyph sized `.themedFont(.callout, ...)`; the glyph grows with Text Size while the tile stays 30 pt and clips at Extra Large. The shared `ModelWorkspaceTitle` tile (ModelWorkspaceTitle.swift:15) has the same pattern, so fix both together. - size the tile from the resolved font or multiply the frame by `AppMetrics.textScale(theme)`.
- [LOW] line 97 (possible) - DESIGN.md 2 (vocabulary: AppProgressBar for progress) - bare `ProgressView().controlSize(.small)` as the refreshing placeholder; the vocabulary routes progress (determinate or not) through AppProgressBar. (Possible: a decorative spinner inside an empty-list state, not a progress display.)
- [LOW] line 206 (possible) - DESIGN.md 4 (Color: the theme decides color) - `tint: .teal` fixed hue into `ModelWorkspaceTitle` ignores custom themes; ModelHubView.swift:62 passes `.appAccent` while this file, ModelManagerHeaderView.swift:19 and AudioDiscoverCatalogView.swift:50 pass `.teal`, so the model workspaces drift. - pass `.appAccent` (or one agreed token) at every call site.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/CatalogSheet.swift
- [HIGH] line 146 - DESIGN.md 2 (Action hierarchy: destructive goes through a confirmationDialog) - `Button(role: .destructive) { model.deleteModel(item) }` deletes the model from disk with no confirmationDialog anywhere in this sheet (`AppModel.deleteModel` starts the delete immediately, AppModel+Models.swift:490). InstalledModelDetailPaneView confirms the same action. - add a `.confirmationDialog` with localized title/message/button before calling deleteModel.
- [MEDIUM] line 195 - DESIGN.md 4 (Color: theme spellings; DesignTokenRatchetTests-counted) - `.foregroundStyle(.tertiary)` on the Download/Disk line ignores custom themes. - `.appSecondary` or a theme token.
- [MEDIUM] line 112 - DESIGN.md 4 (Color/Surfaces) - `.background(.quaternary.opacity(0.3), ...)` on the install progress box: system hierarchical style plus hard-coded dimming on a surface. - `.appElevated` / `.appSurface`.
- [MEDIUM] lines 154-155, 213-214 - DESIGN.md 2 (vocabulary: a card is `.appCardSurface(padding:)`) - hand-rolled row cards (`.padding(8)` + `.background(.appSurface, in: RoundedRectangle(cornerRadius: 8))`). - `.appCardSurface(padding: 8)` or a shared row style.
- [LOW] lines 112, 130, 155, 171, 214 - DESIGN.md 2 (Radii: AppMetrics.cardRadius 10, controlRadius 7) - corner radii 10/8/6 literals (also `ModelLogoView(cornerRadius: 6)` at 130/171). - AppMetrics radii.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelDetailPaneView.swift
- clean. Both destructive actions present confirmationDialogs; dialog strings are pre-localized via `ModelDeleteDialog`/`AppStrings` and `Text(verbatim:)` is correct for them; Cancel uses `Text(..., bundle: .module)`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelSteeringCardView.swift
- [HIGH] line 132 - DESIGN.md 6 (nothing to translate means verbatim; fixed English copy must be a key) - `Text(verbatim: "or")` is fixed English UI copy; the key "or" does not exist in Localizable.xcstrings, so it renders English under all 21 languages. - `Text("or", bundle: .module)`.
- [HIGH] line 220 - DESIGN.md 6 (ternary of bundled Texts; fixed English held in variables is catalog-keyed) - `Text(verbatim: isEnabled ? "Disable" : isSelected ? "Enable" : "Use")` renders fixed English words verbatim; keys "Disable" and "Enable" are absent from the catalog ("Use" exists but is bypassed by verbatim). - `isEnabled ? Text("Disable", bundle: .module) : isSelected ? Text("Enable", bundle: .module) : Text("Use", bundle: .module)`.
- [HIGH] line 154 - DESIGN.md 6 (interpolation becomes a format key) - `Text(verbatim: "Enabled: " + currentPreset.displayName)`: fixed English "Enabled: " prefix; no "Enabled: %@" key exists. - `Text("Enabled: %@", bundle: .module)` with the display name as the (verbatim) argument.
- [HIGH] line 203 - DESIGN.md 6 (fixed English join must be localized) - `Text(verbatim: preset.mode.menuLabel + " at " + String(format: "%.2g", preset.scale))`: the " at " join is fixed English; no "%@ at %@" key exists (possible: `menuLabel` itself may or may not be a catalog word). - build with a localized format key and the two values as arguments.
- [MEDIUM] line 52 - DESIGN.md 4 (Color; ratchet-counted spelling) - `.foregroundStyle(.secondary)` on the unsupported-family notice. - `.appSecondary`.
- [MEDIUM] line 133 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(.tertiary)` on the "or" separator. - `.appSecondary`.
- [MEDIUM] line 197 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(isEnabled ? Color.green : Color.secondary)` on the preset toggle glyph: `Color.secondary` is a counted spelling; green is a system color on a themed surface. - `.appSecondary` plus a theme/accent token for the on state.
- [MEDIUM] line 206 - DESIGN.md 4 (Color; checklist: warning/attention tints via AppAttention.color) - `.foregroundStyle(compatibility.allowsEnabling ? Color.secondary : Color.red)`: theme-blind secondary plus system red for the "would be refused at load" state (the summary text is present, so it is not color alone). - `.appSecondary` / `AppAttention.color`.
- [MEDIUM] lines 180-182 - DESIGN.md 2 (vocabulary: an action is `.appAction`) - `.buttonStyle(.bordered).controlSize(.small).tint(.orange)` bespoke button duplicating `.appAction(.secondary, size: .small)` with a system orange tint. - `.appAction(.secondary, size: .small)` + `AppAttention.color` for the attention tint.
- [MEDIUM] lines 222-223 - DESIGN.md 2 (vocabulary) - `.buttonStyle(.bordered).controlSize(.small)` on the preset row button: a second bespoke treatment in the same card. - `.appAction(.secondary, size: .small)`.
- [MEDIUM] lines 70-76 - DESIGN.md 5 (section headings carry .isHeader) - the card heading `Label("Steering")` has no `.accessibilityAddTraits(.isHeader)`, so VoiceOver heading navigation skips it (vocabulary headings add the trait themselves; a bare Label does not). - add the trait.
- [LOW] lines 83, 140-141 (possible) - DESIGN.md 2 (vocabulary) - `.buttonStyle(.link)` in-row actions twice in one card; borderline against `.appAction(.quiet)`; fine if a link treatment is intended, but it is a second local variant.
- [LOW] lines 205, 227 (possible) - DESIGN.md 6 (a String that reaches the UI uses AppStrings.localized) - `Text(compatibility.summary)` and `.help(compatibility.summary)` render `AppSteeringPolicy.Compatibility.summary`, which is hard-coded English built in State code (AppSteeringPolicy.swift:81), so every language reads English; the fix belongs in AppSteeringPolicy with AppStrings.localized.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelRowView.swift
- [MEDIUM] line 47 - DESIGN.md 4 (Color: SwiftUI spellings count too) - `.foregroundStyle(rowModel.isDimmed ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.primary))`: AnyShapeStyle is used correctly, but `.primary` still ignores a custom theme's text color. - `.appText`.
- [MEDIUM] line 113 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(.tertiary)` on the size glyph. - `.appSecondary`.
- [MEDIUM] lines 155-160 - DESIGN.md 4 (Color) - selection background `Color.teal.opacity(0.14)`: system teal; the theme exposes `theme.selection` (AudioInstalledModelsView.swift:163 uses it), so the two model lists draw selection differently. - `theme.selection`.
- [MEDIUM] lines 161-167 - DESIGN.md 4 (Color; hard-coded opacity borders) - overlay stroke `isSelected ? Color.teal.opacity(0.4) : Color.clear`: theme-blind color and hard-coded border opacity. - `.appBorder` / theme.selection.
- [MEDIUM] lines 143-144 - DESIGN.md 4 (Color) - tag pills: `Color.teal.opacity(0.12)` background and `Color.teal` foreground system colors on themed surfaces. - theme token or the family visuals.
- [MEDIUM] line 53 - DESIGN.md 4 (Color; checklist lists `.yellow` as theme-blind) - `.foregroundStyle(Color.yellow)` favorite star. - one agreed accent token; keep consistent with InstalledModelHeroHeaderView.swift:37 which also uses yellow.
- [LOW] line 195 - DESIGN.md 2 (text-holding frames multiply by AppMetrics.textScale) - `.frame(maxWidth: 120)` on the library badge is not scaled, so a long translated library name keeps a 120 pt box at Extra Large. (maxWidth is the allowed direction; the missing textScale is the issue.) - multiply by `AppMetrics.textScale(theme)`.
- [LOW] lines 152, 159, 162 - DESIGN.md 2 (Radii) - `RoundedRectangle(cornerRadius: 8)` for the row shape; `AppMetrics.controlRadius` is 7. - AppMetrics radius.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ProbeReportCardView.swift
- [MEDIUM] line 157 - DESIGN.md 4 (Color; ratchet-counted `.secondary`; checklist: warning tints via AppAttention.color) - `.foregroundStyle(t.bytes == nil ? Color.orange : .secondary)` on the UNSIZED column. - `AppAttention.color` / `.appSecondary`.
- [MEDIUM] line 180 - DESIGN.md 4 (Color) - `.foregroundStyle(Color.orange)` on missing sidecars. - `AppAttention.color`.
- [MEDIUM] lines 16, 23, 206-217 - DESIGN.md 2/4 (attention surface vocabulary; system colors + hard-coded dimming) - `calloutBox(tone:)` draws system `.red`/`.orange` foreground plus `tone.opacity(0.1)` dimmed background and cornerRadius 8; the vocabulary for a surface that needs attention is `.appAttention` with an `AppAttentionNote`. (Glyph plus text is present, so it is not color alone; the tint spellings and hand-rolled box are the defect.)
- [MEDIUM] lines 44, 161 - DESIGN.md 4 (Color; checklist lists `.red`) - `Color.green : Color.red` verdict tinting on "Would (not) run here" and "has/no kernels" (both paired with glyph/text). - theme/status tokens; red per the destructive convention only.
- [LOW] lines 146, 150, 158, 198 - DESIGN.md 2 (text-holding frames multiply by textScale) - fixed widths 74/90/84/118 on text-holding cells and the specRow label column are not scaled by `AppMetrics.textScale(theme)`; they clip at Extra Large. - scale the widths.
- [LOW] line 27 - DESIGN.md 2 (Radii) - `RoundedRectangle(cornerRadius: 10)` on the card. - `AppMetrics.cardRadius`.
- [LOW] lines 87, 139 (possible) - DESIGN.md 5 (headings carry .isHeader) - the "Memory at ... tokens" and "Block types" subsection headings carry no `.accessibilityAddTraits(.isHeader)` (possible: small report rows).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelCardView.swift
- [MEDIUM] line 49 - DESIGN.md 4 (Color; ratchet-counted `Color.accentColor`) - `.foregroundStyle(Color.accentColor)` on the download percent. - `.appAccent`.
- [MEDIUM] lines 62-63 - DESIGN.md 4 (Color; ratchet-counted) - Active pill: `Color.accentColor.opacity(0.18)` background and `Color.accentColor` foreground. - `.appAccent`.
- [MEDIUM] lines 116-127 - DESIGN.md 4 (Color; hard-coded opacity borders) - selection background `Color.accentColor.opacity(0.14)` and overlay stroke `.opacity(0.4)`: theme-blind accent and hard-coded border opacity; also disagrees with InstalledModelRowView, which selects with teal. - `theme.selection` / `.appBorder`.
- [MEDIUM] line 107 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(.tertiary)` on the download size. - `.appSecondary`.
- [MEDIUM] line 192 - DESIGN.md 4 (Color; checklist: warning tints via AppAttention.color) - `.foregroundStyle(.orange)` on the caveat seal (glyph and help text present, so not color alone). - `AppAttention.color`.
- [LOW] line 45 (possible) - DESIGN.md 2 (vocabulary: AppProgressBar for progress) - `ProgressView().controlSize(.mini)` inline spinner beside the percent text (possible: tiny row indicator; the percentage carries the value).
- [LOW] line 66 - DESIGN.md 4 (Color) - `Circle().fill(.green)` status dot (paired with the "Installed" word, so not color alone).
- [LOW] lines 113, 120, 123 - DESIGN.md 2 (Radii) - `RoundedRectangle(cornerRadius: 10)` literals. - `AppMetrics.cardRadius`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelInstallCards.swift
- [MEDIUM] line 63 - DESIGN.md 4 (DesignTokenRatchetTests explicitly counts hard-coded `.appBorder.opacity(`) - `.stroke(.appBorder.opacity(0.6), lineWidth: 0.5)` on HardwareScanCard. - stroke `.appBorder` as the theme gives it.
- [MEDIUM] lines 94, 106-107, 125-133, 137, 145 - DESIGN.md 4 (Color: system colors and hard-coded dimming on themed surfaces) - MoESpotlightCard MoE branding: `Color.purple` foreground (94, 145), pill `Color.purple.opacity(0.2)` (106-107), card wash `LinearGradient(colors: [Color.purple.opacity(0.12), Color.clear], ...)` (125-133), border `Color.purple.opacity(0.3)` (137). - theme tokens or one brand token from the theme.
- [MEDIUM] line 153 - DESIGN.md 4 (Surfaces: do not hard-code opacity under the theme's surfaceOpacity) - `.background(.appPage.opacity(0.7), ...)` on feature pills bypasses Reduce Transparency/surfaceOpacity handling. - `.appElevated` / `.appSurface`.
- [MEDIUM] lines 200-205 - DESIGN.md 4 (Color; ratchet-counted accentColor; hard-coded opacity border) - ActiveInstallCard: `Color.accentColor.opacity(0.1)` background and `.stroke(Color.accentColor.opacity(0.35), ...)`. - `.appAccent` / `.appBorder`.
- [MEDIUM] lines 56-64, 123-138, 200-205 - DESIGN.md 2 (vocabulary: a card is `.appCardSurface(padding:)`) - all three cards hand-roll padding + background + overlay instead of the shared card surface. - `.appCardSurface(padding: 16)` plus overlay only where a border is needed.
- [LOW] lines 24-25 (possible) - DESIGN.md 4 (Color) - "Detected" pill `Color.green.opacity(0.18)` + `.green` (possible: the word is present, so not color alone).
- [LOW] line 165 (possible) - DESIGN.md 2 (vocabulary) - `ProgressView().controlSize(.small)` beside the stage text while the real progress below already is an AppProgressBar.
- [LOW] lines 59, 133, 201 - DESIGN.md 2 (Radii) - cornerRadius 14 literals; `AppMetrics.cardRadius` is 10. - AppMetrics radius.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelActionBarView.swift
- [MEDIUM] lines 122-125 - DESIGN.md 2 (Action hierarchy: destructive means data is lost; removing a rebuildable list entry is a quiet action, not destructive red) - "Remove from list" uses `Button(role: .destructive)` + `.appAction(.destructive)` (red) though its own help text says files are not deleted. The confirmation dialog does exist in InstalledModelDetailPaneView (lines 201-212), so this is the role/styling defect, not a missing dialog. - keep the `minus.circle` glyph and use `.appAction(.secondary)` without the destructive role.
- [LOW] line 181 (possible) - DESIGN.md 4 (checklist: warning/attention tints via AppAttention.color) - warning glyph `AnyShapeStyle(Color.orange)`: correctly AnyShapeStyle-wrapped and paired with text; `AppAttention.color` is the sanctioned token.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelRecommendationRow.swift
- [MEDIUM] lines 131-195 - DESIGN.md 1/2 (one source of truth; a second local copy is how drift starts) - `RecommendationVerdictBadge` is a fourth hand-written restatement of the verdict presentation that ModelFitPresentation.swift exists to retire (its doc: "THIS EXISTS BECAUSE THE MAPPING WAS WRITTEN OUT THREE TIMES"); it duplicates `ModelFitVerdictPill` with different labels ("Fits Memory"/"Streams Fast"/"Tight Fit"/"Too Large" vs the pill's "Full Resident Fit"/"Resident"/...) and a different shape (RoundedRectangle radius 4 vs Capsule). - use `ModelFitVerdictPill(verdict:)`.
- [MEDIUM] lines 152-153, 165-166, 177-178, 189-190 - DESIGN.md 4 (Color: system colors plus hard-coded dimmed pill backgrounds) - verdict pills: `Color.green/.blue/.orange/.red` foregrounds with `Color.X.opacity(0.18)` backgrounds. - resolved by adopting the shared pill; otherwise theme tokens.
- [MEDIUM] lines 43-44 - DESIGN.md 4 (Color) - MoE pill `Color.purple.opacity(0.18)` background and `Color.purple` foreground. - theme/brand token.
- [MEDIUM] line 116 - DESIGN.md 4 (hard-coded opacity borders) - overlay stroke `AnyShapeStyle(Color.purple.opacity(0.3))` for MoE rows (the AnyShapeStyle itself is correct). - `.appBorder` or a theme token.
- [MEDIUM] lines 73, 87 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(.tertiary)` on the "-" separators. - `.appSecondary`.
- [LOW] line 111 - DESIGN.md 2 (Radii) - `RoundedRectangle(cornerRadius: 12)`. - `AppMetrics.cardRadius`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelHeroHeaderView.swift
- [HIGH] lines 40-41 - DESIGN.md 6 (bundled spellings; String overloads resolve against Bundle.main) - favorite button `.help(isFavorite ? "Remove from Favorites" : "Mark as Favorite")` and `.accessibilityLabel(isFavorite ? "Favorited" : "Favorite")` pass bare String literals through the String overloads, so they render English under every language; keys "Remove from Favorites", "Mark as Favorite" and "Favorited" do not exist in Localizable.xcstrings ("Favorite" does). - ternaries of `Text(..., bundle: .module)` for both.
- [MEDIUM] lines 34-41 - DESIGN.md 2 (an icon-only action uses AppIconButton) - the favorite star is a hand-drawn icon-only `Button` with `.buttonStyle(.plain)`; AppIconButton makes the label both tooltip and VoiceOver name so the two cannot drift. - `AppIconButton`.
- [MEDIUM] line 37 - DESIGN.md 4 (Color; ratchet-counted `Color.secondary`; checklist lists `.yellow`) - `.foregroundStyle(isFavorite ? Color.yellow : Color.secondary)`. - theme tokens for both states.
- [MEDIUM] lines 31-32 - DESIGN.md 4 (Color) - "INSTALLED" badge: `Color.teal.opacity(0.16)` background and `Color.teal` foreground. - theme token.
- [MEDIUM] lines 158, 164-165 - DESIGN.md 4 (Color) - loaded badge: `Circle().fill(Color.green)`, `Color.green.opacity(0.16)` background, `Color.green` foreground. - theme/status tokens (word present, so not color alone).
- [MEDIUM] lines 168, 174-175 - DESIGN.md 4 (Color; ratchet-counted accentColor) - selected badge: `Circle().fill(Color.accentColor)`, `Color.accentColor.opacity(0.14)` background, `Color.accentColor` foreground. - `.appAccent`.
- [MEDIUM] line 178 - DESIGN.md 4 (Color; ratchet-counted) - `Circle().fill(Color.secondary)` on the "Ready to Load" badge. - `.appSecondary`.
- [MEDIUM] lines 60, 68, 106 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(.tertiary)` on the bullet separators and on the plain repo text. - `.appSecondary`.
- [MEDIUM] line 96 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(Color.accentColor)` on the Hugging Face link. - `.appAccent`.
- [LOW] lines 18-46 (possible) - DESIGN.md 3 (a header never truncates) - the hero title row is a bare `HStack` with trailing badges and the title carries `.lineLimit(2)` (line 25), so a long translation loses the tail at narrow widths; not a pane header band, hence possible. - consider AppAdaptiveHeader or let the title wrap.
- [LOW] lines 23-25 (possible) - DESIGN.md 5 (headings carry .isHeader) - the pane's title `Text(installedModel.alias)` at `.title2` carries no `.accessibilityAddTraits(.isHeader)`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelLogoView.swift
- [MEDIUM] line 61 - DESIGN.md 4 (Color; ratchet-counted `Color.primary.opacity`) - `.stroke(Color.primary.opacity(0.12), lineWidth: 0.5)` on the logo tile. - `.appBorder`.
- [LOW] line 17 (possible) - DESIGN.md 4 (Color) - default `accentColor: Color = .accentColor` on the first initializer; callers using it draw the system accent over a custom theme (the `visuals:` initializer overrides it). - default to a theme token or require callers to pass one.
- Note: `.themedFont(fitting: size * 0.45)` (lines 111, 115) is the sanctioned form; `.appSurface` fill and `.accessibilityHidden(true)` are correct.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelHardwareFitCardView.swift
- [MEDIUM] line 32 - DESIGN.md 4 (Color; ratchet-counted `Color.accentColor`) - `.foregroundStyle(Color.accentColor)` on the gauge glyph in the sized card. - `.appAccent`.
- [MEDIUM] lines 34-35, 96-97 - DESIGN.md 5 (section headings carry .isHeader) - both "Hardware Compatibility" headings have no `.accessibilityAddTraits(.isHeader)`; the unprobed card sets an accessibilityLabel on the whole card but heading navigation misses the title. - add the trait.
- [MEDIUM] line 105 - DESIGN.md 4 (Color; ratchet-counted `Color.secondary` plus hard-coded dimming) - "Not sized" pill `.background(Color.secondary.opacity(0.16), in: Capsule())`. - `.appElevated` / `.appSurface`.
- [LOW] line 127 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(.tertiary)` on the probe command hint. - `.appSecondary`.
- Note (out-of-batch definition, affects this file and ModelSteeringCardView): the shared `modelCardStyle()` is defined in ModelTechnicalSpecsCardView.swift:92 and hard-codes `.appBorder.opacity(0.5)` and cornerRadius 12 - one of the ratchet-counted `.appBorder.opacity` sites lives there.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelManagerFilterBarView.swift
- [MEDIUM] line 114 - DESIGN.md 4 (Color; ratchet-counted) - `.foregroundStyle(.tertiary)` on the "Tags:" label. - `.appSecondary`.
- otherwise clean: `AppSearchField` with bundled prompt, shared `ModelFilterMenu` dropdowns, `AppChip` tag chips, FlowLayout wrap for long translations, enum rawValues resolved through `Text(LocalizedStringKey(...), bundle: .module)`, and the icon-only favorites toggle keeps a real (hidden) Label plus `.help`.

---

### Batch batch28_Installation

Audit against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md (read in full). All 13 files read end to end. Localization spot-check: 122 keys used by these files (badge titles/tooltips, verdict labels, filter labels, summary keys, format keys) all exist in Localizable.xcstrings; no bundleless literals and no Text(verbatim:) on fixed English copy were found. No motion violations (no bare withAnimation/.animation/repeatForever), no file exceeds ~400 lines.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ContextLadderView.swift
- [HIGH] line 78 - Theming (section 4, DesignTokenRatchet) - `Color.accentColor.opacity(0.10)` highlight background on the current rung ignores custom themes and is a ratchet-tracked spelling - use `.appAccent.opacity(0.10)`.
- [MEDIUM] lines 57, 60 - Layout (section 2, text-holding frames) - fixed `.frame(width: 68)` / `.frame(width: 78)` on themedFont-sized text columns do not grow with Text Size, so "131,072" can clip at Extra Large - multiply the widths by `AppMetrics.textScale(theme)` (needs `@Environment(\.appTheme)`), as AudioDiscoverCatalogView does for its list column.
- [MEDIUM] lines 119-125 - Theming (section 4) + vocabulary - system `Color.green.opacity(0.16)` background and `.foregroundStyle(.green)` for the "measured" capsule are theme-blind; the same capsule is hand-copied in InstalledModelMemoryFitCardView lines 80-87, so it is also a second local copy of one treatment - extract a shared "measured" badge (or theme-resolved token) and use it in both files.
- [MEDIUM] line 23 - Accessibility (section 5, section 2 heading rule) - card heading built from a `Label` ("Memory by context window") lacks `.accessibilityAddTraits(.isHeader)`; headings built from Label/Text must add it themselves - add the trait.
- [LOW] line 79 - Radii (section 4) - `RoundedRectangle(cornerRadius: 4)` not from AppMetrics - use an AppMetrics constant or a shared chip radius.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelHubFilterBarView.swift
- [LOW] line 112 - Action hierarchy (section 2, "a disabled control says why") - `filterMenu` is disabled when the facet has no options, with no tooltip or adjacent text saying why - add a `.help` such as "No options in the catalog for this filter" (localized).
- Otherwise clean: AppSearchField, ModelFilterMenu, `.appAction(.quiet)`, bundled Text throughout, wrapping FlowLayout, real Picker labels hidden correctly.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelMemoryFitCardView.swift
- [HIGH] line 76 - Theming (section 4, DesignTokenRatchet) - `.foregroundStyle(Color.accentColor)` on the "Allocates" figure ignores the custom theme accent - use `.appAccent`.
- [MEDIUM] lines 80-87 - Theming (section 4) + vocabulary - system `Color.green` capsule for "measured", duplicating ContextLadderView lines 119-125 - share one token/badge treatment between the two files.
- [MEDIUM] lines 57-58 - Vocabulary (section 2) - hand-rolled card (`.padding(14)` + `.appSurface` + `RoundedRectangle(cornerRadius: 10)`) instead of `.appCardSurface(padding: 14)`, which also adds the theme hairline, continuous corners, and Increase Contrast behavior - replace.
- [MEDIUM] line 16 - Accessibility (section 5) - heading Label ("Unified Memory & Working Set") lacks `.accessibilityAddTraits(.isHeader)` - add the trait.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelFeatureBadgesView.swift
- [MEDIUM] lines 50, 61, 83, 93, 104, 118 - Theming (section 4) - fixed system-color badge tints (`.teal`, `.indigo`, `.orange`, `.secondary`, `.pink`) ignore custom themes; `.secondary` is a spelling DESIGN.md names explicitly - route badge tints through a shared palette or theme token (possible: categorical hues may be deliberate, but they bypass theming).
- [MEDIUM] lines 128-129 - Vocabulary (section 2) - hand-rolled card instead of `.appCardSurface(padding: 14)` - replace.
- [MEDIUM] line 10 - Accessibility (section 5) - heading Text ("Model Highlights & Features") lacks `.accessibilityAddTraits(.isHeader)` - add the trait.
- Localization is correct: title/tooltip literals are catalog keys (verified present) rendered through `Text(LocalizedStringKey, bundle: .module)` by ModelFeatureBadgeView.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelFitPresentation.swift
- [MEDIUM] lines 37-45 - Theming (section 4) (possible) - `ModelFitPresentation.color` returns system `.green/.blue/.orange/.red/.secondary`; `.secondary` is an explicitly listed spelling, and the others ignore a custom theme's palette on themed surfaces. Status is paired with glyph and text, so not color-alone - resolve through the theme or shared status tokens.
- Strings are correct: labels/helps are catalog keys (verified) and are only rendered through `Text(LocalizedStringKey, bundle: .module)` at the call sites.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelOrganizationCardView.swift
- [HIGH] lines 37-38, 62 - Theming (section 4, DesignTokenRatchet) - `Color.accentColor.opacity(0.14)` tag background, `.foregroundStyle(Color.accentColor)`, and the accent `.stroke(Color.accentColor, lineWidth: 0.5)` ignore the custom theme accent - use `.appAccent` in all three places.
- [MEDIUM] line 48 - Layout (section 3, no fixed widths in form rows) - `.frame(width: 80)` on the new-tag TextField clips longer tag names and ignores Text Size - use `minWidth`/`idealWidth` or scale by `AppMetrics.textScale(theme)`.
- [MEDIUM] lines 103-104 - Vocabulary (section 2) - hand-rolled card instead of `.appCardSurface(padding: 14)` - replace.
- [MEDIUM] line 15 - Accessibility (section 5) - heading Label ("Organization & Notes") lacks `.accessibilityAddTraits(.isHeader)`; the sub-headings "Custom Tags" (line 20) and "Personal Notes" (line 83) are also bare Text without the trait - add it.
- [LOW] lines 37, 61, 73 - Radii (section 4) - `cornerRadius: 4` chips not from AppMetrics - use a shared chip radius.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/AudioDiscoverCatalogView.swift
- [LOW] line 50 - Theming (section 4) (possible) - `tint: .teal` fixes a system hue for the workspace icon tile and badge; ModelWorkspaceTitle's doc acknowledges a fixed hue, but it bypasses custom themes - prefer `.appAccent` or a theme-resolved hue.
- [LOW] line 100 - Vocabulary (section 2, "Progress, determinate or not") - `ProgressView()` for the refreshing empty state instead of `AppProgressBar` - use `AppProgressBar` (indeterminate).
- [LOW] line 59 - Action hierarchy (section 2) (possible) - Rescan is disabled while refreshing with no tooltip saying why - add a `.help` (localized) or rely on AppProgressBar making the state legible.
- Otherwise clean: list column widths scale with textScale, AppAdaptiveHeader, AppEmptyState with action, section headers carry `.isHeader` (line 84).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelTechnicalSpecsCardView.swift
- [HIGH] line 48 - Theming (section 4, DesignTokenRatchet) - `.foregroundStyle(Color.accentColor)` on the install-path button ignores the custom theme accent - use `.appAccent`.
- [MEDIUM] line 96 - Surfaces (section 4, DesignTokenRatchet baseline 18) - hard-coded `.appBorder.opacity(0.5)` stroke defeats the theme's border (and its Increase Contrast raise) - stroke `.appBorder` as the theme gives it.
- [MEDIUM] lines 41-54 - Vocabulary (section 2, filesystem path in a row) - bespoke path button instead of `SettingsPathText`; it has code font, middle truncation, and the full path in `.help`, but the text is not selectable - use SettingsPathText (embedding it in the Reveal button's label) or add `.textSelection(.enabled)`.
- [MEDIUM] lines 92-97 - Vocabulary (section 2) - local `modelCardStyle()` duplicates `.appCardSurface` with an off-vocabulary radius 12 and the hard-coded border opacity - use `.appCardSurface(padding: 16)` and delete the extension.
- [LOW] lines 95-96 - Radii (section 4) - `cornerRadius: 12` not from AppMetrics (`cardRadius` is 10) - use `AppMetrics.cardRadius` via `.appCardSurface`.
- [MEDIUM] lines 26, 79 - Accessibility (section 5) - headings "Technical Overview" (line 26) and "Benchmark & Architecture Notes" (line 79) lack `.accessibilityAddTraits(.isHeader)` - add the trait.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelManagerHeaderView.swift
- [LOW] line 19 - Theming (section 4) (possible) - `tint: .teal` fixed system hue for the icon tile/badge, bypassing custom themes - prefer `.appAccent` or a theme-resolved hue.
- Otherwise clean: AppAdaptiveHeader with compactActions folding into a labeled More menu, one primary action, `.appAction(.secondary)` on the rest, localized summary built from catalog keys with Int/String interpolation, `AppStrings.localized` for the toast.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelTechnicalSpecsView.swift
- [HIGH] line 60 - Theming (section 4, DesignTokenRatchet) - `.foregroundStyle(Color.accentColor)` on the disk-path button ignores the custom theme accent - use `.appAccent`.
- [MEDIUM] lines 54-66 - Vocabulary (section 2, filesystem path in a row) - bespoke path display instead of `SettingsPathText`: `lineLimit(2)` with tail truncation (not `lineLimit(1)` + middle truncation) and no `.textSelection` - use SettingsPathText in the button label or add selection and middle truncation.
- [MEDIUM] lines 76-77 - Vocabulary (section 2) - hand-rolled card instead of `.appCardSurface(padding: 14)` - replace.
- [MEDIUM] line 12 - Accessibility (section 5) - heading Label ("Technical Specifications") lacks `.accessibilityAddTraits(.isHeader)` - add the trait.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledModelDeveloperCommandsView.swift
- [MEDIUM] lines 46-47 - Vocabulary (section 2) - hand-rolled card instead of `.appCardSurface(padding: 14)` - replace.
- [MEDIUM] line 33 - Accessibility (section 5) - heading Label ("Terminal / CLI Commands") lacks `.accessibilityAddTraits(.isHeader)` - add the trait.
- [LOW] lines 62-68 - Vocabulary (section 2, icon-only action) - copy action built as `Label(...).labelStyle(.iconOnly)` instead of `AppIconButton`, which unifies tooltip and VoiceOver name; `.help` is present so both exist, but the vocabulary is AppIconButton - use `AppIconButton("doc.on.doc", label: Text("Copy command", bundle: .module))`.
- [LOW] line 71 - Radii (section 4) - `cornerRadius: 4` snippet surface not from AppMetrics - use a shared chip radius.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelWorkspaceTitle.swift
- [MEDIUM] lines 23-25 - Layout (section 3, no fixed frame on glyphs sized with themedFont) - the icon tile `.frame(width: 30, height: 30)` stays fixed while the contained `Image` grows with `.themedFont(.callout)` under Extra Large text, so the glyph can overflow the tile - scale the tile (and badge paddings if needed) by `AppMetrics.textScale(theme)` (the view needs the theme environment, or callers pass a scaled size).
- [LOW] line 66 - Text in layouts (section 3) (possible) - `ModelFilterMenu` label `value.lineLimit(1)` truncates the current value while `.help(name)` carries only the filter's name, not the active value - include the value in the tooltip or let it wrap where space allows.
- [LOW] lines 23, 41 - Radii (section 4) - `cornerRadius: 8` tile and `cornerRadius: 4` badge not from AppMetrics - use AppMetrics constants or a shared chip radius.
- ModelFilterMenu's accessibility (`.help(name)`, `.accessibilityLabel(name)`, `.accessibilityValue(value)`) and the title's `.accessibilityAddTraits(.isHeader)` (line 36) are correct.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Installation/InstalledSidebarPanel.swift
- clean (SidebarModePanel used as the vocabulary prescribes; enum titles rendered through `Text(LocalizedStringKey, bundle: .module)` with keys present).

---

### Batch batch30_Server

Audited against /Users/whit3rabbit/Documents/GitHub/turbospark/DESIGN.md (sections 2-8).
Localization keys for every interpolated `Text(_, bundle: .module)` and
`AppStrings.localized` call in these files were verified present in
`swift/TurboSparkApp/Localization/Localizable.xcstrings`; no missing-key
defects are reported. Severity notes: sites matched by
`DesignTokenRatchetTests` rules are marked "(ratchet-counted)".

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerConsoleView.swift
- [MEDIUM] line 72 - section 4 Surfaces ("do not hard-code a lower opacity ... than the theme gives") - `.background(.appElevated.opacity(0.4))` hard-codes dimming on the themed surface, bypassing `surfaceOpacity` (Reduce Transparency / Increase Contrast cannot raise it) - use `.background(.appElevated)` or an existing surface style.
- [MEDIUM] lines 50-57 - section 2 vocabulary ("An empty list or workspace | AppEmptyState") - the console's empty state is a bare `Text("No traffic yet."/"Nothing matches.")` centered in a `Group`, not `AppEmptyState(systemImage:title:message:)` - replace with `AppEmptyState`.
- [MEDIUM] line 120 - section 4 Color (system colors on themed surfaces) - `.foregroundStyle(.orange)` on the dropped-events warning label ignores custom themes - use `AppAttentionNote`/`AppAttention.color` tint (the warning already pairs glyph + text, good).
- [MEDIUM] line 148 - section 4 Color (ratchet-counted: `\(\.secondary\)`) - `.foregroundStyle(copiedAll ? .green : .secondary)` uses system green and system secondary in a bare ternary - use `.appSecondary` for the idle state and a theme-resolved or AppAttention color for the copied state.
- [MEDIUM] lines 212-227 - section 2 vocabulary ("An icon-only action | AppIconButton") - row copy button is a `Button` + `Label.labelStyle(.iconOnly)` + `.buttonStyle(.borderless)` + `.help`, a fourth local copy of the icon-only control (same shape again at lines 124-153, 328-338, 369-379) - use `AppIconButton`, which makes tooltip and VoiceOver name one value (the label ternary at 216-218 and the `.help` ternary at 227 would collapse with it).
- [MEDIUM] line 170 - section 3 ("No fixed .frame(width:) ... on a label") - method column `.frame(width: 38)` on `theme.code(.small)` text clips "DELETE" (6 chars) even at Default size - use `minWidth`/no fixed width, or widen; the status column (line 165, width 34) and duration column (line 209, width 62) have the same pattern and clip at Extra Large (LOW for those two).
- [MEDIUM] lines 232-238 - section 5 ("Every action is reachable by keyboard"; "Hover-only actions are also exposed with .accessibilityAction") - expanding a request row is `.onTapGesture` only: no Button trait, no `.accessibilityAction`, not keyboard reachable; the row is not announced as expandable - add `.accessibilityAddTraits(.isButton)`, an `.accessibilityAction` for expand/collapse, and `.accessibilityValue` for the expanded state.
- [MEDIUM] lines 319, 323, 345 - section 4 Color - `.foregroundStyle(.red)` / `.red.opacity(0.95)` for the error box heading and body ignore custom themes and hard-code opacity - keep red (destructive/error convention, and a glyph is present) but resolve it from the theme's error/attention color rather than the system red.
- [MEDIUM] lines 349, 358, 392 - section 2/4 (hand-rolled attention surface; `AppMetrics` radii) - the error box draws `Color.red.opacity(0.08)`/`.opacity(0.04)` backgrounds with `.cornerRadius(4)`/`.cornerRadius(6)`, and the raw-events box uses `.appSurface.opacity(0.4)` with `.cornerRadius(4)` - none of these radii is an `AppMetrics` value (controlRadius 7, cardRadius 10); the opacities bypass the theme - build these boxes on the shared card/attention vocabulary.
- [MEDIUM] lines 418-423 - section 4 Color (ratchet-counted pattern family) - `statusColor` returns `.secondary`, `.red`, `.orange`, `.green` system colors that ignore a custom theme's palette - resolve from the theme (or AppAttention) and keep pairing with the numeric status text.
- [MEDIUM] line 222 - section 4 Color (ratchet-counted: `Color\.secondary`) - `.foregroundStyle(copiedRecordID == record.id ? Color.green : Color.secondary)` - same fix as line 148.
- [LOW] lines 86, 321, 365 - section 5 ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") - "Console", "Error Details (HTTP ...)", and "Raw Events (N)" are headings without `.isHeader` (line 86), unlike the dashboard's headings which have it - add the trait.
- [LOW] lines 84-157 - section 2 vocabulary ("A pane or workspace header | AppAdaptiveHeader") - the console toolbar is a hand-rolled `HStack + Spacer`; with a long German "Errors only"/"The engine's buffer overran..." string it can overflow instead of reflowing like `AppAdaptiveHeader` (possible) - consider `AppAdaptiveHeader` or a `ViewThatFits` fallback.
- [LOW] lines 218-220, 337-339 - section 5 ("Asynchronous completion and failure are announced") - the error surfaces appear silently; the copy actions already use `showToast` (which announces), so prefer a toast or an announcement for the failure state too.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/AudioAPIPaneView.swift
- [HIGH] line 11 - section 6 Localization ("A String that reaches the UI without a Text (... a seeded chat draft) uses AppStrings.localized") - `@State private var text = "Hello! This voice clone is running locally on TurboSpark."` seeds the editor with English that renders verbatim under all 21 languages - seed via `AppStrings.localized("...")`.
- [MEDIUM] line 127 - section 4 Color (badge tint) - `badgeTint: .purple` is a system color on the themed row - pass nil for the theme accent, or resolve from the theme.
- [MEDIUM] lines 163-167 - section 3 ("TextEditor has no intrinsic height ... measure with a hidden Text") and section 2 (`.appEditorSurface(minHeight:)`) - `.frame(height: 74)` is a FIXED height on the `TextEditor`; at Extra Large text the prompt clips instead of growing - use `.appEditorSurface(minHeight: 74)` and drop the fixed frame.
- [MEDIUM] lines 196-201 - section 2 Action hierarchy ("When all that is missing is input the user can still give ... do not leave the primary action dead ... .appAttention on each unfinished card") - Synthesize is disabled when text is empty, or the reference/consent is missing: exactly the givable-input case (the Voice Cloning reference behavior), while `attached == nil` alone is the legitimate disable - keep the button pressable for missing input and use `.appAttention` + focus on the missing field.
- [MEDIUM] line 219 - section 4 Color - `.foregroundStyle(.red)` on the error message ignores custom themes - resolve the error color from the theme/AppAttention.
- [LOW] lines 114-118 - section 3 ("Every list and workspace has an AppEmptyState that ... offers the action that fills it") - when nothing is installed the pane shows a bare `Text` pointing at Audio Studio with no action button - use `AppEmptyState` with a button that opens Audio Studio.
- [LOW] line 112 - section 2 ("A disabled control says why") - Attach is disabled for four reasons (no server, none selected, already attached, busy) with no tooltip or adjacent text; the serving row below shows attachment, which covers one case - add a `.help` stating the current reason.
- [LOW] lines 218-220 - section 5 ("Asynchronous completion and failure are announced") - the synth failure appears silently as inline text; a toast (which announces) or an announcement is preferred.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/APIPaneComponents.swift
- [HIGH] line 303 - section 4 ("never the bare property (`TurboSparkTheme.accentColor`)", Gotcha 36; ratchet-counted) - `(badgeTint ?? TurboSparkTheme.accentColor)` reads the static accessor inside `body`; it follows `NSApp.effectiveAppearance`, not `.preferredColorScheme`, so a forced dark mode draws the light accent on the badge capsule - use the environment theme's resolved accent (`theme.accent`).
- [MEDIUM] lines 115-124 - section 2 vocabulary ("A card | .appCardSurface(padding:)"; "A second local copy is how the drift starts") - `apiCard()` re-implements the card treatment with drifted values (padding 16 vs 18, hairline 0.5 vs `appCardSurface`'s 1) instead of using `.appCardSurface` - either adopt `.appCardSurface(padding: 16)` or fold `apiCard` into the Theme vocabulary so there is one spelling.
- [MEDIUM] line 291 - section 4 Color / "Status is never color alone" - `Circle().fill(Color.green)` system green as the serving indicator; on a custom theme it clashes and the dot alone carries "live" - resolve from the theme, and the row already carries name/kind text, so pair status with a word or keep the dot decorative (`accessibilityHidden(true)` is already there).
- [MEDIUM] line 47 - section 4 Surfaces - `color.opacity(0.25)` hard-codes the halo ring opacity on the status dot - derive from the theme or drop the ring under Increase Contrast.
- [MEDIUM] line 60 - section 2 vocabulary ("Progress, determinate or not | AppProgressBar") - `ProgressView().controlSize(.small)` while busy is a bare system spinner; the vocabulary's indeterminate progress is `AppProgressBar` (and a live stage indicator is `.tsPulse(active:)`) - use the vocabulary control.
- [MEDIUM] line 177 - section 2 vocabulary (same rule) - `APIRunIndicator` shows a bare `ProgressView` while a request runs; this spinner is the run indicator for Text/Image/Audio/TypeSafe panes - use `AppProgressBar` (indeterminate) or `.tsPulse(active:)` so Reduce Motion is honored.
- [MEDIUM] lines 254-273 - section 2 vocabulary ("An icon-only action | AppIconButton") - `APICopyButton` is another hand-rolled icon-only button (`Label.iconOnly` + `.buttonStyle(.borderless)` + `.help`) duplicating `AppIconButton`'s tooltip-and-VoiceOver contract - rebuild on `AppIconButton`.
- [LOW] line 227-228 - section 4 Surfaces - `.fill(.appPage.opacity(0.6))` hard-codes dimming on the themed page color - use `.appPage` or `.appElevated` as the theme gives it.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/TypeSafePlaygroundView.swift
- [MEDIUM] line 67 - section 3 ("No fixed .frame(width:) ... A fixed width is what wrapped 'Installed model type' into several lines"; text-scale rule) - `examplePanel.frame(width: 180)` is a fixed-width column that does not scale with `AppMetrics.textScale(theme)`; example button labels wrap/clip at Extra Large - use `minWidth`/`maxWidth` or multiply by textScale like the menu bar panel does.
- [MEDIUM] lines 35-41 - section 2 vocabulary ("Switching one pane between a few modes | AppPillPicker ... inside an AppAdaptiveHeader") (possible) - the Playground/Benchmark switch is a segmented `Picker` in the header rather than `AppPillPicker`; same again at lines 103-112 for Form/Raw JSON - adopt `AppPillPicker` or justify the native segmented style as the canonical spelling.
- [MEDIUM] lines 118, 238 - section 4 Color - `.foregroundStyle(.red)` on error messages ignores custom themes - resolve the error color from the theme.
- [LOW] line 126 - section 2 ("A disabled control says why") - Run is disabled while running or when the TypeSafe service is down with no tooltip; the header subtitle only says requests go to the service - add a `.help` naming the reason.
- [LOW] lines 142-144 - section 3 empty-state rule - "Select a model to load" is a bare hint with no action to open the model list - use `AppEmptyState` with the action, or a hint plus button.
- [LOW] line 291 - section 6 ("Do not localize" vs UI strings) (possible) - `errorMessage = "HTTP \(error.status): \(error.message)"` composes an English sentence reaching the UI verbatim; the status prefix is technical but the composed line will not localize - consider a catalog key "HTTP %@: %@" with the message interpolated.
- [LOW] lines 117-120, 237-239 - section 5 ("Asynchronous completion and failure are announced") - errors appear silently inline; the copy actions elsewhere toast, so prefer a toast or announcement for the failure.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerAdvancedSettingsView.swift
- [MEDIUM] line 95 - section 4 Color / section 2 AppAttentionNote - the no-key warning is a `Label` with `.foregroundStyle(.orange)` (system orange on a themed surface) instead of the attention vocabulary - use `AppAttentionNote`/`AppAttention.color` (text + glyph are already paired, good).
- [MEDIUM] lines 225-233 - same rule - the "turbospark CLI not found" warning uses `.foregroundStyle(.orange)` - same fix as line 95.
- [LOW] lines 210, 222 - section 2 (`APIControlSize` is the shared mapping for API panes) - `.appAction(.secondary, size: .small)` uses the literal ControlSize instead of `APIControlSize.row`, drifting from the components file's one mapping - use `APIControlSize.row`.
- Compliance worth noting: every disabled field is explained (line 26 header text for the start-time fields, lines 225-233 for the CLI buttons, lines 45-48 and 108-112 comments document the intent), labels use the keep-real-label + `.labelsHidden()` pattern, and `Text(LocalizedStringKey(...), bundle: .module)` is the correct enum-backed spelling.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerMenuBarView.swift
- clean
- (Verified: all interpolated menu labels and toasts resolve to existing catalog keys, e.g. "CLI Daemon: %@", "Stop Background Daemon (PID %@)", "Copied %@ endpoint"; `Text(verbatim:)` is used only for product names, commands, routes, and the formatted stats where a Double is first formatted to a String - all correct per section 6.)

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerMenuDashboardView.swift
- [MEDIUM] lines 141-143 - section 2 vocabulary ("A surface that still needs the user | .appAttention ... with an AppAttentionNote inside") / section 4 - the failure card hand-draws `RoundedRectangle(...).fill(Color.red.opacity(0.14))` with a hard-coded red and opacity - build it on the attention vocabulary so the theme and Increase Contrast control it (the `AppMetrics.cardRadius` and action buttons are correct).
- [MEDIUM] lines 157-159 - section 5 ("Titles and section headings carry .accessibilityAddTraits(.isHeader)") - the "Memory" section heading lacks `.isHeader` while the sibling "Loaded" heading (line 182) has it - add `.accessibilityAddTraits(.isHeader)`.
- [LOW] lines 185-191 - section 3 empty-state rule (informational for a compact menu panel) - "No model loaded." is a bare `Text` rather than `AppEmptyState`; acceptable at menu-panel scale, but it also offers no path to load beyond the "Load Model" row below it (which exists, so this is minor).
- Compliance worth noting: the 340pt panel width is multiplied by `AppMetrics.textScale(theme)` (line 42), the server Toggle keeps a real label and hides it (lines 63-66), memory uses `AppProgressBar(animated: false)` (line 165), and `Text(verbatim:)` is confined to the product name, glyphs, and formatted stats.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerMenuDashboardRows.swift
- [MEDIUM] line 107 - section 3 ("No fixed .frame(width:) ... nor on a glyph sized with .themedFont (it grows with Text Size)") - the model-kind glyph is `.themedFont(.small)` inside `.frame(width: 26, height: 26)`; the glyph grows with Text Size but the tile does not, so it clips at Extra Large - size the tile by textScale or use `.themedFont(fitting:)` for a frame-dictated glyph.
- [MEDIUM] lines 53-60 - section 4 Color (possible - deliberate) - `MenuBarStatusLine.color(for:)` returns `.secondary`, `.green`, `.red` system colors (warning correctly uses `AppAttention.color`); the doc comment justifies the contrast choice and the glyph shape carries the state, but ok/error still bypass custom themes - resolve from the theme's status colors for consistency with AppAttention.
- [MEDIUM] line 66 - section 2 vocabulary ("A live indicator (recording, active stage) | .tsPulse(active:)"; progress | AppProgressBar) - the starting/working phase shows a bare `ProgressView().controlSize(.mini)` - use `.tsPulse(active:)` on the glyph or `AppProgressBar` so Reduce Motion is honored and the vocabulary is followed.
- [MEDIUM] line 130 - section 2 ("A disabled control says why") - the eject `AppIconButton` is disabled when `canUnload` is false (model held by an external process) with no tooltip or adjacent text explaining it - add a `.help` naming the reason.
- Compliance worth noting: the AnyShapeStyle ternary (line 106), `.tsAnimation(TSMotion.hover, value:)` (line 190), `AppIconButton` (line 128), and `.accessibilityElement(children: .contain)` (line 134) are all correct.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerLoadedModelsView.swift
- [HIGH] line 58 - section 4 ("never the bare property (`TurboSparkTheme.accentColor`)", Gotcha 36; ratchet-counted) - `badgeTint: row.isChatSession ? TurboSparkTheme.accentColor : .purple` reads the static accent accessor inside `body` (wrong mode under `.preferredColorScheme`) and also passes system `.purple` for the audio badge - resolve from the environment theme (`theme.accent`) and drop the system purple.
- [MEDIUM] lines 72-75 - section 4 Color ("In a ternary that mixes two styles, wrap each side in AnyShapeStyle"; system colors) - `.foregroundStyle(status.isActive || status.isUnsupported ? Color.orange : Color.secondary)` uses system orange/secondary in a bare ternary on a themed row - use `.appSecondary` and a theme-resolved attention color (the Label already pairs glyph + text).
- [LOW] line 32 - section 2 ("A disabled control says why") - "Attach Model..." is disabled while busy or with no server; the header band above shows server state, which partially explains - add a `.help` naming the current reason.
- [LOW] line 166 - section 3 ("No window has a fixed frame"; layouts adapt) (possible) - the picker sheet is `.frame(width: 460)`, a fixed width that does not scale with text size; the row content has `minWidth`-free text that wraps, so this is minor, but scaling by `AppMetrics.textScale` would match the dashboard's pattern.
- Compliance worth noting: detach stays `.secondary` (release-from-memory is not data loss), `detailLine` refuses to state unmeasured zeros, and the steering status maps its fixed readings to catalog text.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ImageAPIPaneView.swift
- [HIGH] line 9 - section 6 Localization (seeded draft) - `@State private var prompt = "A red fox in a snowy forest, watercolor illustration"` seeds the editor with English copy rendered verbatim under all languages - seed via `AppStrings.localized("...")` (a catalog key for the example prompt).
- [MEDIUM] lines 94-98 - section 3 / section 2 (`.appEditorSurface(minHeight:)`) - `.frame(height: 74)` is a fixed height on the prompt `TextEditor` - use `.appEditorSurface(minHeight: 74)` and drop the fixed frame (same defect as AudioAPIPaneView line 165).
- [MEDIUM] lines 106-110 - section 2 Action hierarchy (givable-input rule) - Generate is disabled when the prompt is empty, exactly the input the user can still give; only `testing`/no-attached-model are legitimate disables - keep it pressable and use `.appAttention` + focus, per the Voice Cloning reference.
- [MEDIUM] line 114 - section 4 Color - `.foregroundStyle(.red)` on the error message ignores custom themes - resolve the error color from the theme.
- [LOW] lines 67-71 - section 3 empty-state rule - the "Install a supported Z-Image or Qwen-Image model..." hint is bare text with no action button to Image Generation - `AppEmptyState` with the action.
- [LOW] lines 122-129 - section 5 ("Asynchronous completion and failure are announced") - saving to the gallery flips `saved` and disables the button with no toast/announcement; every other completion in the batch uses `showToast` (which announces) - post a toast on save success/failure.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/TypeSafePlaygroundCards.swift
- [MEDIUM] lines 23-24, 53-56, 88-92 - section 2 ("A destructive role with no confirmation dialog is always a defect: either the action is not destructive, or the dialog is missing") - three `Button(role: .destructive)` remove controls (question remove, option remove, level remove) with no `confirmationDialog`; they delete user-typed draft content that cannot be rebuilt - either add the localized confirmation dialog, or if draft edits are considered non-destructive, drop the destructive role for a quiet remove.
- [MEDIUM] lines 99-100, 154-156 - section 2 vocabulary ("A card | .appCardSurface(padding:)") / section 4 radii - both cards hand-roll `.padding(...).background(.appPage, in: RoundedRectangle(cornerRadius: 8))`; radius 8 is not an `AppMetrics` value (controlRadius 7, cardRadius 10) and this is a second card spelling - use `.appCardSurface` or `AppMetrics` radii.
- [MEDIUM] lines 124, 170 - section 2 vocabulary ("Progress, determinate or not | AppProgressBar") - the probability bars are bare `ProgressView(value:total:)` instead of `AppProgressBar(value:...)` (which also brings the label/valueText VoiceOver contract; the first one adds a manual `.accessibilityLabel`, the second none) - use `AppProgressBar`.
- [LOW] line 48 - section 2 ("A disabled control says why") - the option key field is disabled when the key is the reserved "__none__" with no explanation - add a `.help` ("reserved schema keyword").
- [LOW] line 76 - section 3 ("No fixed .frame(width:) ... on a label") - `Text(verbatim: "\(index)").frame(width: 25)` is a fixed-width label in a form row; it holds one digit so the risk is small (possible) - use `minWidth`.
- [LOW] lines 169, 171 - section 3 (fixed widths on row labels/readouts) - probability label `.frame(width: 145)` and percent `.frame(width: 55)` can clip long option keys or "100.0%" at Extra Large - `minWidth` or textScale-scaled widths, with truncation + tooltip for long keys.

### Cross-file notes
- The seeded-draft pattern (HIGH, 2 files) and the bare `TurboSparkTheme.accentColor` read (HIGH, 2 sites) are the only HIGH findings; both have explicit DESIGN.md rules and ratchet coverage.
- `ProgressView` in place of `AppProgressBar`/`.tsPulse` appears in 3 files (APIPaneComponents, ServerMenuDashboardRows, TypeSafePlaygroundCards); fixing it inside `APIRunIndicator` and `MenuBarStatusLine` covers most panes at once.
- Console/dashboard readouts use `Text(verbatim:)` correctly throughout (numbers, tok/s, ms, URLs, model ids, routes); no bare-literal or key-less API violations were found, and every interpolated key used by these files exists in the catalog.

---

### Batch batch31_Server

Audit of 12 Server pane files against DESIGN.md (sections 2 vocabulary/action
hierarchy, 3 layout, 4 theming/type, 5 accessibility/motion, 6 localization).
Every file was read in full. Context checked: `Server/APIPaneComponents.swift`
(APIStatusBand, APIPanel, APIServedModelRow, APICopyButton), `Theme/AppControls.swift`
(AppSectionHeader adds `.isHeader` itself; AppIconButton sets no button style),
`Theme/AppSurfaces.swift` (appCardSurface), `Theme/AppAttention.swift`
(single `static let color = Color.orange`).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/TypeSafeAPIPaneView.swift
- [MEDIUM] line 114 - Theming (sec 4: system colors) - `Text(verbatim: error).foregroundStyle(.red)` uses theme-blind system red for the daemon error line - use the theme's error/destructive style; keep text selection.
- [MEDIUM] line 146 - Theming (sec 4) - `Circle().fill(item.loaded ? Color.green : Color.secondary)` fills the loaded status dot with theme-blind system colors (possible: the same green/secondary dot is also the shared `APIServedModelRow` convention) - resolve via the theme or a shared status-color vocabulary; the paired "Loaded"/"Not loaded" text already keeps status from being color-alone.
- [MEDIUM] lines 33-36 - Theming (sec 4) - `bandColor` returns `.secondary` / `.green` / `.orange` system colors into `APIStatusBand` (possible: this is the convention both band callers share) - the warning/attention tint could come from `AppAttention.color`; green/running needs a themed status spelling.
- [LOW] lines 129-132 - Empty states (sec 3) - the Models panel's empty state is a bare secondary `Text`, not `AppEmptyState(systemImage:title:message:)` - use the vocabulary control.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerHeaderBandView.swift
- [MEDIUM] line 94 - Theming (sec 4) - guardrails Label uses `.foregroundStyle(Color.secondary)`, ignoring the custom theme's secondary color and the Increase Contrast raise - use `.appSecondary`.
- [MEDIUM] line 103 - Theming (sec 4) - `.foregroundStyle(rows.authIsWarning ? Color.orange : Color.secondary)` mixes system colors in a ternary without `AnyShapeStyle` - use `AppAttention.color` for the warning and `.appSecondary` for the normal state (wrap in `AnyShapeStyle` if the types differ).
- [MEDIUM] line 154 - Theming (sec 4) - `stat()` uses `.foregroundStyle(tint ?? .primary)`; `.primary` ignores a custom theme's text color - use `.appText` as the fallback.
- [MEDIUM] lines 55-59 - Theming (sec 4) - `stateColor` returns `.secondary`/`.orange`/`.yellow`/`.green` system status colors (possible: shared band convention); the localized title/subtitle next to the dot does keep status from being color-alone.
- [LOW] lines 135, 142 - Theming (sec 4) - error and log-gap stat tints are hard-coded `.orange` - route warning/attention tints through `AppAttention.color`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerConnectCardView.swift
- [MEDIUM] line 132 - Theming (sec 4: hard-coded dimming) - SSE/NDJSON badge uses `.background(Color.secondary.opacity(0.15), in: Capsule())`, a theme-blind system color with a hard-coded opacity - derive from `.appSecondary`/`.appSurface` per the theme.
- [MEDIUM] line 123 - Layout (sec 3: no fixed width on labels in rows) - `Text(verbatim: endpoint.method).themedCode(...)` sits in a fixed `.frame(width: 34)`; the code font grows with Text Size so `DELETE` can clip under Extra Large - drop the fixed width or use `minWidth`.
- [LOW] lines 39, 105, 113, 139 - Localization (sec 6, possible) - `Text(LocalizedStringKey(...), bundle: .module)` over enum-backed strings (`snippet.title`/`note`, `option.title`, `family.blurb`, `endpoint.summary`); correct pattern, but those enums must be registered in `testEnumBackedDisplayStringsAreCatalogKeys` (not verifiable from this file).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerChartsView.swift
- [MEDIUM] line 22 - Accessibility (sec 5: headings) - the "Traffic" section heading `Text` lacks `.accessibilityAddTraits(.isHeader)` (does not use `AppSectionHeader`, which adds it) - add the trait or use `AppSectionHeader(title:)`.
- [LOW] lines 37-43 - Empty states (sec 3) - hand-rolled empty `Text` instead of `AppEmptyState(systemImage:title:message:)`.
- [LOW] lines 117-140 - Vocabulary (sec 2) - `chartCard` hand-rolls the card surface (padding 12 + `.appSurface` + `.appBorder` stroke at `AppMetrics.cardRadius`) instead of `.appCardSurface(padding:)` or the feature's own `apiCard()`; a second local copy of the card treatment is how drift starts.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/TypeSafeBenchmarkView.swift
- [MEDIUM] line 57 - Vocabulary (sec 2: progress) - determinate `ProgressView(value:total:)` instead of `AppProgressBar(value:label:valueText:size:animated:)`.
- [MEDIUM] line 24 - Accessibility (sec 5: headings) - "Benchmark" heading `Text` lacks `.accessibilityAddTraits(.isHeader)`.
- [MEDIUM] line 64 - Theming (sec 4) - `Text(verbatim: errorMessage).foregroundStyle(.red)` is theme-blind system red - use the themed error style.
- [LOW] lines 26-32 - Accessibility (sec 5) - the repetitions `Stepper` has no `.accessibilityValue` (the count sits inside the label); expose the formatted value.
- [LOW] lines 35-41 - Action hierarchy (sec 2) - Stop is disabled while `stopRequested` and Run when `service == nil || selectedModels.isEmpty`, with no tooltip or adjacent text saying why.
- [LOW] lines 70-72 - Localization (sec 6, possible) - `String(format: "n=%d  mean %.1f ms ...")` renders fixed English stat words (mean/p50/p95/min/max) verbatim in the results list; only the numbers are language-neutral.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/EmbeddingsAPIPaneView.swift
- [LOW] line 71 - Vocabulary (sec 2, possible) - indeterminate `ProgressView().controlSize(.small)` during attach instead of `AppProgressBar`; it matches the in-flight spinner convention already used by `APIStatusBand` and `APIRunIndicator`, so it may be intentional.
- Otherwise clean: the Attach button's disabled reason is given in both `.help` and adjacent text (lines 55, 63-68, 100-116), the reference pattern for this batch.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerInspectorView.swift
- [HIGH] lines 73-77 - Action hierarchy (sec 2: destructive actions) - `Button(role: .destructive)` "Delete" removes a saved server preset with no `confirmationDialog`. DESIGN: "A destructive role with no confirmation dialog is always a defect." A preset (name, host, port) is user-entered and not rebuildable, so it is genuinely destructive - add a `confirmationDialog` with localized title, message, and button.
- [MEDIUM] line 10 - Accessibility (sec 5: headings) - the "Server" `Label` heading lacks `.accessibilityAddTraits(.isHeader)`; hand-built headings must add it themselves.
- [MEDIUM] line 53 - Accessibility (sec 5: headings) - the "Profile" section heading `Text` lacks `.accessibilityAddTraits(.isHeader)`.
- [LOW] lines 56-61 - Action hierarchy (sec 2) - Save Preset is disabled while the name is empty with no tooltip or adjacent text explaining why.
- [LOW] lines 62-82 - Vocabulary (sec 2) - each favorite row hand-rolls a card (`.appSurface` in a `RoundedRectangle(cornerRadius: AppMetrics.cardRadius)`, no hairline, padding 10) instead of `.appCardSurface(padding:)`.
- [LOW] lines 18-21 - Action hierarchy (sec 2) - "Capture text previews" toggle is disabled while the server runs; the adjacent text explains the feature but not why the control is disabled.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerLiveChartsView.swift
- [MEDIUM] line 27 - Layout (sec 3: text in layouts) - `In:/Out:` readout uses `.lineLimit(1).minimumScaleFactor(0.8)`, shrinking text to fit; DESIGN says never shrink text - let it wrap or truncate with the full value in a tooltip.
- [LOW] lines 42, 65 - Vocabulary (sec 2) - `bandwidth` and `memory` hand-roll the card (padding 14 + `.appSurface`, no hairline) instead of `.appCardSurface(padding:)`/`apiCard()`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/EmbeddingsAPITestPanel.swift
- [MEDIUM] line 48 - Theming (sec 4) - the error `Label` uses `.foregroundStyle(.red)`, theme-blind system red - use the themed error style; the octagon glyph already keeps status from being color-alone.
- Otherwise clean: editor uses `.appEditorSurface(minHeight:)` plus an `accessibilityLabel`, Send's disabled state is explained in `.help` and adjacent text (lines 29-40).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerPaneView.swift
- clean. The console drag handle has an `accessibilityLabel` and `.accessibilityAdjustableAction` (lines 61-64), the segmented picker uses `maxWidth` only, and backgrounds use `.appPage`/`.appBorder`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerConnectionView.swift
- [MEDIUM] line 46 - Theming (sec 4) - the port error text uses `.foregroundStyle(.red)`, theme-blind system red - use the themed error style.
- Otherwise clean: fields disabled while the server runs are explained by the adjacent "Stop the server to edit" line (line 48), and widths use min/ideal/max.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Server/APISidebarPanel.swift
- clean. Correctly uses `SidebarModePanel(title:groupLabel:selection:)` with localized `Text` labels.

### Cross-file notes
- No motion defects in this batch: no bare `withAnimation`, no `.animation(_:value:)`, no `repeatForever`, no inline durations.
- No localization gate defects: no bare literals in key-only APIs, no ternary inside a single `Text`, no Double/Date interpolated into localized keys; `Text(verbatim:)` is used only for addresses, model ids, paths, code, and readouts, which is correct.
- No `TurboSparkTheme` static accessors read in view bodies in this batch.
- Recurring pattern worth one shared fix: error text in system `.red` (TypeSafeAPIPaneView 114, TypeSafeBenchmarkView 64, EmbeddingsAPITestPanel 48, ServerConnectionView 46) and hand-rolled cards/empty states (ServerChartsView, ServerLiveChartsView, ServerInspectorView, TypeSafeAPIPaneView).
- All files are well under the ~400-line guideline (29-175 lines).

---

### Batch batch32_Theme

Audit of the Theme vocabulary files against DESIGN.md. These files define the
shared control vocabulary, so `.appBorder.opacity`-style spellings inside
Theme/ are allowed by the ratchet; findings below cover contract drift with
DESIGN.md, the vocabulary rules applied to the vocabulary's own views, and
theme-resolution bugs. Line numbers are 1-based file lines.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/AppControls.swift
- [MEDIUM] lines 364-371 - Increase Contrast / theme resolution - AppChip colors its off-state label with `Color.primary` / `Color.secondary` and hard-codes the fill and border opacities (`theme.accent.opacity(0.14)`, stroke `0.3`, hover `0.04/0.08`) with no `theme.isHighContrast` branch, so custom themes' text colors never reach the chip and Increase Contrast leaves the on-state border faint. Sibling control AppPillPicker does this correctly (`theme.foreground`/`theme.secondaryText`, and a high-contrast stroke at AppPillPicker.swift:98-100). - Use `theme.foreground`/`theme.secondaryText` and raise the border/fill under `theme.isHighContrast` like AppPillPicker.
- [MEDIUM] lines 367, 370 - Radii contract drift - AppChip hard-codes `cornerRadius: 8` while DESIGN.md section 2's Radii row maps chips to `AppMetrics.controlRadius` (7). The file's own doc comment ("radius 8") documents the drift rather than resolving it. - Use `AppMetrics.controlRadius`, or add a named chip radius to AppMetrics and update the DESIGN.md row.
- [MEDIUM] line 308 - Text scale - AppEmptyState caps its message with `.frame(maxWidth: 440)`, a text-holding frame not multiplied by `AppMetrics.textScale(theme)` (DESIGN.md section 2 table). Under Extra Large the measure stays at the 16pt-reference width while every other text frame grows. - `440 * AppMetrics.textScale(theme)`.
- Note: AppSearchField (prompt doubles as accessible name, optional `focus` binding at 128/164-165), AppIconButton (label is both tooltip and VoiceOver name via the Label at 189-192), AppAdaptiveHeader's three ViewThatFits layouts, and AppSectionHeader's `.isHeader` trait all match their documented contracts. `appEditorSurface` goes through `.appSurface`/`.appBorder` so Reduce Transparency and Increase Contrast apply.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/TurboSparkMotion.swift
- [HIGH] lines 200, 231 - Static theme read in a view - TSIdlingSparkView reads bare `TurboSparkTheme.accentColor` (TurboSparkTheme.swift:100, resolved from `NSApp.effectiveAppearance`) for the mascot glow and fallback glyph. This is the exact spelling DESIGN.md section 4 bans for views (Gotcha 36): it will not track `.preferredColorScheme` and ignores custom themes entirely, so the welcome mascot can glow the wrong accent. - Read `@Environment(\.appTheme)` and use `theme.accent`.
- [MEDIUM] lines 213, 245 - Inline durations - the mascot's two `phaseAnimator`s animate with `.easeInOut(duration: 1.7)` and `.easeInOut(duration: 2.3)` instead of TSMotion curves ("No inline durations", DESIGN.md section 5). TSMotion.breathe (0.85) is too fast for this use, which is why the numbers were invented. - Add a documented slow-breathe curve to TSMotion (doc comment + any pinned expectations) and use it here.
- [LOW] line 9 - Doc drift - header comment says "These are the four curves the redesign uses" but the enum defines seven (hover, select, press, pane, sweep, breathe, snap), which matches DESIGN.md section 5's list of seven. - Fix the comment.
- [LOW] line 101 - Motion spelling (possible) - TSEntrance uses a bare `withAnimation(TSMotion.pane.delay(delay))` rather than `TSMotion.animate`; behavior is correct because line 100 guards on `reduceMotion`, but the guarded bare spelling is the pattern DESIGN.md calls out. - Use `TSMotion.animate(TSMotion.pane.delay(delay))`.
- Note: TSPulse (lines 132-142) matches the documented phaseAnimator-only-while-live contract exactly, and `TSMotion.animate`/`.tsAnimation`/TSPressScaleStyle all resolve reduce motion from the two documented sources.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/ThemeCodePreviewView.swift
- [MEDIUM] lines 68, 88 - System color on a themed surface - the preview card's divider and border use `Color(nsColor: .separatorColor)`, which resolves against the ambient appearance, while the card itself resolves its background/text with an explicit `isDark`. One of the two side-by-side Light/Dark cards always disagrees with the ambient appearance, so its divider is tuned for the wrong mode (same class as Gotcha 36, though mode-explicit everywhere else in the file). - Derive the divider/border from `manager.activeForegroundColor(isDark: isDark)` with an opacity, as the text already does.
- [MEDIUM] lines 103, 143 - Same class - gutter line numbers use `Color.secondary.opacity(0.6/0.7)`, ambient-resolved, ignoring the previewed theme's foreground. - `manager.activeForegroundColor(isDark: isDark).opacity(...)`.
- [MEDIUM] lines 121, 156 - Text scale (possible) - code lines are fixed at `.frame(height: 18)` (gutter `.frame(width: 14)`) while `manager.codeFont(isDark:)` scales with both the code font size setting and Text Size (`AppearanceSettings.swift:342` applies `textSize.scaled`), so a large code size clips inside the 18pt row. - Let rows size to content or multiply the frame by `AppMetrics.textScale`.
- [LOW] lines 85, 87, 158 - Radii - hard-coded `cornerRadius: 8/8/3`, not AppMetrics values (controlRadius 7, cardRadius 10), though all use `style: .continuous`. - Use AppMetrics radii or document a preview radius.
- Note (16pt reference check): the demo text sizes derive from the live settings via `manager.codeFont(isDark:)`/`activeForegroundColor(isDark:)`, not from hard point sizes, so the previews track the Appearance code size and Text Size correctly; the 16pt reference is a doc-table device and no contract pins these previews to it. All rendered strings are verbatim code demo content, which is allowed.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/AppProgressBar.swift
- clean
- Note: matches its DESIGN.md contract exactly - signature `(value:label:valueText:size:animated:)`, `animated: false` (paused) and Reduce Motion both hold the sheen/sweep still (lines 46-48, 105, 125-127), Increase Contrast gets a solid fill, track stroke and no glow (92, 97, 105, 108), height scales with `textScale` (51), and VoiceOver gets a native `ProgressView` representation with the optional value text (75-86, 141-151).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/AppAttention.swift
- clean
- Note: rings are a `keyframeAnimator` triggered by the `ping` counter (line 95), not a timer or repeatForever, matching "bump ping only on the user's attempt"; Reduce Motion gets a held static ring with a documented hold (44-59, 74-78); Increase Contrast widens edge and rings (65, 93); the persistent edge is the state, and AppAttentionNote pairs the triangle glyph with localized words and `.accessibilityElement(children: .combine)` (125-147).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/AppPillPicker.swift
- clean
- Note: implements the vocabulary rules correctly - Increase Contrast stroke on the selection capsule (98-100), custom focus ring because styled buttons draw none (109-114), arrow-key navigation with RTL mirroring (64-65, 131-146), TSMotion only (`TSMotion.select` at 61, `TSMotion.animate(TSMotion.hover)` at 122, `.tsPress` at 117, live dot via `tsPulse` at 83), theme colors throughout (`theme.foreground`/`theme.secondaryText`/`theme.selection`), and tooltip plus VoiceOver label/value/selected trait on every segment (120, 126-128). The documented no-ViewThatFits decision is explained at 44-48.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/ThemeTypographyPreviewView.swift
- [MEDIUM] lines 108, 116 - System color on a themed surface - card borders use `Color(nsColor: .separatorColor)` over the previewed theme's background, ambient-resolved while the card resolves isDark explicitly; same wrong-mode risk as ThemeCodePreviewView. - Derive from `manager.activeForegroundColor(isDark: isDark)` with an opacity.
- [MEDIUM] lines 62-63, 70-71 - System color on a themed surface - the family/size/weight badges fill with `Color.primary.opacity(0.06)` and label with `.foregroundStyle(.secondary)`, both ambient spellings that ignore the previewed theme (the banned SwiftUI spellings of DESIGN.md section 4) and can disagree with the card's explicit isDark. - Use `manager.activeForegroundColor(isDark:)` with opacity for both.
- [LOW] lines 83, 93 - Type by role - secondary demo text is sized by numeric multipliers of the base (`manager.uiFontSize * 0.88`, `* 0.82`) instead of AppFontStep factors (`.callout` 0.875, `.small` 0.75); AppFontCatalog is documented as "the ONE change point for font sizing". - Pass the role factors (or `textSize.scaled` roles) instead of ad-hoc numbers.
- [LOW] lines 24-30 - Redundant surface (possible) - `cardBgColor` is the same expression as `bgFillColor` (`activeBackgroundColor(isDark:)`), so the inner "card" fill equals its parent and renders as a floating border with no surface contrast; looks like an unfinished distinction. - Give the inner card a distinct derived fill (e.g. foreground at low opacity) or drop the fill.
- [LOW] lines 56, 105, 113 - Radii - hard-coded `cornerRadius: 4/6/8`, not AppMetrics values (continuous style used). - Use AppMetrics radii or document preview-specific ones.
- Note (16pt reference check): the demo text derives from the live UI font settings (`manager.uiFont(isDark:)`, whose descriptor applies `textSize.scaled` per AppearanceSettings.swift:353), so it follows both the base size and Text Size; no hard point sizes for the demo text itself. All display strings go through `Text(..., bundle: .module)` (43, 77, 82, 92) and the "px" unit is verbatim as allowed.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/AppWaveform.swift
- [LOW] line 69 - Motion pattern note - AppFillWaveform's rippling loop is a `TimelineView(.animation(minimumInterval:paused:))` rather than the documented "phaseAnimator inside a modifier applied only while the state is live" pattern. The contract's intent is honored (paused when not rippling or under Reduce Motion, and static 3pt bars otherwise, lines 69, 95), so this is informational drift, not a defect. - Keep, or note the TimelineView pattern in DESIGN.md section 5 if it is to be a sanctioned alternative.
- Note: both waveforms are `accessibilityHidden(true)` as their doc comment promises (34, 81), scale height with `AppMetrics.textScale(theme)` (21, 66), and handle Increase Contrast (28, 90).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Theme/AppPreferenceOptions.swift
- clean
- Note: enum-backed labels render through `Text(LocalizedStringKey(option.label), bundle: .module)` exactly as DESIGN.md section 6 requires, and AppAppearance/AppTextSize are registered in `LocalizationParityTests` (Tests/TurboSparkAppTests/LocalizationParityTests.swift:568-572); language native names are `Text(verbatim:)` by documented design with only "System" localized.

### Cross-file checks
- `AppMetrics.composerRadius` is defined in AppSurfaces.swift:13, so the DESIGN.md radii row is fully covered; no missing-radius drift.
- No file exceeds ~400 lines (largest is AppControls.swift at 386); no line-count findings.
- No bare `repeatForever`, no `withAnimation` that ignores Reduce Motion, and no phaseAnimator running while its state is not live, in any of the nine files (the only bare `withAnimation` is the guarded TSEntrance one noted above).
- All user-visible strings in these files go through `Text(..., bundle: .module)`, `AppStrings`-style bundled literals, or verbatim demo/code content.

---

### Batch batch05_misc

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioEngineSummary.swift
- clean
  - Checked: the `Text(LocalizedStringKey(engine.title), bundle: .module)` pattern is the DESIGN.md section 6 "fixed English held in a variable" form, and `AudioEngineLabel` IS registered in `testEnumBackedDisplayStringsAreCatalogKeys` (LocalizationParityTests.swift lines 584-586), so this is compliant.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCloneInputsView.swift
- clean

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioNavigationView.swift
- [MEDIUM] line 13 (also line 23) - Section 6, fixed English held in a variable - `Text(LocalizedStringKey(page.title), bundle: .module)` depends on every `AudioWorkspacePage.title` value being a catalog key, but `AudioWorkspacePage` is NOT registered in `testEnumBackedDisplayStringsAreCatalogKeys` (verified: no reference in LocalizationParityTests.swift, while sibling enums incl. AudioEngineLabel are), so nothing holds new titles to the catalog - add `AudioWorkspacePage.allCases` to the parity test in LocalizationParityTests.swift (spot-check confirms the current 9 titles are keys).

#### swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioRecordingPanel.swift
- [MEDIUM] lines 25-26 - Section 2 vocabulary (a card is `.appCardSurface(padding:)`) and Section 4 radii - hand-rolled card: `.padding(20).background(.appSurface, in: RoundedRectangle(cornerRadius: 12))` plus a manual `.appBorder` stroke; radius 12 is not an AppMetrics radius (cardRadius is 10) and the treatment duplicates the card vocabulary - replace with `.appCardSurface(padding: 20)`.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Files/FilesSectionView.swift
- [HIGH] line 214 - Section 4 color - `symbolTint: isSelected ? TurboSparkTheme.accentColor : Color.secondary` reads the static `TurboSparkTheme.accentColor` inside a view body (mode may not match `.preferredColorScheme`, Gotcha 36) and uses theme-blind `Color.secondary` - read `@Environment(\.appTheme)` and pass `theme.accent` / a theme-resolved secondary.
- [HIGH] lines 244-247 - Section 4 color - row background ternary `isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : (isHovering ? Color.primary.opacity(0.04) : Color.clear)` combines a static accessor read in the body with `Color.primary.opacity`, both of which ignore a custom theme; if kept as a style ternary each side needs `AnyShapeStyle` - use theme-resolved colors (`theme.accent.opacity(...)`, a themed hover fill).
- [MEDIUM] line 99 - Section 4 color (status is never color alone; warning tint) - `.foregroundStyle(.orange)` on the error-row glyph is a system color that ignores the theme - use `AppAttention.color`.
- [MEDIUM] line 234 - Section 4 color - `.foregroundStyle(.orange)` on the missing-source glyph - use `AppAttention.color`.
- [MEDIUM] lines 275-277 - Section 2 action hierarchy - `AppIconButton("trash", ..., role: .destructive)` for Detach with no confirmationDialog anywhere in the file; detaching an attachment is rebuildable (source file remains), so per rule it is either a quiet action with a neutral glyph or it needs a dialog - drop the destructive role (neutral glyph) or add a localized confirmationDialog.
- [MEDIUM] lines 310-315 - Section 2 action hierarchy - context-menu `Button(role: .destructive)` "Detach" with no confirmationDialog - same rule as above; make both sites consistent.
- [MEDIUM] (possible) line 238 - Section 5 accessibility - row actions (Go to chat, Reveal in Finder, Detach) render only `if isHovering || isSelected`; VoiceOver focus does not set `isHovering`, and there is no `.accessibilityAction` exposing them on the row element, so they are hover-only - expose them with `.accessibilityAction` and keep them keyboard-reachable.
- [LOW] lines 247-248 - Section 4 radii - `RoundedRectangle(cornerRadius: 8, style: .continuous)` twice; 8 is not an AppMetrics radius - use `AppMetrics.controlRadius` (7).
- [LOW] lines 84-85 - Section 2 action hierarchy (a disabled control says why) - the Add files button disables while `isExtracting || model.isRunning`, but `.help` describes the action, not the disable reason - state why in the tooltip or adjacent text.
- [LOW] (possible) line 77 - Section 2 vocabulary - `ProgressView().controlSize(.mini)` as the button glyph instead of the progress vocabulary; acceptable as an inline activity spinner in a label, noting only because the vocabulary table routes progress through AppProgressBar/AppLevelWaveform.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Files/FilePreviewView.swift
- [MEDIUM] lines 251-256 - Section 2 action hierarchy and Section 4 color - `AppIconButton("trash", ..., role: .destructive)` Remove (detach) with no confirmationDialog, plus `.foregroundStyle(.red)` painted over the role treatment with a theme-blind system color - drop the red (the role already communicates and the red label reaches only sighted users), and either make the action quiet or add a localized dialog, consistent with the FilesSectionView decision.
- [MEDIUM] line 208 - Section 4 color - `.foregroundStyle(.quaternary)` on the eye.slash glyph is a SwiftUI semantic spelling that ignores a custom theme's text colors - use `.appSecondary` or a theme-resolved tint.
- [LOW] (possible) lines 130-139 - Section 2 vocabulary - bespoke `.buttonStyle(.plain)` + `.foregroundStyle(.appAccent)` "View all" button duplicating a quiet-action treatment - consider `.appAction(.quiet, size: .small)` like the neighboring footers use.
- [LOW] line 293 - Section 3 layout (layouts adapt instead of clipping) - fixed `.frame(width: 1100, height: 800)` on the maximize sheet does not adapt to small screens or text scale; the comment says it deliberately matches the artifact panel's sheet, so the fix must move both - derive from available size or clamp to the visible frame.
- Note: lines 34 and 294 use the mode-explicit `TurboSparkTheme.barBackgroundColor(isDark: theme.isDark)`, which is the correct form per Section 4; not flagged.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Files/AttachmentThumbnailStore.swift
- (store/caching logic lines 1-96 audited as non-UI; UI-facing parts below)
- [MEDIUM] line 108 - Section 4 color - `var symbolTint: Color = .secondary` defaults the fallback glyph to theme-blind `Color.secondary` (and FileRowView passes `Color.secondary` explicitly) - default to a theme-resolved color (resolve via `\.appTheme` in the view) or require callers to pass one.
- [MEDIUM] line 139 - Section 5 motion - bare `withAnimation(.easeIn(duration: 0.15))` with an inline duration; it animates even under Reduce Motion and bypasses the TSMotion curves - use `.tsAnimation(TSMotion.select, value:)` or `TSMotion.animate`.
- [MEDIUM] (possible) lines 121+125 - Section 3 layout - fixed `.frame(width: pixelSize, height: pixelSize)` around a glyph sized with `.themedFont(symbolStep)`: the fallback symbol grows with Text Size inside the fixed frame and clips - size the symbol to the frame via `.themedFont(fitting:)` or scale the frame by `AppMetrics.textScale(theme)`; marked possible because the frame is also the thumbnail pixel contract (cache key includes pixelSize).
- [LOW] line 118 - Section 4 radii - `RoundedRectangle(cornerRadius: 3, style: .continuous)` thumbnail clip; 3 is not an AppMetrics radius.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Files/FilePreviewLeaves.swift
- clean
  - `view.backgroundColor = .clear` (line 47) is transparency, not a system palette color; both NSViewRepresentables carry no UI copy.

#### swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/ChatMessageMarkdownView.swift
- [HIGH] line 323 - Section 6 localization - `.accessibilityLabel(isCopied ? "Copied \(language ?? "code")" : "Copy \(language ?? "code") block")` passes bare String literals (the StringProtocol overload never consults the catalog) and interpolates a dynamic language into them - use the Text ternary form with two bundled keys (`Text("Copied %@", bundle: .module)` / `Text("Copy %@ block", bundle: .module)`), matching line 316 which already does the Text ternary correctly.
- [MEDIUM] line 363 - Section 6 localization (a String that reaches the UI) - VoiceOver announcement `.init("Copied \(language ?? "code") block to clipboard")` is fixed English - build with `AppStrings.localized` and a `%@` format key.
- [MEDIUM] lines 365-366 and 370-371 - Section 5 motion - bare `withAnimation(.easeInOut(duration: 0.15))` / `withAnimation(.easeOut(duration: 0.15))` with inline durations; they ignore Reduce Motion - use `TSMotion.animate` / `.tsAnimation`.
- [MEDIUM] lines 230, 237, 239 (and 247, 254, 256) - Sections 4 and 6 (chart preview chrome) - the generated HTML pages show fixed English UI chrome ("Loading ECharts from cdn.jsdelivr.net (grant network access to render)...", "ECharts failed to load...", "Chart error: ...", and the Chart.js equivalents) painted on hard-coded light colors (`background:#fff`, `color:#666`) that ignore the app theme and dark mode - localize the notices via `AppStrings.localized` before embedding and derive the colors from the resolved theme.
- [LOW] lines 351-354 - Section 4 radii - `RoundedRectangle(cornerRadius: 8, style: .continuous)` used for both clip and border stroke; 8 is not an AppMetrics radius (cardRadius 10) - use `AppMetrics.cardRadius`.
- [LOW] (possible) lines 278-324 - Section 2 vocabulary - the code-block chrome actions (HTML Preview, chart Preview, Copy) are bespoke `.buttonStyle(.plain)` quiet buttons rather than `.appAction(.quiet, size: .small)`; marked possible because they sit inside a MarkdownUI-rendered surface where the row bezel treatment may not fit, but the treatment is a third local copy of a quiet action.
- Note: line 268 `Text(language.lowercased())` resolves to the StringProtocol (verbatim) initializer, which is correct for a language tag; lines 316 and 319 show the correct Text-ternary and theme-resolved foreground forms.
