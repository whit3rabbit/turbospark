# Components/Hooks

## Overview
The SwiftUI settings surface for lifecycle hooks. It renders the Hooks settings pane (source groups, review status, discovery diagnostics), a per-group detail sheet that lists hooks by lifecycle event with trust, enable, edit, and delete controls, an editor sheet for custom hooks, and an options sheet for plugin `userConfig` values. It holds no hook logic of its own: all state and mutation go through `AppHookStore.shared` (see `Tools/Hooks`), and the views only present and forward user actions.

## Interface
- [HooksSettingsPaneView.swift:HooksSettingsPaneView] (`init(model: AppModel)`): the pane. Refreshes the store for the selected project on appear and after the plugin marketplace sheet closes, searches groups by title, subtitle, hook name, or command, and presents the other sheets and `PluginMarketplaceSheet`.
- [HookSourceDetailSheet.swift:HookSourceDetailSheet] (`init(groupID:)`): per-group detail, looked up live from `hookStore.sourceGroups` by id. Offers Trust, Trust All in Group, enable toggle, reveal source file, expand details, and (custom hooks only) Edit and Delete.
- [HookEditorSheet.swift:HookEditorSheet] (`init(existingHook:onSave:onDismiss:)`, `selectableTypes(existing:)`): create or edit a hook; emits an `AppHookCommand` through `onSave`.
- [HookOptionsConfigSheet.swift:HookOptionsConfigSheet] (`init(group:)`): edits `AppHookOptionSpec` values for a group (boolean, number, string, file, directory).
- Subdirectories: none.

## Invariants
- The views never write hook files or Keychain entries directly; every change goes through `AppHookStore` (`addCustomHook`, `updateCustomHook`, `deleteCustomHook`, `trustHook`, `trustAllInGroup`, `toggleHookEnabled`, `updateOptionValue`). [HooksSettingsPaneView.swift], [HookSourceDetailSheet.swift], [HookOptionsConfigSheet.swift]
- Only custom hooks are editable or deletable here: the detail sheet saves edits only when `updated.sourceType == .custom` and shows Edit and Delete only for custom hooks. Deletion requires a confirmation dialog. [HookSourceDetailSheet.swift]
- A new `.prompt` hook cannot be created. `selectableTypes` hides it unless the hook being edited already has that type, and the Save button is disabled for a prompt type on a non-prompt hook, because the engine never evaluates prompt hooks. Existing prompt hooks show a "not evaluated" badge. [HookEditorSheet.swift:selectableTypes], [HookSourceDetailSheet.swift]
- The async toggle is disabled for `AppHookExecutionEngine.blockingEvents`, since the engine ignores async for them. Save is disabled for an empty command. A non-positive timeout is saved as 30 seconds. [HookEditorSheet.swift]
- Editing preserves identity and provenance: `id`, `isEnabled`, `sourceType`, `sourcePath`, `pluginName`, and `options` are carried from the existing hook; a new hook defaults to `.custom`. [HookEditorSheet.swift:save]
- Trust state shown to the user comes from `hookStore.isHookTrusted`; "Needs review" counts come from `AppHookSourceGroup.unreviewedCount`. The view must not compute trust itself. [HooksSettingsPaneView.swift], [HookSourceDetailSheet.swift]
- Sensitive options use `SecureField` and pass `isSensitive: true` so the store routes the value to the Keychain. The displayed env variable name uses `AppHookExecutionEngine.optionEnvSuffix`, which must match what hooks receive. [HookOptionsConfigSheet.swift]
- Option edits persist on every change (no Save step); the Done button only dismisses. [HookOptionsConfigSheet.swift]
- Per the project design rules, UI strings use bundled `Text(..., bundle: .module)` or `AppStrings.localized`, and colors, fonts, and controls come from theme helpers (`themedFont`, `appAction`, `AppAttention.color`). `LocalizedStringKey(t.title)` is used for hook type titles that are catalog keys. [HookEditorSheet.swift], [HooksSettingsPaneView.swift]

## Side Effects
- Triggers `AppHookStore.refresh(projectDirectory:)` using `model.selectedProject?.rootDirectoryPath`, which rescans config files and plugins on disk. [HooksSettingsPaneView.swift:refresh]
- Store mutations written to disk and Keychain by the store, not by the views. [HookSourceDetailSheet.swift], [HookOptionsConfigSheet.swift]
- Presents an `NSOpenPanel` to choose a file or directory option value. [HookOptionsConfigSheet.swift:choosePath]
- Reveals a hook's source file in Finder via `NSWorkspace.shared.selectFile`. [HookSourceDetailSheet.swift]
- Opens the Claude Code hooks documentation URL through the intro component's documentation link. [HooksSettingsPaneView.swift:header]
- Animates row expansion through `TSMotion`. [HookSourceDetailSheet.swift]

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
- For UI changes also run `make design-check` (see `DESIGN.md`).
