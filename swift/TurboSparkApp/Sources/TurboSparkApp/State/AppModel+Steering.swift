import Foundation
import TurboSpark

/// Directional steering: which direction is selected, whether it can run
/// here, and whether the model already loaded is running it.
///
/// The DECISIONS live in `AppSteeringPolicy`, a pure type, for
/// `swift/CLAUDE.md` Gotcha 26's reason: an assertion written against
/// `AppModel` needs a real install to reach and therefore never runs. What is
/// here is the plumbing that reads this model's state and hands it over.
extension AppModel {
    /// The preset a session would open with: the selected one, or the
    /// Inspector's raw knobs as an implicit "Custom" preset.
    ///
    /// **ONE ANSWER, READ BY BOTH THE UI AND `buildOpenOptions`.** Two
    /// spellings would let the pill say one thing and the open do another,
    /// which is the shape `activeLoadGuard` exists to prevent for the memory
    /// tier (Gotcha 25).
    public var resolvedSteeringPreset: AppSteeringPreset? {
        if let id = activeSteeringPresetID,
            let preset = steeringPresets.first(where: { $0.id == id })
        {
            return preset
        }
        let raw = (runtimeOptions.steeringPath ?? steeringPath)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let raw, !raw.isEmpty else { return nil }
        return AppSteeringPreset(
            name: "Custom (Inspector)",
            vectorPath: raw,
            mode: runtimeOptions.steeringMode,
            scale: runtimeOptions.steeringScale,
            layers: runtimeOptions.steeringLayers,
            target: runtimeOptions.steeringTarget,
            gate: runtimeOptions.steeringGate,
            // The raw knobs never read the file, so the shape is genuinely
            // unknown here rather than zero (Gotcha 23).
            vectorHidden: nil,
            vectorSpannedLayers: nil
        )
    }

    /// Whether the LOADED session's family dispatches the edit at all, or nil
    /// before any session exists.
    ///
    /// Read off `info.steering.supported` and never off a family table here:
    /// the engine owns that predicate and refuses an open on it, so a second
    /// copy could tell a user a family steers while the load refuses it
    /// (root Gotcha 61's shape).
    public var steeringFamilySupported: Bool? { info?.steering.supported }

    /// The selected install's own hidden size, read off its `manifest.json`.
    ///
    /// `nil` when no model is selected or the manifest declares none, which
    /// makes the shape check report `unknown` rather than inventing a zero
    /// (Gotcha 23).
    public var selectedModelHiddenSize: Int? {
        selected.flatMap { ModelFeatureDescriptor.resolve(installedModel: $0).hiddenSize }
    }

    /// The selected install's own layer count, from the same manifest.
    public var selectedModelLayerCount: Int? {
        selected.flatMap { ModelFeatureDescriptor.resolve(installedModel: $0).layerCount }
    }

    /// The selected preset's shape against the selected install's.
    public var steeringCompatibility: AppSteeringPolicy.Compatibility {
        AppSteeringPolicy.compatibility(
            preset: resolvedSteeringPreset,
            modelHidden: selectedModelHiddenSize,
            modelLayers: selectedModelLayerCount
        )
    }

    /// Re-opens the selected model so a steering change takes effect.
    ///
    /// Goes through `reloadModel()` rather than opening directly, so the
    /// guards every other reload observes (a turn in flight, an open already
    /// running, the server attachment) apply here too.
    public func reloadForSteering() {
        reloadModel()
    }

    /// Why the steering control is disabled, or nil when it is usable.
    public var steeringDisabledReason: String? {
        AppSteeringPolicy.disabledReason(
            familySupported: steeringFamilySupported,
            familyReason: info?.steering.reason,
            preset: resolvedSteeringPreset,
            compatibility: steeringCompatibility
        )
    }

    /// Whether the open session is running what the settings currently ask
    /// for. **Steering resolves once, at open**, so a change here does
    /// nothing until the model is reloaded, and saying so is the whole point.
    public var steeringNeedsReload: Bool {
        AppSteeringPolicy.needsReload(
            wantEnabled: steeringEnabled,
            wantPreset: resolvedSteeringPreset,
            sessionSteering: info?.steering
        )
    }

    /// What the LOADED session is actually steering with, for a status row.
    /// Read back rather than restated: `summary` is the engine's own line.
    public var activeSteeringSummary: String? {
        guard let steering = info?.steering, steering.active else { return nil }
        return steering.summary
    }

    public func setSteeringEnabled(_ enabled: Bool) {
        steeringEnabled = enabled
        persistSettings()
    }

    public func selectSteeringPreset(_ id: UUID?) {
        activeSteeringPresetID = id
        persistSettings()
    }

    /// Adds or replaces a preset, keyed on `id`.
    public func upsertSteeringPreset(_ preset: AppSteeringPreset) {
        if let index = steeringPresets.firstIndex(where: { $0.id == preset.id }) {
            steeringPresets[index] = preset
        } else {
            steeringPresets.append(preset)
        }
        persistSettings()
    }

    /// Removes a preset, and clears the selection if it was the one selected.
    ///
    /// **AND TURNS STEERING OFF when the selection goes**, rather than
    /// silently falling through to the Inspector's raw knobs. Deleting the
    /// direction you were steering with and then steering with a different
    /// one is exactly the outcome `resolvedSteeringPreset`'s fallback would
    /// produce here.
    public func deleteSteeringPreset(_ id: UUID) {
        steeringPresets.removeAll { $0.id == id }
        if activeSteeringPresetID == id {
            activeSteeringPresetID = nil
            steeringEnabled = false
        }
        persistSettings()
    }

    /// Reads a control vector's header and records its shape on `preset`.
    ///
    /// Goes through the engine's own parser (`TurboSparkCatalog
    /// .controlVectorInfo`), so this app never reimplements the GGUF layout
    /// and cannot disagree with what the open will make of the same file.
    /// Milliseconds: a vector is around 1.3 MB and no model is involved.
    public static func readingVectorShape(into preset: AppSteeringPreset) -> (
        preset: AppSteeringPreset, info: ControlVectorInfo?, error: String?
    ) {
        var updated = preset
        let path = preset.vectorPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else {
            updated.vectorHidden = nil
            updated.vectorSpannedLayers = nil
            return (updated, nil, nil)
        }
        do {
            let info = try TurboSparkCatalog.controlVectorInfo(path: path)
            updated.vectorHidden = info.hidden
            updated.vectorSpannedLayers = info.spannedLayers
            // The file's own declared mode is used only when the preset is
            // still at its default, so re-reading a vector never overwrites a
            // mode the operator chose.
            if let declared = info.declaredMode,
                let mode = AppSteeringModeOption(rawValue: declared),
                preset.mode == .ablate
            {
                updated.mode = mode
            }
            return (updated, info, nil)
        } catch {
            // The shape is cleared rather than left stale: a preset carrying
            // the width of a file it no longer points at would pass a
            // compatibility check the open then fails.
            updated.vectorHidden = nil
            updated.vectorSpannedLayers = nil
            return (updated, nil, (error as? TurboSparkError)?.message ?? "\(error)")
        }
    }
}
