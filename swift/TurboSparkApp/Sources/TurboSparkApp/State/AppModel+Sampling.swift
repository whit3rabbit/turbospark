import Foundation
import TurboSpark

/// Per-conversation sampling: resolution, the chat-override mutators, the
/// app-wide snapshot writer, and the preset registry over
/// `AppSamplingSettings`.
///
/// `samplingOptions()` (the app-wide builder this file does not touch)
/// stays the source `AppToolRegistry.subagentSamplingOptionsProvider`
/// hands to `SubagentRunner`: a subagent launch carries no chat id, so
/// subagents deliberately read the app-wide settings rather than the
/// originating chat's override.
extension AppModel {
    /// A snapshot of the app-wide Generation Sampling properties.
    ///
    /// One builder for the Inspector's read side, `applyGlobalSampling`'s
    /// write side's sibling, and `samplingOptions(for:)`'s fallback -- three
    /// readers, one spelling (the reason `resolvedSteeringPreset` exists).
    func globalSamplingSettings() -> AppSamplingSettings {
        AppSamplingSettings(
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            maxNewTokens: maxNewTokens,
            repetitionPenaltyEnabled: repetitionPenaltyEnabled,
            repetitionPenalty: repetitionPenalty,
            seedEnabled: seedEnabled,
            seed: seed,
            stopSequences: stopSequences
        )
    }

    /// Writes a complete snapshot back into the app-wide sampling
    /// properties, the inverse of `globalSamplingSettings()`.
    ///
    /// Called per slider tick from the Inspector's "app defaults" scope, so
    /// this persists DEBOUNCED, not immediately -- the same reason
    /// `persistChatsDebounced` exists for the chat archive.
    func applyGlobalSampling(_ settings: AppSamplingSettings) {
        let clamped = settings.clamped()
        temperature = clamped.temperature
        topKEnabled = clamped.topKEnabled
        topK = clamped.topK
        topPEnabled = clamped.topPEnabled
        topP = clamped.topP
        maxNewTokens = clamped.maxNewTokens
        repetitionPenaltyEnabled = clamped.repetitionPenaltyEnabled
        repetitionPenalty = clamped.repetitionPenalty
        seedEnabled = clamped.seedEnabled
        seed = clamped.seed
        stopSequences = clamped.stopSequences
        persistSettingsDebounced()
    }

    /// The sampling a turn in THIS chat runs with: the chat's own override
    /// when one is set, the app-wide settings otherwise.
    ///
    /// **BY ID, NEVER THE SELECTION** (state#17): a turn's continuation
    /// resolves long after the user became free to click another chat.
    func effectiveSamplingSettings(chatID: UUID?) -> AppSamplingSettings {
        guard let chatID,
            let chat = chats.first(where: { $0.id == chatID }),
            let override = chat.samplingOverride
        else {
            return globalSamplingSettings()
        }
        return override
    }

    /// `samplingOptions()` for a specific conversation, with `maxNewTokens`
    /// folded in: the call site in `executeGenerationTurn` used to read the
    /// global property directly, which is precisely the read a per-chat
    /// override has to replace.
    func samplingOptions(chatID: UUID) -> GenerateOptions {
        Self.makeOptions(from: effectiveSamplingSettings(chatID: chatID))
    }

    /// The one place an `AppSamplingSettings` becomes engine options.
    ///
    /// Same shape as the app-wide `samplingOptions()`: a toggle that is off
    /// leaves the engine default standing, and `UInt32(clamping:)` because
    /// a plain conversion TRAPS on an out-of-range value (state#35).
    /// `clamped()` has already bounded everything, but the clamps are cheap
    /// and the property they protect is "no persisted integer can kill the
    /// process", not "the decoder is careful".
    private static func makeOptions(from settings: AppSamplingSettings) -> GenerateOptions {
        var options = GenerateOptions()
        let clamped = settings.clamped()
        options.temperature = clamped.temperature
        if clamped.topKEnabled {
            options.topK = UInt32(clamping: clamped.topK)
        }
        if clamped.topPEnabled {
            options.topP = clamped.topP
        }
        if clamped.repetitionPenaltyEnabled {
            options.repetitionPenalty = clamped.repetitionPenalty
        }
        if clamped.seedEnabled {
            options.seed = clamped.seed
        }
        let customStops = clamped.stopSequences
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if !customStops.isEmpty {
            options.stop = customStops
        }
        options.maxNewTokens = UInt32(clamping: clamped.maxNewTokens)
        return options
    }

    /// Sets THIS chat's own sampling override, the per-chat sibling of
    /// `setChatSystemPrompt`. Called per slider tick from the Inspector, so
    /// the archive write is debounced.
    public func setChatSamplingOverride(id: UUID, settings: AppSamplingSettings) {
        if chats.firstIndex(where: { $0.id == id }) == nil {
            // The selected chat can be an unmaterialized DRAFT (no row yet;
            // `selectedChat`'s `activeDraftChat` fallback). An explicit
            // sampling edit IS content worth promoting the row for, which
            // `materializeDraftChatIfNeeded`'s has-content predicate counts
            // -- set on the draft first, then let it insert.
            guard id == selectedChatID else { return }
            activeDraftChat.samplingOverride = settings.clamped()
            materializeDraftChatIfNeeded()
            return
        }
        guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
        chats[index].samplingOverride = settings.clamped()
        chats[index].updatedAt = Date()
        persistChatsDebounced()
    }

    /// Clears THIS chat's override, returning it to the app-wide settings.
    /// A no-op (and no write) when the chat carries none.
    public func removeChatSamplingOverride(id: UUID) {
        guard let index = chats.firstIndex(where: { $0.id == id }) else {
            // Same draft case as `setChatSamplingOverride`: the override may
            // sit on the not-yet-inserted draft row.
            if id == selectedChatID {
                activeDraftChat.samplingOverride = nil
            }
            return
        }
        guard chats[index].samplingOverride != nil else { return }
        chats[index].samplingOverride = nil
        chats[index].updatedAt = Date()
        persistChatsDebounced()
    }

    /// Adds or replaces a sampling preset, keyed on `id`. A save naming an
    /// existing preset overwrites that preset (the id is looked up by name
    /// by the caller), which is the predictable reading of "Save".
    public func upsertSamplingPreset(_ preset: AppSamplingPreset) {
        if let index = samplingPresets.firstIndex(where: { $0.id == preset.id }) {
            samplingPresets[index] = preset
        } else {
            samplingPresets.append(preset)
        }
        persistSettings()
    }

    /// Removes a sampling preset. A no-op (and no write) when the id is
    /// already gone, so a double-delete cannot dirty the settings file.
    public func deleteSamplingPreset(_ id: UUID) {
        guard samplingPresets.contains(where: { $0.id == id }) else { return }
        samplingPresets.removeAll { $0.id == id }
        persistSettings()
    }
}
