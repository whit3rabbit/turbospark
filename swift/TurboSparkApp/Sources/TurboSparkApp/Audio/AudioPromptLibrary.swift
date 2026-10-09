import Foundation
import TurboSpark

extension AudioWorkspaceController {
    var canSavePrompt: Bool {
        guard !isShutDown else { return false }
        let content = task == .music ? recipe.caption : recipe.text
        return (task == .music || task == .textToSpeech)
            && !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func prompts(matching query: String) -> [AudioPreset] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return presets.filter { preset in
            preset.recipe.task == task.rawValue && (query.isEmpty
                || [preset.name, preset.recipe.caption, preset.recipe.lyrics, preset.recipe.text]
                    .contains { $0.localizedCaseInsensitiveContains(query) })
        }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    @discardableResult func savePrompt(name: String) -> Bool {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSavePrompt, !name.isEmpty else { return false }
        var savedRecipe = recipe
        savedRecipe.task = task.rawValue
        return persistPresets(preferences.presets + [AudioPreset(name: name, recipe: savedRecipe)])
    }

    func loadPrompt(_ prompt: AudioPreset) {
        guard !isBusy, !isShutDown, prompt.recipe.task == task.rawValue else { return }
        // Prompt reuse must not silently replace the user's model, duration,
        // seed, or experimental backend. Full recipes remain available as presets.
        if task == .music { recipe.caption = prompt.recipe.caption; recipe.lyrics = prompt.recipe.lyrics }
        else if task == .textToSpeech { recipe.text = prompt.recipe.text }
    }

    @discardableResult func renamePrompt(_ prompt: AudioPreset, name: String) -> Bool {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, let index = preferences.presets.firstIndex(where: { $0.id == prompt.id }) else { return false }
        var presets = preferences.presets
        presets[index].name = name
        return persistPresets(presets)
    }

    @discardableResult func deletePrompt(_ prompt: AudioPreset) -> Bool {
        persistPresets(preferences.presets.filter { $0.id != prompt.id })
    }

    private func persistPresets(_ presets: [AudioPreset]) -> Bool {
        guard !isShutDown else { return false }
        var next = preferences
        next.page = page; next.recipe = recipe; next.presets = presets
        do {
            try library.savePreferences(next)
            preferences = next
            return true
        } catch { fail(error); return false }
    }
}
