import Foundation

/// One selectable image style from the bundled catalog: a display name plus
/// the prompt text it injects into the image composer.
public struct AppImageStyle: Identifiable, Sendable, Hashable {
    public let id: String
    public let name: String
    public let prompt: String
    public let categoryID: String
    public let categoryName: String

    init(
        id: String, name: String, prompt: String, categoryID: String, categoryName: String
    ) {
        self.id = id
        self.name = name
        self.prompt = prompt
        self.categoryID = categoryID
        self.categoryName = categoryName
    }
}

/// The bundled image style catalog (`Resources/image-styles.json`), grouped
/// the way the pickers present it.
public enum AppImageStyleCatalog {
    public struct Group: Identifiable, Sendable {
        public let id: String
        public let name: String
        public let styles: [AppImageStyle]
    }

    private struct CatalogFile: Decodable {
        let version: Int
        let categories: [RawCategory]
    }

    private struct RawCategory: Decodable {
        let id: String
        let name: String
        let styles: [RawStyle]
    }

    private struct RawStyle: Decodable {
        let id: String
        let name: String
        let prompt: String
    }

    /// The groups, read from the bundle ONCE (same rule as `AppPromptPreset`,
    /// state#110): these accessors run from SwiftUI bodies, and an empty
    /// decode is not a catalog. Unlike the prompt presets there is no
    /// meaningful hard-coded fallback, so a failed read yields an empty
    /// catalog and the pickers degrade to "None" plus the browse entry.
    public static let groups: [Group] = {
        guard let url = Bundle.module.url(forResource: "image-styles", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let file = try? JSONDecoder().decode(CatalogFile.self, from: data)
        else { return [] }
        var groups: [Group] = []
        for category in file.categories where !category.styles.isEmpty {
            let styles = category.styles
                .filter { !$0.id.isEmpty && !$0.name.isEmpty && !$0.prompt.isEmpty }
                .map {
                    AppImageStyle(
                        id: $0.id, name: $0.name, prompt: $0.prompt,
                        categoryID: category.id, categoryName: category.name)
                }
            guard !styles.isEmpty else { continue }
            groups.append(Group(id: category.id, name: category.name, styles: styles))
        }
        return groups
    }()

    /// Every style flat, in catalog order.
    public static let all: [AppImageStyle] = groups.flatMap(\.styles)

    private static let byID: [String: AppImageStyle] = Dictionary(all.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })

    /// Resolves a style ID (a favorites entry or a stale selection) to its style.
    public static func style(id: String) -> AppImageStyle? {
        byID[id]
    }

    /// Case- and diacritic-insensitive filter over name, category, and prompt
    /// text, used by the searchable browser.
    public static func search(_ query: String) -> [AppImageStyle] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return all }
        return all.filter {
            $0.name.localizedStandardContains(trimmed)
                || $0.categoryName.localizedStandardContains(trimmed)
                || $0.prompt.localizedStandardContains(trimmed)
        }
    }
}
