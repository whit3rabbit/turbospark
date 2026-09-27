import Foundation
import XCTest

@testable import TurboSparkApp

@MainActor
final class ImageStyleTests: XCTestCase {
    // MARK: Catalog

    func testBundledCatalogDecodesAllGroupsAndStyles() {
        XCTAssertEqual(AppImageStyleCatalog.groups.count, 8)
        XCTAssertEqual(AppImageStyleCatalog.all.count, 286)
        let ids = AppImageStyleCatalog.all.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count, "style IDs must be unique")
        XCTAssertTrue(
            AppImageStyleCatalog.all.allSatisfy {
                !$0.name.isEmpty && !$0.prompt.isEmpty && !$0.categoryName.isEmpty
            })
    }

    func testGroupsCarryTheirCategoryStyles() {
        let anime = AppImageStyleCatalog.groups.first { $0.id == "anime" }
        XCTAssertEqual(anime?.name, "Anime")
        XCTAssertEqual(anime?.styles.first?.name, "Anime Style")
        XCTAssertTrue(anime!.styles.allSatisfy { $0.categoryID == "anime" })
    }

    func testStyleLookupResolvesByID() {
        let ghibli = AppImageStyleCatalog.style(id: "ghibli-style")
        XCTAssertEqual(ghibli?.name, "Ghibli Style")
        XCTAssertNil(AppImageStyleCatalog.style(id: "does-not-exist"))
    }

    func testSearchMatchesNameCategoryAndPromptCaseInsensitively() {
        XCTAssertEqual(AppImageStyleCatalog.search("").count, AppImageStyleCatalog.all.count)
        XCTAssertEqual(
            Set(AppImageStyleCatalog.search("GHIBLI").map(\.id)),
            Set(AppImageStyleCatalog.all.filter { $0.name == "Ghibli Style" }.map(\.id)))
        // A diacritic-free query still finds accented names.
        XCTAssertTrue(AppImageStyleCatalog.search("dessinee").contains {
            $0.name == "Bande Dessinée (Hergé/Goscinny Tradition)"
        })
        XCTAssertTrue(AppImageStyleCatalog.search("Akira Toriyama").contains {
            $0.name == "Akira Toriyama Style"
        })
        // Both distinct Film Noir entries stay findable; prompt-text mentions
        // elsewhere may match too, so the count is a lower bound.
        let noirNames = AppImageStyleCatalog.search("Film Noir").map(\.name)
        XCTAssertEqual(noirNames.filter { $0 == "Film Noir" }.count, 2)
    }

    // MARK: Favorites

    func testFavoriteStoreRoundTripsAndToggles() {
        let original = ImageStyleFavoriteStore.ids()
        defer { ImageStyleFavoriteStore.setIDs(original) }

        ImageStyleFavoriteStore.setIDs([])
        let favorites = ImageStyleFavorites()
        XCTAssertFalse(favorites.contains("ghibli-style"))

        favorites.toggle("ghibli-style")
        XCTAssertTrue(favorites.contains("ghibli-style"))
        XCTAssertTrue(ImageStyleFavoriteStore.ids().contains("ghibli-style"))

        favorites.toggle("ghibli-style")
        XCTAssertFalse(favorites.contains("ghibli-style"))
        XCTAssertFalse(ImageStyleFavoriteStore.ids().contains("ghibli-style"))
    }

    func testFavoriteStylesFollowCatalogOrder() {
        let original = ImageStyleFavoriteStore.ids()
        defer { ImageStyleFavoriteStore.setIDs(original) }

        // Photographed later in the catalog than the anime entries.
        ImageStyleFavoriteStore.setIDs(["35mm-photography", "ghibli-style"])
        XCTAssertEqual(
            ImageStyleFavorites().styles.map(\.id),
            ["ghibli-style", "35mm-photography"])
    }

    // MARK: Prompt composition

    private func makeModel() -> AppModel {
        let model = AppModel()
        model.chats = [AppChat(id: UUID(), title: "Image")]
        model.selectedChatID = model.chats[0].id
        return model
    }

    private var ghibli: AppImageStyle { AppImageStyleCatalog.style(id: "ghibli-style")! }
    private var manga: AppImageStyle { AppImageStyleCatalog.style(id: "manga-style")! }

    func testSelectingAStyleAppendsToExistingPromptText() {
        let model = makeModel()
        model.promptText = "a cat on a roof"

        model.setImageStyle(id: ghibli.id)

        XCTAssertEqual(model.promptText, "a cat on a roof, " + ghibli.prompt)
        XCTAssertEqual(model.imageStyleID, ghibli.id)
        XCTAssertEqual(model.appliedImageStyleID, ghibli.id)
    }

    func testSelectingAStyleOnAnEmptyPromptFillsItEntirely() {
        let model = makeModel()

        model.setImageStyle(id: ghibli.id)

        XCTAssertEqual(model.promptText, ghibli.prompt)
    }

    func testSwitchingStylesSwapsTheInjectedTextWithoutStacking() {
        let model = makeModel()
        model.promptText = "a cat on a roof"

        model.setImageStyle(id: ghibli.id)
        model.setImageStyle(id: manga.id)

        XCTAssertEqual(model.promptText, "a cat on a roof, " + manga.prompt)
        XCTAssertFalse(model.promptText.contains(ghibli.prompt))
    }

    func testSelectingNoneStripsTheStyleTextAndKeepsTheUserText() {
        let model = makeModel()
        model.promptText = "a cat on a roof"

        model.setImageStyle(id: ghibli.id)
        model.setImageStyle(id: nil)

        XCTAssertEqual(model.promptText, "a cat on a roof")
        XCTAssertNil(model.imageStyleID)
        XCTAssertNil(model.appliedImageStyleID)
    }

    func testSelectingNoneWithoutAnAppliedStyleLeavesTheDraftAlone() {
        let model = makeModel()
        model.promptText = "a cat on a roof"

        model.setImageStyle(id: nil)

        XCTAssertEqual(model.promptText, "a cat on a roof")
    }

    func testUnknownStyleIDBehavesLikeNone() {
        let model = makeModel()
        model.promptText = "a cat on a roof"

        model.setImageStyle(id: "does-not-exist")

        XCTAssertEqual(model.promptText, "a cat on a roof")
        XCTAssertNil(model.imageStyleID)
    }

    func testRemovingAppliedStyleTextHandlesAllThreeShapes() {
        XCTAssertEqual(AppModel.removingAppliedStyleText("S", from: "S"), "")
        XCTAssertEqual(AppModel.removingAppliedStyleText("S", from: "a cat, S"), "a cat")
        // A rewritten draft belongs to the user, not to the style.
        XCTAssertEqual(AppModel.removingAppliedStyleText("S", from: "S but edited"), "S but edited")
    }
}
