import XCTest
@testable import TurboSpark
@testable import TurboSparkApp

final class ModelManagerTests: XCTestCase {
    func testModelFeatureDescriptorMoEDetection() {
        let gemma = InstalledModel(
            alias: "gemma4-26b",
            repo: "google/gemma-4-26b",
            path: "/path/to/gemma4",
            family: "gemma4",
            installBytes: 13_000_000_000
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: gemma)
        XCTAssertEqual(desc.routingType, .moe)
        XCTAssertTrue(desc.isSlotCacheStreaming)
        XCTAssertTrue(desc.isSteeringReady)
        XCTAssertTrue(desc.supportsChunkedPrefill)
    }

    func testModelFeatureDescriptorDenseDetection() {
        let mistral = InstalledModel(
            alias: "mistral7b",
            repo: "mistralai/Mistral-7B-Instruct-v0.3",
            path: "/path/to/mistral7b.gguf",
            family: "mistral",
            installBytes: 4_300_000_000
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: mistral)
        XCTAssertEqual(desc.routingType, .dense)
        XCTAssertFalse(desc.isSlotCacheStreaming)
        XCTAssertEqual(desc.format, .gguf)
    }

    func testModelFeatureDescriptorDrafters() {
        let dflashModel = InstalledModel(
            alias: "qwen38-27b-dflash2",
            repo: "qwen/qwen38-27b-dflash2",
            path: "/path/to/qwen-dflash",
            family: "qwen38"
        )
        let dflashDesc = ModelFeatureDescriptor.resolve(installedModel: dflashModel)
        XCTAssertEqual(dflashDesc.speculativeDrafter, .dynamicSloth)

        let mtpModel = InstalledModel(
            alias: "qwen38-27b-mtp",
            repo: "qwen/qwen38-27b-mtp",
            path: "/path/to/qwen-mtp",
            family: "qwen38"
        )
        let mtpDesc = ModelFeatureDescriptor.resolve(installedModel: mtpModel)
        XCTAssertEqual(mtpDesc.speculativeDrafter, .mtp)
    }

    func testModelFeatureDescriptorQuantFormats() {
        let ternaryModel = InstalledModel(
            alias: "ternary-bonsai-27b",
            repo: "microsoft/ternary-bonsai",
            path: "/path/to/ternary",
            family: "custom"
        )
        let ternaryDesc = ModelFeatureDescriptor.resolve(installedModel: ternaryModel)
        XCTAssertTrue(ternaryDesc.quantFormat.contains("Ternary") || ternaryDesc.quantFormat.contains("1.58"))

        let bonsaiModel = InstalledModel(
            alias: "bonsai27b-1bit",
            repo: "meta/bonsai",
            path: "/path/to/bonsai",
            family: "llama"
        )
        let bonsaiDesc = ModelFeatureDescriptor.resolve(installedModel: bonsaiModel)
        XCTAssertTrue(bonsaiDesc.quantFormat.contains("1-Bit"))

        let iq3Model = InstalledModel(
            alias: "gemma4-iq3_xxs",
            repo: "unsloth/gemma4-iq3",
            path: "/path/to/gemma4-iq3.gguf",
            family: "gemma4"
        )
        let iq3Desc = ModelFeatureDescriptor.resolve(installedModel: iq3Model)
        XCTAssertTrue(iq3Desc.quantFormat.contains("IQ3_XXS"))
    }

    func testModelFeatureDescriptorLinearAttention() {
        let qwen36 = InstalledModel(
            alias: "qwen36-35b",
            repo: "qwen/qwen36-35b",
            path: "/path/to/qwen36",
            family: "qwen36"
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: qwen36)
        XCTAssertTrue(desc.hasLinearAttention)
        XCTAssertTrue(desc.isSteeringReady)
    }

    func testModelFeatureDescriptorSources() {
        let lmStudio = InstalledModel(
            alias: "llama3-lmstudio",
            repo: "lm studio/llama3",
            path: "/Users/user/.lmstudio/models/llama3.gguf",
            family: "llama"
        )
        let lmDesc = ModelFeatureDescriptor.resolve(installedModel: lmStudio)
        XCTAssertEqual(lmDesc.storageSource, .lmStudio)

        let custom = InstalledModel(
            alias: "custom-model",
            repo: "custom/model",
            path: "/Volumes/External/models/model.gturbo",
            family: "custom"
        )
        let customDesc = ModelFeatureDescriptor.resolve(installedModel: custom)
        XCTAssertEqual(customDesc.storageSource, .custom)

        let turbospark = InstalledModel(
            alias: "qwen36",
            repo: "qwen/qwen36",
            path: "/Users/user/.turbospark/models/qwen36.gturbo",
            family: "qwen36"
        )
        let tsDesc = ModelFeatureDescriptor.resolve(installedModel: turbospark)
        XCTAssertEqual(tsDesc.storageSource, .turboSpark)
    }

    // MARK: - Catalog Join for Format/Quant (U1)

    func testCatalogInstalledGGUFRowResolvesItsRealFormatAndQuant() {
        // "qwen3moe" is a real curated row (models.json) named
        // "Qwen3-30B-A3B (GGUF Q4_K_M)". Its actual on-disk install is a
        // repacked `.gturbo` bundle, NOT a literal `.gguf` file, so the
        // path-extension heuristic alone cannot see its original format --
        // only the catalog's own descriptive name can. No `catalogEntry` is
        // passed here on purpose: `resolve` must join it itself (U1).
        let installed = InstalledModel(
            alias: "qwen3moe",
            repo: "Qwen/Qwen3-30B-A3B-GGUF",
            path: "/Users/test/.turbospark/models/qwen3moe.gturbo",
            family: "qwen3moe",
            installBytes: 18_000_000_000
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: installed)
        XCTAssertEqual(desc.format, .gguf, "A catalog row whose name states GGUF must not render as MLX just because its resident install is a .gturbo bundle.")
        XCTAssertTrue(desc.quantFormat.contains("Q4_K_M"), "Expected the catalog's own stated quant in quantFormat, got '\(desc.quantFormat)'")
    }

    func testCatalogInstalledMLXRowResolvesItsRealFormatAndQuant() {
        let installed = InstalledModel(
            alias: "gemma4",
            repo: "google/gemma-4",
            path: "/Users/test/.turbospark/models/gemma4.gturbo",
            family: "gemma4",
            installBytes: 13_000_000_000
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: installed)
        XCTAssertEqual(desc.format, .mlx)
        XCTAssertTrue(desc.quantFormat.contains("INT4"), "Expected the catalog's own stated quant, got '\(desc.quantFormat)'")
    }

    // MARK: - Routing Determination Is Not Fooled by Aliases (U2)

    func testDenseFamilyIsNotForcedToMoEByAnAliasSubstring() {
        // qwen38 is this repo's DENSE MTP/DFlash2 family
        // (`CLAUDE.local.md`), not MoE -- the old alias-substring chain
        // included `lAlias.contains("qwen38")` in its MoE check.
        let dense = InstalledModel(
            alias: "qwen38-27b-mtp",
            repo: "qwen/qwen38-27b",
            path: "/path/to/qwen38.gturbo",
            family: "qwen38"
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: dense)
        XCTAssertEqual(desc.routingType, .dense)
    }

    func testASideLoadedModelWithGemmaInItsAliasIsNotForcedToMoE() {
        // A side-loaded model whose USER-CHOSEN alias happens to contain
        // "gemma" but whose actual scanner-assigned family is something
        // else must not be classified MoE on alias text alone.
        let sideLoaded = InstalledModel(
            alias: "my-gemma-finetune",
            repo: "custom/my-gemma-finetune",
            path: "/Volumes/External/my-gemma-finetune.gguf",
            family: "llama"
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: sideLoaded)
        XCTAssertEqual(desc.routingType, .dense)
    }

    // MARK: - Unmeasured Facts Are Nil, Not Guessed (U3)

    func testContextLimitIsNilWithoutARealManifestReading() {
        let noManifest = InstalledModel(
            alias: "no-manifest-model",
            repo: "someone/model",
            path: "/tmp/does-not-exist-\(UUID().uuidString)",
            family: "mistral"
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: noManifest)
        XCTAssertNil(desc.contextLimit, "An unread trained context must be nil, not a guessed 8192/2048/4096 default.")
    }

    func testRamEstimateIsNilForAnUnrecognizedMoEFamily() {
        let unknownMoE = InstalledModel(
            alias: "some-future-moe-model",
            repo: "someone/future-moe",
            path: "/tmp/does-not-exist-\(UUID().uuidString)",
            family: "mixtral" // in the known set, but with 0 install bytes and no manifest
        )
        // Force through the MoE branch via a recognized family, but with no
        // real data behind the estimate for anything outside the two
        // grounded buckets (gemma4, qwen36).
        let desc = ModelFeatureDescriptor.resolve(installedModel: unknownMoE)
        XCTAssertEqual(desc.routingType, .moe)
        XCTAssertNil(desc.estimatedWorkingSetRAM, "An MoE family with no measured oracle ceiling must report nil, not an invented average.")
    }

    func testRamEstimateUsesRealInstallBytesForDenseModels() {
        let dense = InstalledModel(
            alias: "dense-with-bytes",
            repo: "someone/dense",
            path: "/tmp/does-not-exist-\(UUID().uuidString)",
            family: "mistral",
            installBytes: 4_300_000_000
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: dense)
        XCTAssertEqual(desc.estimatedWorkingSetRAM, 4_300_000_000)
    }

    func testRamEstimateIsNilForADenseModelWithNoKnownSize() {
        let dense = InstalledModel(
            alias: "dense-no-bytes",
            repo: "someone/dense",
            path: "/tmp/does-not-exist-\(UUID().uuidString)",
            family: "mistral",
            installBytes: 0
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: dense)
        XCTAssertNil(desc.estimatedWorkingSetRAM, "Zero install bytes must not become a flat 4GB guess.")
    }

    // MARK: - Storage Source Respects TURBOSPARK_HOME (U4)

    func testStorageSourceRespectsTurboSparkHomeOverride() {
        let customHome = "/Volumes/Data/turbospark-store"
        setenv("TURBOSPARK_HOME", customHome, 1)
        defer { unsetenv("TURBOSPARK_HOME") }

        let model = InstalledModel(
            alias: "under-custom-home",
            repo: "someone/model",
            path: "\(customHome)/models/under-custom-home.gturbo",
            family: "mistral"
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: model)
        XCTAssertEqual(desc.storageSource, .turboSpark, "A path under a configured TURBOSPARK_HOME must resolve as the TurboSpark store, not 'Custom Folder'.")
    }

    @MainActor
    func testModelOrganizationStore() {
        let store = ModelOrganizationStore.shared
        let alias = "test-model-\(UUID().uuidString)"
        let path = "/tmp/\(alias)"

        // Initial state
        XCTAssertFalse(store.isFavorite(alias: alias, path: path))
        XCTAssertEqual(store.notes(alias: alias, path: path), "")
        XCTAssertEqual(store.tags(alias: alias, path: path), [])

        // Favorite toggle
        store.toggleFavorite(alias: alias, path: path)
        XCTAssertTrue(store.isFavorite(alias: alias, path: path))
        store.toggleFavorite(alias: alias, path: path)
        XCTAssertFalse(store.isFavorite(alias: alias, path: path))

        // Notes
        store.setNotes("Best model for Swift testing", for: alias, path: path)
        XCTAssertEqual(store.notes(alias: alias, path: path), "Best model for Swift testing")

        // Nickname
        store.setNickname("Fast Swift Drafter", for: alias, path: path)
        XCTAssertEqual(store.nickname(alias: alias, path: path), "Fast Swift Drafter")

        // Tags
        store.addTag("Coding", for: alias, path: path)
        store.addTag("Fast", for: alias, path: path)
        XCTAssertEqual(store.tags(alias: alias, path: path), ["Coding", "Fast"])
        XCTAssertTrue(store.allKnownTags.contains("Coding"))
        XCTAssertTrue(store.allKnownTags.contains("Fast"))

        // Remove tag
        store.removeTag("Coding", for: alias, path: path)
        XCTAssertEqual(store.tags(alias: alias, path: path), ["Fast"])

        // Cleanup
        store.removeMetadata(for: alias, path: path)
        XCTAssertFalse(store.isFavorite(alias: alias, path: path))
        XCTAssertEqual(store.notes(alias: alias, path: path), "")
    }

    func testAppNavigationSectionAllCases() {
        let sections = AppModel.AppNavigationSection.allCases
        XCTAssertEqual(sections.count, 4)
        XCTAssertTrue(sections.contains(.chat))
        XCTAssertTrue(sections.contains(.files))
        XCTAssertTrue(sections.contains(.modelManager))
        XCTAssertTrue(sections.contains(.modelHub))

        XCTAssertEqual(AppModel.AppNavigationSection.modelManager.shortcutKey, "3")
        XCTAssertEqual(AppModel.AppNavigationSection.modelHub.shortcutKey, "4")
    }
}
