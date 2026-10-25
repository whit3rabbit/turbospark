import Combine
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 5.5 (requirement 8.1): organization metadata and reasoning defaults
/// follow a model's `install_id`, not the absolute path it happens to be at.
///
/// Every store here is an injected one over a temp directory, so nothing
/// touches the test process's shared `ModelOrganizationStore`. The few cases
/// that go through `AppModel` use the shared store and clean up after
/// themselves with unique aliases.
@MainActor
final class ModelOrganizationIdentityTests: XCTestCase {
    private var directory: URL!
    private var metadataFile: URL { directory.appendingPathComponent("model_organization.json") }
    private var exclusionsFile: URL { directory.appendingPathComponent("excluded_scan_paths.json") }

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("org-identity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: - Fixtures

    private let oldPath = "/Volumes/T7/TurboSpark/text/qwen.gturbo"
    private let newPath = "/Users/test/.turbospark/models/text/qwen.gturbo"
    private var legacyPathKey: String { "qwen::\(oldPath)" }

    /// A row the engine listed inside library `library`.
    private func libraryRow(
        alias: String = "qwen", library: String = "T7", path: String? = nil
    ) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "owner/\(alias)",
            path: path ?? "/Volumes/\(library)/TurboSpark/text/\(alias).gturbo",
            family: "qwen", library: library, libraryName: library,
            installId: "\(library):text/\(alias)", locationKind: .library)
    }

    /// A This Mac row as the engine lists it (`library` is "internal").
    private func internalRow(alias: String = "qwen", path: String? = nil) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "owner/\(alias)",
            path: path ?? "/Users/test/.turbospark/models/text/\(alias).gturbo",
            family: "qwen", library: "internal", libraryName: "This Mac",
            installId: "internal:text/\(alias)", locationKind: .thisMac)
    }

    /// A row the app built itself for another app's folder: the memberwise
    /// defaults, so `library` is nil and the id is the placeholder.
    private func placeholderRow(alias: String = "lm-model", path: String? = nil) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "lmstudio/\(alias)", revision: "local",
            path: path ?? "/Users/test/.lmstudio/models/\(alias).gguf", family: "qwen")
    }

    private func openStore(seed: [String: ModelCustomMetadata]? = nil) throws
        -> ModelOrganizationStore
    {
        if let seed {
            try JSONEncoder().encode(seed).write(to: metadataFile, options: .atomic)
        }
        return ModelOrganizationStore(metadataFile: metadataFile, exclusionsFile: exclusionsFile)
    }

    private func persisted() throws -> [String: ModelCustomMetadata] {
        try JSONDecoder().decode(
            [String: ModelCustomMetadata].self, from: Data(contentsOf: metadataFile))
    }

    /// Any mutation saves the file, which is when retired legacy keys go.
    /// Using an unrelated model keeps the assertion about the model under test.
    private func forceASave(_ store: ModelOrganizationStore) {
        store.setNotes("unrelated", for: libraryRow(alias: "unrelated-save-trigger"))
    }

    private struct FieldKind {
        let name: String
        let seeded: ModelCustomMetadata
        let observe: (ModelOrganizationStore, InstalledModel) -> String
        let expected: String
    }

    private var fieldKinds: [FieldKind] {
        [
            FieldKind(
                name: "favorite", seeded: ModelCustomMetadata(isFavorite: true),
                observe: { "\($0.isFavorite($1))" }, expected: "true"),
            FieldKind(
                name: "tags", seeded: ModelCustomMetadata(tags: ["coding", "fast"]),
                observe: { $0.tags($1).joined(separator: ",") }, expected: "coding,fast"),
            FieldKind(
                name: "notes", seeded: ModelCustomMetadata(notes: "best for swift"),
                observe: { $0.notes($1) }, expected: "best for swift"),
            FieldKind(
                name: "nickname", seeded: ModelCustomMetadata(nickname: "Drafter"),
                observe: { $0.nickname($1) ?? "nil" }, expected: "Drafter"),
            FieldKind(
                name: "reasoning default", seeded: ModelCustomMetadata(reasoningDefault: "high"),
                observe: { $0.reasoningDefault(for: $1) ?? "nil" }, expected: "high"),
        ]
    }

    // MARK: - Lookup by install id

    func testEachFieldKindIsFoundByInstallId() throws {
        let model = libraryRow()
        for kind in fieldKinds {
            try? FileManager.default.removeItem(at: metadataFile)
            let store = try openStore(seed: [model.installId: kind.seeded])
            XCTAssertEqual(kind.observe(store, model), kind.expected, kind.name)
            XCTAssertEqual(
                Set(store.metadataByModelKey.keys), [model.installId],
                "\(kind.name): a hit by install id has nothing to copy")
        }
    }

    // MARK: - Legacy fallback, copy on first read, removal on the next save

    func testEachFieldKindIsFoundByALegacyKeyCopiedOnReadAndRetiredOnTheNextSave() throws {
        let model = libraryRow(path: oldPath)
        let legacyShapes: [(name: String, key: String)] = [
            ("alias::path", legacyPathKey),
            ("bare alias", "qwen"),
        ]
        for kind in fieldKinds {
            for shape in legacyShapes {
                let label = "\(kind.name) via \(shape.name)"
                try? FileManager.default.removeItem(at: metadataFile)
                let store = try openStore(seed: [shape.key: kind.seeded])

                XCTAssertEqual(kind.observe(store, model), kind.expected, label)
                XCTAssertEqual(
                    store.metadataByModelKey[model.installId], kind.seeded,
                    "\(label): the legacy hit is copied to the install id key on first read")
                XCTAssertNotNil(
                    store.metadataByModelKey[shape.key],
                    "\(label): a read alone does not remove the legacy key")

                forceASave(store)
                XCTAssertNil(
                    store.metadataByModelKey[shape.key],
                    "\(label): the legacy key is removed on the next save")
                XCTAssertEqual(store.metadataByModelKey[model.installId], kind.seeded, label)
                XCTAssertNil(try persisted()[shape.key], "\(label): and gone from the file")

                // Stable: a second read changes nothing, here and after a relaunch.
                let before = store.metadataByModelKey
                XCTAssertEqual(kind.observe(store, model), kind.expected, label)
                XCTAssertEqual(store.metadataByModelKey, before, "\(label): second read is stable")
                let relaunched = try openStore()
                XCTAssertEqual(kind.observe(relaunched, model), kind.expected, "\(label): relaunch")
            }
        }
    }

    func testTheSpecificLegacyKeyShadowsTheSharedBareAliasRow() throws {
        // The old lookup returned the exact key and never merged the bare
        // alias row, which is shared by every model of that alias (state#64).
        // Folding both would make tags appear that this model never showed.
        let model = libraryRow(path: oldPath)
        let store = try openStore(seed: [
            legacyPathKey: ModelCustomMetadata(tags: ["specific"]),
            "qwen": ModelCustomMetadata(tags: ["shared"]),
        ])
        XCTAssertEqual(store.tags(model), ["specific"])
        forceASave(store)
        XCTAssertEqual(
            store.metadataByModelKey["qwen"]?.tags, ["shared"],
            "the shared bare alias row is left alone when the specific key answered")
    }

    func testAnEntryUnderAnotherLibraryOfTheSameModelIsAdopted() throws {
        // A CLI move, or a row that gained its library later (tasks.md 1.2 note).
        let store = try openStore(seed: [
            "internal:text/qwen": ModelCustomMetadata(tags: ["moved"], isFavorite: true)
        ])
        let model = libraryRow()
        XCTAssertTrue(store.isFavorite(model))
        XCTAssertEqual(store.tags(model), ["moved"])
        XCTAssertNotNil(store.metadataByModelKey[model.installId])
        forceASave(store)
        XCTAssertNil(store.metadataByModelKey["internal:text/qwen"])
    }

    func testAmbiguousEntriesUnderTwoOtherLibrariesAreNotGuessed() throws {
        let store = try openStore(seed: [
            "internal:text/qwen": ModelCustomMetadata(tags: ["a"]),
            "T8:text/qwen": ModelCustomMetadata(tags: ["b"]),
        ])
        XCTAssertEqual(store.tags(libraryRow()), [])
        forceASave(store)
        XCTAssertNotNil(store.metadataByModelKey["internal:text/qwen"], "kept, not dropped")
        XCTAssertNotNil(store.metadataByModelKey["T8:text/qwen"], "kept, not dropped")
    }

    // MARK: - Both keys present

    func testWhenBothKeysExistTheInstallIdWinsOnlyWhereItIsNonEmpty() throws {
        // MERGE RULE: field by field, the install id entry wins where it has
        // a value; the legacy entry fills the gaps. A favorite is a value only
        // when true.
        let model = libraryRow(path: oldPath)
        let store = try openStore(seed: [
            model.installId: ModelCustomMetadata(
                nickname: nil, notes: "", tags: ["a"], isFavorite: false, reasoningDefault: "low"),
            legacyPathKey: ModelCustomMetadata(
                nickname: "Legacy", notes: "legacy notes", tags: ["b"], isFavorite: true,
                reasoningDefault: "high"),
        ])
        let merged = store.metadata(for: model)
        XCTAssertEqual(merged.nickname, "Legacy", "filled from legacy")
        XCTAssertEqual(merged.notes, "legacy notes", "filled from legacy")
        XCTAssertEqual(merged.tags, ["a"], "install id value kept")
        XCTAssertTrue(merged.isFavorite, "filled from legacy")
        XCTAssertEqual(merged.reasoningDefault, "low", "install id value kept")
        forceASave(store)
        XCTAssertEqual(try persisted()[model.installId], merged)
        XCTAssertNil(try persisted()[legacyPathKey])
    }

    func testAFullInstallIdEntryIsNeverOverwrittenByTheLegacyOne() throws {
        let model = libraryRow(path: oldPath)
        let full = ModelCustomMetadata(
            nickname: "Id", notes: "id notes", tags: ["x"], isFavorite: true,
            reasoningDefault: "xhigh")
        let store = try openStore(seed: [
            model.installId: full,
            legacyPathKey: ModelCustomMetadata(
                nickname: "Legacy", notes: "legacy notes", tags: ["b"], isFavorite: true,
                reasoningDefault: "high"),
        ])
        XCTAssertEqual(store.metadata(for: model), full)
    }

    func testFillingGapsFollowsTheDocumentedRule() {
        let preferred = ModelCustomMetadata(
            nickname: "", notes: "", tags: [], isFavorite: false, reasoningDefault: "")
        let filler = ModelCustomMetadata(
            nickname: "N", notes: "T", tags: ["g"], isFavorite: true, reasoningDefault: "low")
        XCTAssertEqual(preferred.filling(gapsFrom: filler), filler)
        XCTAssertEqual(filler.filling(gapsFrom: preferred), filler)
    }

    // MARK: - Per-model reasoning defaults (settings.json legacy table)

    func testLegacyReasoningDefaultsAreFoundByPathThenAliasAndRekeyed() throws {
        let model = libraryRow(path: oldPath)
        let store = try openStore()
        var legacy = [oldPath: "high", "qwen": "low"]

        XCTAssertEqual(store.reasoningDefault(for: model, migratingLegacyFrom: &legacy), "high")
        XCTAssertEqual(store.reasoningDefault(for: model), "high", "copied to the install id key")
        XCTAssertEqual(
            store.metadataByModelKey[model.installId]?.reasoningDefault, "high")
        XCTAssertNil(legacy[oldPath], "the path key is retired")

        // Stable, and the leftover alias row is dropped now that the id answers.
        XCTAssertEqual(store.reasoningDefault(for: model, migratingLegacyFrom: &legacy), "high")
        XCTAssertTrue(legacy.isEmpty)
        XCTAssertEqual(store.reasoningDefault(for: model, migratingLegacyFrom: &legacy), "high")
    }

    func testALegacyReasoningDefaultFoundOnlyByBareAliasIsRekeyed() throws {
        let model = libraryRow(path: oldPath)
        let store = try openStore()
        var legacy = ["qwen": "low", "other-model": "high"]

        XCTAssertEqual(store.reasoningDefault(for: model, migratingLegacyFrom: &legacy), "low")
        XCTAssertEqual(store.reasoningDefault(for: model), "low")
        XCTAssertEqual(legacy, ["other-model": "high"], "only this model's key is retired")
    }

    func testTheInstallIdReasoningDefaultBeatsALegacyOneAndTheLegacyKeyIsRetired() throws {
        let model = libraryRow(path: oldPath)
        let store = try openStore(seed: [model.installId: ModelCustomMetadata(reasoningDefault: "medium")])
        var legacy = [oldPath: "high"]
        XCTAssertEqual(store.reasoningDefault(for: model, migratingLegacyFrom: &legacy), "medium")
        XCTAssertTrue(legacy.isEmpty)
    }

    func testNoReasoningDefaultLeavesEverythingAlone() throws {
        let store = try openStore()
        var legacy = ["someone-else": "low"]
        XCTAssertNil(store.reasoningDefault(for: libraryRow(), migratingLegacyFrom: &legacy))
        XCTAssertEqual(legacy, ["someone-else": "low"])
        XCTAssertTrue(store.metadataByModelKey.isEmpty)
    }

    // MARK: - Move and reconnect

    func testMetadataSurvivesAMoveToAnotherLibraryThroughRekey() throws {
        // Written under the legacy path key by an older build, read once
        // (which copies it), then the model moves and its id changes.
        let full = ModelCustomMetadata(
            nickname: "Drafter", notes: "keep", tags: ["coding"], isFavorite: true,
            reasoningDefault: "high")
        let store = try openStore(seed: [legacyPathKey: full])
        let before = libraryRow(path: oldPath)
        XCTAssertEqual(store.metadata(for: before), full)

        store.rekey(from: "T7:text/qwen", to: "internal:text/qwen")

        let after = internalRow(path: newPath)
        XCTAssertEqual(store.metadata(for: after), full, "nothing is lost")
        XCTAssertNil(store.metadataByModelKey["T7:text/qwen"])
        XCTAssertNil(try persisted()["T7:text/qwen"])
        XCTAssertEqual(try persisted()["internal:text/qwen"], full)
    }

    func testRekeyCarriesEveryFieldKind() throws {
        for kind in fieldKinds {
            try? FileManager.default.removeItem(at: metadataFile)
            let store = try openStore(seed: ["T7:text/qwen": kind.seeded])
            store.rekey(from: "T7:text/qwen", to: "internal:text/qwen")
            XCTAssertEqual(kind.observe(store, internalRow()), kind.expected, kind.name)
            XCTAssertNil(store.metadataByModelKey["T7:text/qwen"], kind.name)
        }
    }

    func testRekeyFindsAnEntryThatWasNeverReadWhenGivenTheOldPath() throws {
        let full = ModelCustomMetadata(notes: "never read", tags: ["t"], isFavorite: true)
        let store = try openStore(seed: [legacyPathKey: full])
        store.rekey(from: "T7:text/qwen", to: "internal:text/qwen", legacyPath: oldPath)
        XCTAssertEqual(store.metadata(for: internalRow(path: newPath)), full)
        XCTAssertNil(store.metadataByModelKey[legacyPathKey])
    }

    func testRekeyingTwiceChangesNothing() throws {
        let full = ModelCustomMetadata(notes: "n", tags: ["t"], isFavorite: true)
        let store = try openStore(seed: ["T7:text/qwen": full])
        store.rekey(from: "T7:text/qwen", to: "internal:text/qwen")
        let once = store.metadataByModelKey
        let fileOnce = try Data(contentsOf: metadataFile)
        store.rekey(from: "T7:text/qwen", to: "internal:text/qwen")
        XCTAssertEqual(store.metadataByModelKey, once)
        XCTAssertEqual(try Data(contentsOf: metadataFile), fileOnce)
    }

    func testRekeyOntoANonEmptyDestinationMergesByTheSameRule() throws {
        let store = try openStore(seed: [
            "T7:text/qwen": ModelCustomMetadata(
                nickname: "Moved", notes: "moved notes", tags: ["moved"], isFavorite: true,
                reasoningDefault: "high"),
            "internal:text/qwen": ModelCustomMetadata(
                nickname: nil, notes: "dest notes", tags: [], isFavorite: false,
                reasoningDefault: nil),
        ])
        store.rekey(from: "T7:text/qwen", to: "internal:text/qwen")
        let merged = try XCTUnwrap(store.metadataByModelKey["internal:text/qwen"])
        XCTAssertEqual(merged.notes, "dest notes", "the destination wins where it has a value")
        XCTAssertEqual(merged.nickname, "Moved", "the moved entry fills the gaps")
        XCTAssertEqual(merged.tags, ["moved"])
        XCTAssertTrue(merged.isFavorite)
        XCTAssertEqual(merged.reasoningDefault, "high")
        XCTAssertNil(store.metadataByModelKey["T7:text/qwen"])
    }

    func testRekeyWithNothingToMoveOrTheSameIdIsANoOp() throws {
        let store = try openStore(seed: ["T7:text/qwen": ModelCustomMetadata(tags: ["t"])])
        store.rekey(from: "T7:text/qwen", to: "T7:text/qwen")
        store.rekey(from: "", to: "internal:text/qwen")
        store.rekey(from: "T7:text/qwen", to: "")
        store.rekey(from: "T8:text/other", to: "internal:text/other")
        XCTAssertEqual(Set(store.metadataByModelKey.keys), ["T7:text/qwen"])
    }

    // MARK: - Placeholder ids

    func testAPlaceholderRowKeysLikeAnInternalRowAndNeverTakesALibraryKey() throws {
        let store = try openStore(seed: [
            "T7:text/lm-model": ModelCustomMetadata(tags: ["belongs-to-a-library-row"])
        ])
        let row = placeholderRow()
        XCTAssertEqual(store.tags(row), [], "a placeholder never scans other libraries")
        store.setFavorite(true, for: row)
        store.addTag("mine", for: row)
        XCTAssertTrue(store.isFavorite(row))
        XCTAssertEqual(store.tags(row), ["mine"])
        XCTAssertNotNil(store.metadataByModelKey["internal:text/lm-model"])
        XCTAssertEqual(
            store.metadataByModelKey["T7:text/lm-model"]?.tags, ["belongs-to-a-library-row"],
            "the library entry is untouched")
    }

    func testAPlaceholderRowsLegacyEntryIsCopiedToItsOwnInternalKey() throws {
        let path = "/Users/test/.lmstudio/models/lm-model.gguf"
        let store = try openStore(seed: [
            "lm-model::\(path)": ModelCustomMetadata(isFavorite: true)
        ])
        let row = placeholderRow(path: path)
        XCTAssertTrue(store.isFavorite(row))
        XCTAssertTrue(
            store.metadataByModelKey.keys.allSatisfy { !$0.hasPrefix("T7:") },
            "never copied onto a library id")
        XCTAssertEqual(store.metadataByModelKey["internal:text/lm-model"]?.isFavorite, true)
    }

    // MARK: - Removal

    func testRemovingAModelDropsItsInstallIdAndPathEntriesButNotTheSharedAliasRow() throws {
        let model = libraryRow(path: oldPath)
        let store = try openStore(seed: [
            model.installId: ModelCustomMetadata(tags: ["id"]),
            legacyPathKey: ModelCustomMetadata(tags: ["path"]),
            "qwen": ModelCustomMetadata(tags: ["shared"]),
        ])
        store.removeMetadata(for: model)
        XCTAssertNil(store.metadataByModelKey[model.installId])
        XCTAssertNil(store.metadataByModelKey[legacyPathKey])
        XCTAssertEqual(store.metadataByModelKey["qwen"]?.tags, ["shared"], "state#64")
    }

    // MARK: - Mutations by model

    func testEveryMutatorWritesTheInstallIdKey() throws {
        let store = try openStore()
        let model = libraryRow()
        store.toggleFavorite(model)
        XCTAssertTrue(store.isFavorite(model))
        store.setFavorite(false, for: model)
        XCTAssertFalse(store.isFavorite(model))
        store.setNotes("n", for: model)
        store.setNickname("  Nick  ", for: model)
        store.addTag("a", for: model)
        store.addTag("b", for: model)
        store.removeTag("a", for: model)
        store.setReasoningDefault("high", for: model)
        XCTAssertEqual(store.notes(model), "n")
        XCTAssertEqual(store.nickname(model), "Nick")
        XCTAssertEqual(store.tags(model), ["b"])
        XCTAssertEqual(store.reasoningDefault(for: model), "high")
        XCTAssertEqual(Set(store.metadataByModelKey.keys), [model.installId])
        store.setReasoningDefault(nil, for: model)
        XCTAssertNil(store.reasoningDefault(for: model))
    }

    func testAMutationFoldsTheLegacyEntryInsteadOfShadowingIt() throws {
        // A user edits a model whose data is still under the legacy key. The
        // edit must start from that data, not from an empty record.
        let model = libraryRow(path: oldPath)
        let store = try openStore(seed: [
            legacyPathKey: ModelCustomMetadata(notes: "old notes", tags: ["old"])
        ])
        store.addTag("new", for: model)
        XCTAssertEqual(store.tags(model), ["old", "new"])
        XCTAssertEqual(store.notes(model), "old notes")
        XCTAssertNil(store.metadataByModelKey[legacyPathKey], "the save retired it")
    }

    // MARK: - Reads do not publish

    func testReadingALegacyEntryDoesNotNotifyObservers() throws {
        // A view body reads this store; publishing from inside a body is a
        // SwiftUI runtime warning. The copy leaves every reader's answer the
        // same, so it has nothing to announce.
        let store = try openStore(seed: [legacyPathKey: ModelCustomMetadata(isFavorite: true)])
        var announcements = 0
        let subscription = store.objectWillChange.sink { announcements += 1 }
        defer { subscription.cancel() }
        _ = store.isFavorite(libraryRow(path: oldPath))
        XCTAssertEqual(announcements, 0)
        store.setNotes("x", for: libraryRow(path: oldPath))
        XCTAssertGreaterThan(announcements, 0)
    }

    // MARK: - Persistence format

    func testReadingNeverLosesDataEvenWhenNothingIsSaved() throws {
        // The copy is in memory until the next save. A launch that reads and
        // then quits writes nothing, and the next launch must find the same
        // answer in the untouched legacy file.
        let model = libraryRow(path: oldPath)
        let seed = [legacyPathKey: ModelCustomMetadata(tags: ["kept"], isFavorite: true)]
        let first = try openStore(seed: seed)
        XCTAssertEqual(first.tags(model), ["kept"])
        let onDisk = try Data(contentsOf: metadataFile)

        let second = try openStore()
        XCTAssertEqual(second.tags(model), ["kept"])
        XCTAssertTrue(second.isFavorite(model))
        XCTAssertEqual(try Data(contentsOf: metadataFile), onDisk, "a read never writes")
    }

    func testAFileWrittenBeforeTheReasoningFieldLoadsAndStaysAFlatDictionary() throws {
        let legacy = """
            {"qwen::\(oldPath)": {"isFavorite": true, "tags": ["x"], "notes": "n"}}
            """
        try Data(legacy.utf8).write(to: metadataFile)
        let store = try openStore()
        let model = libraryRow(path: oldPath)
        XCTAssertTrue(store.isFavorite(model))
        XCTAssertNil(store.reasoningDefault(for: model))
        forceASave(store)

        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: metadataFile))
        let object = try XCTUnwrap(raw as? [String: Any], "the root stays the flat dictionary")
        XCTAssertNotNil(object["T7:text/qwen"])
        XCTAssertNil(object["qwen::\(oldPath)"])
        XCTAssertNil(object["version"], "the existing format has no version field to bump")
    }

    func testACorruptFileIsQuarantinedAndTheStoreStartsEmpty() throws {
        try Data("{ this is not json".utf8).write(to: metadataFile)
        let failuresBefore = AppJSONStore.readFailureCount
        defer { AppJSONStore.clearLastReadError() }

        let store = try openStore()
        XCTAssertTrue(store.metadataByModelKey.isEmpty)
        XCTAssertGreaterThan(AppJSONStore.readFailureCount, failuresBefore)
        XCTAssertFalse(FileManager.default.fileExists(atPath: metadataFile.path))
        let siblings = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertTrue(siblings.contains { $0.contains("corrupt-") }, "\(siblings)")

        // The next save writes beside the quarantined bytes, never over them.
        store.setFavorite(true, for: libraryRow())
        XCTAssertTrue(FileManager.default.fileExists(atPath: metadataFile.path))
        let after = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertTrue(after.contains { $0.contains("corrupt-") }, "\(after)")
    }

    // MARK: - The key helper (task 5.9 reuses it)

    func testTheKeyHelperSpellsEveryLegacyShapeAndTheLibraryIndependentId() {
        let key = ModelIdentityKey(libraryRow(path: "/Volumes/T7/a/../TurboSpark/text/qwen.gturbo"))
        XCTAssertEqual(key.installId, "T7:text/qwen")
        XCTAssertEqual(key.modalityAndAlias, "text/qwen")
        XCTAssertEqual(
            key.legacyMetadataKeys, ["qwen::/Volumes/T7/TurboSpark/text/qwen.gturbo", "qwen"],
            "the metadata path key is standardized, as the old key(for:path:) did")
        XCTAssertEqual(
            key.legacyReasoningKeys, ["/Volumes/T7/a/../TurboSpark/text/qwen.gturbo", "qwen"],
            "the reasoning path key was the raw path")
        XCTAssertFalse(key.isPlaceholder)

        let bare = ModelIdentityKey(installId: "internal:text/qwen", alias: "qwen", path: nil)
        XCTAssertEqual(bare.legacyMetadataKeys, ["qwen"])
        XCTAssertEqual(bare.legacyReasoningKeys, ["qwen"])

        XCTAssertTrue(ModelIdentityKey(placeholderRow()).isPlaceholder)
        XCTAssertFalse(ModelIdentityKey(internalRow()).isPlaceholder)
        XCTAssertEqual(ModelIdentityKey.alias(inInstallId: "T7:text/qwen"), "qwen")
        XCTAssertNil(ModelIdentityKey.alias(inInstallId: "not-an-id"))
    }

    // MARK: - Wiring: the filters use the identity

    func testTheFavoritesAndTagFiltersFindAModelAfterItsDriveReconnectedUnderAnotherName() throws {
        // The library id is stable, the mount name is not: the same install
        // now lives at a different path.
        let store = try openStore(seed: [
            "T7:text/qwen": ModelCustomMetadata(
                notes: "swift helper", tags: ["coding"], isFavorite: true)
        ])
        let reconnected = libraryRow(path: "/Volumes/T7 1/TurboSpark/text/qwen.gturbo")
        let unrelated = libraryRow(alias: "gemma")

        func filtered(
            search: String = "", favoritesOnly: Bool = false, tag: String? = nil
        ) -> [String] {
            ModelManagerView.filterModels(
                installed: [reconnected, unrelated], searchText: search,
                showFavoritesOnly: favoritesOnly, architectureFilter: .all, drafterFilter: .all,
                sourceFilter: .all, selectedTag: tag, sortOption: .name, orgStore: store
            ).map(\.alias)
        }
        XCTAssertEqual(filtered(favoritesOnly: true), ["qwen"])
        XCTAssertEqual(filtered(tag: "coding"), ["qwen"])
        XCTAssertEqual(filtered(search: "swift helper"), ["qwen"])
    }

    // MARK: - Wiring: reasoning defaults in AppModel

    func testSettingTheReasoningLevelStoresItAgainstTheInstallIdNotThePath() {
        let appModel = AppModel()
        let alias = "reasoning-\(UUID().uuidString.prefix(8))"
        let model = libraryRow(alias: alias)
        defer { ModelOrganizationStore.shared.removeMetadata(for: model) }

        appModel.selected = model
        appModel.setReasoning(.low)

        XCTAssertEqual(ModelOrganizationStore.shared.reasoningDefault(for: model), "low")
        XCTAssertNil(appModel.modelReasoningDefaults[model.path])
        XCTAssertNil(appModel.modelReasoningDefaults[alias])

        // The same install at a new path (moved, or its drive renamed).
        let moved = libraryRow(
            alias: alias, path: "/Volumes/T7 1/TurboSpark/text/\(alias).gturbo")
        appModel.reasoning = .off
        appModel.restoreReasoningPreference(for: moved)
        XCTAssertEqual(appModel.reasoning, .low)
    }

    func testAppModelRekeysALegacyReasoningDefaultOnRestore() {
        let appModel = AppModel()
        let alias = "legacy-reasoning-\(UUID().uuidString.prefix(8))"
        let model = libraryRow(alias: alias)
        defer { ModelOrganizationStore.shared.removeMetadata(for: model) }

        appModel.modelReasoningDefaults = [model.path: "high", "unrelated": "low"]
        appModel.restoreReasoningPreference(for: model)

        XCTAssertEqual(appModel.reasoning, .high)
        XCTAssertEqual(ModelOrganizationStore.shared.reasoningDefault(for: model), "high")
        XCTAssertNil(appModel.modelReasoningDefaults[model.path], "the legacy key is retired")
        XCTAssertEqual(appModel.modelReasoningDefaults["unrelated"], "low")
    }
}
