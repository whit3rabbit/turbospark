import CTurboSpark
import XCTest

@testable import TurboSpark

/// The model-library wire shapes, decoded WITH and WITHOUT the fields the
/// engine gained in the model-libraries work, so a payload an older engine
/// wrote still reads as a This Mac row that can be opened.
///
/// Shapes follow crates/ffi/src/api/libraries.rs (camelCase), the text rows of
/// crates/ffi/src/models/catalog.rs (snake_case, like the keys beside them),
/// and the asserts in crates/ffi/tests/library_reports.rs. The engine's own
/// output for a This Mac-only home is decoded in `LibraryEngineTests`.
final class LibraryWireTests: XCTestCase {

    private func parse<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    private let libraryID = "3f2c9d6e-7a41-4b0e-9c55-0d1e2f3a4b5c"

    // MARK: - Output copied from the release engine

    /// Captured from `ts_installed_json`, `ts_libraries_json` and
    /// `ts_audio_installed_json` of the release archive against a throwaway
    /// home (2026-10-16), with the home's path shortened to /h. The rows are a
    /// This Mac folder that is there, one that is gone, and a custom folder
    /// under a volume that is not mounted. Hand-written fixtures can drift
    /// from the engine; these are what it sent.
    func testRowsAsTheReleaseEngineWroteThem() throws {
        let installed = """
            [{"alias":"away","repo":"owner/away","revision":"main","path":"/Volumes/NoSuchDrive-scratch/models/away.gturbo","family":"llama","install_bytes":4096,"installed_on":"2026-10-14","status":"verified","kind":null,"modality":"text","library":null,"library_name":"custom location","install_id":"internal:text/away","availability":"not_connected","location_kind":"custom","connect_name":"NoSuchDrive-scratch"},{"alias":"gone","repo":"owner/gone","revision":"main","path":"/h/models/text/gone.gturbo","family":"llama","install_bytes":4096,"installed_on":"2026-10-14","status":"verified","kind":null,"modality":"text","library":"internal","library_name":"This Mac","install_id":"internal:text/gone","availability":"missing","location_kind":"internal","connect_name":null},{"alias":"here","repo":"owner/here","revision":"main","path":"/h/models/text/here.gturbo","family":"llama","install_bytes":4096,"installed_on":"2026-10-14","status":"verified","kind":null,"modality":"text","library":"internal","library_name":"This Mac","install_id":"internal:text/here","availability":"available","location_kind":"internal","connect_name":null}]
            """
        let rows = try parse([InstalledModel].self, installed)
        XCTAssertEqual(rows.map(\.availability), [.notConnected, .missing, .available])
        XCTAssertEqual(rows.map(\.locationKind), [.custom, .thisMac, .thisMac])
        XCTAssertEqual(rows.map(\.library), [nil, "internal", "internal"])
        XCTAssertEqual(rows[0].connectName, "NoSuchDrive-scratch")
        XCTAssertEqual(rows[0].libraryName, "custom location")

        let libraries = """
            [{"id":"internal","name":"This Mac","root":"/h/models","available":true,"duplicate":false,"modelCount":2,"isDefault":true,"lastSeen":null,"isInternal":true}]
            """
        XCTAssertEqual(try parse([ModelLibrary].self, libraries).first?.modelCount, 2)

        let audio = #"{"installed":[],"needs_adoption":[],"incompatible":[],"other_audio":[],"not_connected":[]}"#
        let report = try audioDecoder().decode(AudioInstalledReport.self, from: Data(audio.utf8))
        XCTAssertTrue(report.notConnected.isEmpty)
    }

    // MARK: - Libraries

    func testLibraryListDecodesThisMacAndADisconnectedDrive() throws {
        let json = """
            [{"id":"internal","name":"This Mac","root":"/Users/me/.turbospark/models",
              "available":true,"duplicate":false,"modelCount":2,"isDefault":true,
              "lastSeen":null,"isInternal":true},
             {"id":"\(libraryID)","name":"Studio","root":null,"available":false,
              "duplicate":false,"modelCount":1,"isDefault":false,
              "lastSeen":"2026-10-12","isInternal":false}]
            """
        let libraries = try parse([ModelLibrary].self, json)
        XCTAssertEqual(libraries.count, 2)

        let home = libraries[0]
        XCTAssertEqual(home.id, "internal")
        XCTAssertTrue(home.isInternal && home.isDefault && home.available)
        XCTAssertEqual(home.root, "/Users/me/.turbospark/models")
        XCTAssertNil(home.lastSeen)
        XCTAssertEqual(home.modelCount, 2)

        let studio = libraries[1]
        XCTAssertEqual(studio.id, libraryID)
        XCTAssertEqual(studio.name, "Studio")
        XCTAssertNil(studio.root, "a library that is not connected has no root")
        XCTAssertFalse(studio.available)
        XCTAssertFalse(studio.isInternal)
        XCTAssertFalse(studio.isDefault)
        XCTAssertEqual(studio.lastSeen, "2026-10-12")
    }

    func testDuplicateFlagIsReadFromTheListing() throws {
        let json = """
            {"id":"\(libraryID)","name":"Studio","root":"/Volumes/T7/TurboSpark",
             "available":true,"duplicate":true,"modelCount":0,"isDefault":false,
             "lastSeen":"2026-10-12","isInternal":false}
            """
        XCTAssertTrue(try parse(ModelLibrary.self, json).duplicate)
    }

    func testAddedLibraryKeepsTheSkippedAliases() throws {
        let json = """
            {"library":{"id":"\(libraryID)","name":"Studio","root":"/Volumes/T7/TurboSpark",
              "available":true,"duplicate":false,"modelCount":3,"isDefault":false,
              "lastSeen":"2026-10-16","isInternal":false},
             "skipped":["qwen","gemma4"]}
            """
        let added = try parse(AddedLibrary.self, json)
        XCTAssertEqual(added.library.id, libraryID)
        XCTAssertEqual(added.skipped, ["qwen", "gemma4"])
    }

    func testDiscoverySummaryDecodes() throws {
        let json = """
            {"id":"\(libraryID)","name":"Studio","createdOn":"2026-09-01",
             "modelCount":4,"readOnly":true}
            """
        let summary = try parse(LibraryMarkerSummary.self, json)
        XCTAssertEqual(summary.id, libraryID)
        XCTAssertEqual(summary.createdOn, "2026-09-01")
        XCTAssertEqual(summary.modelCount, 4)
        XCTAssertTrue(summary.readOnly, "a library a newer build wrote can be listed, not changed")
    }

    func testDiscoveryAnswersNullAsNoLibrary() throws {
        // `null` is the answer for no marker, an unreadable marker, and a path
        // that is not there: the wrapper must read it as "nothing found",
        // never as a decoding failure.
        let found = try decode(LibraryMarkerSummary?.self, from: "null")
        XCTAssertNil(found)
    }

    // MARK: - Importing a moved store

    private func importedJSON(failures: String) -> String {
        """
        {"library":{"id":"\(libraryID)","name":"Old store","root":"/Volumes/T7/models",
          "available":true,"duplicate":false,"modelCount":5,"isDefault":false,
          "lastSeen":"2026-10-16","isInternal":false},
         "skipped":["a"],"custom":["b"],"needsAdoption":["c"],
         "copiedToHome":["embeddings/e5"],"copyFailures":\(failures),
         "restoredCatalogOverride":true}
        """
    }

    func testImportedStoreWithCopyFailuresIsNotComplete() throws {
        let store = try parse(
            ImportedStore.self,
            importedJSON(failures: #"[{"key":"ocr/paddle","reason":"permission denied"}]"#))
        XCTAssertEqual(store.skipped, ["a"])
        XCTAssertEqual(store.custom, ["b"])
        XCTAssertEqual(store.needsAdoption, ["c"])
        XCTAssertEqual(store.copiedToHome, ["embeddings/e5"])
        XCTAssertEqual(store.copyFailures, [LibraryCopyFailure(key: "ocr/paddle", reason: "permission denied")])
        XCTAssertTrue(store.restoredCatalogOverride)
        // Requirement 9.5: a failed home copy leaves the conversion pending,
        // and it is retried only when the import is called again.
        XCTAssertFalse(store.isComplete)
    }

    func testImportedStoreWithoutFailuresIsComplete() throws {
        let store = try parse(ImportedStore.self, importedJSON(failures: "[]"))
        XCTAssertTrue(store.copyFailures.isEmpty)
        XCTAssertTrue(store.isComplete)
    }

    // MARK: - Move outcome

    func testMovedOutcomeCarriesTheNewInstallID() throws {
        let outcome = try parse(
            MoveOutcome.self, #"{"outcome":"moved","installId":"\#(libraryID):text/qwen"}"#)
        XCTAssertEqual(outcome, .moved(installId: "\(libraryID):text/qwen"))
        XCTAssertEqual(outcome.installId, "\(libraryID):text/qwen")
    }

    func testSourceMarkerStaleOutcomeNamesTheLibraryStillListingIt() throws {
        let outcome = try parse(
            MoveOutcome.self,
            #"{"outcome":"source_marker_stale","installId":"internal:text/qwen","staleMarker":"Studio"}"#)
        XCTAssertEqual(outcome, .sourceMarkerStale(installId: "internal:text/qwen", staleMarker: "Studio"))
    }

    func testSourceCleanupFailedOutcomeCarriesTheLeftoverFolder() throws {
        let outcome = try parse(
            MoveOutcome.self,
            #"{"outcome":"source_cleanup_failed","installId":"internal:text/qwen","path":"/Volumes/T7/TurboSpark/text/qwen.gturbo","staleMarker":null}"#)
        XCTAssertEqual(
            outcome,
            .sourceCleanupFailed(
                installId: "internal:text/qwen", staleMarker: nil,
                path: "/Volumes/T7/TurboSpark/text/qwen.gturbo"))
        XCTAssertEqual(outcome.installId, "internal:text/qwen")
    }

    func testAnUnknownOutcomeStillReportsTheInstallInItsDestination() throws {
        // Every outcome means the install is in the destination. Failing the
        // decode on a spelling this build does not know would show a failure
        // for a move that happened.
        let outcome = try parse(
            MoveOutcome.self, #"{"outcome":"something_new","installId":"x:text/a"}"#)
        XCTAssertEqual(outcome, .unrecognized(outcome: "something_new", installId: "x:text/a"))
        XCTAssertEqual(outcome.installId, "x:text/a")
    }

    // MARK: - Move progress

    /// A final holder, because a `@Sendable` closure cannot capture a `var`.
    private final class Progress: @unchecked Sendable {
        var events: [[UInt64]] = []
    }

    func testMoveProgressForwardsByteEventsAndIgnoresStageLines() throws {
        // A move sends only TS_INSTALL_BYTES, first (0, total) and then the
        // cumulative bytes; the callback must forward exactly those.
        let seen = Progress()
        let box = MoveBox { done, total in seen.events.append([done, total]) }
        let userdata = Unmanaged.passUnretained(box).toOpaque()
        let callback = try XCTUnwrap(moveCallback)
        callback(userdata, TS_INSTALL_BYTES, nil, 0, 0, 100)
        callback(userdata, TS_INSTALL_BYTES, nil, 0, 40, 100)
        callback(userdata, TS_INSTALL_STAGE, nil, 0, 7, 7)
        callback(userdata, TS_INSTALL_BYTES, nil, 0, 100, 100)
        callback(nil, TS_INSTALL_BYTES, nil, 0, 1, 1)
        withExtendedLifetime(box) {}
        XCTAssertEqual(seen.events, [[0, 100], [40, 100], [100, 100]])
    }

    // MARK: - Text rows

    func testInstalledRowFromAnOlderEngineReadsAsAnAvailableThisMacRow() throws {
        let json = """
            {"alias":"qwen","repo":"owner/qwen","revision":"main","path":"/p/qwen.gturbo",
             "family":"qwen3","install_bytes":4096,"installed_on":"2026-10-14"}
            """
        let row = try parse(InstalledModel.self, json)
        XCTAssertNil(row.library)
        XCTAssertNil(row.libraryName)
        XCTAssertNil(row.connectName)
        XCTAssertEqual(row.installId, "internal:text/qwen")
        XCTAssertEqual(row.availability, .available)
        XCTAssertEqual(row.locationKind, .thisMac)
    }

    func testInstalledRowOfThisMacKeepsEveryOldKeyAndTheNewOnes() throws {
        let json = """
            {"alias":"homey","repo":"owner/homey","revision":"main","path":"/h/models/text/homey.gturbo",
             "family":"llama","install_bytes":4096,"installed_on":"2026-10-14","status":"verified",
             "modality":"text","library":"internal","library_name":"This Mac",
             "install_id":"internal:text/homey","availability":"available",
             "location_kind":"internal","connect_name":null}
            """
        let row = try parse(InstalledModel.self, json)
        XCTAssertEqual(row.path, "/h/models/text/homey.gturbo")
        XCTAssertEqual(row.installBytes, 4096)
        XCTAssertEqual(row.status, "verified")
        XCTAssertEqual(row.library, "internal")
        XCTAssertEqual(row.libraryName, "This Mac")
        XCTAssertEqual(row.installId, "internal:text/homey")
        XCTAssertEqual(row.availability, .available)
        XCTAssertEqual(row.locationKind, .thisMac)
        XCTAssertNil(row.connectName)
    }

    func testInstalledRowInADisconnectedLibraryNamesTheDriveToConnect() throws {
        let json = """
            {"alias":"qwen","repo":"owner/qwen","revision":"main",
             "path":"/Volumes/T7/TurboSpark/text/qwen.gturbo","family":"llama",
             "install_bytes":4096,"installed_on":"2026-10-14","status":"verified",
             "library":"\(libraryID)","library_name":"Studio",
             "install_id":"\(libraryID):text/qwen","availability":"not_connected",
             "location_kind":"library","connect_name":"Studio"}
            """
        let row = try parse(InstalledModel.self, json)
        XCTAssertEqual(row.library, libraryID)
        XCTAssertEqual(row.availability, .notConnected)
        XCTAssertEqual(row.locationKind, .library)
        XCTAssertEqual(row.connectName, "Studio")
        XCTAssertEqual(row.installId, "\(libraryID):text/qwen")
    }

    func testMissingAndCustomAndUnknownLocationKindsDecode() throws {
        func row(_ availability: String, _ kind: String, library: String) throws -> InstalledModel {
            try parse(
                InstalledModel.self,
                """
                {"alias":"x","repo":"o/x","revision":"main","path":"/p","family":"llama",
                 "install_bytes":1,"installed_on":"2026-10-14","library":\(library),
                 "library_name":"n","install_id":"i","availability":"\(availability)",
                 "location_kind":"\(kind)","connect_name":null}
                """)
        }
        let gone = try row("missing", "library", library: #""abc""#)
        XCTAssertEqual(gone.availability, .missing)
        // A custom location has a null library, which is not This Mac.
        let custom = try row("available", "custom", library: "null")
        XCTAssertNil(custom.library)
        XCTAssertEqual(custom.locationKind, .custom)
        let orphan = try row("missing", "unknown", library: #""gone""#)
        XCTAssertEqual(orphan.locationKind, .unknown)
    }

    func testAnInstallResultIntoALibraryIsReadAsALibraryRow() throws {
        // The install result gains library, library_name and install_id only.
        // Defaulting its place to This Mac would label a library install as
        // internal.
        let json = """
            {"alias":"qwen","repo":"owner/qwen","revision":"abc","path":"/Volumes/T7/TurboSpark/text/qwen.gturbo",
             "family":"llama","install_bytes":4096,"installed_on":"2026-10-16","status":"verified",
             "library":"\(libraryID)","library_name":"Studio","install_id":"\(libraryID):text/qwen"}
            """
        let row = try parse(InstalledModel.self, json)
        XCTAssertEqual(row.library, libraryID)
        XCTAssertEqual(row.installId, "\(libraryID):text/qwen")
        XCTAssertEqual(row.availability, .available)
        XCTAssertEqual(row.locationKind, .library)
    }

    func testAnInstallResultIntoThisMacIsReadAsAThisMacRow() throws {
        let json = """
            {"alias":"qwen","repo":"owner/qwen","revision":"abc","path":"/h/models/text/qwen.gturbo",
             "family":"llama","install_bytes":4096,"installed_on":"2026-10-16","status":"verified",
             "library":"internal","library_name":"This Mac","install_id":"internal:text/qwen"}
            """
        let row = try parse(InstalledModel.self, json)
        XCTAssertEqual(row.locationKind, .thisMac)
        XCTAssertEqual(row.availability, .available)
    }

    func testMemberwiseInitKeepsEveryExistingCallShape() {
        // The app and its tests build rows with this call. New parameters
        // follow the old ones and default to a row an older engine listed.
        let row = InstalledModel(
            alias: "one", repo: "x/y", path: "/models/one.gturbo", family: "gemma4")
        XCTAssertEqual(row.installId, "internal:text/one")
        XCTAssertEqual(row.availability, .available)
        XCTAssertEqual(row.locationKind, .thisMac)
        XCTAssertNil(row.library)
    }

    // MARK: - Image rows

    func testImageRowFromAnOlderEngineHasNoLibraryKeys() throws {
        let json = """
            {"alias":"pic","modelID":"o/m","revision":"r","path":"/p/pic.gturbo",
             "width":512,"height":512,"schedulerSteps":8,"quantization":"4bit",
             "sourcePath":null,"family":"z-image"}
            """
        let row = try parse(ImageInstalledModel.self, json)
        XCTAssertNil(row.library)
        XCTAssertNil(row.libraryName)
        XCTAssertNil(row.installId, "a row of a scanned folder names no registry install")
    }

    func testImageRowInALibraryCarriesItsInstallID() throws {
        let json = """
            {"alias":"pic","modelID":"o/m","revision":"r","path":"/Volumes/T7/TurboSpark/image/pic.gturbo",
             "width":512,"height":512,"schedulerSteps":8,"quantization":"4bit",
             "sourcePath":null,"family":"z-image",
             "library":"\(libraryID)","libraryName":"Studio","installId":"\(libraryID):image/pic"}
            """
        let row = try parse(ImageInstalledModel.self, json)
        XCTAssertEqual(row.library, libraryID)
        XCTAssertEqual(row.libraryName, "Studio")
        XCTAssertEqual(row.installId, "\(libraryID):image/pic")
    }

    func testImageInstallResultCarriesTheLibraryKeysToo() throws {
        let json = """
            {"alias":"pic","modelID":"o/m","revision":"r","path":"/h/models/image/pic.gturbo",
             "width":512,"height":512,"schedulerSteps":8,"quantization":"4bit",
             "sourcePath":null,"family":"z-image",
             "library":"internal","libraryName":"This Mac","installId":"internal:image/pic"}
            """
        let row = try parse(ImageInstalledModel.self, json)
        XCTAssertEqual(row.library, "internal")
        XCTAssertEqual(row.installId, "internal:image/pic")
    }

    func testImageMemberwiseInitKeepsEveryExistingCallShape() {
        let row = ImageInstalledModel(
            alias: "a", modelID: "o/m", revision: "r", path: "/p", width: 1, height: 1,
            schedulerSteps: 1)
        XCTAssertNil(row.installId)
    }

    // MARK: - Audio

    private func audioDecoder() -> JSONDecoder {
        // The same strategy `AudioCatalog` decodes with.
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    private let audioIdentity = """
        {"task":"music","alias":"minimax-music3-4bit","repository":"o/m","revision":"r",
         "asset_fingerprint":"f"}
        """

    private func audioRecord(_ extra: String = "") -> String {
        """
        {"alias":"minimax-music3-4bit","repo":"o/m","revision":"r","path":"/p/a.gturbo",
         "family":"minimax-music","install_bytes":10,"installed_on":"2026-10-14",
         "status":"verified","modality":"audio"\(extra)}
        """
    }

    func testAudioReportWithTheNotConnectedBucket() throws {
        let json = """
            {"installed":[],"needs_adoption":[],"incompatible":[],"other_audio":[],
             "not_connected":[{"record":\(audioRecord(#","library":"\#(libraryID)""#)),
                              "identity":\(audioIdentity),"library_name":"Studio"}]}
            """
        let report = try audioDecoder().decode(AudioInstalledReport.self, from: Data(json.utf8))
        XCTAssertEqual(report.notConnected.count, 1)
        XCTAssertEqual(report.notConnected[0].libraryName, "Studio")
        XCTAssertEqual(report.notConnected[0].record.library, libraryID)
        XCTAssertEqual(report.notConnected[0].identity?.alias, "minimax-music3-4bit")
    }

    func testAudioReportFromAnOlderEngineHasAnEmptyBucket() throws {
        let json = #"{"installed":[],"needs_adoption":[],"incompatible":[],"other_audio":[]}"#
        let report = try audioDecoder().decode(AudioInstalledReport.self, from: Data(json.utf8))
        XCTAssertTrue(report.notConnected.isEmpty)
    }

    func testAudioNotConnectedRowMayHaveNoIdentity() throws {
        let json = """
            {"installed":[],"needs_adoption":[],"incompatible":[],"other_audio":[],
             "not_connected":[{"record":\(audioRecord(#","library":"x""#)),
                              "identity":null,"library_name":"Studio"}]}
            """
        let report = try audioDecoder().decode(AudioInstalledReport.self, from: Data(json.utf8))
        XCTAssertNil(report.notConnected[0].identity)
    }

    func testAudioRecordWithAndWithoutLibraryKeys() throws {
        let old = try audioDecoder().decode(
            AudioInstalledRecord.self, from: Data(audioRecord().utf8))
        XCTAssertNil(old.library)
        XCTAssertNil(old.libraryName)
        XCTAssertNil(old.installId)

        let result = try audioDecoder().decode(
            AudioInstalledRecord.self,
            from: Data(
                audioRecord(
                    #","library":"\#(libraryID)","library_name":"Studio","install_id":"\#(libraryID):audio/minimax-music3-4bit""#
                ).utf8))
        XCTAssertEqual(result.library, libraryID)
        XCTAssertEqual(result.libraryName, "Studio")
        XCTAssertEqual(result.installId, "\(libraryID):audio/minimax-music3-4bit")
    }

    // MARK: - Install destinations

    func testNoDestinationSendsNoOptions() throws {
        XCTAssertNil(try TurboSparkCatalog.installDestinationOptions(library: nil))
    }

    func testADestinationIsSentAsTheLibraryKeyAndNothingElse() throws {
        let json = try XCTUnwrap(TurboSparkCatalog.installDestinationOptions(library: libraryID))
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(object.count, 1)
        XCTAssertEqual(object["library"] as? String, libraryID)
    }

    func testADestinationWithQuotesIsEscapedNotInterpolated() throws {
        let hostile = #"a"b\c"#
        let json = try XCTUnwrap(TurboSparkCatalog.installDestinationOptions(library: hostile))
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(object["library"] as? String, hostile)
    }

    func testTheOldInstallCallShapesStillCompile() {
        // The app passes `TurboSparkCatalog.install` as a function value
        // (AppModel+Downloads) and calls the others with no destination.
        // These lines fail to COMPILE if an overload makes a reference
        // ambiguous or a new parameter loses its default. The calls sit in a
        // function that is never run: creating the stream starts a real
        // install.
        let reference: (String) -> AsyncThrowingStream<InstallEvent, Error> =
            TurboSparkCatalog.install
        let image: (String) -> AsyncThrowingStream<ImageInstallEvent, Error> =
            TurboSparkCatalog.installImage
        func neverCalled() {
            _ = TurboSparkCatalog.install(repo: "o/r", alias: "a")
            _ = TurboSparkCatalog.install(repo: "o/r", alias: "a", file: "f.gguf", sidecarRepo: nil)
            _ = TurboSparkCatalog.install(repo: "o/r", alias: "a", library: nil)
            _ = TurboSparkCatalog.install("alias")
            _ = TurboSparkCatalog.install("alias", library: "id")
            _ = TurboSparkCatalog.installImage("alias", library: nil)
            _ = try? AudioCatalog.install(
                AudioProfileIdentity(task: .music, alias: "a", repository: "r", revision: "v", assetFingerprint: "f"),
                acknowledged: [], progress: { _ in })
        }
        _ = (reference, image, neverCalled)
    }

    func testAudioInstallOptionsAddTheLibraryOnlyWhenOneIsGiven() throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let without = try encoder.encode(AudioInstallOptions(acknowledged: ["a"], library: nil))
        XCTAssertEqual(String(decoding: without, as: UTF8.self), #"{"acknowledged":["a"]}"#)
        let with = try encoder.encode(AudioInstallOptions(acknowledged: [], library: "lib"))
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: with) as? [String: Any])
        XCTAssertEqual(object["library"] as? String, "lib")
        XCTAssertEqual((object["acknowledged"] as? [Any])?.count, 0)
    }
}
