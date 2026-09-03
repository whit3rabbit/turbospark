import XCTest

@testable import TurboSparkApp

/// G1/G2: the expert-slot picker offered `[0, 4, 8, 16, 32, 64, 128]` against
/// an engine whose `ALLOWED_CACHE_SLOTS` is `[8, 16, 24, 32]`. Four of those
/// values are refused by the engine and the refusal is a PANIC, in this
/// process -- the engine is linked in, so picking 4 aborts the whole app
/// rather than showing an error. The picker also omitted the legal 24.
///
/// **THE SET IS PINNED BY HAND HERE, AND THAT IS THE POINT.** Swift cannot
/// read a Rust `const`, so the two spellings can only be kept in step by a
/// test that states the engine's and fails when this app's drifts -- the same
/// job `SurfaceTests` does for `turbospark.h`. Change `ALLOWED_CACHE_SLOTS` in
/// `crates/core/src/runtime_config.rs` and this file is what tells you the
/// picker needs changing too.
final class RuntimeOptionRangeTests: XCTestCase {
    /// `foundation::runtime_config::ALLOWED_CACHE_SLOTS`, verbatim.
    private let engineAllowedSlots = [8, 16, 24, 32]

    func testEveryOfferedSlotCountExceptAutomaticIsOneTheEngineAccepts() {
        let offered = AppRuntimeOptions.allowedSlotCounts.filter { $0 != 0 }
        XCTAssertEqual(
            offered, engineAllowedSlots,
            "The picker must offer exactly the engine's set. A value it refuses is an abort, "
                + "and a value it accepts but the picker omits is a capability the user cannot reach.")
    }

    func testAutomaticIsOfferedAndIsSpeltZero() {
        XCTAssertEqual(
            AppRuntimeOptions.allowedSlotCounts.first, 0,
            "0 is `Auto` and is the default; `buildOpenOptions` sends no slot count for it.")
    }

    @MainActor
    func testAutomaticSizingSendsNoSlotCountAtAll() {
        let appModel = AppModel()
        appModel.runtimeOptions.expertCacheSlots = 0
        XCTAssertNil(
            appModel.buildOpenOptions().expertCacheSlots,
            "0 must mean `Auto`, not a literal zero slots.")
    }

    // MARK: - G2: a persisted value is clamped rather than trapped on

    @MainActor
    func testAnUnrecognizedPersistedSlotCountFallsBackToAutomatic() throws {
        // `settings.json` is user-editable and outlives the versions that
        // wrote it, so a value the current build refuses is reachable without
        // anyone writing a bug.
        let file = AppStorageRoot.file("settings.json")
        let saved = try? Data(contentsOf: file)
        defer {
            if let saved {
                try? saved.write(to: file)
            } else {
                try? FileManager.default.removeItem(at: file)
            }
        }

        var settings = MacAppSettingsFileStore.load()
        settings.expertCacheSlots = 4  // legal before this fix, an abort after
        MacAppSettingsFileStore.save(settings)

        let appModel = AppModel()
        XCTAssertEqual(
            appModel.runtimeOptions.expertCacheSlots, 0,
            "A slot count the engine refuses must resolve to automatic sizing, not reach `open()`.")
    }

    @MainActor
    func testANegativePersistedTopKIsClampedRatherThanTrapped() throws {
        let file = AppStorageRoot.file("settings.json")
        let saved = try? Data(contentsOf: file)
        defer {
            if let saved {
                try? saved.write(to: file)
            } else {
                try? FileManager.default.removeItem(at: file)
            }
        }

        var settings = MacAppSettingsFileStore.load()
        settings.topK = -1
        settings.contextTokens = -1
        MacAppSettingsFileStore.save(settings)

        let appModel = AppModel()
        // `UInt32(-1)` traps. These two are the values that reach one.
        XCTAssertGreaterThanOrEqual(appModel.topK, 0)
        XCTAssertGreaterThanOrEqual(appModel.maxContextTokens, 0)
        XCTAssertNotNil(UInt32(exactly: Double(appModel.topK)))
    }
}
