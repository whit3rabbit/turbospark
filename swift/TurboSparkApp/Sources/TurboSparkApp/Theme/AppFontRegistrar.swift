import AppKit
import CoreText
import Foundation

/// Registers the bundled font files with this process at launch.
///
/// **This cannot be an `Info.plist` `ATSApplicationFontsPath`**, for two
/// independent reasons, and either one alone would be enough:
///
/// 1. `swift run` produces a bare executable with no `Info.plist` at all
///    (`swift/CLAUDE.md` Gotcha 12), so a developer build would have no
///    bundled fonts and the pickers would offer families that do not render --
///    exactly the bug this is here to fix, reintroduced on the path most
///    likely to be used while working on it.
/// 2. In the shipped `.app` these files live inside SwiftPM's generated
///    resource bundle under `Contents/Resources/TurboSparkApp_TurboSparkApp.bundle`,
///    not directly under `Contents/Resources`, which is the only place that
///    key looks.
///
/// Registering through Core Text at `.process` scope works identically in
/// both, and needs no packaging change: `scripts/make-app-bundle.sh` already
/// copies every `*.bundle` from `.build/release` into `Contents/Resources`.
public enum AppFontRegistrar {
    private static var registeredCount: Int?

    /// The bundled font files, whether or not they have been registered.
    ///
    /// Separate from `registerBundledFonts()` so a caller can tell "there are
    /// no font files" from "registration failed", which are the same zero
    /// otherwise. That distinction is not hypothetical: the first version of
    /// this type reported zero because of the bug below, and the test that was
    /// meant to catch it SKIPPED on the same zero and passed.
    ///
    /// `Package.swift` declares `.process("Resources")`, and that rule
    /// FLATTENS subdirectories: these sit at the bundle root rather than under
    /// `Fonts/`, exactly as `Resources/Logos/*.svg` do (`ModelLogoView`
    /// carries the same fallback chain, so this is the house pattern).
    ///
    /// **`??` is the wrong operator for that fallback.** A missing
    /// subdirectory yields an EMPTY ARRAY rather than nil, so a nil-coalescing
    /// chain never falls through to the root and finds nothing, with no error
    /// anywhere to say so.
    public static var bundledFontURLs: [URL] {
        let inSubdirectory = Bundle.module.urls(forResourcesWithExtension: "ttf", subdirectory: "Fonts") ?? []
        let atRoot = Bundle.module.urls(forResourcesWithExtension: "ttf", subdirectory: nil) ?? []
        return inSubdirectory.isEmpty ? atRoot : inSubdirectory
    }

    /// Registers every bundled `.ttf`, once per process.
    ///
    /// Returns how many faces are now registered.
    @discardableResult
    public static func registerBundledFonts() -> Int {
        if let registeredCount { return registeredCount }

        let urls = bundledFontURLs
        var count = 0
        for url in urls {
            var cfError: Unmanaged<CFError>?
            if CTFontManagerRegisterFontsForURL(url as CFURL, .process, &cfError) {
                count += 1
                continue
            }

            // Already-registered is a success, not a failure: a test target can
            // reach this before the app delegate does, and Core Text reports
            // the second call as an error rather than as a no-op.
            let code = cfError.map { CFErrorGetCode($0.takeRetainedValue()) } ?? 0
            if code == CTFontManagerError.alreadyRegistered.rawValue {
                count += 1
            } else {
                FileHandle.standardError.write(
                    Data("TurboSpark: could not register font \(url.lastPathComponent) (Core Text error \(code))\n".utf8))
            }
        }

        registeredCount = count
        return count
    }
}

