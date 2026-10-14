// swift-tools-version: 5.9
import PackageDescription

// The Rust staticlib and its header are COPIED into
// `Sources/CTurboSpark/` by `make swift-lib` rather than referenced across
// the repository. A SwiftPM target may not reach outside its own directory,
// and a modulemap pointing at `../../../../crates/ffi/include` works until
// someone builds from a different root. Both copies are gitignored; the
// canonical header is `crates/ffi/include/turbospark.h`.
//
// Consequence: `swift build` FAILS with a missing-header error until
// `make swift-lib` has run once. That is the honest ordering -- there is no
// Swift here without a Rust library under it.
let package = Package(
    name: "TurboSpark",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "TurboSpark", targets: ["TurboSpark"]),
        // syntext's code-search index and its context engine. Their C ABIs
        // are compiled INTO libturbospark_ffi.a (crates/ffi links
        // syntext-context), because a process can link only one Rust
        // staticlib. The wrappers are vendored from the revision Cargo.lock
        // pins (`scripts/vendor-syntext.sh`; `swift/docs/SYNTEXT.md`).
        .library(name: "Syntext", targets: ["Syntext"]),
        .library(name: "SyntextContext", targets: ["SyntextContext"])
    ],
    targets: [
        // The C surface: `module.modulemap` plus the copied header.
        .systemLibrary(name: "CTurboSpark", path: "Sources/CTurboSpark"),

        .target(
            name: "TurboSpark",
            dependencies: ["CTurboSpark"],
            linkerSettings: [
                .linkedLibrary("turbospark_ffi"),
                // A Rust `staticlib` does NOT record the frameworks its
                // dependencies asked for, so the final link has to supply
                // them. Metal and Foundation come from `metal-rs`;
                // QuartzCore carries `CAMetalLayer`'s symbols, which that
                // crate references even though nothing here draws.
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
                .linkedFramework("QuartzCore"),
            ]
        ),

        // syntext's hand-written headers, unmodified, beside the modulemap
        // that names them. A system-library target builds no object file.
        .systemLibrary(name: "CSyntext", path: "Vendor/syntext/CSyntext"),

        // `Syntext` and `SyntextContext` share `package`-access helpers, so
        // they have to live in one package; the symbols come from the same
        // archive the TurboSpark target links.
        .target(
            name: "Syntext",
            dependencies: ["CSyntext"],
            path: "Vendor/syntext/Syntext",
            linkerSettings: [
                .linkedLibrary("turbospark_ffi"),
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
                .linkedFramework("QuartzCore"),
            ]
        ),
        .target(
            name: "SyntextContext",
            dependencies: ["Syntext", "CSyntext"],
            path: "Vendor/syntext/SyntextContext"
        ),

        // Links the staticlib and calls into it. This is what proves the
        // HAND-WRITTEN header matches the Rust side -- a mismatch is a link
        // error or a wrong answer here, and nowhere else.
        .testTarget(
            name: "TurboSparkTests",
            dependencies: ["TurboSpark"],
            linkerSettings: [
                // `unsafeFlags` resolves relative to the package being built.
                // Placed on the test target so it does not leak as an invalid
                // relative search path to downstream consumers like TurboSparkApp.
                .unsafeFlags([
                    "-LSources/CTurboSpark",
                    "-Xlinker", "-force_load",
                    "-Xlinker", "Sources/CTurboSpark/libturbospark_ffi.a",
                ])
            ]
        ),

        // syntext's own suites, vendored with the wrappers: the only proof
        // that its hand-written headers match the archive this repository
        // builds, for the same reason TurboSparkTests is for turbospark.h.
        .testTarget(
            name: "SyntextTests",
            dependencies: ["Syntext"],
            path: "Vendor/syntext/Tests/SyntextTests",
            linkerSettings: [
                .unsafeFlags([
                    "-LSources/CTurboSpark",
                    "-Xlinker", "-force_load",
                    "-Xlinker", "Sources/CTurboSpark/libturbospark_ffi.a",
                ])
            ]
        ),
        .testTarget(
            name: "SyntextContextTests",
            dependencies: ["SyntextContext"],
            path: "Vendor/syntext/Tests/SyntextContextTests",
            linkerSettings: [
                .unsafeFlags([
                    "-LSources/CTurboSpark",
                    "-Xlinker", "-force_load",
                    "-Xlinker", "Sources/CTurboSpark/libturbospark_ffi.a",
                ])
            ]
        ),
    ]
)
