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
        .library(name: "TurboSpark", targets: ["TurboSpark"])
    ],
    targets: [
        // The C surface: `module.modulemap` plus the copied header.
        .systemLibrary(name: "CTurboSpark", path: "Sources/CTurboSpark"),

        .target(
            name: "TurboSpark",
            dependencies: ["CTurboSpark"],
            linkerSettings: [
                // `unsafeFlags` blocks this package from being consumed as a
                // versioned dependency by another package. Accepted: it is
                // the only way to point at a staticlib whose path is a build
                // artifact, and the consumer here is an app in this repo.
                .unsafeFlags(["-LSources/CTurboSpark"]),
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

        // Links the staticlib and calls into it. This is what proves the
        // HAND-WRITTEN header matches the Rust side -- a mismatch is a link
        // error or a wrong answer here, and nowhere else.
        .testTarget(name: "TurboSparkTests", dependencies: ["TurboSpark"]),
    ]
)
