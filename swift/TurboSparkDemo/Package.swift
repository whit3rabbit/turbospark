// swift-tools-version: 5.9
import PackageDescription

// A deliberately small SwiftUI app. Its job is to VERIFY THE BINDING end to
// end -- streaming, cancellation, telemetry, model management -- not to be a
// product. Anything here that looks under-designed is under-designed on
// purpose.
//
// `swift run TurboSparkDemo` launches it. A SwiftPM executable has no app
// bundle, so it starts as an accessory process and has to promote itself to
// a regular one (see `TurboSparkDemoApp`); a shipping app would carry an
// Xcode project and an Info.plist instead.
let package = Package(
    name: "TurboSparkDemo",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../TurboSpark")
    ],
    targets: [
        .executableTarget(
            name: "TurboSparkDemo",
            dependencies: [.product(name: "TurboSpark", package: "TurboSpark")],
            linkerSettings: [
                // EVERY CONSUMER OF TurboSpark HAS TO REPEAT THIS, and that
                // is a SwiftPM limitation rather than a mistake here: a
                // library search path in `unsafeFlags` is resolved against
                // the root of the package being BUILT, not the package that
                // declared it. So TurboSpark's own `-LSources/CTurboSpark`
                // is correct when its tests link and wrong for everyone
                // else, and the path below is this package's view of the
                // same directory.
                //
                // The alternative is an `.xcframework` binary target, which
                // resolves paths for its consumers properly. That is what a
                // published version of this package should ship; a
                // two-package repository does not need the packaging step.
                .unsafeFlags(["-L../TurboSpark/Sources/CTurboSpark"])
            ]
        )
    ]
)
