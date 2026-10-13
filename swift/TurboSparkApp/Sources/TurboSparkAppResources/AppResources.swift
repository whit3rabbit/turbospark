import Foundation

/// Where the app's resource bundle lives, found without trusting SwiftPM's
/// generated `Bundle.module`.
///
/// The generated accessor depends on the toolchain. The one that Xcode 16's
/// native build system emits only looks at `Bundle.main.bundleURL`, which in
/// a `.app` is the bundle ROOT, and then at a path baked in from the build
/// machine. A release built that way crashes at launch with "could not load
/// resource bundle" on every other Mac. Putting the bundle at the root is
/// not an option either: `codesign` refuses a bundle with anything but
/// `Contents/` there ("unsealed contents present in the bundle root"), and it
/// refuses a symlink for the same reason. So the lookup has to change, and it
/// lives in its own target because a target with resources always gets a
/// generated `Bundle.module`, which cannot be overridden in place.
///
/// `Bundle.module` is only evaluated as the fallback, so a toolchain whose
/// accessor would crash is never asked. The fallback is what `swift run` and
/// the test runner use, where the bundle sits next to the build products and
/// `Bundle.main` is not an app.
public enum AppResources {
    public static let bundle: Bundle = {
        let name = "TurboSparkApp_TurboSparkAppResources.bundle"
        if let url = Bundle.main.resourceURL?.appendingPathComponent(name),
           let bundle = Bundle(url: url) {
            return bundle
        }
        return Bundle.module
    }()
}
