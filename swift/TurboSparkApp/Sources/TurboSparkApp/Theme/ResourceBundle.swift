import Foundation
import TurboSparkAppResources

extension Bundle {
    /// The app's resources. Hand-written on purpose: this target declares no
    /// resources of its own, so SwiftPM generates no `Bundle.module` here and
    /// every `Bundle.module` / `bundle: .module` call site resolves to this.
    /// See `AppResources` for why the generated accessor is not used.
    static var module: Bundle { AppResources.bundle }
}
