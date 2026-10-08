import CTurboSpark
import Foundation

/// What a model family can do, as the engine itself answers it.
public struct FamilyCapabilities: Decodable, Sendable, Equatable {
    public let family: String
    /// False for a family this build does not recognise; every capability is
    /// then false too.
    public let known: Bool
    /// Whether the family's decode flow dispatches the steering edit. The
    /// engine refuses a control vector on a family where this is false, by
    /// name, at open.
    public let steeringSupported: Bool
}

/// Capability predicates answered in Rust, so a host does not keep a second
/// copy that is right only until the next family lands.
public enum TurboSparkCapabilities {
    /// Capabilities for a family spelled as `installed.json` and
    /// `manifest.json` spell it (for example `gemma4`, `gptOss`). Never throws:
    /// if the answer cannot be read the family is reported unknown with every
    /// capability false, which is the safe direction.
    public static func family(_ name: String) -> FamilyCapabilities {
        let fallback = FamilyCapabilities(family: name, known: false, steeringSupported: false)
        guard
            let json = try? takeString({ out in name.withCString { ts_family_capabilities_json($0, out) } }),
            let decoded = try? decode(FamilyCapabilities.self, from: json)
        else { return fallback }
        return decoded
    }

    /// Whether `kvBits` would be accepted for an install with these
    /// `manifest.json` `arch` facts, by the rule `ts_session_open` applies.
    /// `layerMask` is `fullAttentionLayerMask` (1 full attention, 0 sliding
    /// window). Any value that does not fit a byte, or a non-positive count,
    /// answers false.
    public static func kvQuantSupported(
        fullHeadDim: Int, layerMask: [Int], numLayers: Int
    ) -> Bool {
        guard numLayers > 0, !layerMask.isEmpty,
            layerMask.allSatisfy({ (0...255).contains($0) })
        else { return false }
        let bytes = layerMask.map { UInt8($0) }
        return bytes.withUnsafeBufferPointer { buffer in
            ts_kv_quant_supported(Int64(fullHeadDim), buffer.baseAddress, buffer.count, numLayers) == 1
        }
    }
}
