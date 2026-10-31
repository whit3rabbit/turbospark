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
    /// Whether the family supports the ANE+Steel prefill split (dense Qwen).
    public let anePrefillSupported: Bool
    /// Whether the family has a mapped expert residency arm. An install also
    /// needs routed experts and an expert table that fits Metal's working set;
    /// `TurboSparkCapabilities.expertResidency(forInstallAt:requested:)`
    /// answers for one install.
    public let mappedResidencySupported: Bool
    /// Whether mapped residency has a real-install memory and speed
    /// measurement on this family, not only a correctness check.
    public let mappedResidencyMeasured: Bool
}

/// What opening one install would do with an expert residency request.
public struct ExpertResidencyPreflight: Decodable, Sendable, Equatable {
    /// False when the install has no routed experts: residency does not apply,
    /// so send `.auto`.
    public let applicable: Bool
    public let requested: String
    /// `"streamed"` or `"mapped"` when the open would take the request, nil
    /// when it would refuse it.
    public let resolved: String?
    /// Why the open would refuse the request, or nil.
    public let refusal: String?

    public init(applicable: Bool, requested: String, resolved: String?, refusal: String?) {
        self.applicable = applicable
        self.requested = requested
        self.resolved = resolved
        self.refusal = refusal
    }
}

/// Capability predicates answered in Rust, so a host does not keep a second
/// copy that is right only until the next family lands.
public enum TurboSparkCapabilities {
    /// Capabilities for a family spelled as `installed.json` and
    /// `manifest.json` spell it (for example `gemma4`, `gptOss`). Never throws:
    /// if the answer cannot be read the family is reported unknown with every
    /// capability false, which is the safe direction.
    public static func family(_ name: String) -> FamilyCapabilities {
        let fallback = FamilyCapabilities(
            family: name, known: false, steeringSupported: false, anePrefillSupported: false,
            mappedResidencySupported: false, mappedResidencyMeasured: false)
        guard
            let json = try? takeString({ out in name.withCString { ts_family_capabilities_json($0, out) } }),
            let decoded = try? decode(FamilyCapabilities.self, from: json)
        else { return fallback }
        return decoded
    }

    /// What `TurboSparkSession.init` would do with `requested` residency on
    /// the install at `path`, answered by the same resolver and family rule
    /// the open applies, without opening it. Nil when the install's manifest
    /// cannot be read.
    public static func expertResidency(
        forInstallAt path: String, requested: OpenOptions.ExpertResidency
    ) -> ExpertResidencyPreflight? {
        guard
            let json = try? takeString({ out in
                path.withCString { p in
                    requested.rawValue.withCString { r in ts_expert_residency_for_install_json(p, r, out) }
                }
            })
        else { return nil }
        return try? decode(ExpertResidencyPreflight.self, from: json)
    }

    /// Whether the host chip is an Apple Silicon M4 or newer chip capable of ANE prefill.
    public static var chipSupportsAnePrefill: Bool {
        ts_chip_supports_ane_prefill() == 1
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
