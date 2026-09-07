import Foundation
import TurboSpark

/// Options for selecting model context window length limits.
public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    /// Use automatic context window configured from model checkpoint and available RAM.
    case auto = 0
    /// 2,048 tokens.
    case c2k = 2048
    /// 4,096 tokens.
    case c4k = 4096
    /// 8,192 tokens.
    case c8k = 8192
    /// 16,384 tokens.
    case c16k = 16384
    /// 32,768 tokens.
    case c32k = 32768
    /// 65,536 tokens.
    case c64k = 65536
    /// 131,072 tokens.
    case c128k = 131072

    /// Unique integer identifier.
    public var id: Int { rawValue }
    /// Token count corresponding to the option.
    public var tokens: Int { rawValue }

    /// Human-readable menu display label.
    public var menuLabel: String {
        switch self {
        case .auto: return "Auto"
        case .c2k: return "2,048 (2K)"
        case .c4k: return "4,096 (4K)"
        case .c8k: return "8,192 (8K)"
        case .c16k: return "16,384 (16K)"
        case .c32k: return "32,768 (32K)"
        case .c64k: return "65,536 (64K)"
        case .c128k: return "131,072 (128K)"
        }
    }

    /// Display label showing the resolved token count for auto mode.
    public func formattedLabel(resolvedAuto: Int? = nil) -> String {
        if self == .auto {
            if let resolvedAuto {
                return "Auto (\(resolvedAuto.formatted()))"
            }
            return "Auto"
        }
        return menuLabel
    }
}

/// Power profile options governing GPU clock frequencies and core selection.
public enum AppPowerProfileOption: String, CaseIterable, Identifiable, Sendable {
    /// Automatically managed by macOS and thermal state.
    case auto = "auto"
    /// Highest throughput and clock frequencies.
    case performance = "performance"
    /// Balanced efficiency and performance.
    case balanced = "balanced"
    /// Maximizes energy efficiency.
    case efficiency = "efficiency"

    /// Unique string identifier.
    public var id: String { rawValue }

    /// Human-readable menu display label.
    public var menuLabel: String {
        switch self {
        case .auto: return "Auto (OS Managed)"
        case .performance: return "Performance"
        case .balanced: return "Balanced"
        case .efficiency: return "Efficiency"
        }
    }

    /// Converts to the underlying TurboSpark open option.
    public var powerProfile: OpenOptions.PowerProfile? {
        switch self {
        case .auto: return nil
        case .performance: return .performance
        case .balanced: return .balanced
        case .efficiency: return .efficiency
        }
    }
}

/// How much of the machine a model may commit when it loads.
///
/// **Deliberately NOT named `AppGuardrailsMode`**, which already exists in
/// `MacAppSettings` for Forge TOOL-CALL guardrails and is an unrelated
/// feature. These are memory guardrails; see `docs/LOAD_GUARD.md`.
public enum AppLoadGuardOption: String, CaseIterable, Identifiable, Sendable {
    /// No precautions against overcommitting memory.
    case off = "off"
    /// Mild precautions. The default, and what shipped before the setting.
    case relaxed = "relaxed"
    /// Moderate precautions.
    case balanced = "balanced"
    /// Strong precautions.
    case strict = "strict"
    /// A user-set ceiling on what the engine may allocate.
    case custom = "custom"

    /// Unique string identifier.
    public var id: String { rawValue }

    /// Human-readable menu display label.
    public var menuLabel: String {
        switch self {
        case .off: return "Off (Not Recommended)"
        case .relaxed: return "Relaxed"
        case .balanced: return "Balanced"
        case .strict: return "Strict"
        case .custom: return "Custom"
        }
    }

    /// One-line explanation shown under the label.
    public var detailText: String {
        switch self {
        case .off: return "No precautions against system overload"
        case .relaxed: return "Mild precautions against system overload"
        case .balanced: return "Moderate precautions against system overload"
        case .strict: return "Strong precautions against system overload"
        case .custom: return "Set your own limit for maximum model size that can be loaded"
        }
    }

    /// Converts to the underlying TurboSpark open option.
    ///
    /// `customBytes` is read only by `.custom`; a zero there falls back to
    /// `.relaxed` rather than to a ceiling of nothing, which would refuse
    /// every model.
    public func loadGuard(customBytes: UInt64) -> OpenOptions.LoadGuard {
        switch self {
        case .off: return .off
        case .relaxed: return .relaxed
        case .balanced: return .balanced
        case .strict: return .strict
        case .custom: return customBytes > 0 ? .custom(customBytes) : .relaxed
        }
    }
}

/// Speculative decoding block size configuration options.
public enum AppSpeculationOption: String, CaseIterable, Identifiable, Sendable {
    /// Automatically resolve speculative block size.
    case auto = "auto"
    /// Disable speculative decoding.
    case off = "off"
    /// Draft 2 tokens per forward pass.
    case block2 = "2"
    /// Draft 4 tokens per forward pass.
    case block4 = "4"
    /// Draft 8 tokens per forward pass.
    case block8 = "8"

    /// Unique string identifier.
    public var id: String { rawValue }

    /// Human-readable menu display label.
    public var menuLabel: String {
        switch self {
        case .auto: return "Auto (Recommended)"
        case .off: return "Off"
        case .block2: return "Block 2"
        case .block4: return "Block 4"
        case .block8: return "Block 8"
        }
    }

    /// Converts to the underlying TurboSpark speculation option.
    public var speculation: OpenOptions.Speculation {
        switch self {
        case .auto: return .auto
        case .off: return .off
        case .block2: return .block(2)
        case .block4: return .block(4)
        case .block8: return .block(8)
        }
    }
}

/// Speculative decoding drafter architecture options.
public enum AppSpeculativeDrafterOption: String, CaseIterable, Identifiable, Sendable {
    /// Use automatic drafter selection.
    case auto = "auto"
    /// Use the checkpoint's multi-token prediction head.
    case mtp = "mtp"
    /// Use the DFlash2 diffusion-based block drafter.
    case dflash = "dflash"

    /// Unique string identifier.
    public var id: String { rawValue }

    /// Human-readable menu display label.
    public var menuLabel: String {
        switch self {
        case .auto: return "Auto (MTP Default)"
        case .mtp: return "MTP Head"
        case .dflash: return "DFlash2"
        }
    }

    /// Converts to the underlying TurboSpark drafter option.
    public var drafter: OpenOptions.SpeculativeDrafter {
        switch self {
        case .auto: return .auto
        case .mtp: return .mtp
        case .dflash: return .dflash
        }
    }
}

/// TurboQuant KV-cache quantization width configuration.
///
/// **`.auto` IS NOT THE ENGINE'S `KvQuant` -- there is no such case there.**
/// `--kv-bits` REFUSES at open by name when a checkpoint's `head_dim` is not
/// a power of two in 32...512, or when every eligible layer is excluded by
/// the last-full-attention-layer rule (`docs/TRUBOQUANT.md`), and a refused
/// open takes the whole session down with it -- there is no soft "fall back
/// to off" inside the engine the way `--speculative auto` has one. So `.auto`
/// here means "ask for 4-bit, the one width the assessment calls safe, but
/// only on a checkpoint `ModelFeatureDescriptor.supportsKvQuant` already says
/// would accept it" -- resolved CLIENT-SIDE in `AppModel.buildOpenOptions()`
/// against the install's own manifest, before the option is even built, so
/// this app never sends a request the engine would refuse for a default
/// nobody asked for. An EXPLICIT width is sent unconditionally and refused
/// exactly like any other named request in this app (the same shape
/// `AppSpeculationOption.block2/4/8` already have against an install with no
/// drafter): a user who chose 2-bit on purpose should see the refusal, not
/// have it silently swallowed.
public enum AppKvBitsOption: String, CaseIterable, Identifiable, Sendable {
    /// 4-bit when the checkpoint supports it, off otherwise. The default.
    case auto = "auto"
    /// Never quantize the KV cache.
    case off = "off"
    /// K2/V2.
    case two = "2"
    /// K3/V3.
    case three = "3"
    /// K3/V4, TurboQuant's own split for its one fractional rate.
    case threePointFive = "3.5"
    /// K4/V4.
    case four = "4"

    /// Unique string identifier.
    public var id: String { rawValue }

    /// Human-readable menu display label.
    public var menuLabel: String {
        switch self {
        case .auto: return "Auto (4-bit when supported)"
        case .off: return "Off"
        case .two: return "2-bit"
        case .three: return "3-bit"
        case .threePointFive: return "3.5-bit (K3/V4)"
        case .four: return "4-bit"
        }
    }

    /// Converts to the underlying TurboSpark open option, for an install
    /// this app has already checked (or the caller has decided not to
    /// check) is eligible. `.auto` resolves to 4-bit HERE -- the eligibility
    /// gate is a separate, earlier decision made against the install's own
    /// manifest, not something this accessor can see.
    public var kvBits: OpenOptions.KvBits? {
        switch self {
        case .auto: return .four
        case .off: return nil
        case .two: return .two
        case .three: return .three
        case .threePointFive: return .threePointFive
        case .four: return .four
        }
    }
}

/// Activation steering operation modes.
public enum AppSteeringModeOption: String, CaseIterable, Identifiable, Sendable {
    /// Suppress target activations.
    case ablate = "ablate"
    /// Amplify target activations.
    case add = "add"
    /// Clamp activations to a target value.
    case clamp = "clamp"
    /// Renormalize activations following steering.
    case renorm = "renorm"

    /// Unique string identifier.
    public var id: String { rawValue }

    /// Human-readable menu display label.
    public var menuLabel: String {
        switch self {
        case .ablate: return "Ablate (Suppress)"
        case .add: return "Add (Amplify)"
        case .clamp: return "Clamp"
        case .renorm: return "Renorm"
        }
    }

    /// Converts to the underlying TurboSpark steering mode option.
    public var steeringMode: OpenOptions.SteeringMode {
        switch self {
        case .ablate: return .ablate
        case .add: return .add
        case .clamp: return .clamp
        case .renorm: return .renorm
        }
    }
}

/// Configurable runtime parameters for opening model sessions.
public struct AppRuntimeOptions: Equatable, Sendable {
    /// Allowed expert slot counts for MoE caching, 0 being automatic.
    ///
    /// **THIS SET IS THE ENGINE'S, NOT THIS APP'S** (state#21), and it must stay equal to
    /// `foundation::runtime_config::ALLOWED_CACHE_SLOTS` (`[8, 16, 24, 32]`)
    /// with 0 prepended for `Auto`. It read `[0, 4, 8, 16, 32, 64, 128]`, so
    /// the picker offered four values the engine refuses -- and a refused
    /// value there is a PANIC, in this process, taking the whole app with it,
    /// because the engine is linked in rather than reached over a socket.
    /// It also omitted the legal 24.
    ///
    /// 8 is legal and is still a trap on a top-8 model: root Gotcha 64
    /// records that chunked prefill needs `slots >= 2 * top_k` and panics at
    /// `slots == top_k` on the first multi-token prompt. That is a per-model
    /// fact this static set cannot express, which is why `ts_session_open`
    /// now validates too rather than trusting any GUI to.
    public static let allowedSlotCounts = [0, 8, 16, 24, 32]

    /// Number of expert cache slots (0 for automatic).
    public var expertCacheSlots: Int = 0
    /// Selected power and thermal profile.
    public var powerProfile: AppPowerProfileOption = .auto
    /// How much of the machine a model may commit when it loads.
    public var loadGuard: AppLoadGuardOption = .relaxed
    /// The ceiling `AppLoadGuardOption.custom` applies, in bytes. Read by
    /// that tier alone.
    public var loadGuardCustomBytes: UInt64 = 0
    /// Refuse to load when an automatically-sized context window resolves
    /// below this. 0 imposes no floor and does not constrain an explicit
    /// context length.
    public var minAutoContextTokens: UInt32 = 0
    /// Selected speculative decoding block configuration.
    public var speculation: AppSpeculationOption = .auto
    /// Selected speculative drafter architecture.
    public var speculativeDrafter: AppSpeculativeDrafterOption = .auto
    /// TurboQuant KV-cache quantization width. `.auto` (the default) asks
    /// for 4-bit only on a checkpoint `ModelFeatureDescriptor.supportsKvQuant`
    /// already says would accept it, and stays off otherwise -- see
    /// `AppKvBitsOption`'s own doc for why that check happens here rather
    /// than being left to the engine's open-time refusal.
    public var kvBits: AppKvBitsOption = .auto
    /// Maximum tokens per second rate cap (0 for uncapped).
    public var maxTokensPerSec: Double = 0
    /// Filesystem path to steering control vector file.
    public var steeringPath: String? = nil
    /// Steering transformation mode.
    public var steeringMode: AppSteeringModeOption = .ablate
    /// Steering scale multiplier.
    public var steeringScale: Double = 1.0
    /// Layer range expression for steering vector application.
    public var steeringLayers: String = ""
    /// Target coefficient for clamp steering mode.
    public var steeringTarget: Double = 0.0
    /// Activation magnitude gate threshold.
    public var steeringGate: Double = 0.0

    /// Creates runtime options with configured defaults.
    public init(
        expertCacheSlots: Int = 0,
        powerProfile: AppPowerProfileOption = .auto,
        loadGuard: AppLoadGuardOption = .relaxed,
        loadGuardCustomBytes: UInt64 = 0,
        minAutoContextTokens: UInt32 = 0,
        speculation: AppSpeculationOption = .auto,
        speculativeDrafter: AppSpeculativeDrafterOption = .auto,
        kvBits: AppKvBitsOption = .auto,
        maxTokensPerSec: Double = 0,
        steeringPath: String? = nil,
        steeringMode: AppSteeringModeOption = .ablate,
        steeringScale: Double = 1.0,
        steeringLayers: String = "",
        steeringTarget: Double = 0.0,
        steeringGate: Double = 0.0
    ) {
        self.expertCacheSlots = expertCacheSlots
        self.powerProfile = powerProfile
        self.loadGuard = loadGuard
        self.loadGuardCustomBytes = loadGuardCustomBytes
        self.minAutoContextTokens = minAutoContextTokens
        self.speculation = speculation
        self.speculativeDrafter = speculativeDrafter
        self.kvBits = kvBits
        self.maxTokensPerSec = maxTokensPerSec
        self.steeringPath = steeringPath
        self.steeringMode = steeringMode
        self.steeringScale = steeringScale
        self.steeringLayers = steeringLayers
        self.steeringTarget = steeringTarget
        self.steeringGate = steeringGate
    }

    /// Formats an expert cache slot count for menu display.
    public static func slotsLabel(for slots: Int) -> String {
        slots == 0 ? "Auto" : "\(slots) slots"
    }
}
