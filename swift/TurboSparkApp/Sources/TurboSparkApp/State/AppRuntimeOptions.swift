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
    /// Allowed expert slot counts for MoE caching.
    public static let allowedSlotCounts = [0, 4, 8, 16, 32, 64, 128]

    /// Number of expert cache slots (0 for automatic).
    public var expertCacheSlots: Int = 0
    /// Whether prompt prefill acceleration is enabled.
    public var prefillEnabled: Bool = true
    /// Selected power and thermal profile.
    public var powerProfile: AppPowerProfileOption = .auto
    /// Selected speculative decoding block configuration.
    public var speculation: AppSpeculationOption = .auto
    /// Selected speculative drafter architecture.
    public var speculativeDrafter: AppSpeculativeDrafterOption = .auto
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
        prefillEnabled: Bool = true,
        powerProfile: AppPowerProfileOption = .auto,
        speculation: AppSpeculationOption = .auto,
        speculativeDrafter: AppSpeculativeDrafterOption = .auto,
        maxTokensPerSec: Double = 0,
        steeringPath: String? = nil,
        steeringMode: AppSteeringModeOption = .ablate,
        steeringScale: Double = 1.0,
        steeringLayers: String = "",
        steeringTarget: Double = 0.0,
        steeringGate: Double = 0.0
    ) {
        self.expertCacheSlots = expertCacheSlots
        self.prefillEnabled = prefillEnabled
        self.powerProfile = powerProfile
        self.speculation = speculation
        self.speculativeDrafter = speculativeDrafter
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
