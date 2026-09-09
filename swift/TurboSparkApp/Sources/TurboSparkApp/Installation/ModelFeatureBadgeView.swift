import SwiftUI
import TurboSpark

/// Visual badge component for rendering model capabilities and architectural features.
public struct ModelFeatureBadgeView: View {
    public enum BadgeStyle {
        case compact
        case regular
        case prominent
    }

    let title: String
    let iconSystemName: String?
    let tintColor: Color
    let tooltip: String?
    let style: BadgeStyle

    public init(
        title: String,
        iconSystemName: String? = nil,
        tintColor: Color = .secondary,
        tooltip: String? = nil,
        style: BadgeStyle = .compact
    ) {
        self.title = title
        self.iconSystemName = iconSystemName
        self.tintColor = tintColor
        self.tooltip = tooltip
        self.style = style
    }

    public var body: some View {
        HStack(spacing: iconSpacing) {
            if let iconSystemName {
                Image(systemName: iconSystemName)
                    .themedFont(fitting: iconSize, weight: .semibold)
            }
            Text(title)
                .themedFont(fontStep, weight: fontWeight)
                .lineLimit(1)
        }
        .padding(.horizontal, horizontalPadding)
        .padding(.vertical, verticalPadding)
        .background(
            tintColor.opacity(backgroundOpacity),
            in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        )
        .overlay {
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .stroke(tintColor.opacity(borderOpacity), lineWidth: 0.5)
        }
        .foregroundStyle(tintColor)
        .help(tooltip ?? title)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(tooltip ?? title)
    }

    private var iconSpacing: CGFloat {
        switch style {
        case .compact: return 3
        case .regular: return 4
        case .prominent: return 5
        }
    }

    private var iconSize: CGFloat {
        switch style {
        case .compact: return 8
        case .regular: return 10
        case .prominent: return 11
        }
    }

    private var fontStep: AppFontStep {
        switch style {
        case .compact: return .tiny
        case .regular: return .small
        case .prominent: return .small
        }
    }

    private var fontWeight: Font.Weight {
        switch style {
        case .compact: return .medium
        case .regular: return .semibold
        case .prominent: return .bold
        }
    }

    private var horizontalPadding: CGFloat {
        switch style {
        case .compact: return 5
        case .regular: return 7
        case .prominent: return 10
        }
    }

    private var verticalPadding: CGFloat {
        switch style {
        case .compact: return 2
        case .regular: return 3.5
        case .prominent: return 5
        }
    }

    private var cornerRadius: CGFloat {
        switch style {
        case .compact: return 4
        case .regular: return 6
        case .prominent: return 8
        }
    }

    private var backgroundOpacity: Double {
        switch style {
        case .compact: return 0.12
        case .regular: return 0.15
        case .prominent: return 0.18
        }
    }

    private var borderOpacity: Double {
        switch style {
        case .compact: return 0.25
        case .regular: return 0.35
        case .prominent: return 0.45
        }
    }
}

// MARK: - Convenient Feature Badge Presets

extension ModelFeatureBadgeView {
    /// MoE badge with routing details.
    public static func moe(details: String = "MoE", style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        ModelFeatureBadgeView(
            title: details,
            iconSystemName: "circle.grid.cross.fill",
            tintColor: .indigo,
            tooltip: "Mixture-of-Experts: streams routed experts via slot cache into ~2 GB RAM",
            style: style
        )
    }

    /// Dense model badge.
    public static func dense(details: String = "Dense", style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        ModelFeatureBadgeView(
            title: details,
            iconSystemName: "cube.fill",
            tintColor: .blue,
            tooltip: "Dense Transformer: full resident weights mapped into memory",
            style: style
        )
    }

    /// Dynamic Sloth / DFlash2 block drafter badge.
    public static func dynamicSloth(style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        ModelFeatureBadgeView(
            title: "Dynamic Sloth",
            iconSystemName: "bolt.badge.clock.fill",
            tintColor: .emeraldAccent,
            tooltip: "DFlash2 (Dynamic Sloth): proposes entire token blocks in one forward pass",
            style: style
        )
    }

    /// MTP (Multi-Token Prediction) head speculative badge.
    public static func mtp(style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        ModelFeatureBadgeView(
            title: "MTP Drafter",
            iconSystemName: "bolt.fill",
            tintColor: .cyan,
            tooltip: "Multi-Token-Prediction Head: fast speculative drafting for up to 1.66x speedup",
            style: style
        )
    }

    /// Quantization format badge.
    public static func quant(_ quant: String, style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        let isLowBit = quant.contains("1-Bit") || quant.contains("Ternary") || quant.contains("IQ3")
        let tint: Color = isLowBit ? .orange : .teal
        return ModelFeatureBadgeView(
            title: quant,
            iconSystemName: "number.square.fill",
            tintColor: tint,
            tooltip: "Weight Quantization: \(quant)",
            style: style
        )
    }

    /// Format container badge (MLX vs GGUF).
    public static func format(_ format: ModelFeatureDescriptor.WeightContainerFormat, style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        switch format {
        case .mlx:
            return ModelFeatureBadgeView(
                title: "MLX",
                iconSystemName: "shippingbox.fill",
                tintColor: .purple,
                tooltip: "Native Apple MLX .gturbo format with zero-copy Metal execution",
                style: style
            )
        case .gguf:
            return ModelFeatureBadgeView(
                title: "GGUF",
                iconSystemName: "doc.zipper",
                tintColor: .orange,
                tooltip: "GGUF format parsed and streamed directly via custom Metal kernels",
                style: style
            )
        }
    }

    /// Live directional steering badge.
    public static func steeringReady(style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        ModelFeatureBadgeView(
            title: "Steering Ready",
            iconSystemName: "arrow.triangle.swap",
            tintColor: .mint,
            tooltip: "Live Directional Steering: supports residual stream abliteration vectors without writing weights",
            style: style
        )
    }

    /// Linear attention / Gated DeltaNet badge.
    public static func linearAttention(style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        ModelFeatureBadgeView(
            title: "DeltaNet Attention",
            iconSystemName: "bolt.horizontal.fill",
            tintColor: .blue,
            tooltip: "Gated-DeltaNet linear attention layers for extreme throughput",
            style: style
        )
    }

    /// Storage source badge.
    public static func source(_ source: ModelFeatureDescriptor.StorageSource, style: BadgeStyle = .compact) -> ModelFeatureBadgeView {
        switch source {
        case .turboSpark:
            return ModelFeatureBadgeView(
                title: "TurboSpark Store",
                iconSystemName: "sparkles",
                tintColor: .secondary,
                tooltip: "Stored in ~/.turbospark/models",
                style: style
            )
        case .lmStudio:
            return ModelFeatureBadgeView(
                title: "LM Studio",
                iconSystemName: "desktopcomputer",
                tintColor: .purple,
                tooltip: "Indexed from LM Studio (~/.lmstudio/models) without duplicating files",
                style: style
            )
        case .custom:
            return ModelFeatureBadgeView(
                title: "Custom Folder",
                iconSystemName: "folder.fill",
                tintColor: .orange,
                tooltip: "Scanned from custom external filesystem folder",
                style: style
            )
        }
    }
}

private extension Color {
    static var emeraldAccent: Color {
        Color(red: 16 / 255, green: 185 / 255, blue: 129 / 255)
    }
}
