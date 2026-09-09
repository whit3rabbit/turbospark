import AppKit
import SwiftUI

/// View rendering authentic provider and model family logos from bundled assets.
public struct ModelLogoView: View {
    public let logoName: String?
    public let fallbackLetter: String
    public let fallbackIcon: String
    public let accentColor: Color
    public let size: CGFloat
    public let cornerRadius: CGFloat

    public init(
        logoName: String?,
        fallbackLetter: String = "",
        fallbackIcon: String = "cpu.fill",
        accentColor: Color = .accentColor,
        size: CGFloat = 36,
        cornerRadius: CGFloat = 8
    ) {
        self.logoName = logoName
        self.fallbackLetter = fallbackLetter
        self.fallbackIcon = fallbackIcon
        self.accentColor = accentColor
        self.size = size
        self.cornerRadius = cornerRadius
    }

    public init(
        visuals: ModelFamilyVisuals,
        size: CGFloat = 36,
        cornerRadius: CGFloat = 8
    ) {
        self.logoName = visuals.logoAssetName
        self.fallbackLetter = visuals.letter
        self.fallbackIcon = visuals.iconSystemName
        self.accentColor = visuals.accentColor
        self.size = size
        self.cornerRadius = cornerRadius
    }

    public var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))

            if let image = resolvedImage {
                Image(nsImage: image)
                    .renderingMode(Self.isMonochromeMark(logoName) ? .template : .original)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .foregroundStyle(.primary)
                    .padding(size * 0.18)
            } else {
                fallbackView
            }
        }
        .frame(width: size, height: size)
        .overlay(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .stroke(Color.primary.opacity(0.12), lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .accessibilityHidden(true)
    }

    /// Marks with no colour of their own, which the asset stores as pure
    /// black. The tile behind them is `.controlBackgroundColor`, i.e. dark
    /// in dark mode, so drawn `.original` they are black on near-black --
    /// legible in light mode and invisible in dark. `NSImage(contentsOf:)`
    /// does not produce a template image, so the rendering mode has to be
    /// stated here.
    ///
    /// An explicit list rather than a heuristic: the other bundled marks
    /// (`hf.svg`, `microsoft.svg`, `mistral.svg`, `google.png`, `qwen.png`)
    /// carry their own brand colours, and templating one would flatten it
    /// to a single tint. Add a name here only after checking the asset has
    /// no `fill` of its own.
    static func isMonochromeMark(_ name: String?) -> Bool {
        guard let name else { return false }
        return monochromeMarks.contains((name as NSString).lastPathComponent.lowercased())
    }

    private static let monochromeMarks: Set<String> = [
        "openai.svg",
        "meta.svg",
        "deepseek.svg",
        "xai.svg",
        "zai.svg",
        "z-ai.svg",
        "llama_cpp.svg",
        "ollama.svg",
    ]

    private var resolvedImage: NSImage? {
        guard let logoName, !logoName.isEmpty else { return nil }
        return ModelLogoAssetCache.shared.image(named: logoName)
    }

    @ViewBuilder
    private var fallbackView: some View {
        ZStack {
            LinearGradient(
                colors: [accentColor.opacity(0.85), accentColor.opacity(0.6)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            if !fallbackLetter.isEmpty {
                Text(fallbackLetter)
                    .themedFont(fitting: size * 0.45, weight: .bold)
                    .foregroundStyle(.white)
            } else {
                Image(systemName: fallbackIcon)
                    .themedFont(fitting: size * 0.45, weight: .semibold)
                    .foregroundStyle(.white)
            }
        }
    }
}

/// Cache and asset loader for bundled model provider logos.
public final class ModelLogoAssetCache {
    public static let shared = ModelLogoAssetCache()
    private var cache: [String: NSImage] = [:]

    private init() {}

    public func image(named name: String) -> NSImage? {
        if let img = cache[name] {
            return img
        }

        let filename = (name as NSString).lastPathComponent
        let baseName = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension

        let candidates: [URL?] = [
            Bundle.module.url(forResource: filename, withExtension: nil, subdirectory: "Logos"),
            Bundle.module.url(forResource: filename, withExtension: nil, subdirectory: "Resources/Logos"),
            Bundle.module.url(forResource: filename, withExtension: nil),
            Bundle.module.url(forResource: baseName, withExtension: ext.isEmpty ? nil : ext, subdirectory: "Logos"),
            Bundle.module.url(forResource: baseName, withExtension: ext.isEmpty ? nil : ext)
        ]

        for case let url? in candidates {
            if let image = NSImage(contentsOf: url) {
                cache[name] = image
                return image
            }
        }

        return nil
    }
}
