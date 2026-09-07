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
                    .resizable()
                    .aspectRatio(contentMode: .fit)
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
                    .themedFont(points: size * 0.45, weight: .bold)
                    .foregroundStyle(.white)
            } else {
                Image(systemName: fallbackIcon)
                    .themedFont(points: size * 0.45, weight: .semibold)
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
