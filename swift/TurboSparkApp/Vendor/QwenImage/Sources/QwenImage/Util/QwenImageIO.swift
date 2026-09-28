import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX

/// PNG encoding for decoded RGB tensors.
enum QwenImageIO {
  /// Converts `[1, channels, height, width]` float32 values in `[0, 1]` to
  /// PNG data. The alpha channel is dropped: the first three channels are
  /// taken as RGB, which is what opaque prompts produce.
  static func pngData(from image: MLXArray) throws -> Data {
    precondition(image.ndim == 4, "expected [1, C, H, W]")
    let height = image.dim(2)
    let width = image.dim(3)
    precondition(image.dim(1) >= 3, "expected at least RGB channels")
    let rgb = image[0..., 0..<3, 0..., 0...].asType(.float32)
    MLX.eval(rgb)
    let values = rgb.asArray(Float32.self)

    var pixels = [UInt8](repeating: 255, count: height * width * 4)
    let plane = height * width
    for index in 0..<plane {
      let offset = index * 4
      pixels[offset] = Self.toByte(values[index])
      pixels[offset + 1] = Self.toByte(values[plane + index])
      pixels[offset + 2] = Self.toByte(values[2 * plane + index])
    }

    let cgImage = pixels.withUnsafeMutableBytes { buffer -> CGImage? in
      guard let context = CGContext(
        data: buffer.baseAddress,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
      ) else { return nil }
      return context.makeImage()
    }
    guard let cgImage else {
      throw QwenImagePipelineError.weightsMissing("could not encode the decoded image")
    }

    let output = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(
      output, UTType.png.identifier as CFString, 1, nil
    ) else {
      throw QwenImagePipelineError.weightsMissing("could not create PNG destination")
    }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw QwenImagePipelineError.weightsMissing("could not finalize PNG data")
    }
    return output as Data
  }

  private static func toByte(_ value: Float32) -> UInt8 {
    UInt8(max(0, min(255, Double(value) * 255.0 + 0.5)))
  }
}
