import Foundation
import Hub
import MLX
import Tokenizers

enum QwenTokenizerError: Error {
  case directoryNotFound(URL)
  case fileNotFound(URL)
  case padTokenMissing
}

/// Thin wrapper over the swift-transformers tokenizer for the Qwen
/// BPE vocabulary shared by the Qwen3-VL processor assets.
///
/// The Qwen-Image-2.1 pipeline tokenizes raw template strings, so this
/// wrapper exposes plain encoding plus the special-token ids the pipeline
/// needs, and performs the envelope's right truncation.
public final class QwenTokenizer {
  private let tokenizer: Tokenizer
  private let encodeFunction: (String) -> [Int]

  public let padTokenId: Int
  public let imageTokenId: Int

  public init(
    padTokenId: Int,
    imageTokenId: Int,
    tokenizer: Tokenizer,
    encode: @escaping (String) -> [Int]
  ) {
    self.padTokenId = padTokenId
    self.imageTokenId = imageTokenId
    self.tokenizer = tokenizer
    self.encodeFunction = encode
  }

  /// Loads the tokenizer from a snapshot whose tokenizer assets live under
  /// `processor/` (Qwen-Image-2.1 layout) or `tokenizer/`.
  public static func load(from snapshot: URL) throws -> QwenTokenizer {
    let processorDirectory = snapshot.appending(path: "processor", directoryHint: .isDirectory)
    let tokenizerDirectory =
      FileManager.default.fileExists(atPath: processorDirectory.path)
      ? processorDirectory
      : snapshot.appending(path: "tokenizer", directoryHint: .isDirectory)

    guard FileManager.default.fileExists(atPath: tokenizerDirectory.path) else {
      throw QwenTokenizerError.directoryNotFound(tokenizerDirectory)
    }
    let tokenizerConfigURL = tokenizerDirectory.appending(path: "tokenizer_config.json")
    guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
      throw QwenTokenizerError.fileNotFound(tokenizerConfigURL)
    }
    let tokenizerDataURL = tokenizerDirectory.appending(path: "tokenizer.json")
    guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
      throw QwenTokenizerError.fileNotFound(tokenizerDataURL)
    }

    let tokenizerConfig = try decodeConfig(fileURL: tokenizerConfigURL)
    let tokenizerData = try decodeConfig(fileURL: tokenizerDataURL)
    let tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)

    let padTokenNode = tokenizerConfig["pad_token"]
    let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
    guard let padToken = padTokenString,
      let padId = tokenizer.convertTokenToId(padToken) ?? tokenizer.eosTokenId
    else {
      throw QwenTokenizerError.padTokenMissing
    }

    let imagePadId = tokenizer.convertTokenToId("<|image_pad|>") ?? 151_655
    return QwenTokenizer(
      padTokenId: padId,
      imageTokenId: imagePadId,
      tokenizer: tokenizer
    ) { text in
      tokenizer.encode(text: text)
    }
  }

  /// Encodes a raw string (special tokens parsed from the text, as the
  /// upstream template expects) and applies the envelope's right truncation.
  public func encodeRaw(_ text: String, maxLength: Int) -> [Int] {
    let tokens = encodeFunction(text)
    return Array(tokens.prefix(maxLength))
  }

  public func tokenCount(of text: String) -> Int {
    encodeFunction(text).count
  }

  private static func decodeConfig(fileURL: URL) throws -> Config {
    let data = try Data(contentsOf: fileURL)
    return try JSONDecoder().decode(Config.self, from: data)
  }
}
