import SwiftUI
import TurboSpark

/// Visual metadata and capability descriptors for catalog model families.
public struct ModelFamilyVisuals {
    public let family: String
    public let letter: String
    public let iconSystemName: String
    public let logoAssetName: String?
    public let accentColor: Color
    public let parameterTag: String
    public let architectureType: String
    public let formatLabel: String
    public let capabilities: [String]
    public let description: String

    /// The quantization format, READ OFF the catalog row's own name.
    ///
    /// Every branch below used to state this by hand and most of them were
    /// wrong, because a family and a format are independent: `mistral7b` and
    /// `tinyllama` are GGUF rows that the Mistral and TinyLlama branches both
    /// labelled "MLX INT4", `gemma4-gguf` is Q8_0 shown as "Q4_K_M", and
    /// `bonsai27b` is MLX affine 1-bit shown as INT4. The name is where the
    /// catalog states the format ("Gemma 4 26B-A4B Instruct (MLX INT4, group
    /// 64)"), so the parenthesised suffix up to the first comma IS the label
    /// and there is nothing to keep in sync.
    public static func formatLabel(alias: String, name: String) -> String {
        if let open = name.lastIndex(of: "("),
           let close = name[open...].firstIndex(of: ")") {
            let inside = name[name.index(after: open)..<close]
            let head = inside.split(separator: ",", maxSplits: 1).first.map(String.init) ?? ""
            let trimmed = head.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { return trimmed }
        }
        // No parenthesised format: fall back to the coarse signal rather than
        // inventing a precision the row never stated.
        let haystack = (name + " " + alias).lowercased()
        return haystack.contains("gguf") ? "GGUF" : "MLX"
    }

    public static func resolve(alias: String, family: String, name: String) -> ModelFamilyVisuals {
        let lAlias = alias.lowercased()
        let lFamily = family.lowercased()
        let derivedFormat = formatLabel(alias: alias, name: name)

        if lAlias.contains("gemma") || lFamily.contains("gemma") {
            return ModelFamilyVisuals(
                family: "Gemma 4",
                letter: "G",
                iconSystemName: "sparkles",
                logoAssetName: "google.png",
                accentColor: Color.purple,
                parameterTag: "26B-A4B",
                architectureType: "MoE 4/128",
                formatLabel: derivedFormat,
                capabilities: ["Conversational", "Text Generation", "Coding", "MoE"],
                description: "Google Gemma 4 mixture-of-experts model optimized with INT4 quantization and fast local streaming."
            )
        } else if lAlias.contains("qwen38") {
            return ModelFamilyVisuals(
                family: "Qwen 3.8 MTP",
                letter: "Q",
                iconSystemName: "bolt.badge.clock.fill",
                logoAssetName: "qwen.png",
                accentColor: Color.cyan,
                parameterTag: "27B-A2B",
                architectureType: "MTP Speculative",
                formatLabel: derivedFormat,
                capabilities: ["Conversational", "Speculative Drafter", "MTP Head", "MoE"],
                description: "Qwen 3.8 with multi-token-prediction drafter head achieving up to 1.66x speculative speedup."
            )
        } else if lAlias.contains("qwen36") || lFamily == "qwen36" {
            return ModelFamilyVisuals(
                family: "Qwen 3.6",
                letter: "Q",
                iconSystemName: "bolt.horizontal.fill",
                logoAssetName: "qwen.png",
                accentColor: Color.blue,
                parameterTag: "35B-A3B",
                architectureType: "Gated-DeltaNet MoE",
                formatLabel: derivedFormat,
                capabilities: ["Conversational", "Linear Attention", "Reasoning", "MoE"],
                description: "Alibaba Qwen 3.6 with Gated-DeltaNet linear attention on 30 of 40 layers for extreme throughput."
            )
        } else if lAlias.contains("qwen3moe") || lFamily == "qwen3moe" {
            return ModelFamilyVisuals(
                family: "Qwen 3 MoE",
                letter: "Q",
                iconSystemName: "circle.grid.cross.fill",
                logoAssetName: "qwen.png",
                accentColor: Color.indigo,
                parameterTag: "30B-A3B",
                architectureType: "MoE 48-Layer",
                formatLabel: derivedFormat,
                capabilities: ["Conversational", "General", "MoE"],
                description: "Qwen3 30B MoE packed in GGUF format with 48 layers and efficient slot caching."
            )
        } else if lAlias.contains("gptoss") || lFamily.contains("gptoss") {
            return ModelFamilyVisuals(
                family: "GPT-OSS",
                letter: "O",
                iconSystemName: "brain.head.profile",
                logoAssetName: "openai.svg",
                accentColor: Color.green,
                parameterTag: "20B",
                architectureType: "Harmony MoE",
                formatLabel: derivedFormat,
                capabilities: ["Reasoning", "Chain of Thought", "Conversational", "MoE"],
                description: "GPT-OSS 20B with Harmony reasoning analysis channel before output generation."
            )
        } else if lAlias.contains("ternary") {
            return ModelFamilyVisuals(
                family: "Ternary",
                letter: "T",
                iconSystemName: "number.square.fill",
                logoAssetName: "microsoft.svg",
                accentColor: Color.orange,
                parameterTag: "27B",
                architectureType: "1.58-bit MoE",
                formatLabel: derivedFormat,
                capabilities: ["Ultra Low Bit", "Experimental", "MoE"],
                description: "Extreme low-bit quantized model using ternary 1.58-bit weights for minimal memory consumption."
            )
        } else if lAlias.contains("museglimmer") || lFamily.contains("museglimmer") {
            return ModelFamilyVisuals(
                family: "Muse Glimmer",
                letter: "M",
                iconSystemName: "wand.and.stars",
                logoAssetName: "hf.svg",
                accentColor: Color.pink,
                parameterTag: "30B",
                architectureType: "Dense Reasoning",
                formatLabel: derivedFormat,
                capabilities: ["Reasoning", "Dense 30B", "Conversational", "Long Context"],
                description: "High-parameter dense reasoning model featuring internal self-reflection and 8k context window."
            )
        } else if lAlias.contains("ornith") || (lFamily == "qwen35" && lAlias.contains("ornith")) {
            let is9b = lAlias.contains("9b")
            return ModelFamilyVisuals(
                family: is9b ? "Ornith 9B" : "Ornith 35B",
                letter: "O",
                iconSystemName: "bird.fill",
                logoAssetName: "qwen.png",
                accentColor: Color.teal,
                parameterTag: is9b ? "9B-A1B" : "35B-A3B",
                architectureType: is9b ? "Lightweight MoE" : "35B MoE",
                formatLabel: derivedFormat,
                capabilities: is9b ? ["Fast", "Small MoE", "Low Memory"] : ["Conversational", "MoE", "General"],
                description: is9b ? "Compact Ornith MoE designed for instant response and low memory footprint." : "Full-sized Ornith MoE balanced for high generation quality."
            )
        } else if lAlias.contains("mistral") {
            return ModelFamilyVisuals(
                family: "Mistral 7B",
                letter: "M",
                iconSystemName: "wind",
                logoAssetName: "mistral.svg",
                accentColor: Color.red,
                parameterTag: "7B Dense",
                architectureType: "Dense Transformer",
                formatLabel: derivedFormat,
                capabilities: ["Conversational", "Dense 7B", "Coding", "Fast"],
                description: "Mistral 7B Instruct v0.3 dense transformer with 8,192 token context window and high single-stream speed."
            )
        } else if lAlias.contains("tinyllama") {
            return ModelFamilyVisuals(
                family: "TinyLlama",
                letter: "L",
                iconSystemName: "flame.fill",
                logoAssetName: "meta.svg",
                accentColor: Color.yellow,
                parameterTag: "1.1B Dense",
                architectureType: "Compact Dense",
                formatLabel: derivedFormat,
                capabilities: ["Ultra Compact", "Fast", "Low Memory", "1.1B"],
                description: "Ultra-compact 1.1B parameter model with instant load times and negligible RAM footprint."
            )
        } else if lAlias.contains("mixtral") {
            return ModelFamilyVisuals(
                family: "Mixtral",
                letter: "M",
                iconSystemName: "square.split.2x2.fill",
                logoAssetName: "mistral.svg",
                accentColor: Color.orange,
                parameterTag: "8x7B MoE",
                architectureType: "Classic MoE",
                formatLabel: derivedFormat,
                capabilities: ["Conversational", "8x7B MoE", "Code"],
                description: "Mixtral 8x7B classic mixture-of-experts model running via GGUF format."
            )
        } else if lAlias.contains("bonsai") {
            return ModelFamilyVisuals(
                family: "Bonsai 27B",
                letter: "B",
                iconSystemName: "leaf.fill",
                logoAssetName: "meta.svg",
                accentColor: Color.mint,
                parameterTag: "27B Dense",
                architectureType: "Dense 27B",
                formatLabel: derivedFormat,
                capabilities: ["Dense 27B", "General", "Text Generation"],
                description: "Bonsai 27B dense model offering broad capabilities and deep knowledge."
            )
        } else if lFamily == "llama" || lAlias.contains("llama") {
            return ModelFamilyVisuals(
                family: "LLaMA",
                letter: "L",
                iconSystemName: "cpu.fill",
                logoAssetName: "meta.svg",
                accentColor: Color.orange,
                parameterTag: "Dense LLaMA",
                architectureType: "LLaMA Architecture",
                formatLabel: derivedFormat,
                capabilities: ["Dense", "Text Generation", "General"],
                description: name
            )
        } else if lAlias.contains("deepseek") || lFamily.contains("deepseek") {
            return ModelFamilyVisuals(
                family: "DeepSeek",
                letter: "D",
                iconSystemName: "water.waves",
                logoAssetName: "deepseek.svg",
                accentColor: Color.blue,
                parameterTag: "DeepSeek MoE",
                architectureType: "DeepSeek Architecture",
                formatLabel: derivedFormat,
                capabilities: ["Reasoning", "Coding", "MoE"],
                description: name
            )
        } else if lAlias.contains("phi") || lFamily.contains("phi") {
            return ModelFamilyVisuals(
                family: "Phi",
                letter: "P",
                iconSystemName: "circle.grid.2x2.fill",
                logoAssetName: "microsoft.svg",
                accentColor: Color.blue,
                parameterTag: "Phi Dense",
                architectureType: "SLM Architecture",
                formatLabel: derivedFormat,
                capabilities: ["Reasoning", "Compact", "Coding"],
                description: name
            )
        }

        // Generic fallback
        let initial = String(alias.prefix(1)).uppercased()
        return ModelFamilyVisuals(
            family: family.isEmpty ? "Transformer" : family.capitalized,
            letter: initial.isEmpty ? "M" : initial,
            iconSystemName: "cpu.fill",
            logoAssetName: "hf.svg",
            accentColor: Color.accentColor,
            parameterTag: "Transformer",
            architectureType: family.isEmpty ? "Neural Model" : family,
            formatLabel: derivedFormat,
            capabilities: ["Text Generation", "Conversational"],
            description: name
        )
    }
}
