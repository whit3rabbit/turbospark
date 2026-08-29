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

    public static func resolve(alias: String, family: String, name: String) -> ModelFamilyVisuals {
        let lAlias = alias.lowercased()
        let lName = name.lowercased()
        let lFamily = family.lowercased()

        if lAlias.contains("gemma") || lFamily.contains("gemma") {
            let isGguf = lAlias.contains("gguf") || lAlias.contains("iq3")
            return ModelFamilyVisuals(
                family: "Gemma 4",
                letter: "G",
                iconSystemName: "sparkles",
                logoAssetName: "google.png",
                accentColor: Color.purple,
                parameterTag: "26B-A4B",
                architectureType: "MoE 4/128",
                formatLabel: isGguf ? (lAlias.contains("iq3") ? "GGUF IQ3_M" : "GGUF Q4_K_M") : "MLX INT4",
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
                formatLabel: "MLX INT4",
                capabilities: ["Conversational", "Speculative Drafter", "MTP Head", "MoE"],
                description: "Qwen 3.8 with multi-token-prediction drafter head achieving up to 1.66x speculative speedup."
            )
        } else if lAlias.contains("qwen36") || lFamily == "qwen36" {
            let isGguf = lAlias.contains("gguf")
            return ModelFamilyVisuals(
                family: "Qwen 3.6",
                letter: "Q",
                iconSystemName: "bolt.horizontal.fill",
                logoAssetName: "qwen.png",
                accentColor: Color.blue,
                parameterTag: "35B-A3B",
                architectureType: "Gated-DeltaNet MoE",
                formatLabel: isGguf ? "GGUF Q4_K_M" : "MLX INT4",
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
                formatLabel: "GGUF Q4_K_M",
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
                formatLabel: "MLX INT4",
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
                formatLabel: "MLX 1.58b",
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
                formatLabel: "MLX INT4",
                capabilities: ["Reasoning", "Dense 30B", "Conversational", "Long Context"],
                description: "High-parameter dense reasoning model featuring internal self-reflection and 8k context window."
            )
        } else if lAlias.contains("ornith") || (lFamily == "qwen35" && lAlias.contains("ornith")) {
            let is9b = lAlias.contains("9b")
            let isGguf = lAlias.contains("gguf")
            return ModelFamilyVisuals(
                family: is9b ? "Ornith 9B" : "Ornith 35B",
                letter: "O",
                iconSystemName: "bird.fill",
                logoAssetName: "qwen.png",
                accentColor: Color.teal,
                parameterTag: is9b ? "9B-A1B" : "35B-A3B",
                architectureType: is9b ? "Lightweight MoE" : "35B MoE",
                formatLabel: isGguf ? "GGUF Q4_K_M" : "MLX INT4",
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
                formatLabel: "MLX INT4",
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
                formatLabel: "MLX INT4",
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
                formatLabel: "GGUF Q4_K_M",
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
                formatLabel: "MLX INT4",
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
                formatLabel: "MLX INT4",
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
                formatLabel: "MLX INT4",
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
                formatLabel: "MLX INT4",
                capabilities: ["Reasoning", "Compact", "Coding"],
                description: name
            )
        }

        // Generic fallback
        let initial = String(alias.prefix(1)).uppercased()
        let isGguf = lName.contains("gguf") || lAlias.contains("gguf")
        return ModelFamilyVisuals(
            family: family.isEmpty ? "Transformer" : family.capitalized,
            letter: initial.isEmpty ? "M" : initial,
            iconSystemName: "cpu.fill",
            logoAssetName: "hf.svg",
            accentColor: Color.accentColor,
            parameterTag: "Transformer",
            architectureType: family.isEmpty ? "Neural Model" : family,
            formatLabel: isGguf ? "GGUF" : "MLX INT4",
            capabilities: ["Text Generation", "Conversational"],
            description: name
        )
    }
}
