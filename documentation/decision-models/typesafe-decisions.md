---
description: Typed local decision models via the bundled OpenKind daemon (Noul, Choice, Score).
icon: brain-circuit
---

# TypeSafe Decision Models

The **TypeSafe** tab in TurboSpark's API workspace provides structured, typed classification and scoring answers from small local models instead of unconstrained conversational free text.

TypeSafe is powered by [OpenKind](https://github.com/whit3rabbit/openkind), a specialized local decision engine. The TurboSpark desktop application bundles the `openkindd` daemon and runs it on a separate loopback listener (`127.0.0.1`).

## Decision Primitives

TypeSafe models return strongly typed JSON structures with calibrated probability distributions:

1. **Noul (Yes / No)**:
   * Binary verification or boolean classification.
   * Answers questions like "Is this comment spam?" or "Does this invoice require approval?".
   * Returns `{"decision": true, "confidence": 0.94}`.

2. **Choice (Categorical)**:
   * Selects the single best option from a user-defined set of named labels.
   * Example: Routing support tickets to `["billing", "technical", "sales", "general"]`.
   * Returns the selected choice alongside softmax probabilities for all options.

3. **Score (Ordinal Grading)**:
   * Evaluates input against structured rubrics or named performance tiers.
   * Example: Grading code review quality across `["poor", "acceptable", "good", "exceptional"]`.

---

## The OpenKind Daemon & Security Architecture

* **Process Isolation**: The OpenKind daemon (`openkindd`) runs as an isolated subprocess on `127.0.0.1` and does not share memory with the TurboSpark inference process.
* **Ephemeral API Key**: At each launch, TurboSpark generates a fresh, random cryptographic API key in memory for daemon communication.
* **Optional Keychain Storage**: In the TypeSafe tab, toggle **Remember this key in the Keychain** to preserve the credential securely across app launches.
* **Endpoint**: Requests go directly to OpenKind's `/v1/systemone` loopback endpoint.

---

## TypeSafe UI in TurboSpark

The TypeSafe tab includes:

* **Dual Request Editors**:
  * **Form Editor**: Visual builder with dropdowns for decision type, options, rubrics, and premises.
  * **JSON Editor**: Direct raw payload inspection and editing with syntax highlighting.
  * **Templates**: Pre-populated examples for sentiment, moderation, priority routing, and compliance.
* **Model Manager**:
  * Load and unload decision checkpoints (such as `laya:typed-decisions`).
  * Monitor memory residency for loaded decision models.
* **Benchmarking Suite**:
  * Send 1 to 200 sequential classification queries to measure local client wall-clock latency, Swift client overhead, and daemon throughput.

---

## Recommended Local Models

On Apple Silicon, OpenKind recommends **Laya Typed Decisions** (`laya:typed-decisions`). It is compact (~1.5 GB), fast, and specialized for zero-shot decision classification.
