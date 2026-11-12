---
description: Rewrite drafts to sound natural using a local 12B model, user-defined rules, and an automatic audit.
icon: pen-nib
---

# Humanize (Text Rewriting)

The **Humanize** feature rewrites drafts so they read more naturally and less like generic machine-generated text. It runs entirely on your Mac using `humanizer-12b`, a dense Gemma 4 12B checkpoint (MLX INT8, ~12.7 GB on disk), within an 8,192-token context window.

## Architecture: Why Rules Run Outside the Model

The `humanizer-12b` model was trained on a single, fixed instruction objective. Consequently, the core model refuses prompt modifications or conversational system prompts.

To give you complete control without degrading output quality, TurboSpark executes all styling and guardrail rules **outside the model**, on the finished rewrite:

* **Protected Terms**: Words, trademarks, acronyms, or proper nouns that must appear in the final output exactly as originally written.
* **Words to Avoid**: Vocabulary that you prefer not to see; matches are flagged or substituted.
* **Replacements**: Specific `old -> new` phrase mappings applied systematically to the output.
* **Deterministic Seed**: Setting a fixed numeric seed ensures that identical drafts produce bit-identical rewrites.

---

## Automatic Number & Fact Slip Audit

Every rewrite automatically runs through an automated verification audit before being shown:

1. **Dropped Numbers**: Flags any dates, currency amounts, statistics, or numerical measurements present in your draft that the model omitted.
2. **Invented Numbers**: Alerts you if the model hallucinated figures or percentages not present in your draft.
3. **Script Switching**: Flags unintended switches between Latin, Cyrillic, or CJK scripts.
4. **Verbatim Leaks**: Warns if large paragraphs were copied verbatim without restructuring.

*Important*: The audit alerts you to potential issues; it never silently edits your text. Always review the side-by-side comparison before publishing.

---

## Using Humanize in the Desktop App

1. Select **Humanize** from the app sidebar.
2. Paste your draft in the left pane.
3. (Optional) Expand **Styling Rules** to enter protected terms or forbidden words.
4. Click **Humanize**.
5. The rewritten draft appears in the right pane with dynamic visual diff highlights showing added and omitted phrasing.

---

## Using Humanize via CLI

```sh
# Pull the model
turbospark-model pull humanizer-12b

# Rewrite a draft with a fixed seed
turbospark humanize humanizer-12b draft.txt --seed 42 --output rewritten.txt
```

Long documents exceeding the model context are automatically chunked into overlapping passages, rewritten sequentially, and stitched together cleanly.
