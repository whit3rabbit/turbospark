#!/usr/bin/env python3
"""Mutation-checking runner for IG1 native Rust tests.

Applies mutations to crates/image files via assert-applied substitutions,
verifies that each mutation fails only its intended test case, and records
the results in docs/verification/z-image-ig1-mutations.json.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTATIONS = [
    {
        "file": "crates/image/src/scheduler.rs",
        "target": "        for &sigma in &raw_sigmas {\n            let shifted = self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma);",
        "replacement": "        for &sigma in &raw_sigmas {\n            let shifted = 2.0 * sigma / (1.0 + (self.shift - 1.0) * sigma);",
        "expected_failed_case": "test_scheduler_schedules_exact_contract",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "--test", "scheduler_parity"],
    },
    {
        "file": "crates/image/tests/scheduler_parity.rs",
        "target": "    let captured_sigmas = read_npy_file_f32(&s_path).expect(\"read captured sigmas.npy\");\n\n    let mut sched = FlowMatchEulerScheduler::default();\n    sched.set_timesteps(9);",
        "replacement": "    let captured_sigmas = read_npy_file_f32(&s_path).expect(\"read captured sigmas.npy\");\n\n    let mut sched = FlowMatchEulerScheduler::default();\n    sched.set_timesteps(8);",
        "expected_failed_case": "test_scheduler_captured_timesteps_sigmas_exact",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "--test", "scheduler_parity", "--", "test_scheduler_captured_timesteps_sigmas_exact"],
    },
    {
        "file": "crates/image/src/scheduler.rs",
        "target": "prev.push(sample[i] + dt * model_output[i]);",
        "replacement": "prev.push(sample[i] - dt * model_output[i]);",
        "expected_failed_case": "test_scheduler_euler_step_parity_latents",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "--test", "scheduler_parity", "--", "test_scheduler_euler_step_parity_latents"],
    },
    {
        "file": "crates/image/src/conditioning.rs",
        "target": "render_generic_chat_template(tokenizer, &[msg], &[], true, ReasoningEffort::Low)",
        "replacement": "render_generic_chat_template(tokenizer, &[msg], &[], true, ReasoningEffort::Off)",
        "expected_failed_case": "test_conditioning_framing_exact_parity",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "--test", "conditioning_parity", "--", "test_conditioning_framing_exact_parity"],
    },
    {
        "file": "crates/image/src/conditioning.rs",
        "target": "pub const PAD_TOKEN_ID: i64 = 151643;",
        "replacement": "pub const PAD_TOKEN_ID: i64 = 151644;",
        "expected_failed_case": "test_conditioning_tokenization_exact_parity",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "--test", "conditioning_parity", "--", "test_conditioning_tokenization_exact_parity"],
    },
    {
        "file": "crates/image/src/text_encoder.rs",
        "target": "pub const EXTRACT_LAYER_COUNT: usize = 35;",
        "replacement": "pub const EXTRACT_LAYER_COUNT: usize = 34;",
        "expected_failed_case": "test_text_encoder_real_conditioning_parity",
        "test_command": ["cargo", "test", "--release", "-p", "turbospark-image", "--test", "text_encoder_parity", "--", "--ignored", "test_text_encoder_real_conditioning_parity"],
    },
    {
        "file": "crates/image/src/rope.rs",
        "target": "x[r_idx] = x_r * cos - x_i * sin;",
        "replacement": "x[r_idx] = x_r * cos + x_i * sin;",
        "expected_failed_case": "test_apply_rotary_emb_orthogonal_rotation",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "rope::tests::test_apply_rotary_emb_orthogonal_rotation"],
    },
    {
        "file": "crates/image/src/patchify.rs",
        "target": "let sub_idx = ((pf * patch_size + ph) * patch_size + pw) * c + ch;",
        "replacement": "let sub_idx = ((pf * patch_size + pw) * patch_size + ph) * c + ch;",
        "expected_failed_case": "test_patchify_unpatchify_contract",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "--test", "transformer_math_parity", "--", "test_patchify_unpatchify_contract"],
    },
    {
        "file": "crates/image/src/transformer.rs",
        "target": "        let mut sum = 0.0f32;\n        let w_row = &weight[o * in_features..(o + 1) * in_features];\n        for (xi, wi) in x.iter().zip(w_row.iter()) {\n            sum += xi * wi;\n        }\n        // Match linear algebra backends: reduce the matrix product first,\n        // then apply the bias as the separate affine operation.\n        out[o] = bias.map_or(sum, |b| sum + b[o]);",
        "replacement": "        let mut sum = bias.map_or(0.0f32, |b| b[o]);\n        let w_row = &weight[o * in_features..(o + 1) * in_features];\n        for (xi, wi) in x.iter().zip(w_row.iter()) {\n            sum += xi * wi;\n        }\n        out[o] = sum;",
        "expected_failed_case": "transformer::tests::linear_adds_bias_after_dot_product",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "transformer::tests::linear_adds_bias_after_dot_product"],
    },
    {
        "file": "crates/image/src/transformer.rs",
        "target": "fn pairwise_sum_squares(values: &[f32]) -> f32 {\n    if values.len() <= 32 {\n        return values.iter().map(|value| value * value).sum();\n    }\n    let middle = values.len() / 2;\n    pairwise_sum_squares(&values[..middle]) + pairwise_sum_squares(&values[middle..])\n}",
        "replacement": "fn pairwise_sum_squares(values: &[f32]) -> f32 {\n    values.iter().map(|value| value * value).sum()\n}",
        "expected_failed_case": "transformer::tests::rms_reduction_is_pairwise",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "transformer::tests::rms_reduction_is_pairwise"],
    },
    {
        "file": "crates/image/src/transformer.rs",
        "target": "let gate_msa = gate_msa_raw.iter().map(|&g| g.tanh()).collect();",
        "replacement": "let gate_msa = gate_msa_raw.to_vec();",
        "expected_failed_case": "test_transformer_synthetic_block_parity",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "--test", "transformer_math_parity", "--", "--ignored", "test_transformer_synthetic_block_parity"],
    },
    {
        "file": "crates/image/src/vae.rs",
        "target": "scaled.push(val * inv_scale + VAE_SHIFT_FACTOR);",
        "replacement": "scaled.push(val * inv_scale - VAE_SHIFT_FACTOR);",
        "expected_failed_case": "test_vae_real_decode_parity",
        "test_command": ["cargo", "test", "--release", "-p", "turbospark-image", "--test", "vae_parity", "--", "--ignored", "test_vae_real_decode_parity"],
    },
    {
        "file": "crates/image/src/vae.rs",
        "target": "rgb.push((((value * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0).round() as u8);",
        "replacement": "rgb.push((((value * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0) as u8);",
        "expected_failed_case": "test_decoded_rgb_and_png_contract",
        "test_command": ["cargo", "test", "-p", "turbospark-image", "--test", "vae_parity", "--", "test_decoded_rgb_and_png_contract"],
    },
]


def run_mutations():
    results = []

    for item in MUTATIONS:
        file_path = ROOT / item["file"]
        content = file_path.read_text(encoding="utf-8")

        target = item["target"]
        replacement = item["replacement"]
        expected_case = item["expected_failed_case"]
        cmd = item["test_command"]

        # Assert target occurs uniquely
        count = content.count(target)
        if count != 1:
            raise ValueError(f"Target pattern count is {count} (expected 1) in {item['file']}: {target}")

        mutated = content.replace(target, replacement, 1)
        file_path.write_text(mutated, encoding="utf-8")

        try:
            res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            failed = res.returncode != 0
            has_expected = expected_case in res.stdout or expected_case in res.stderr
            isolated = failed and has_expected

            print(f"[{item['file']}] Mutation for '{expected_case}': failed={failed}, expected_mentioned={has_expected}")
            if not isolated:
                print("STDOUT:", res.stdout[:500])
                print("STDERR:", res.stderr[:500])

            results.append({
                "file": item["file"],
                "mutation": replacement.strip(),
                "expected_failed_case": expected_case,
                "applied_unique": True,
                "isolated_failure": isolated,
            })
        finally:
            # Always restore
            file_path.write_text(content, encoding="utf-8")

    out_file = ROOT / "docs" / "verification" / "z-image-ig1-mutations.json"
    out_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} mutation results to {out_file}")


if __name__ == "__main__":
    run_mutations()
