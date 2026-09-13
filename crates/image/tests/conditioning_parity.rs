use std::fs;
use turbospark_image::conditioning::{
    frame_prompt, load_tokenizer, tokenize_prompt, MAX_SEQUENCE_LENGTH,
};
use turbospark_image::fixtures::{
    capture_manifest_path, contracts_json_path, model_subpath, read_npy_file_i64, run_array_path,
};

const CASES: [(&str, &str); 7] = [
    ("empty", ""),
    (
        "composition",
        "A red ceramic teapot to the left of a blue cup on a wooden table, a window behind them.",
    ),
    (
        "typography",
        "A shop sign with the exact words \"FRESH BREAD\" in clear black letters.",
    ),
    (
        "detail",
        "Macro photograph of a honeybee on lavender, fine hairs and translucent wing veins.",
    ),
    (
        "lighting",
        "A lighthouse in winter at dusk, warm windows reflected on wet snow, cold blue shadows.",
    ),
    (
        "unicode",
        "\u{96ea}\u{4e2d}\u{306e}\u{706f}\u{53f0}, caf\u{00e9} au cr\u{00e9}puscule",
    ),
    ("overlong", ""), // Will be constructed dynamically as "small red lighthouse " * 600
];

#[test]
fn test_conditioning_framing_exact_parity() {
    let tok_dir = model_subpath("tokenizer");
    if !tok_dir.exists() {
        eprintln!(
            "NOTE: skipping test_conditioning_framing_exact_parity; tokenizer checkout missing at {}",
            tok_dir.display()
        );
        return;
    }

    let tok = load_tokenizer(&tok_dir).expect("load tokenizer");

    for (case_name, raw_prompt) in CASES {
        let prompt_str = if case_name == "overlong" {
            "small red lighthouse ".repeat(600)
        } else {
            raw_prompt.to_string()
        };

        let manifest_path = capture_manifest_path(case_name, "encode.json");
        assert!(
            manifest_path.exists(),
            "manifest missing at {}",
            manifest_path.display()
        );

        let manifest_str = fs::read_to_string(&manifest_path).expect("read encode manifest");
        let manifest_json: serde_json::Value =
            serde_json::from_str(&manifest_str).expect("parse encode manifest");
        let expected_framed = manifest_json["framed_prompt"]
            .as_str()
            .expect("framed_prompt field");

        let actual_framed = frame_prompt(&prompt_str, &tok).expect("render chat template");
        assert_eq!(
            actual_framed, expected_framed,
            "case {case_name} framed prompt mismatch"
        );
    }
}

#[test]
fn test_conditioning_tokenization_exact_parity() {
    let tok_dir = model_subpath("tokenizer");
    if !tok_dir.exists() {
        eprintln!(
            "NOTE: skipping test_conditioning_tokenization_exact_parity; tokenizer checkout missing at {}",
            tok_dir.display()
        );
        return;
    }

    let tok = load_tokenizer(&tok_dir).expect("load tokenizer");

    for (case_name, _raw_prompt) in CASES {
        let manifest_path = capture_manifest_path(case_name, "encode.json");
        let manifest_str = fs::read_to_string(&manifest_path).expect("read encode manifest");
        let manifest_json: serde_json::Value =
            serde_json::from_str(&manifest_str).expect("parse encode manifest");
        let framed = manifest_json["framed_prompt"]
            .as_str()
            .expect("framed_prompt field");

        let (token_ids, attention_mask) = tokenize_prompt(framed, &tok, MAX_SEQUENCE_LENGTH);
        assert_eq!(token_ids.len(), MAX_SEQUENCE_LENGTH);
        assert_eq!(attention_mask.len(), MAX_SEQUENCE_LENGTH);

        // Compare with captured arrays if present
        let ids_path = run_array_path(case_name, "token_ids.npy");
        let mask_path = run_array_path(case_name, "attention_mask.npy");

        if ids_path.exists() && mask_path.exists() {
            let captured_ids = read_npy_file_i64(&ids_path).expect("read captured token_ids.npy");
            let captured_mask =
                read_npy_file_i64(&mask_path).expect("read captured attention_mask.npy");

            assert_eq!(
                token_ids, captured_ids.data,
                "case {case_name} token_ids mismatch"
            );
            assert_eq!(
                attention_mask, captured_mask.data,
                "case {case_name} attention_mask mismatch"
            );
        }

        // Verify untruncated and retained counts
        let raw_encoded = tok.encode(framed, false);
        let untruncated_count = raw_encoded.len();
        let retained_count: i64 = attention_mask.iter().sum();

        let contracts_path = contracts_json_path();
        if contracts_path.exists() {
            let contracts_str = fs::read_to_string(&contracts_path).expect("read contracts json");
            let contracts_json: serde_json::Value =
                serde_json::from_str(&contracts_str).expect("parse contracts json");
            if let Some(case_info) = contracts_json["token_cases"].get(case_name) {
                let expected_untruncated =
                    case_info["untruncated_tokens"].as_u64().unwrap() as usize;
                let expected_retained = case_info["retained_tokens"].as_i64().unwrap();
                assert_eq!(
                    untruncated_count, expected_untruncated,
                    "case {case_name} untruncated tokens mismatch"
                );
                assert_eq!(
                    retained_count, expected_retained,
                    "case {case_name} retained tokens mismatch"
                );
            }
        }
    }
}
