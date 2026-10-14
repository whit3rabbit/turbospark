//! Parity tests against `tools/gen_siglip2_fixtures.py` goldens.
//!
//! Tolerances: preprocessing is exact (the uint8 resize is integer math and
//! normalize is the same two f32 ops). Tower activations allow 1e-5 absolute
//! on values of order 1-10: the remaining differences are f32 reduction
//! order (sequential sums here vs blocked BLAS in torch) and the f32 vs
//! f64 weights of the position-grid resize, measured at a few 1e-6 at most.

use std::path::PathBuf;

use serde_json::{json, Value};
use turbospark_vision_io::decode_image_file;

use super::config::{ProcessorConfig, Siglip2Config};
use super::processing::{preprocess, resized_image, target_size};
use super::vision::resize_grid_aa_bilinear;
use super::Siglip2;
use crate::similarity::cosine_similarity;
use crate::testutil::{max_abs_diff, model_dir, read_json, read_npy, read_npy_u32, testdata};

const F: &str = "siglip2";
const ACT_TOL: f32 = 1e-5;

fn tiny_dir() -> PathBuf {
    testdata(F, "tiny")
}

fn npy(name: &str) -> Vec<f32> {
    read_npy(F, name).1
}

fn check(label: &str, got: &[f32], want: &[f32], tol: f32) {
    let d = max_abs_diff(got, want);
    eprintln!("{label}: max abs diff {d:e}");
    assert!(d <= tol, "{label}: max abs diff {d} > {tol}");
}

fn real_processor() -> ProcessorConfig {
    // The pinned checkpoint's preprocessor_config.json, verbatim.
    ProcessorConfig::from_json(&json!({
        "do_convert_rgb": null, "do_normalize": true, "do_rescale": true,
        "do_resize": true, "image_mean": [0.5, 0.5, 0.5],
        "image_processor_type": "Siglip2ImageProcessorFast",
        "image_std": [0.5, 0.5, 0.5], "max_num_patches": 256, "patch_size": 16,
        "processor_class": "Siglip2Processor", "resample": 2,
        "rescale_factor": 0.00392156862745098
    }))
    .unwrap()
}

#[test]
fn config_fills_hf_defaults() {
    // The pinned checkpoint's config.json, which leaves nearly every
    // field to the HF config classes.
    let cfg = Siglip2Config::from_json(&json!({
        "architectures": ["Siglip2Model"], "initializer_factor": 1.0,
        "model_type": "siglip2",
        "text_config": {"model_type": "siglip2_text_model", "vocab_size": 256000},
        "torch_dtype": "float32",
        "vision_config": {"model_type": "siglip2_vision_model"}
    }))
    .unwrap();
    assert_eq!(cfg.text.vocab_size, 256000);
    assert_eq!(cfg.text.hidden_size, 768);
    assert_eq!(cfg.text.projection_size, 768);
    assert_eq!(cfg.text.max_position_embeddings, 64);
    assert_eq!(cfg.text.num_hidden_layers, 12);
    assert_eq!(cfg.vision.num_patches, 256);
    assert_eq!(cfg.vision.position_grid(), 16);
    assert_eq!(cfg.vision.patch_size, 16);
    assert_eq!(cfg.vision.layer_norm_eps, 1e-6);
    let p = real_processor();
    assert_eq!((p.patch_size, p.max_num_patches), (16, 256));
}

#[test]
fn config_refuses_unverified_values() {
    let base = || json!({"model_type": "siglip2", "text_config": {}, "vision_config": {}});
    let refuse = |v: Value, what: &str| {
        let e = Siglip2Config::from_json(&v).expect_err(what);
        eprintln!("{what}: {e}");
    };
    refuse(
        json!({"text_config": {}, "vision_config": {}}),
        "missing model_type",
    );
    let mut v = base();
    v["model_type"] = json!("siglip");
    refuse(v, "siglip1 model_type");
    let mut v = base();
    v["text_config"]["hidden_act"] = json!("gelu");
    refuse(v, "erf gelu");
    let mut v = base();
    v["vision_config"]["vision_use_head"] = json!(false);
    refuse(v, "no pooling head");
    let mut v = base();
    v["vision_config"]["num_patches"] = json!(250);
    refuse(v, "non-square position grid");
    let mut v = base();
    v["vision_config"]["num_channels"] = json!(1);
    refuse(v, "grayscale");
    let mut v = base();
    v["text_config"]["projection_size"] = json!(512);
    refuse(v, "projection width mismatch");
    let mut v = base();
    v["vision_config"]["num_attention_heads"] = json!(7);
    refuse(v, "heads do not divide");
    let mut v = base();
    v["quantization"] = json!({"bits": 4, "group_size": 64});
    refuse(v, "quantized");

    let proc = |patch: &dyn Fn(&mut Value)| {
        let mut v = json!({"image_processor_type": "Siglip2ImageProcessorFast"});
        patch(&mut v);
        ProcessorConfig::from_json(&v)
    };
    assert!(proc(&|_| {}).is_ok());
    assert!(proc(&|v| v["image_processor_type"] = json!("Siglip2ImageProcessor")).is_err());
    assert!(proc(&|v| v["resample"] = json!(3)).is_err());
    assert!(proc(&|v| v["do_normalize"] = json!(false)).is_err());
    assert!(proc(&|v| v["image_std"] = json!([0.5, 0.0, 0.5])).is_err());
}

#[test]
fn naflex_target_sizes_match_reference() {
    let meta = read_json(F, "real/meta.json");
    for img in meta["images"].as_array().unwrap() {
        let (w, h) = (
            img["width"].as_u64().unwrap(),
            img["height"].as_u64().unwrap(),
        );
        let want = &img["target_hw"];
        let got = target_size(h as usize, w as usize, 16, 256);
        assert_eq!(
            [got.0 as u64, got.1 as u64],
            [want[0].as_u64().unwrap(), want[1].as_u64().unwrap()],
            "{}",
            img["name"]
        );
    }
    let tiny = read_json(F, "tiny/meta.json");
    for img in tiny["images"].as_array().unwrap() {
        let (w, h) = (
            img["width"].as_u64().unwrap(),
            img["height"].as_u64().unwrap(),
        );
        let s = &img["spatial_shape"];
        let got = target_size(h as usize, w as usize, 4, 24);
        assert_eq!(
            [got.0 as u64 / 4, got.1 as u64 / 4],
            [s[0].as_u64().unwrap(), s[1].as_u64().unwrap()]
        );
    }
}

#[test]
fn real_processor_resize_and_patches_are_exact() {
    let cfg = real_processor();
    let meta = read_json(F, "real/meta.json");
    for img in meta["images"].as_array().unwrap() {
        let name = img["name"].as_str().unwrap();
        let src = decode_image_file(&testdata(F, &format!("real/{name}.png"))).unwrap();
        let want = decode_image_file(&testdata(F, &format!("real/{name}_resized.png"))).unwrap();
        let got = resized_image(&cfg, &src).unwrap();
        assert_eq!((got.height, got.width), (want.height, want.width), "{name}");
        let diff = got
            .data
            .iter()
            .zip(&want.data)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff, 0,
            "{name}: {diff} resized samples differ from torchvision"
        );

        let input = preprocess(&cfg, &src).unwrap();
        let s = &img["spatial_shape"];
        assert_eq!(
            [input.spatial_shape.0 as u64, input.spatial_shape.1 as u64],
            [s[0].as_u64().unwrap(), s[1].as_u64().unwrap()]
        );
        let mask = input.pixel_attention_mask();
        assert_eq!(mask.len(), 256);
        assert_eq!(
            mask.iter().sum::<u32>() as u64,
            img["mask_sum"].as_u64().unwrap()
        );
        assert_eq!(input.padded_pixel_values().len(), 256 * 768);
        let dim = input.patch_dim();
        let rows: Vec<usize> = img["pixel_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_u64().unwrap() as usize)
            .collect();
        let got_rows: Vec<f32> = rows
            .iter()
            .flat_map(|&r| input.patches[r * dim..(r + 1) * dim].iter().copied())
            .collect();
        let want_rows = npy(&format!("real/{name}_pixel_rows.npy"));
        assert_eq!(
            max_abs_diff(&got_rows, &want_rows),
            0.0,
            "{name} pixel rows"
        );
        let sum: f64 = input.patches.iter().map(|&v| f64::from(v)).sum();
        let want_sum = img["pixel_values_sum_f64"].as_f64().unwrap();
        assert!(
            (sum - want_sum).abs() < 1e-6,
            "{name}: sum {sum} vs {want_sum}"
        );
    }
}

#[test]
fn grid_resize_identity_and_constant() {
    let src: Vec<f32> = (0..4 * 4 * 2).map(|i| i as f32 * 0.25).collect();
    assert_eq!(resize_grid_aa_bilinear(&src, 4, 4, 2, 4, 4), src);
    // A constant field stays constant through up- and down-sampling.
    let c = vec![1.5f32; 16 * 16 * 3];
    for &(h, w) in &[(5, 42), (25, 9), (16, 16), (1, 24)] {
        let out = resize_grid_aa_bilinear(&c, 16, 16, 3, h, w);
        assert_eq!(out.len(), h * w * 3);
        assert!(out.iter().all(|v| (v - 1.5).abs() < 1e-6));
    }
}

#[test]
fn tiny_model_vision_parity() {
    let model = Siglip2::load(tiny_dir()).unwrap();
    assert!(model.tokenizer().is_none());
    let meta = read_json(F, "tiny/meta.json");
    let images = meta["images"].as_array().unwrap();
    let mut embeds = Vec::new();
    for img in images {
        let name = img["name"].as_str().unwrap();
        let src = decode_image_file(&testdata(F, &format!("tiny/{name}.png"))).unwrap();
        let input = model.preprocess(&src).unwrap();
        let n = img["valid_patches"].as_u64().unwrap() as usize;
        assert_eq!(input.num_patches(), n, "{name}");
        assert_eq!(
            input.pixel_attention_mask().iter().sum::<u32>() as usize,
            n,
            "{name} mask"
        );
        check(
            &format!("tiny {name} pixel_values"),
            &input.patches,
            &npy(&format!("tiny/{name}_pixel_values.npy")),
            0.0,
        );
        let t = model.vision().forward_trace(&input).unwrap();
        check(
            &format!("tiny {name} pos_resized"),
            &t.pos_resized,
            &npy(&format!("tiny/{name}_pos_resized.npy")),
            ACT_TOL,
        );
        check(
            &format!("tiny {name} embeddings"),
            &t.embeddings,
            &npy(&format!("tiny/{name}_embeddings.npy")),
            ACT_TOL,
        );
        check(
            &format!("tiny {name} last_hidden"),
            &t.last_hidden,
            &npy(&format!("tiny/{name}_last_hidden.npy")),
            ACT_TOL,
        );
        check(
            &format!("tiny {name} pooled"),
            &t.pooled,
            &npy(&format!("tiny/{name}_pooled.npy")),
            ACT_TOL,
        );
        embeds.push(model.embed_pixels(&input).unwrap());
    }
    let got: Vec<f32> = embeds.concat();
    check(
        "tiny image_embeds",
        &got,
        &npy("tiny/image_embeds.npy"),
        ACT_TOL,
    );
}

#[test]
fn tiny_model_text_parity_and_logits() {
    let model = Siglip2::load(tiny_dir()).unwrap();
    let (shape, ids) = read_npy_u32(F, "tiny/text_ids.npy");
    let (n, len) = (shape[0], shape[1]);
    let h = model.config().text.hidden_size;
    let want_emb = npy("tiny/text_embeddings.npy");
    let want_last = npy("tiny/text_last_hidden.npy");
    let want_pool = npy("tiny/text_pooled.npy");
    let mut text_embeds = Vec::new();
    for i in 0..n {
        let row = &ids[i * len..(i + 1) * len];
        let t = model.text().forward_trace(row).unwrap();
        let s = i * len * h..(i + 1) * len * h;
        check(
            &format!("tiny text {i} embeddings"),
            &t.embeddings,
            &want_emb[s.clone()],
            ACT_TOL,
        );
        check(
            &format!("tiny text {i} last_hidden"),
            &t.last_hidden,
            &want_last[s],
            ACT_TOL,
        );
        let p = model.config().text.projection_size;
        check(
            &format!("tiny text {i} pooled"),
            &t.pooled,
            &want_pool[i * p..(i + 1) * p],
            ACT_TOL,
        );
        text_embeds.push(model.embed_ids(row).unwrap());
    }
    check(
        "tiny text_embeds",
        &text_embeds.concat(),
        &npy("tiny/text_embeds.npy"),
        ACT_TOL,
    );

    let (ishape, img) = read_npy(F, "tiny/image_embeds.npy");
    let images: Vec<Vec<f32>> = img.chunks(ishape[1]).map(<[f32]>::to_vec).collect();
    let logits = model.logits(&images, &text_embeds).unwrap();
    check(
        "tiny logits_per_image",
        &logits,
        &npy("tiny/logits_per_image.npy"),
        1e-4,
    );
    let probs = model.probabilities(&images, &text_embeds).unwrap();
    for (p, l) in probs.iter().zip(&logits) {
        assert!((p - 1.0 / (1.0 + (-l).exp())).abs() < 1e-7);
    }
}

#[test]
fn text_rejects_overlong_and_out_of_vocab_ids() {
    let model = Siglip2::load(tiny_dir()).unwrap();
    assert!(model.embed_ids(&[1; 17]).is_err());
    assert!(model.embed_ids(&[]).is_err());
    assert!(model.embed_ids(&[128]).is_err());
    assert!(model.embed_text("needs a tokenizer").is_err());
}

fn real_model() -> Option<Siglip2> {
    let dir = model_dir("TURBOSPARK_SIGLIP2_DIR", "siglip2-base-patch16-naflex")?;
    Some(Siglip2::load(&dir).expect("load real SigLIP2"))
}

fn real_texts() -> Vec<String> {
    read_json(F, "real/meta.json")["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap().to_owned())
        .collect()
}

#[test]
#[ignore = "needs the real checkpoint (TURBOSPARK_SIGLIP2_DIR)"]
fn real_tokenizer_ids_match_processor() {
    let Some(dir) = model_dir("TURBOSPARK_SIGLIP2_DIR", "siglip2-base-patch16-naflex") else {
        return;
    };
    let tok = super::Siglip2Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    assert_eq!((tok.pad_id(), tok.eos_id()), (0, 1));
    let (shape, want) = read_npy_u32(F, "real/text_ids.npy");
    for (i, text) in real_texts().iter().enumerate() {
        let got = tok.encode_padded(text, 64).unwrap();
        assert_eq!(
            got,
            want[i * shape[1]..(i + 1) * shape[1]],
            "text {i}: {text:?}"
        );
    }
}

#[test]
#[ignore = "needs the real checkpoint (TURBOSPARK_SIGLIP2_DIR)"]
fn real_checkpoint_parity() {
    let t0 = std::time::Instant::now();
    let Some(model) = real_model() else {
        return;
    };
    eprintln!("load: {:.2?}", t0.elapsed());
    let meta = read_json(F, "real/meta.json");
    let (_, want_img) = read_npy(F, "real/image_embeds.npy");
    let (_, want_txt) = read_npy(F, "real/text_embeds.npy");
    let d = 768;
    let mut images = Vec::new();
    for (i, img) in meta["images"].as_array().unwrap().iter().enumerate() {
        let name = img["name"].as_str().unwrap();
        let t = std::time::Instant::now();
        let e = model
            .embed_image_file(&testdata(F, &format!("real/{name}.png")))
            .unwrap();
        let el = t.elapsed();
        let want = &want_img[i * d..(i + 1) * d];
        let diff = max_abs_diff(&e, want);
        let cos = cosine_similarity(&e, want);
        eprintln!("image {name}: max abs {diff:e}, cosine {cos:.8}, {el:.2?}");
        assert!(diff < 1e-5 && cos > 0.99999, "{name}");
        images.push(e);
    }
    let mut texts = Vec::new();
    for (i, text) in real_texts().iter().enumerate() {
        let t = std::time::Instant::now();
        let e = model.embed_text(text).unwrap();
        let el = t.elapsed();
        let want = &want_txt[i * d..(i + 1) * d];
        let diff = max_abs_diff(&e, want);
        let cos = cosine_similarity(&e, want);
        eprintln!("text {i}: max abs {diff:e}, cosine {cos:.8}, {el:.2?}");
        assert!(diff < 1e-5 && cos > 0.99999, "text {i}");
        texts.push(e);
    }
    let logits = model.logits(&images, &texts).unwrap();
    let want = npy("real/logits_per_image.npy");
    let ld = max_abs_diff(&logits, &want);
    eprintln!("logits max abs {ld:e}");
    eprintln!("logits rust {logits:?}");
    eprintln!("logits ref  {want:?}");
    assert!(ld < 1e-3, "logits");
}
