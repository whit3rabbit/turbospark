//! `pull-vision` of a LLaVA 1.6 repository, end to end against a loopback
//! server: the `llava_next` dispatch, the ranged shard reads including
//! `image_newline`, the written sidecar, and what `auto` then does with it
//! (`docs/LLAVA_NEXT_VISION_PHASE0.md`). Nothing leaves the machine.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

use repack::synthetic_llava_next_checkpoint;
use turbospark_catalog::{
    install, resolve_vision_sidecar, resolve_vision_sidecar_auto, set_hf_endpoint_override,
    AutoVisionSidecar, Client, InstallPlan, RepoRef, Store,
};

/// Serve `files` by their basename, honoring one `Range: bytes=a-b` header.
fn serve(files: Vec<(&'static str, Vec<u8>)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let url = format!("http://{}", listener.local_addr().expect("address"));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let files = files.clone();
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().expect("clone"));
                let mut request = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    let done = line == "\r\n";
                    request.push_str(&line);
                    if done {
                        break;
                    }
                }
                let mut words = request.split_whitespace();
                let (method, path) = (words.next().unwrap_or(""), words.next().unwrap_or(""));
                let range = request.lines().find_map(|l| {
                    let (name, value) = l.split_once(':')?;
                    if !name.eq_ignore_ascii_case("range") {
                        return None;
                    }
                    let (a, b) = value.trim().strip_prefix("bytes=")?.split_once('-')?;
                    Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))
                });
                let name = path.rsplit('/').next().unwrap_or("");
                let (status, extra, body) = match files.iter().find(|(n, _)| *n == name) {
                    None => ("404 Not Found", String::new(), Vec::new()),
                    Some((_, data)) => match range {
                        Some((a, b)) if a < data.len() => {
                            let end = (b + 1).min(data.len());
                            (
                                "206 Partial Content",
                                format!("Content-Range: bytes {a}-{}/{}\r\n", end - 1, data.len()),
                                data[a..end].to_vec(),
                            )
                        }
                        _ => ("200 OK", String::new(), data.clone()),
                    },
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                if method != "HEAD" {
                    let _ = stream.write_all(&body);
                }
            });
        }
    });
    url
}

#[test]
fn a_llava_next_repository_installs_as_a_clip_next_sidecar_that_auto_skips() {
    let root =
        std::env::temp_dir().join(format!("turbospark-llava-next-pull-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("temp root");
    // This test binary holds one test, so the process environment is ours: no
    // stored token, and no cache outside the temp directory.
    std::env::set_var("HOME", &root);
    std::env::set_var("TURBOSPARK_HOME", &root);
    std::env::remove_var("HF_TOKEN");
    std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
    std::env::remove_var("HF_HOME");

    let ck = synthetic_llava_next_checkpoint(64);
    set_hf_endpoint_override(Some(serve(vec![
        ("config.json", ck.config_json.clone().into_bytes()),
        (
            "preprocessor_config.json",
            ck.preprocessor_json.clone().into_bytes(),
        ),
        ("model.safetensors", ck.safetensors.clone()),
    ])));

    let store = Store::new(root.clone());
    let dir = root.join("models").join("next.gturbo-vision");
    let plan = InstallPlan::for_vision_tower("next", RepoRef::new("test/llava-next", "main"), None);
    let installed =
        install(&plan, &dir, &Client::with_token(None), |_| {}).expect("the tower installs");
    set_hf_endpoint_override(None);

    let (record, vision) = model_io::load_vision_sidecar(&dir).expect("loads");
    assert_eq!(vision.tower, model_io::VisionTowerKind::ClipNext);
    assert_eq!(
        (
            record.pairs_with.family.as_str(),
            record.pairs_with.hidden_size
        ),
        ("llama", 64)
    );
    assert_eq!(record.tower_blocks, 2);
    let index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    assert!(
        index
            .entries
            .keys()
            .any(|k| k.as_str() == "vision.image_newline"),
        "the newline is fetched and written"
    );
    // The runtime reads the processor config from the sidecar directory.
    assert!(dir.join("preprocessor_config.json").is_file());

    store
        .record(&installed.model)
        .expect("records the installed tower");
    let family = model_io::ModelFamily::Llama;
    assert!(resolve_vision_sidecar(&store, family, 64).is_ok());
    match resolve_vision_sidecar_auto(&store, false, family, 64).expect("auto") {
        AutoVisionSidecar::TextOnly(reason) => {
            assert!(reason.contains("LLaVA 1.6"), "{reason}")
        }
        other => panic!("auto must not attach a 1.6 tower: {other:?}"),
    }

    // A LLaVA-1.5 tower beside it still installs as `clip` through the same
    // function, without the newline, and `auto` attaches it as before.
    let mut config: serde_json::Value = serde_json::from_str(&ck.config_json).expect("config json");
    config["model_type"] = serde_json::json!("llava");
    set_hf_endpoint_override(Some(serve(vec![
        ("config.json", config.to_string().into_bytes()),
        ("preprocessor_config.json", b"{}".to_vec()),
        ("model.safetensors", ck.safetensors.clone()),
    ])));
    let old_dir = root.join("models").join("old.gturbo-vision");
    let plan = InstallPlan::for_vision_tower("old", RepoRef::new("test/llava", "main"), None);
    let installed =
        install(&plan, &old_dir, &Client::with_token(None), |_| {}).expect("1.5 installs");
    set_hf_endpoint_override(None);
    let (_, vision) = model_io::load_vision_sidecar(&old_dir).expect("loads");
    assert_eq!(vision.tower, model_io::VisionTowerKind::Clip);
    let index = model_io::load_resident_index(&old_dir.join("model_weights.bin")).expect("index");
    assert!(!index
        .entries
        .keys()
        .any(|k| k.as_str() == "vision.image_newline"));
    store.record(&installed.model).expect("records");
    assert_eq!(
        resolve_vision_sidecar_auto(&store, false, family, 64).expect("auto"),
        AutoVisionSidecar::Attached(old_dir.canonicalize().expect("canonical"))
    );
    // Two towers pair by width, so the explicit lookup refuses to pick.
    assert!(resolve_vision_sidecar(&store, family, 64).is_err());
    let _ = std::fs::remove_dir_all(&root);
}
