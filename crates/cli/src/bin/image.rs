//! Image generation CLI for the frozen IG2 envelope.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use image::{
    build_image_install, generate, CancellationToken, CpuReferenceBackend, ImageBackend,
    ImageInstallSpec, ImageManifest, ImageProgress, ImageRequest, IMAGE_HEIGHT, IMAGE_QUANTIZATION,
    IMAGE_STEPS, IMAGE_WIDTH,
};

const USAGE: &str = "\
usage: turbospark-image generate --model MODEL --prompt TEXT --output PNG [OPTIONS]
       turbospark-image pack --source ROOT --output INSTALL --model-id ID --model-revision REV

Generates one 1024x1024 PNG using native Metal on macOS. The CPU reference
backend is diagnostic-only and must be selected explicitly.

Required:
    --model PATH|ALIAS        complete image install, or its local-store alias
    --prompt TEXT             prompt to condition
    --output PATH             new PNG path, never overwritten

Options:
    --seed U64                explicit seed, generated when omitted
    --width 1024              only 1024 is supported in IG2
    --height 1024             only 1024 is supported in IG2
    --steps 9                 only 9 scheduler steps are supported in IG2
    --backend native|reference
                              native Metal (default), or explicit CPU reference
    --help                    print this help

Pack source components into a complete checked image install:
    --source ROOT             Diffusers export with tokenizer/, text_encoder/,
                              transformer/, scheduler/, and vae/ components
    --output INSTALL          new install directory, never overwritten
    --model-id ID             source model identifier
    --model-revision REV      immutable source revision
";

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let result = match args.first().map(String::as_str) {
        Some("generate") => run_generate(&args[1..]),
        Some("pack") => run_pack(&args[1..]),
        _ => Err(format!(
            "expected the generate or pack subcommand\n\n{USAGE}"
        )),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_pack(args: &[String]) -> Result<(), String> {
    let mut source = None;
    let mut output = None;
    let mut model_id = None;
    let mut model_revision = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = |index: &mut usize| -> Result<String, String> {
            *index += 1;
            args.get(*index)
                .cloned()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag {
            "--source" => source = Some(PathBuf::from(value(&mut index)?)),
            "--output" => output = Some(PathBuf::from(value(&mut index)?)),
            "--model-id" => model_id = Some(value(&mut index)?),
            "--model-revision" => model_revision = Some(value(&mut index)?),
            other => return Err(format!("unknown option {other}")),
        }
        index += 1;
    }
    let spec = ImageInstallSpec {
        source_root: source.ok_or_else(|| "--source is required".to_string())?,
        output_root: output.ok_or_else(|| "--output is required".to_string())?,
        model_id: model_id.ok_or_else(|| "--model-id is required".to_string())?,
        model_revision: model_revision.ok_or_else(|| "--model-revision is required".to_string())?,
    };
    let report = build_image_install(&spec)?;
    println!(
        "packed {} files ({} bytes) to {}",
        report.file_count,
        report.total_bytes,
        report.output_root.display()
    );
    Ok(())
}

fn run_generate(args: &[String]) -> Result<(), String> {
    let mut model = None;
    let mut prompt = None;
    let mut output = None;
    let mut seed = None;
    let mut width = IMAGE_WIDTH;
    let mut height = IMAGE_HEIGHT;
    let mut steps = IMAGE_STEPS;
    let mut backend_name = "native".to_string();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = |index: &mut usize| -> Result<String, String> {
            *index += 1;
            args.get(*index)
                .cloned()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag {
            "--model" => model = Some(value(&mut index)?),
            "--prompt" => prompt = Some(value(&mut index)?),
            "--output" => output = Some(PathBuf::from(value(&mut index)?)),
            "--seed" => seed = Some(parse_u64("--seed", &value(&mut index)?)?),
            "--width" => width = parse_u32("--width", &value(&mut index)?)?,
            "--height" => height = parse_u32("--height", &value(&mut index)?)?,
            "--steps" => steps = parse_u32("--steps", &value(&mut index)?)?,
            "--backend" => backend_name = value(&mut index)?,
            other => return Err(format!("unknown option {other}")),
        }
        index += 1;
    }

    let model_arg = model.ok_or_else(|| "--model is required".to_string())?;
    let prompt = prompt.ok_or_else(|| "--prompt is required".to_string())?;
    let output = output.ok_or_else(|| "--output is required".to_string())?;
    if backend_name == "native" {
        #[cfg(not(target_os = "macos"))]
        {
            return Err(
                "native image generation requires macOS Metal; pass --backend reference for diagnostics"
                    .to_string(),
            );
        }
    }
    let noise_provenance = match backend_name.as_str() {
        "native" => "zimage_metal_xorshift_box_muller_v1".to_string(),
        "reference" => "zimage_cpu_xorshift_box_muller_v1".to_string(),
        other => {
            return Err(format!(
                "--backend expects native or reference, got {other:?}"
            ))
        }
    };
    let seed = seed.unwrap_or_else(random_seed);
    // Validate the user-controlled envelope before resolving the model. A
    // malformed request should not touch a manifest or a large install.
    let preflight = ImageRequest {
        model_id: model_arg.clone(),
        model_revision: "preflight".to_string(),
        component_revisions: BTreeMap::new(),
        prompt: prompt.clone(),
        width,
        height,
        batch: 1,
        scheduler_steps: steps,
        guidance_scale: 0.0,
        seed,
        quantization: IMAGE_QUANTIZATION.to_string(),
        noise_provenance: noise_provenance.clone(),
    };
    preflight.validate()?;
    if output.exists() {
        return Err(format!(
            "refusing to overwrite existing output {}",
            output.display()
        ));
    }
    let model_path = catalog::resolve_model_arg(&model_arg);
    let manifest = ImageManifest::load(&model_path)?;
    let model_revision = manifest
        .source
        .get("model_revision")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("installed")
        .to_string();
    let component_revisions = manifest
        .components
        .iter()
        .filter_map(|(name, component)| {
            component
                .metadata
                .get("canonical_revision")
                .and_then(serde_json::Value::as_str)
                .map(|revision| (name.clone(), revision.to_string()))
        })
        .collect::<BTreeMap<_, _>>();
    let quantization = image::image_quantization_label(&manifest)?;

    let request = ImageRequest {
        model_id: model_arg,
        model_revision,
        component_revisions,
        prompt,
        width,
        height,
        batch: 1,
        scheduler_steps: steps,
        guidance_scale: 0.0,
        seed,
        quantization,
        noise_provenance,
    };
    request.validate()?;

    let cancellation = CancellationToken::new();
    install_sigint_handler();
    let mut backend: Box<dyn ImageBackend> = match backend_name.as_str() {
        "reference" => Box::new(CpuReferenceBackend::open(&model_path)?),
        "native" => {
            #[cfg(target_os = "macos")]
            {
                Box::new(image::MetalImageBackend::open(&model_path)?)
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err(
                    "native image generation requires macOS Metal; pass --backend reference for diagnostics"
                        .to_string(),
                );
            }
        }
        _ => unreachable!("backend name was validated while building the request"),
    };
    if INTERRUPTED.load(Ordering::Acquire) {
        cancellation.cancel();
    }
    let done = Arc::new(AtomicBool::new(false));
    let watcher_done = Arc::clone(&done);
    let watcher_cancel = cancellation.clone();
    let watcher = thread::spawn(move || {
        while !watcher_done.load(Ordering::Acquire) {
            if INTERRUPTED.load(Ordering::Acquire) {
                watcher_cancel.cancel();
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
    });

    let result = generate(backend.as_mut(), &request, &cancellation, print_progress);
    done.store(true, Ordering::Release);
    watcher
        .join()
        .map_err(|_| "cancellation watcher thread failed".to_string())?;
    let result = result?;
    write_new_file(&output, &result.png)?;
    println!("{}", output.display());
    Ok(())
}

fn print_progress(progress: ImageProgress) {
    eprintln!(
        "image {:?}: {}/{}",
        progress.stage, progress.completed, progress.total
    );
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(format!(
            "output parent is not a directory: {}",
            parent.display()
        ));
    }
    if path.exists() {
        return Err(format!(
            "refusing to overwrite existing output {}",
            path.display()
        ));
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| "output path has no file name".to_string())?
        .to_string_lossy();
    let temporary = parent.join(format!(
        ".{file_name}.turbospark-partial-{}",
        std::process::id()
    ));
    let write_result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| format!("failed to create temporary PNG: {e}"))?;
        file.write_all(bytes)
            .map_err(|e| format!("failed to write PNG: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("failed to sync PNG: {e}"))?;
        drop(file);
        if path.exists() {
            return Err(format!(
                "output appeared while generating: {}",
                path.display()
            ));
        }
        fs::hard_link(&temporary, path)
            .map_err(|e| format!("failed to publish PNG {}: {e}", path.display()))?;
        fs::remove_file(&temporary)
            .map_err(|e| format!("failed to finalize PNG {}: {e}", path.display()))
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects an unsigned integer, got {value:?}"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects an unsigned 64-bit integer, got {value:?}"))
}

fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
}

fn install_sigint_handler() {
    // libc is already a CLI dependency. The handler only flips an atomic flag,
    // which is async-signal-safe; all cleanup stays in the generation thread.
    #[cfg(unix)]
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_sigint as *const () as libc::sighandler_t,
        );
    }
}

extern "C" fn handle_sigint(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("turbospark-image-{name}-{}", std::process::id()))
    }

    #[test]
    fn output_is_published_only_after_the_complete_file_is_written() {
        let root = test_root("output");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create output test directory");
        let output = root.join("image.png");
        write_new_file(&output, b"complete png").expect("publish output");
        assert_eq!(fs::read(&output).expect("read output"), b"complete png");
        assert_eq!(
            fs::read_dir(&root).expect("read output directory").count(),
            1,
            "the temporary publication file must not remain"
        );
        fs::remove_dir_all(root).expect("remove output test directory");
    }

    #[test]
    fn output_refuses_to_overwrite_a_completed_file() {
        let root = test_root("overwrite");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create overwrite test directory");
        let output = root.join("image.png");
        fs::write(&output, b"original").expect("write sentinel");
        let error = write_new_file(&output, b"replacement").expect_err("overwrite must fail");
        assert!(error.contains("refusing to overwrite"), "{error}");
        assert_eq!(fs::read(&output).expect("read sentinel"), b"original");
        fs::remove_dir_all(root).expect("remove overwrite test directory");
    }
}
