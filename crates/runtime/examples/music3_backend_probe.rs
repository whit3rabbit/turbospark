//! Dump AR codes, f32 hiddens and planar waveforms for the pinned MLX probe.
//! Usage: <model-dir> <reference-dir> <output-dir> [--hiddens-file PATH] [--noise-file PATH]

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, path::PathBuf, time::Instant};
    fn write_f32(path: PathBuf, values: &[f32]) -> std::io::Result<()> {
        fs::write(
            path,
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    fn read_f32(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let data = fs::read(path)?;
        if data.len() % 4 != 0 {
            return Err("supplied array is not little-endian f32".into());
        }
        let values: Vec<f32> = data
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("supplied array contains non-finite values".into());
        }
        Ok(values)
    }
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().ok_or("model directory required")?);
    let reference = PathBuf::from(args.next().ok_or("reference directory required")?);
    let output = PathBuf::from(args.next().ok_or("output directory required")?);
    let mut hiddens_file = None;
    let mut noise_file = None;
    while let Some(flag) = args.next() {
        let slot = match flag.as_str() {
            "--hiddens-file" => &mut hiddens_file,
            "--noise-file" => &mut noise_file,
            _ => return Err(format!("unknown option {flag}").into()),
        };
        if slot.is_some() {
            return Err(format!("duplicate option {flag}").into());
        }
        *slot = Some(PathBuf::from(args.next().ok_or("option requires a path")?));
    }
    let request: serde_json::Value =
        serde_json::from_slice(&fs::read(reference.join("request.json"))?)?;
    let ids: Vec<i32> = serde_json::from_value(request["ids"].clone())?;
    let frames = request["frames"].as_u64().ok_or("frames required")? as usize;
    let steps = request["steps"].as_u64().ok_or("steps required")? as usize;
    let seed = request["seed"].as_u64().ok_or("seed required")?;
    fs::create_dir_all(&output)?;
    let started = Instant::now();
    let precision = match request["precision"].as_str().ok_or("precision required")? {
        "float32" => audio::music::minimax_music3::Music3Precision::Float32,
        "checkpoint" => audio::music::minimax_music3::Music3Precision::Checkpoint,
        _ => return Err("unknown reference precision".into()),
    };
    let runner = turbospark_runtime::Music3Runner::open_with_precision(&model, precision)?;
    if std::env::var_os("TURBOSPARK_MUSIC3_TRACE").is_some() {
        let trace_dir = output.join("stages");
        fs::create_dir_all(&trace_dir)?;
        let mut counter = 0;
        let mut calls = std::collections::HashMap::<String, usize>::new();
        let limit = std::env::var("TURBOSPARK_MUSIC3_TRACE_MAX_CALLS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        runner.set_trace_observer(move |stage| {
            let call=calls.entry(stage.name.to_owned()).or_default();
            if limit>0 && *call>=limit {return;} *call+=1;
            let stem=format!("{counter:06}_{}",stage.name.replace('/',"_")); counter+=1;
            write_f32(trace_dir.join(format!("{stem}.f32")),stage.values).unwrap();
            fs::write(trace_dir.join(format!("{stem}.json")),serde_json::to_vec(&serde_json::json!({"stage":stage.name,"dtype":format!("{:?}",stage.dtype),"shape":stage.shape})).unwrap()).unwrap();
        });
    }
    let supplied_noise = if let Some(path) = &noise_file {
        let flat = read_f32(path)?;
        let shapes = request["noise_shapes"]
            .as_array()
            .ok_or("reference noise_shapes required")?;
        let mut offset = 0usize;
        let mut chunks = Vec::new();
        for shape in shapes {
            let count = shape
                .as_array()
                .ok_or("invalid noise shape")?
                .iter()
                .try_fold(1usize, |n, d| {
                    n.checked_mul(
                        d.as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or("invalid noise dimension")?,
                    )
                    .ok_or("noise shape overflows")
                })?;
            let end = offset.checked_add(count).ok_or("noise shape overflows")?;
            chunks.push(
                flat.get(offset..end)
                    .ok_or("supplied noise is too short")?
                    .to_vec(),
            );
            offset = end;
        }
        if offset != flat.len() {
            return Err("supplied noise contains unused values".into());
        }
        Some(chunks)
    } else {
        None
    };
    if let Some(path) = &hiddens_file {
        let fixed = read_f32(path)?;
        let width = runner.config().num_codebooks * runner.config().hidden_size;
        if fixed.is_empty() || fixed.len() % width != 0 {
            return Err("supplied hidden shape mismatch".into());
        }
        let frames = fixed.len() / width;
        let wave = match &supplied_noise {
            Some(noise) => runner.run_flow_with_noise(&fixed, frames, steps, noise)?,
            None => runner.run_flow(&fixed, frames, steps, seed)?,
        };
        write_f32(output.join("hiddens.f32"), &fixed)?;
        write_f32(output.join("wave.f32"), &wave)?;
        println!(
            "{}",
            serde_json::json!({"frames":frames,"samples":wave.len()/2,"precision":format!("{precision:?}"),"supplied_hiddens":path,"supplied_noise":noise_file,"elapsed_seconds":started.elapsed().as_secs_f64()})
        );
        return Ok(());
    }
    let mut warmup = Vec::new();
    let (hiddens, codes) = runner.generate_frame_hiddens_traced(&ids, frames, seed, |t| {
        if t.frame == 0 {
            warmup.push((
                t.codebook,
                t.key,
                t.sampled,
                t.hiddens.to_vec(),
                t.guided_logits.to_vec(),
            ));
        }
    })?;
    let mut decisions = Vec::new();
    for (codebook, key, sampled, hidden, logits) in warmup {
        write_f32(
            output.join(format!("warmup_{codebook}_hidden.f32")),
            &hidden,
        )?;
        write_f32(
            output.join(format!("warmup_{codebook}_logits.f32")),
            &logits,
        )?;
        decisions.push(
            serde_json::json!({"codebook": codebook, "key": [key.0,key.1], "sampled": sampled}),
        );
    }
    fs::write(
        output.join("warmup.json"),
        serde_json::to_vec_pretty(&decisions)?,
    )?;
    write_f32(output.join("hiddens.f32"), &hiddens)?;
    fs::write(
        output.join("codes.json"),
        serde_json::to_vec_pretty(&codes)?,
    )?;
    if std::env::var_os("TURBOSPARK_MUSIC3_AR_ONLY").is_some() {
        println!(
            "{}",
            serde_json::json!({"frames":codes.len(),"precision":format!("{precision:?}"),"elapsed_seconds":started.elapsed().as_secs_f64()})
        );
        return Ok(());
    }
    let frames = codes.len();
    let wave = match &supplied_noise {
        Some(noise) => runner.run_flow_with_noise(&hiddens, frames, steps, noise)?,
        None => runner.run_flow(&hiddens, frames, steps, seed)?,
    };
    write_f32(output.join("wave.f32"), &wave)?;
    // Fixed reference hiddens isolate the flow/vocoder from stochastic AR.
    let fixed = read_f32(&reference.join("hiddens.f32"))?;
    let fixed_frames = request["emitted_frames"]
        .as_u64()
        .ok_or("emitted_frames required")? as usize;
    let fixed_wave = match &supplied_noise {
        Some(noise) => runner.run_flow_with_noise(&fixed, fixed_frames, steps, noise)?,
        None => runner.run_flow(&fixed, fixed_frames, steps, seed)?,
    };
    write_f32(output.join("fixed_wave.f32"), &fixed_wave)?;
    println!(
        "{}",
        serde_json::json!({
            "frames": frames, "samples": wave.len() / 2,
            "resident_bytes": runner.resident_weight_bytes(),
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        })
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Music 3 Metal probe requires macOS");
    std::process::exit(1);
}
