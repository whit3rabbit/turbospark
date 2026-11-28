#![cfg(target_os = "macos")]
//! ACE-Step 1.5 through the native audio session: request validation by
//! default, and an ignored real session (TURBOSPARK_ACE_STEP15_ROOT, with
//! `silence_latent.safetensors` converted) that renders a song end to end
//! and checks the admission estimate against the measured footprint.

use std::path::PathBuf;

use turbospark_runtime::native_audio::{
    validate_request, AudioCancel, AudioEvent, AudioRequest, AudioSession, AudioTask,
    ACE_STEP15_FAMILY, MUSIC3_FAMILY,
};

fn ace() -> AudioRequest {
    AudioRequest {
        task: AudioTask::Music,
        caption: "orchestral strings, solo cello, slow and warm, classical".into(),
        lyrics: "[Instrumental]".into(),
        duration_seconds: Some(10.0),
        steps: None,
        seed: Some(7),
        ..AudioRequest::default()
    }
}

#[test]
fn ace_requests_validate_and_refuse_foreign_settings() {
    assert_eq!(validate_request(ACE_STEP15_FAMILY, &ace(), &[]), Ok(()));
    let ok_steps = AudioRequest {
        steps: Some(8),
        language: Some("zh".into()),
        ..ace()
    };
    assert_eq!(validate_request(ACE_STEP15_FAMILY, &ok_steps, &[]), Ok(()));
    let refused = [
        ("pcm", ace(), vec![0.0f32; 4]),
        (
            "steps",
            AudioRequest {
                steps: Some(30),
                ..ace()
            },
            vec![],
        ),
        (
            "context",
            AudioRequest {
                context_size: Some(512),
                ..ace()
            },
            vec![],
        ),
        (
            "max_frames",
            AudioRequest {
                max_frames: Some(250),
                ..ace()
            },
            vec![],
        ),
        (
            "no duration",
            AudioRequest {
                duration_seconds: None,
                ..ace()
            },
            vec![],
        ),
        (
            "zero duration",
            AudioRequest {
                duration_seconds: Some(0.0),
                ..ace()
            },
            vec![],
        ),
        (
            "long",
            AudioRequest {
                duration_seconds: Some(600.5),
                ..ace()
            },
            vec![],
        ),
        (
            "nan",
            AudioRequest {
                duration_seconds: Some(f64::NAN),
                ..ace()
            },
            vec![],
        ),
        (
            "language",
            AudioRequest {
                language: Some("en us".into()),
                ..ace()
            },
            vec![],
        ),
        (
            "temperature",
            AudioRequest {
                temperature: Some(0.7),
                ..ace()
            },
            vec![],
        ),
    ];
    for (name, r, pcm) in refused {
        assert!(
            validate_request(ACE_STEP15_FAMILY, &r, &pcm).is_err(),
            "{name}: should be refused"
        );
    }
    // Music 3 keeps its own rules: steps 30 and a context size are fine.
    let music3 = AudioRequest {
        steps: Some(30),
        context_size: Some(2048),
        ..ace()
    };
    assert_eq!(validate_request(MUSIC3_FAMILY, &music3, &[]), Ok(()));
}

#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_ROOT; loads the real checkpoint on Metal"]
fn real_session_renders_a_song() {
    let root = PathBuf::from(std::env::var("TURBOSPARK_ACE_STEP15_ROOT").unwrap());
    let before = gpu::process_footprint().unwrap();
    let session = AudioSession::open(root.clone(), AudioTask::Music, false).unwrap();
    let opened = gpu::process_footprint().unwrap();
    assert_eq!(session.family(), ACE_STEP15_FAMILY);
    let estimate = turbospark_runtime::ace_step15::open_bytes(&root).unwrap();
    let mut stages = Vec::new();
    let mut pcm = 0usize;
    let result = session
        .execute(
            ace(),
            Vec::new(),
            AudioCancel::default(),
            |event| match event {
                AudioEvent::Progress(p) => stages.push((p.stage, p.completed, p.total)),
                AudioEvent::Pcm(chunk) => pcm += chunk.len(),
            },
        )
        .unwrap();
    let peak = gpu::process_footprint().unwrap();
    let format = result.pcm_format.unwrap();
    eprintln!(
        "session: open footprint {:.2} GiB (estimate {:.2} GiB), after one 10 s song {:.2} GiB, \
         stages {}",
        (opened - before) as f64 / (1u64 << 30) as f64,
        estimate as f64 / (1u64 << 30) as f64,
        (peak - before) as f64 / (1u64 << 30) as f64,
        stages.len()
    );
    assert_eq!(
        (format.sample_rate, format.channels, format.interleaved),
        (48_000, 2, true)
    );
    assert_eq!(pcm, 2 * 480_000);
    assert_eq!(result.sample_count, pcm);
    assert_eq!(result.seed, Some(7));
    // The session announces `running`, then the pipeline's own stages.
    let mut want = vec![
        ("running".to_string(), 0, None),
        ("text".to_string(), 0, None),
        ("condition".to_string(), 0, None),
    ];
    want.extend((0..8).map(|i| ("dit_step".to_string(), i, Some(8))));
    want.push(("decode".to_string(), 0, None));
    assert_eq!(stages, want);
    // The estimate must not undercount what opening actually costs.
    assert!(
        estimate >= opened - before,
        "admission estimate {estimate} below the measured open footprint {}",
        opened - before
    );
}
