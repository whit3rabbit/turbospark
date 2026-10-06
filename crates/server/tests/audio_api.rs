//! Audio API integration tests against a scripted provider. These prove the
//! HTTP layer (routing, auth, validation, wire formats, jobs, metrics,
//! WebSocket framing). They say nothing about real model output: that needs
//! the macOS smoke run documented in docs/API_WORKSPACE.md.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use tokio::sync::Semaphore;
use turbospark_audio::waveform::Waveform;
use turbospark_server::registry::StaticRegistry;
use turbospark_server::{
    build_router_with_options, AudioError, AudioModelInfo, AudioProvider, AudioTask,
    GenerateRequest, GeneratedAudio, RouterOptions, ServerState, SpeechRequest, SpeechStream,
    TranscribeRequest, TranscribedSegment, Transcription,
};

type Fut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Sets a flag when the future holding it is dropped, so a test can tell a
/// cancelled job from a finished one.
struct DropFlag(Arc<AtomicBool>);
impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct Fake {
    models: Mutex<Vec<AudioModelInfo>>,
    transcribed: Mutex<Vec<TranscribeRequest>>,
    generated: Mutex<Vec<GenerateRequest>>,
    /// `*-slow` models wait for a permit before answering.
    gate: Semaphore,
    started: AtomicUsize,
    dropped: Arc<AtomicBool>,
}

impl Fake {
    fn new() -> Arc<Self> {
        let m = |id: &str, task| AudioModelInfo {
            id: id.into(),
            task,
        };
        Arc::new(Self {
            models: Mutex::new(vec![
                m("stt-test", AudioTask::SpeechToText),
                m("stt-busy", AudioTask::SpeechToText),
                m("stt-slow", AudioTask::SpeechToText),
                m("tts-test", AudioTask::TextToSpeech),
                m("music-test", AudioTask::Music),
            ]),
            transcribed: Mutex::default(),
            generated: Mutex::default(),
            gate: Semaphore::new(0),
            started: AtomicUsize::new(0),
            dropped: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl AudioProvider for Fake {
    fn models(&self) -> Vec<AudioModelInfo> {
        self.models.lock().unwrap().clone()
    }

    fn transcribe(&self, request: TranscribeRequest) -> Fut<'_, Result<Transcription, AudioError>> {
        Box::pin(async move {
            let model = request.model.clone();
            self.transcribed.lock().unwrap().push(request);
            match model.as_str() {
                "stt-busy" => return Err(AudioError::Busy),
                "stt-slow" => {
                    self.started.fetch_add(1, Ordering::SeqCst);
                    let _flag = DropFlag(Arc::clone(&self.dropped));
                    self.gate.acquire().await.unwrap().forget();
                    std::mem::forget(_flag);
                }
                _ => {}
            }
            Ok(Transcription {
                text: "hello world".into(),
                language: Some("en".into()),
                segments: vec![
                    TranscribedSegment {
                        start_seconds: 0.0,
                        end_seconds: 1.5,
                        text: " hello".into(),
                    },
                    TranscribedSegment {
                        start_seconds: 1.5,
                        end_seconds: 3725.25,
                        text: " world".into(),
                    },
                ],
            })
        })
    }

    fn speak(&self, request: SpeechRequest) -> Fut<'_, Result<SpeechStream, AudioError>> {
        Box::pin(async move {
            if request.voice != "af_heart" {
                return Err(AudioError::Invalid(format!(
                    "unknown voice {:?}",
                    request.voice
                )));
            }
            let (tx, rx) = tokio::sync::mpsc::channel(4);
            tokio::spawn(async move {
                for _ in 0..2 {
                    if tx.send(Ok(vec![0.25f32; 240])).await.is_err() {
                        return;
                    }
                }
            });
            Ok(SpeechStream {
                sample_rate: 24_000,
                chunks: rx,
            })
        })
    }

    fn generate(&self, request: GenerateRequest) -> Fut<'_, Result<GeneratedAudio, AudioError>> {
        Box::pin(async move {
            self.generated.lock().unwrap().push(request);
            Ok(GeneratedAudio {
                sample_rate: 44_100,
                channels: 2,
                samples: vec![0.1; 4410 * 2],
            })
        })
    }

    fn load(&self, alias: String) -> Fut<'_, Result<AudioModelInfo, AudioError>> {
        Box::pin(async move {
            if alias != "installed-stt" {
                return Err(AudioError::NotFound(format!(
                    "{alias:?} is not an installed audio model"
                )));
            }
            let info = AudioModelInfo {
                id: alias,
                task: AudioTask::SpeechToText,
            };
            self.models.lock().unwrap().push(info.clone());
            Ok(info)
        })
    }

    fn unload(&self, id: String) -> Fut<'_, Result<(), AudioError>> {
        Box::pin(async move {
            self.models.lock().unwrap().retain(|m| m.id != id);
            Ok(())
        })
    }
}

async fn spawn_with(fake: Option<Arc<Fake>>, key: Option<&str>) -> String {
    let registry = Arc::new(StaticRegistry::new(vec![]).unwrap());
    let mut state = ServerState::new(registry);
    if let Some(fake) = fake {
        state = state.with_audio_provider(fake);
    }
    let router = build_router_with_options(
        state,
        RouterOptions {
            api_key: key.map(str::to_string),
            ..Default::default()
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

const KEY: &str = "secret";

async fn spawn(fake: &Arc<Fake>) -> String {
    spawn_with(Some(Arc::clone(fake)), Some(KEY)).await
}

fn wav(rate: u32, channels: u16, seconds: f32) -> Vec<u8> {
    let frames = (rate as f32 * seconds) as usize;
    let mut samples = Vec::with_capacity(frames * channels as usize);
    for i in 0..frames {
        let v = (i as f32 * 0.05).sin() * 0.3;
        for _ in 0..channels {
            samples.push(v);
        }
    }
    turbospark_audio::wav::write_wav_i16(&Waveform::new(rate, channels, samples).unwrap())
}

fn multipart(fields: &[(&str, &str)], file: Option<&[u8]>) -> (String, Vec<u8>) {
    let boundary = "----tsaudioboundary";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    }
    if let Some(file) = file {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(file);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn post_stt(base: &str, fields: &[(&str, &str)], file: Option<&[u8]>) -> reqwest::Response {
    let (content_type, body) = multipart(fields, file);
    reqwest::Client::new()
        .post(format!("{base}/v1/audio/transcriptions"))
        .bearer_auth(KEY)
        .header("content-type", content_type)
        .body(body)
        .send()
        .await
        .unwrap()
}

async fn error_of(response: reqwest::Response) -> (u16, serde_json::Value) {
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

async fn until<F: FnMut() -> bool>(mut done: F) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("condition not reached in 5 s");
}

#[tokio::test]
async fn audio_routes_do_not_exist_without_a_provider() {
    let base = spawn_with(None, None).await;
    let client = reqwest::Client::new();
    for path in ["/v1/audio/models", "/v1/metrics", "/v1/audio/jobs/x"] {
        let r = client.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(r.status(), 404, "{path}");
    }
    let r = client
        .post(format!("{base}/v1/audio/speech"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    let health: serde_json::Value = client
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["state"], "empty");
}

#[tokio::test]
async fn audio_routes_require_the_api_key_but_health_does_not() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let client = reqwest::Client::new();
    for path in ["/v1/audio/models", "/v1/metrics"] {
        assert_eq!(
            client
                .get(format!("{base}{path}"))
                .send()
                .await
                .unwrap()
                .status(),
            401,
            "{path}"
        );
    }
    let health: serde_json::Value = client
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["state"], "ready");
}

#[tokio::test]
async fn models_are_listed_on_both_surfaces() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let client = reqwest::Client::new();
    let audio: serde_json::Value = client
        .get(format!("{base}/v1/audio/models"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let tasks: HashMap<String, String> = audio["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["id"].as_str().unwrap().into(),
                m["task"].as_str().unwrap().into(),
            )
        })
        .collect();
    assert_eq!(tasks["stt-test"], "speech_to_text");
    assert_eq!(tasks["tts-test"], "text_to_speech");
    assert_eq!(tasks["music-test"], "music_generation");

    let all: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let music = all["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "music-test")
        .unwrap();
    assert_eq!(music["capabilities"][0], "music_generation");
}

#[tokio::test]
async fn transcription_formats() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let file = wav(16_000, 1, 1.0);

    let r = post_stt(
        &base,
        &[("model", "stt-test"), ("language", "EN")],
        Some(&file),
    )
    .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "application/json");
    let v: serde_json::Value = r.json().await.unwrap();
    assert_eq!(v, serde_json::json!({"text": "hello world"}));
    {
        let seen = fake.transcribed.lock().unwrap();
        assert_eq!(seen[0].samples.len(), 16_000);
        assert_eq!(seen[0].language.as_deref(), Some("en"));
    }

    let v: serde_json::Value = post_stt(
        &base,
        &[
            ("model", "stt-test"),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "segment"),
        ],
        Some(&file),
    )
    .await
    .json()
    .await
    .unwrap();
    assert_eq!(v["task"], "transcribe");
    assert_eq!(v["language"], "en");
    assert_eq!(v["segments"][1]["id"], 1);
    assert!((v["duration"].as_f64().unwrap() - 1.0).abs() < 1e-6);

    let text = post_stt(
        &base,
        &[("model", "stt-test"), ("response_format", "text")],
        Some(&file),
    )
    .await
    .text()
    .await
    .unwrap();
    assert_eq!(text, "hello world\n");

    let srt = post_stt(
        &base,
        &[("model", "stt-test"), ("response_format", "srt")],
        Some(&file),
    )
    .await
    .text()
    .await
    .unwrap();
    assert_eq!(
        srt,
        "1\n00:00:00,000 --> 00:00:01,500\nhello\n\n2\n00:00:01,500 --> 01:02:05,250\nworld\n\n"
    );

    let r = post_stt(
        &base,
        &[("model", "stt-test"), ("response_format", "vtt")],
        Some(&file),
    )
    .await;
    assert_eq!(r.headers()["content-type"], "text/vtt; charset=utf-8");
    let vtt = r.text().await.unwrap();
    assert!(
        vtt.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nhello\n"),
        "{vtt}"
    );
}

#[tokio::test]
async fn transcription_downmixes_and_resamples_to_16k_mono() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let file = wav(48_000, 2, 0.5);
    assert_eq!(
        post_stt(&base, &[("model", "stt-test")], Some(&file))
            .await
            .status(),
        200
    );
    let seen = fake.transcribed.lock().unwrap();
    assert_eq!(seen[0].samples.len(), 8_000);
    assert!(seen[0].samples.iter().all(|s| s.is_finite()));
}

#[tokio::test]
async fn transcription_accepts_a_raw_wav_body_with_query_options() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let r = reqwest::Client::new()
        .post(format!(
            "{base}/v1/audio/transcriptions?model=stt-test&response_format=text"
        ))
        .bearer_auth(KEY)
        .header("content-type", "audio/wav")
        .body(wav(16_000, 1, 0.25))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.text().await.unwrap(), "hello world\n");
    assert_eq!(fake.transcribed.lock().unwrap()[0].samples.len(), 4_000);
}

#[tokio::test]
async fn transcription_refuses_what_it_cannot_do() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let file = wav(16_000, 1, 0.1);
    // (name, form fields, file part, expected status)
    type Case<'a> = (&'a str, Vec<(&'a str, &'a str)>, Option<&'a [u8]>, u16);
    let cases: Vec<Case> = vec![
        (
            "not wav",
            vec![("model", "stt-test")],
            Some(b"ID3\x03not a wav at all"),
            415,
        ),
        ("no model", vec![], Some(&file), 400),
        ("unknown model", vec![("model", "nope")], Some(&file), 404),
        ("wrong task", vec![("model", "tts-test")], Some(&file), 400),
        (
            "prompt",
            vec![("model", "stt-test"), ("prompt", "names: Zed")],
            Some(&file),
            400,
        ),
        (
            "temperature",
            vec![("model", "stt-test"), ("temperature", "0.7")],
            Some(&file),
            400,
        ),
        (
            "word timestamps",
            vec![("model", "stt-test"), ("timestamp_granularities[]", "word")],
            Some(&file),
            400,
        ),
        (
            "unknown field",
            vec![("model", "stt-test"), ("diarize", "true")],
            Some(&file),
            400,
        ),
        (
            "bad format",
            vec![("model", "stt-test"), ("response_format", "mp3")],
            Some(&file),
            400,
        ),
        (
            "stream+async",
            vec![("model", "stt-test"), ("stream", "true"), ("async", "true")],
            Some(&file),
            400,
        ),
        (
            "bad language",
            vec![("model", "stt-test"), ("language", "en; drop")],
            Some(&file),
            400,
        ),
        ("no file", vec![("model", "stt-test")], None, 400),
    ];
    for (name, fields, file, want) in cases {
        let (status, body) = error_of(post_stt(&base, &fields, file).await).await;
        assert_eq!(status, want, "{name}: {body}");
        assert!(body["error"]["message"].is_string(), "{name}: {body}");
    }
    assert!(
        fake.transcribed.lock().unwrap().is_empty(),
        "a refused request reached the model"
    );

    let r = reqwest::Client::new()
        .post(format!("{base}/v1/audio/transcriptions?model=stt-test"))
        .bearer_auth(KEY)
        .header("content-type", "text/plain")
        .body("hi")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 415);

    let r = reqwest::Client::new()
        .post(format!("{base}/v1/audio/translations"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 501);
}

#[tokio::test]
async fn transcription_rejects_audio_longer_than_the_cap() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    // 31 minutes at 8 kHz mono: ~30 MB, under the upload cap, over the duration cap.
    let file = wav(8_000, 1, 31.0 * 60.0);
    let (status, body) =
        error_of(post_stt(&base, &[("model", "stt-test")], Some(&file)).await).await;
    assert_eq!(status, 400, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("limit"));
}

#[tokio::test]
async fn upload_limit_is_larger_than_the_chat_limit_and_still_enforced() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    // 30 MiB is over the 25 MiB chat limit. If the audio route inherited it,
    // this would be a 413; the duration is ~16 minutes, under the cap.
    let size = 30 * 1024 * 1024;
    let mut big = wav(16_000, 1, 0.1);
    big.resize(size, 0);
    let len = (size - 8) as u32;
    big[4..8].copy_from_slice(&len.to_le_bytes());
    // Patch the data chunk length so the file is well formed.
    let data_at = big.windows(4).position(|w| w == b"data").unwrap();
    let data_len = (size - data_at - 8) as u32;
    big[data_at + 4..data_at + 8].copy_from_slice(&data_len.to_le_bytes());
    let r = post_stt(&base, &[("model", "stt-test")], Some(&big)).await;
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());

    // One byte past the audio cap on a raw body: a JSON 413, not axum's plain text.
    let r = reqwest::Client::new()
        .post(format!("{base}/v1/audio/transcriptions?model=stt-test"))
        .bearer_auth(KEY)
        .header("content-type", "audio/wav")
        .body(vec![0u8; 128 * 1024 * 1024 + 1])
        .send()
        .await
        .unwrap();
    let (status, body) = error_of(r).await;
    assert_eq!(status, 413);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("128 MiB"));
}

#[tokio::test]
async fn busy_is_a_429_with_retry_after() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let r = post_stt(&base, &[("model", "stt-busy")], Some(&wav(16_000, 1, 0.1))).await;
    assert_eq!(r.status(), 429);
    assert_eq!(r.headers()["retry-after"], "2");
}

#[tokio::test]
async fn ndjson_streams_start_segments_and_complete() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let r = post_stt(
        &base,
        &[("model", "stt-test"), ("stream", "true")],
        Some(&wav(16_000, 1, 1.0)),
    )
    .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "application/x-ndjson");
    let lines: Vec<serde_json::Value> = r
        .text()
        .await
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let kinds: Vec<_> = lines.iter().map(|l| l["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["start", "segment", "segment", "complete"]);
    assert_eq!(lines[3]["text"], "hello world");
}

#[tokio::test]
async fn async_transcription_job_lifecycle() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let client = reqwest::Client::new();
    let r = post_stt(
        &base,
        &[
            ("model", "stt-slow"),
            ("async", "true"),
            ("response_format", "text"),
        ],
        Some(&wav(16_000, 1, 0.1)),
    )
    .await;
    assert_eq!(r.status(), 202);
    let accepted: serde_json::Value = r.json().await.unwrap();
    let id = accepted["job_id"].as_str().unwrap().to_string();

    until(|| fake.started.load(Ordering::SeqCst) == 1).await;
    let job: serde_json::Value = client
        .get(format!("{base}/v1/audio/jobs/{id}"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(job["status"], "running");
    assert_eq!(job["kind"], "transcription");
    let early = client
        .get(format!("{base}/v1/audio/jobs/{id}/result"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(early.status(), 409);

    fake.gate.add_permits(1);
    let mut status = String::new();
    for _ in 0..200 {
        let job: serde_json::Value = client
            .get(format!("{base}/v1/audio/jobs/{id}"))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        status = job["status"].as_str().unwrap().to_string();
        if status != "running" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(status, "succeeded");
    let result = client
        .get(format!("{base}/v1/audio/jobs/{id}/result"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 200);
    // The result is rendered in the format requested at submission.
    assert_eq!(result.text().await.unwrap(), "hello world\n");

    let del = client
        .delete(format!("{base}/v1/audio/jobs/{id}"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(del.status(), 200);
    let gone = client
        .get(format!("{base}/v1/audio/jobs/{id}"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(gone.status(), 404);
}

#[tokio::test]
async fn cancelling_a_running_job_drops_its_work_and_blocks_unload() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let client = reqwest::Client::new();
    let accepted: serde_json::Value = post_stt(
        &base,
        &[("model", "stt-slow"), ("async", "true")],
        Some(&wav(16_000, 1, 0.1)),
    )
    .await
    .json()
    .await
    .unwrap();
    let id = accepted["job_id"].as_str().unwrap().to_string();
    until(|| fake.started.load(Ordering::SeqCst) == 1).await;

    // A model with a running job cannot be unloaded.
    let r = client
        .delete(format!("{base}/v1/audio/models/stt-slow"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 409);

    let del: serde_json::Value = client
        .delete(format!("{base}/v1/audio/jobs/{id}"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(del["status"], "cancelled");
    // Aborting the task drops the provider future, which is how queued work
    // is cancelled.
    until(|| fake.dropped.load(Ordering::SeqCst)).await;
    let result = client
        .get(format!("{base}/v1/audio/jobs/{id}/result"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 410);
}

#[tokio::test]
async fn too_many_unfinished_jobs_is_a_429() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let file = wav(16_000, 1, 0.05);
    for i in 0..8 {
        let r = post_stt(
            &base,
            &[("model", "stt-slow"), ("async", "true")],
            Some(&file),
        )
        .await;
        assert_eq!(r.status(), 202, "job {i}");
    }
    let r = post_stt(
        &base,
        &[("model", "stt-slow"), ("async", "true")],
        Some(&file),
    )
    .await;
    assert_eq!(r.status(), 429);
    assert_eq!(r.headers()["retry-after"], "2");
    fake.gate.add_permits(8);
}

#[tokio::test]
async fn unknown_job_is_404() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let r = reqwest::Client::new()
        .get(format!("{base}/v1/audio/jobs/job_0000000000000000"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

async fn post_json(base: &str, path: &str, body: serde_json::Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{base}{path}"))
        .bearer_auth(KEY)
        .json(&body)
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn speech_wav_pcm_and_chunked_pcm() {
    let fake = Fake::new();
    let base = spawn(&fake).await;

    let r = post_json(
        &base,
        "/v1/audio/speech",
        serde_json::json!({"model": "tts-test", "input": "hi"}),
    )
    .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "audio/wav");
    assert_eq!(r.headers()["x-audio-sample-rate"], "24000");
    let bytes = r.bytes().await.unwrap();
    let wave = turbospark_audio::wav::read_wav_f32_bytes(&bytes).unwrap();
    assert_eq!(
        (wave.sample_rate, wave.channels, wave.frame_count()),
        (24_000, 1, 480)
    );

    // `text` is accepted as an alias of `input`.
    let r = post_json(
        &base,
        "/v1/audio/speech",
        serde_json::json!({"model": "tts-test", "text": "hi", "response_format": "pcm"}),
    )
    .await;
    assert_eq!(r.headers()["content-type"], "audio/pcm");
    assert_eq!(r.bytes().await.unwrap().len(), 480 * 2);

    let r = post_json(
        &base,
        "/v1/audio/speech",
        serde_json::json!({"model": "tts-test", "input": "hi", "response_format": "pcm", "stream": true}),
    )
    .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "audio/pcm");
    assert_eq!(r.bytes().await.unwrap().len(), 480 * 2);
}

#[tokio::test]
async fn speech_validation() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let long = "x".repeat(4097);
    let cases = [
        (
            "empty input",
            serde_json::json!({"model": "tts-test", "input": "  "}),
            400,
        ),
        (
            "too long",
            serde_json::json!({"model": "tts-test", "input": long}),
            400,
        ),
        (
            "speed high",
            serde_json::json!({"model": "tts-test", "input": "a", "speed": 3.0}),
            400,
        ),
        (
            "speed low",
            serde_json::json!({"model": "tts-test", "input": "a", "speed": 0.1}),
            400,
        ),
        (
            "mp3",
            serde_json::json!({"model": "tts-test", "input": "a", "response_format": "mp3"}),
            400,
        ),
        (
            "stream wav",
            serde_json::json!({"model": "tts-test", "input": "a", "stream": true}),
            400,
        ),
        (
            "unknown field",
            serde_json::json!({"model": "tts-test", "input": "a", "instructions": "x"}),
            400,
        ),
        (
            "wrong task",
            serde_json::json!({"model": "stt-test", "input": "a"}),
            400,
        ),
        (
            "unknown model",
            serde_json::json!({"model": "nope", "input": "a"}),
            404,
        ),
        (
            "provider rejects voice",
            serde_json::json!({"model": "tts-test", "input": "a", "voice": "alloy"}),
            400,
        ),
    ];
    for (name, body, want) in cases {
        let (status, body) = error_of(post_json(&base, "/v1/audio/speech", body).await).await;
        assert_eq!(status, want, "{name}: {body}");
    }
}

#[tokio::test]
async fn generate_sync_and_async() {
    let fake = Fake::new();
    let base = spawn(&fake).await;

    let r = post_json(
        &base,
        "/v1/audio/generate",
        serde_json::json!({"model": "music-test", "prompt": "lofi beat", "audio_length": 5.0,
                           "num_inference_steps": 4, "seed": 9}),
    )
    .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "audio/wav");
    let wave = turbospark_audio::wav::read_wav_f32_bytes(&r.bytes().await.unwrap()).unwrap();
    assert_eq!((wave.sample_rate, wave.channels), (44_100, 2));
    {
        let seen = fake.generated.lock().unwrap();
        assert_eq!(seen[0].caption, "lofi beat");
        // The runner rejects empty lyrics, so the route fills the instrumental marker.
        assert_eq!(seen[0].lyrics, "[instrumental]");
        assert_eq!(
            (seen[0].duration_seconds, seen[0].steps, seen[0].seed),
            (Some(5.0), Some(4), Some(9))
        );
    }

    let accepted: serde_json::Value = post_json(
        &base,
        "/v1/audio/generate",
        serde_json::json!({"model": "music-test", "prompt": "x", "lyrics": "la la", "async": true}),
    )
    .await
    .json()
    .await
    .unwrap();
    let id = accepted["job_id"].as_str().unwrap();
    let client = reqwest::Client::new();
    for _ in 0..200 {
        let job: serde_json::Value = client
            .get(format!("{base}/v1/audio/jobs/{id}"))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if job["status"] == "succeeded" {
            let r = client
                .get(format!("{base}/v1/audio/jobs/{id}/result"))
                .bearer_auth(KEY)
                .send()
                .await
                .unwrap();
            assert_eq!(r.headers()["content-type"], "audio/wav");
            assert!(turbospark_audio::wav::read_wav_f32_bytes(&r.bytes().await.unwrap()).is_ok());
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("generation job did not finish");
}

#[tokio::test]
async fn generate_validation() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let m = "music-test";
    let cases = [
        (
            "empty prompt",
            serde_json::json!({"model": m, "prompt": " "}),
        ),
        (
            "steps high",
            serde_json::json!({"model": m, "prompt": "x", "num_inference_steps": 31}),
        ),
        (
            "steps zero",
            serde_json::json!({"model": m, "prompt": "x", "num_inference_steps": 0}),
        ),
        (
            "length zero",
            serde_json::json!({"model": m, "prompt": "x", "audio_length": 0}),
        ),
        (
            "length high",
            serde_json::json!({"model": m, "prompt": "x", "audio_length": 361}),
        ),
        (
            "audio_start",
            serde_json::json!({"model": m, "prompt": "x", "audio_start": 1.0}),
        ),
        (
            "guidance",
            serde_json::json!({"model": m, "prompt": "x", "guidance_scale": 3.0}),
        ),
        (
            "mp3",
            serde_json::json!({"model": m, "prompt": "x", "response_format": "mp3"}),
        ),
        (
            "wrong task",
            serde_json::json!({"model": "stt-test", "prompt": "x"}),
        ),
    ];
    for (name, body) in cases {
        let (status, body) = error_of(post_json(&base, "/v1/audio/generate", body).await).await;
        assert_eq!(status, 400, "{name}: {body}");
    }
    let r = post_json(
        &base,
        "/v1/audio/generate",
        serde_json::json!({"model": "nope", "prompt": "x"}),
    )
    .await;
    assert_eq!(r.status(), 404);
    assert!(
        fake.generated.lock().unwrap().is_empty(),
        "a refused request reached the model"
    );
}

#[tokio::test]
async fn model_management_loads_aliases_and_unloads() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let client = reqwest::Client::new();

    let r = post_json(
        &base,
        "/v1/audio/models",
        serde_json::json!({"model_id": "installed-stt"}),
    )
    .await;
    assert_eq!(r.status(), 201);
    assert_eq!(
        r.json::<serde_json::Value>().await.unwrap()["id"],
        "installed-stt"
    );

    // A path is just an unknown alias to the provider; the real one never
    // opens it.
    let r = post_json(
        &base,
        "/v1/audio/models",
        serde_json::json!({"model_id": "/etc"}),
    )
    .await;
    assert_eq!(r.status(), 404);
    let r = post_json(
        &base,
        "/v1/audio/models",
        serde_json::json!({"model_id": ""}),
    )
    .await;
    assert_eq!(r.status(), 400);
    let r = post_json(
        &base,
        "/v1/audio/models",
        serde_json::json!({"model_id": "a", "dir": "/tmp"}),
    )
    .await;
    assert_eq!(r.status(), 400);

    let r = client
        .delete(format!("{base}/v1/audio/models/installed-stt"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = client
        .delete(format!("{base}/v1/audio/models/installed-stt"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn metrics_count_requests_by_endpoint_and_status() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let file = wav(16_000, 1, 0.1);
    post_stt(&base, &[("model", "stt-test")], Some(&file)).await;
    post_stt(&base, &[("model", "stt-test")], Some(&file)).await;
    post_stt(&base, &[("model", "stt-busy")], Some(&file)).await;
    post_stt(&base, &[("model", "nope")], Some(&file)).await;

    let r = reqwest::Client::new()
        .get(format!("{base}/v1/metrics"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/plain"));
    let text = r.text().await.unwrap();
    for want in [
        "turbospark_audio_requests_total{endpoint=\"transcriptions\",status=\"200\"} 2",
        "turbospark_audio_requests_total{endpoint=\"transcriptions\",status=\"429\"} 1",
        "turbospark_audio_requests_total{endpoint=\"transcriptions\",status=\"404\"} 1",
        "turbospark_audio_busy_total 1",
        "turbospark_audio_models_loaded 5",
        "turbospark_audio_jobs{status=\"running\"} 0",
    ] {
        assert!(text.contains(want), "missing {want:?} in\n{text}");
    }
}

mod realtime {
    use super::*;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::{Error, Message};

    type Socket = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    #[allow(clippy::result_large_err)]
    async fn connect(
        base: &str,
        query: &str,
        key: Option<&str>,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        Error,
    > {
        let url = format!(
            "{}/v1/audio/transcriptions/realtime?{query}",
            base.replace("http://", "ws://")
        );
        let mut request = url.into_client_request().unwrap();
        if let Some(key) = key {
            request
                .headers_mut()
                .insert("authorization", format!("Bearer {key}").parse().unwrap());
        }
        tokio_tungstenite::connect_async(request)
            .await
            .map(|(s, _)| s)
    }

    fn status(e: Error) -> u16 {
        match e {
            Error::Http(r) => r.status().as_u16(),
            other => panic!("expected an HTTP rejection, got {other:?}"),
        }
    }

    fn pcm16(samples: usize) -> Vec<u8> {
        (0..samples)
            .flat_map(|i| (((i % 100) as i16 - 50) * 100).to_le_bytes())
            .collect()
    }

    async fn json_reply(socket: &mut Socket) -> serde_json::Value {
        loop {
            match socket.next().await.unwrap().unwrap() {
                Message::Text(t) => return serde_json::from_str(&t).unwrap(),
                Message::Ping(_) | Message::Pong(_) => {}
                other => panic!("unexpected {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn commit_transcribes_everything_buffered_since_the_last_commit() {
        let fake = Fake::new();
        let base = spawn(&fake).await;
        let mut socket = connect(&base, "model=stt-test", Some(KEY)).await.unwrap();
        socket.send(Message::Binary(pcm16(8_000))).await.unwrap();
        socket.send(Message::Binary(pcm16(8_000))).await.unwrap();
        socket
            .send(Message::Text(r#"{"type":"commit"}"#.into()))
            .await
            .unwrap();
        let reply = json_reply(&mut socket).await;
        assert_eq!(reply["type"], "complete");
        assert_eq!(reply["segment_id"], 0);
        assert_eq!(reply["text"], "hello world");
        assert_eq!(fake.transcribed.lock().unwrap()[0].samples.len(), 16_000);

        // The buffer was consumed: a second commit has nothing to send.
        socket
            .send(Message::Text(r#"{"type":"commit"}"#.into()))
            .await
            .unwrap();
        assert_eq!(json_reply(&mut socket).await["type"], "error");

        socket.send(Message::Binary(pcm16(100))).await.unwrap();
        socket
            .send(Message::Text(r#"{"type":"commit"}"#.into()))
            .await
            .unwrap();
        let reply = json_reply(&mut socket).await;
        assert_eq!(reply["segment_id"], 1);

        socket.send(Message::Text("not json".into())).await.unwrap();
        assert_eq!(json_reply(&mut socket).await["type"], "error");
    }

    #[tokio::test]
    async fn f32le_encoding_is_accepted_and_odd_frames_close_the_socket() {
        let fake = Fake::new();
        let base = spawn(&fake).await;
        let mut socket = connect(&base, "model=stt-test&encoding=f32le", Some(KEY))
            .await
            .unwrap();
        let frame: Vec<u8> = (0..400).flat_map(|_| 0.25f32.to_le_bytes()).collect();
        socket.send(Message::Binary(frame)).await.unwrap();
        socket
            .send(Message::Text(r#"{"type":"commit"}"#.into()))
            .await
            .unwrap();
        assert_eq!(json_reply(&mut socket).await["type"], "complete");
        assert_eq!(fake.transcribed.lock().unwrap()[0].samples.len(), 400);

        socket.send(Message::Binary(vec![0, 0, 0])).await.unwrap();
        assert_eq!(json_reply(&mut socket).await["type"], "error");
    }

    #[tokio::test]
    async fn handshake_is_authenticated_and_validated() {
        let fake = Fake::new();
        let base = spawn(&fake).await;
        assert_eq!(
            status(connect(&base, "model=stt-test", None).await.unwrap_err()),
            401
        );
        assert_eq!(
            status(
                connect(&base, "model=stt-test", Some("wrong"))
                    .await
                    .unwrap_err()
            ),
            401
        );
        assert_eq!(
            status(connect(&base, "model=nope", Some(KEY)).await.unwrap_err()),
            404
        );
        assert_eq!(
            status(
                connect(&base, "model=tts-test", Some(KEY))
                    .await
                    .unwrap_err()
            ),
            400
        );
        assert_eq!(
            status(
                connect(&base, "model=stt-test&encoding=mp3", Some(KEY))
                    .await
                    .unwrap_err()
            ),
            400
        );
        assert_eq!(
            status(
                connect(&base, "model=stt-test&bogus=1", Some(KEY))
                    .await
                    .unwrap_err()
            ),
            400
        );
        assert_eq!(
            status(connect(&base, "", Some(KEY)).await.unwrap_err()),
            400
        );
    }
}

/// The OpenAPI file and the router must name the same routes. Without this,
/// the file is a hand-written claim that drifts the first time a route moves.
#[test]
fn openapi_file_lists_exactly_the_audio_routes() {
    let yaml = include_str!("../../../docs/openapi/audio.openapi.yaml");
    let re = regex_lite::Regex::new(r"(?m)^  (/v1/\S+):\s*$").unwrap();
    let mut documented: Vec<String> = re.captures_iter(yaml).map(|c| c[1].to_string()).collect();
    let mut routed: Vec<String> = turbospark_server::audio::AUDIO_ROUTE_PATHS
        .iter()
        .map(|p| p.to_string())
        .collect();
    documented.sort();
    routed.sort();
    assert_eq!(documented, routed);
}

#[tokio::test]
async fn every_listed_audio_route_is_actually_routed() {
    let fake = Fake::new();
    let base = spawn(&fake).await;
    let client = reqwest::Client::new();
    for path in turbospark_server::audio::AUDIO_ROUTE_PATHS {
        let url = format!("{base}{}", path.replace("{id}", "probe"));
        let r = client.get(&url).bearer_auth(KEY).send().await.unwrap();
        let status = r.status().as_u16();
        let body = r.bytes().await.unwrap();
        // An unrouted path is axum's empty 404; a routed one answers with a
        // JSON error, a body, or 405 for a POST-only route.
        assert!(!(status == 404 && body.is_empty()), "{path} is not routed");
    }
}
