use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use turbospark_catalog::{
    Catalog, Client, HubClient, HubGgufVariant, Machine, ProbeReport, QuantLabel, RepoFile,
    RepoRef, ShardSetIssue, SourceKind, VariantInstallability, Verdict,
};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());

enum FixtureResponse {
    Json(String),
    Head(Option<u64>),
}

fn fixture_server(responses: Vec<FixtureResponse>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    let handle = thread::spawn(move || {
        responses
            .into_iter()
            .map(|response| {
                let (mut stream, _) = listener.accept().expect("accept hub request");
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .expect("set request timeout");
                let mut request = String::new();
                BufReader::new(stream.try_clone().expect("clone fixture stream"))
                    .read_line(&mut request)
                    .expect("read request line");
                let wire = match response {
                    FixtureResponse::Json(body) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    ),
                    FixtureResponse::Head(Some(size)) => format!(
                        "HTTP/1.1 200 OK\r\nx-linked-size: {size}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    ),
                    FixtureResponse::Head(None) => {
                        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_string()
                    }
                };
                stream.write_all(wire.as_bytes()).expect("write response");
                request.trim_end().to_string()
            })
            .collect()
    });
    (format!("http://{address}"), handle)
}

struct EndpointReset;

impl Drop for EndpointReset {
    fn drop(&mut self) {
        turbospark_catalog::set_hf_endpoint_override(None);
    }
}

fn client_for(endpoint: String) -> (EndpointReset, Client) {
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    (EndpointReset, Client::with_timeout(Duration::from_secs(2)))
}

fn hub_for(client: &Client) -> HubClient<'_> {
    HubClient::new(
        client,
        Catalog::embedded().expect("embedded catalog"),
        Machine {
            physical_bytes: 32 * 1024 * 1024 * 1024,
            ..Machine::default()
        },
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
}

fn metadata(files: Vec<serde_json::Value>) -> String {
    serde_json::json!({
        "id": "owner/model",
        "sha": REVISION,
        "siblings": files,
    })
    .to_string()
}

fn listed_file(name: &str, size: u64) -> serde_json::Value {
    serde_json::json!({ "rfilename": name, "size": size })
}

fn unlisted_file(name: &str) -> serde_json::Value {
    serde_json::json!({ "rfilename": name })
}

fn report(
    repo: RepoRef,
    file: &str,
    verdict: Verdict,
    arch: Option<model_io::ArchConfig>,
) -> ProbeReport {
    let expert_stride = arch
        .as_ref()
        .filter(|arch| arch.num_experts > 0)
        .map(|_| 1_024);
    ProbeReport {
        repo,
        kind: SourceKind::Gguf,
        file: Some(file.to_string()),
        download_bytes: None,
        architecture: Some("llama".to_string()),
        family: Some(model_io::ModelFamily::Llama),
        arch,
        types: Vec::new(),
        affine: None,
        expert_stride,
        trained_context: None,
        sidecars_present: Vec::new(),
        sidecars_missing: Vec::new(),
        chat_template: None,
        verdict,
        warnings: Vec::new(),
    }
}

fn variant<'a>(variants: &'a [HubGgufVariant], label: &str) -> &'a HubGgufVariant {
    variants
        .iter()
        .find(|variant| variant.label == QuantLabel(label.to_string()))
        .expect("variant label")
}

#[test]
fn uses_listed_sizes_and_heads_only_files_missing_a_size() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let first = "model-Q4_K_M-00001-of-00002.gguf";
    let second = "model-Q4_K_M-00002-of-00002.gguf";
    let (endpoint, server) = fixture_server(vec![
        FixtureResponse::Json(metadata(vec![
            listed_file(first, 10),
            unlisted_file(second),
        ])),
        FixtureResponse::Head(Some(12)),
    ]);
    let (_reset, client) = client_for(endpoint);
    let hub = hub_for(&client);

    let variants = hub
        .repo_variants(&RepoRef::new("owner/model", REVISION))
        .expect("variants");

    assert_eq!(variants.len(), 1);
    let q4 = variant(&variants, "Q4_K_M");
    assert_eq!(q4.total_bytes, Some(22));
    assert_eq!(
        q4.files,
        vec![
            RepoFile {
                name: first.to_string(),
                size: Some(10)
            },
            RepoFile {
                name: second.to_string(),
                size: Some(12)
            },
        ]
    );
    assert_eq!(q4.fit.verdict, "unknown (probe it)");
    assert_eq!(q4.fit.counted_source, "unknown");
    assert!(matches!(
        q4.installability,
        VariantInstallability::SupportUnverified { .. }
    ));
    let requests = server.join().expect("fixture server joins");
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with("GET /api/models/owner/model/revision/"));
    assert!(requests[1].starts_with("HEAD /owner/model/resolve/"));
    assert!(requests[1]
        .split_whitespace()
        .nth(1)
        .is_some_and(|target| target.ends_with(second)));
}

#[test]
fn a_missing_content_length_keeps_the_variant_unknown_size() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let file = "model-Q4_K_M.gguf";
    let (endpoint, server) = fixture_server(vec![
        FixtureResponse::Json(metadata(vec![unlisted_file(file)])),
        FixtureResponse::Head(None),
    ]);
    let (_reset, client) = client_for(endpoint);
    let hub = hub_for(&client);

    let variants = hub
        .repo_variants(&RepoRef::new("owner/model", REVISION))
        .expect("variants");

    let q4 = variant(&variants, "Q4_K_M");
    assert_eq!(q4.total_bytes, None);
    assert_eq!(q4.fit.verdict, "unknown (probe it)");
    assert_eq!(
        q4.installability,
        VariantInstallability::UnknownSize {
            files: vec![file.to_string()]
        }
    );
    assert_eq!(server.join().expect("fixture server joins").len(), 2);
}

#[test]
fn incomplete_and_inconsistent_groups_have_specific_disabled_reasons() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, server) = fixture_server(vec![FixtureResponse::Json(metadata(vec![
        listed_file("alpha-Q4_K_M-00001-of-00002.gguf", 10),
        listed_file("beta-Q4_K_M-00001-of-00002.gguf", 10),
        listed_file("beta-Q4_K_M-00001-of-00003.gguf", 11),
    ]))]);
    let (_reset, client) = client_for(endpoint);
    let hub = hub_for(&client);

    let variants = hub
        .repo_variants(&RepoRef::new("owner/model", REVISION))
        .expect("variants");

    let incomplete = variants
        .iter()
        .find(|variant| variant.files[0].name.starts_with("alpha-"))
        .expect("incomplete group");
    assert_eq!(incomplete.total_bytes, None);
    assert_eq!(
        incomplete.installability,
        VariantInstallability::IncompleteShardSet {
            files: vec!["alpha-Q4_K_M-00001-of-00002.gguf".to_string()],
            expected_count: 2,
            present_indices: vec![1],
        }
    );
    let inconsistent = variants
        .iter()
        .find(|variant| variant.files[0].name.starts_with("beta-"))
        .expect("inconsistent group");
    assert_eq!(inconsistent.total_bytes, None);
    assert!(matches!(
        &inconsistent.installability,
        VariantInstallability::InconsistentShardSet { issues, .. }
            if issues.contains(&ShardSetIssue::DuplicateIndex { index: 1 })
                && issues.contains(&ShardSetIssue::ConflictingCounts {
                    declared_counts: vec![2, 3]
                })
    ));
    assert_eq!(server.join().expect("fixture server joins").len(), 1);
}

#[test]
fn a_matching_probe_binds_to_the_complete_file_group_and_labels_fit_as_estimated() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let first = "model-Q4_K_M-00001-of-00002.gguf";
    let second = "model-Q4_K_M-00002-of-00002.gguf";
    let (endpoint, server) = fixture_server(vec![FixtureResponse::Json(metadata(vec![
        listed_file(first, 10),
        listed_file(second, 12),
        listed_file("other-Q5_K_M.gguf", 8),
    ]))]);
    let (_reset, client) = client_for(endpoint);
    let hub = hub_for(&client);
    let probe = report(
        RepoRef::new("owner/model", REVISION),
        first,
        Verdict::Runnable,
        Some(model_io::known_architecture(model_io::ModelFamily::Llama)),
    );

    let variants = hub
        .repo_variants_with_probe(&RepoRef::new("owner/model", REVISION), &probe)
        .expect("variants");

    let q4 = variant(&variants, "Q4_K_M");
    assert_eq!(q4.total_bytes, Some(22));
    assert_eq!(q4.files.len(), 2);
    assert_eq!(q4.installability, VariantInstallability::Ready);
    assert!(q4.fit.is_estimate);
    assert_eq!(q4.fit.counted_source, "estimated");
    assert_ne!(q4.fit.verdict, "unknown (probe it)");
    let q5 = variant(&variants, "Q5_K_M");
    assert!(matches!(
        q5.installability,
        VariantInstallability::SupportUnverified { .. }
    ));
    assert_eq!(q5.fit.verdict, "unknown (probe it)");
    assert_eq!(server.join().expect("fixture server joins").len(), 1);
}

#[test]
fn a_matching_probe_does_not_apply_to_another_group_with_the_same_quant_label() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let alpha_file = "alpha-Q4_K_M.gguf";
    let beta_file = "beta-Q4_K_M.gguf";
    let (endpoint, server) = fixture_server(vec![FixtureResponse::Json(metadata(vec![
        listed_file(alpha_file, 10),
        listed_file(beta_file, 12),
    ]))]);
    let (_reset, client) = client_for(endpoint);
    let hub = hub_for(&client);
    let probe = report(
        RepoRef::new("owner/model", REVISION),
        alpha_file,
        Verdict::Runnable,
        Some(model_io::known_architecture(model_io::ModelFamily::Llama)),
    );

    let variants = hub
        .repo_variants_with_probe(&RepoRef::new("owner/model", REVISION), &probe)
        .expect("variants");

    assert_eq!(variants.len(), 2);
    let alpha = variants
        .iter()
        .find(|variant| variant.files[0].name == alpha_file)
        .expect("alpha group");
    assert_eq!(alpha.label, QuantLabel("Q4_K_M".to_string()));
    assert_eq!(alpha.installability, VariantInstallability::Ready);
    assert!(alpha.fit.is_estimate);

    let beta = variants
        .iter()
        .find(|variant| variant.files[0].name == beta_file)
        .expect("beta group");
    assert_eq!(beta.label, QuantLabel("Q4_K_M".to_string()));
    assert!(matches!(
        beta.installability,
        VariantInstallability::SupportUnverified { .. }
    ));
    assert_eq!(beta.fit.verdict, "unknown (probe it)");
    assert_eq!(server.join().expect("fixture server joins").len(), 1);
}

#[test]
fn mismatched_revision_or_floating_probe_cannot_clear_support_unverified() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let other_revision = "fedcba9876543210fedcba9876543210fedcba98";
    let (endpoint, server) = fixture_server(vec![
        FixtureResponse::Json(metadata(vec![listed_file("model-Q4_K_M.gguf", 10)])),
        FixtureResponse::Json(metadata(vec![listed_file("model-Q4_K_M.gguf", 10)])),
    ]);
    let (_reset, client) = client_for(endpoint);
    let hub = hub_for(&client);
    let wrong_revision = report(
        RepoRef::new("owner/model", other_revision),
        "model-Q4_K_M.gguf",
        Verdict::Runnable,
        Some(model_io::known_architecture(model_io::ModelFamily::Llama)),
    );
    let floating = report(
        RepoRef::new("owner/model", "main"),
        "model-Q4_K_M.gguf",
        Verdict::Runnable,
        Some(model_io::known_architecture(model_io::ModelFamily::Llama)),
    );

    for (repo, probe) in [
        (RepoRef::new("owner/model", REVISION), &wrong_revision),
        (RepoRef::new("owner/model", "main"), &floating),
    ] {
        let variants = hub
            .repo_variants_with_probe(&repo, probe)
            .expect("variants");
        let q4 = variant(&variants, "Q4_K_M");
        assert_eq!(q4.repo, RepoRef::new("owner/model", REVISION));
        assert!(matches!(
            q4.installability,
            VariantInstallability::SupportUnverified { .. }
        ));
        assert_eq!(q4.fit.verdict, "unknown (probe it)");
    }
    assert_eq!(server.join().expect("fixture server joins").len(), 2);
}

#[test]
fn non_gguf_probe_cannot_clear_support_unverified() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let file = "model-Q4_K_M.gguf";
    let (endpoint, server) =
        fixture_server(vec![FixtureResponse::Json(metadata(vec![listed_file(
            file, 10,
        )]))]);
    let (_reset, client) = client_for(endpoint);
    let hub = hub_for(&client);
    let mut probe = report(
        RepoRef::new("owner/model", REVISION),
        file,
        Verdict::Runnable,
        Some(model_io::known_architecture(model_io::ModelFamily::Llama)),
    );
    probe.kind = SourceKind::Mlx;

    let variants = hub
        .repo_variants_with_probe(&RepoRef::new("owner/model", REVISION), &probe)
        .expect("variants");

    let q4 = variant(&variants, "Q4_K_M");
    assert!(matches!(
        q4.installability,
        VariantInstallability::SupportUnverified { .. }
    ));
    assert_eq!(q4.fit.verdict, "unknown (probe it)");
    assert_eq!(server.join().expect("fixture server joins").len(), 1);
}

#[test]
fn a_probe_refusal_disables_the_variant_with_its_reason() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let file = "model-Q5_0.gguf";
    let (endpoint, server) =
        fixture_server(vec![FixtureResponse::Json(metadata(vec![listed_file(
            file, 10,
        )]))]);
    let (_reset, client) = client_for(endpoint);
    let hub = hub_for(&client);
    let probe = report(
        RepoRef::new("owner/model", REVISION),
        file,
        Verdict::Refused("unsupported quantization: no kernels for q5_0".to_string()),
        None,
    );

    let variants = hub
        .repo_variants_with_probe(&RepoRef::new("owner/model", REVISION), &probe)
        .expect("variants");

    let q5 = variant(&variants, "Q5_0");
    assert_eq!(
        q5.installability,
        VariantInstallability::Unsupported {
            files: vec![file.to_string()],
            reason: "unsupported quantization: no kernels for q5_0".to_string(),
        }
    );
    assert_eq!(q5.fit.verdict, "unknown (probe it)");
    assert_eq!(server.join().expect("fixture server joins").len(), 1);
}
