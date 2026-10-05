use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use turbospark_catalog::{
    Catalog, Client, HubClient, Machine, PinnedSourceFile, PinnedSourceGroup, RepoRef,
    ResolvedSourceFile, ResolvedSourceIdentity, SourceIdentityMismatch,
};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const SHA_ONE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_TWO: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
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
                let mut reader = BufReader::new(stream.try_clone().expect("clone fixture stream"));
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).expect("read request header");
                    if line.is_empty() {
                        break;
                    }
                    let done = line == "\r\n" || line == "\n";
                    request.push_str(&line);
                    if done {
                        break;
                    }
                }
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
                request
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

fn set_endpoint(endpoint: String) -> EndpointReset {
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    EndpointReset
}

fn hub_for(client: &Client) -> HubClient<'_> {
    HubClient::new(
        client,
        Catalog::embedded().expect("embedded catalog"),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
}

fn source_group(repo: &str, revision: &str, files: Vec<PinnedSourceFile>) -> PinnedSourceGroup {
    PinnedSourceGroup {
        role: "opaque-adapter-role".to_string(),
        repo: RepoRef::new(repo, revision),
        files,
    }
}

fn source_file(path: &str, size: Option<u64>, sha256: Option<&str>) -> PinnedSourceFile {
    PinnedSourceFile {
        path: path.to_string(),
        expected_size: size,
        expected_sha256: sha256.map(str::to_string),
    }
}

fn listed_file(path: &str, size: Option<u64>, sha256: Option<&str>) -> serde_json::Value {
    let mut file = serde_json::json!({ "rfilename": path });
    if let Some(size) = size {
        file["size"] = serde_json::json!(size);
    }
    if let Some(sha256) = sha256 {
        file["lfs"] = serde_json::json!({ "sha256": sha256 });
    }
    file
}

fn listed_file_with_oid(path: &str, size: Option<u64>, sha256: &str) -> serde_json::Value {
    let mut file = serde_json::json!({
        "rfilename": path,
        "lfs": { "oid": format!("sha256:{sha256}") }
    });
    if let Some(size) = size {
        file["size"] = serde_json::json!(size);
    }
    file
}

fn metadata(repo: &str, revision: &str, files: Vec<serde_json::Value>) -> String {
    serde_json::json!({
        "id": repo,
        "sha": revision,
        "siblings": files,
    })
    .to_string()
}

#[test]
fn resolves_exact_owner_source_to_canonical_revision_and_authoritative_files() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (_reset, fixture) = {
        let first = "adapter/model.safetensors";
        let second = "adapter/config file.json";
        let body = metadata(
            "owner/model",
            &REVISION.to_ascii_lowercase(),
            vec![
                listed_file_with_oid(second, None, SHA_TWO),
                listed_file(first, Some(8), Some(SHA_ONE)),
                listed_file("unowned/sidecar.txt", Some(2), None),
            ],
        );
        let (endpoint, server) = fixture_server(vec![
            FixtureResponse::Json(body),
            FixtureResponse::Head(Some(12)),
        ]);
        (set_endpoint(endpoint), server)
    };
    let client = Client::with_token(Some("hf_source_fixture".to_string()));
    let hub = hub_for(&client);
    let expected = source_group(
        "OWNER/Model",
        &REVISION.to_ascii_uppercase(),
        vec![
            source_file(
                "adapter/model.safetensors",
                Some(8),
                Some(&SHA_ONE.to_ascii_uppercase()),
            ),
            source_file("adapter/config file.json", Some(12), None),
        ],
    );

    let resolved = hub
        .resolve_source_identity(&expected)
        .expect("exact source resolves");

    assert_eq!(resolved.repo, RepoRef::new("owner/model", REVISION));
    assert_eq!(
        resolved.files,
        vec![
            ResolvedSourceFile {
                path: "adapter/model.safetensors".to_string(),
                authoritative_size: Some(8),
                source_sha256: Some(SHA_ONE.to_string()),
            },
            ResolvedSourceFile {
                path: "adapter/config file.json".to_string(),
                authoritative_size: Some(12),
                source_sha256: Some(SHA_TWO.to_string()),
            },
        ]
    );
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &resolved),
        Ok(())
    );
    let requests = fixture.join().expect("fixture server joins");
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with("GET /api/models/OWNER/Model/revision/"));
    assert!(requests[1].starts_with("HEAD /OWNER/Model/resolve/"));
    assert!(requests[1].contains("adapter/config%20file.json"));
}

#[test]
fn exact_match_rejects_repository_revision_and_complete_file_set_mismatches() {
    let expected = source_group(
        "owner/model",
        REVISION,
        vec![source_file(
            "weights/model.safetensors",
            Some(8),
            Some(SHA_ONE),
        )],
    );
    let candidate = ResolvedSourceIdentity {
        repo: RepoRef::new("OWNER/MODEL", REVISION.to_ascii_uppercase()),
        files: vec![ResolvedSourceFile {
            path: "weights/model.safetensors".to_string(),
            authoritative_size: Some(8),
            source_sha256: Some(SHA_ONE.to_string()),
        }],
    };

    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &candidate),
        Ok(())
    );
    let wrong_repo = ResolvedSourceIdentity {
        repo: RepoRef::new("owner/other", REVISION),
        ..candidate.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &wrong_repo),
        Err(SourceIdentityMismatch::Repository)
    );
    let wrong_revision = ResolvedSourceIdentity {
        repo: RepoRef::new("owner/model", "f".repeat(40)),
        ..candidate.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &wrong_revision),
        Err(SourceIdentityMismatch::Revision)
    );
    let missing_file = ResolvedSourceIdentity {
        files: Vec::new(),
        ..candidate.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &missing_file),
        Err(SourceIdentityMismatch::FileSet)
    );
    let extra_file = ResolvedSourceIdentity {
        files: vec![
            candidate.files[0].clone(),
            ResolvedSourceFile {
                path: "weights/extra.safetensors".to_string(),
                authoritative_size: Some(3),
                source_sha256: None,
            },
        ],
        ..candidate.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &extra_file),
        Err(SourceIdentityMismatch::FileSet)
    );
    let case_changed_path = ResolvedSourceIdentity {
        files: vec![ResolvedSourceFile {
            path: "weights/Model.safetensors".to_string(),
            ..candidate.files[0].clone()
        }],
        ..candidate.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &case_changed_path),
        Err(SourceIdentityMismatch::FileSet)
    );
    let duplicate_file = ResolvedSourceIdentity {
        files: vec![candidate.files[0].clone(), candidate.files[0].clone()],
        ..candidate.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &duplicate_file),
        Err(SourceIdentityMismatch::DuplicatePath)
    );
}

#[test]
fn exact_match_requires_each_authoritative_pin_without_reordering_paths() {
    let expected = source_group(
        "owner/model",
        REVISION,
        vec![
            source_file("a.bin", Some(8), Some(SHA_ONE)),
            source_file("b.bin", Some(12), Some(SHA_TWO)),
        ],
    );
    let exact_other_order = ResolvedSourceIdentity {
        repo: RepoRef::new("owner/model", REVISION),
        files: vec![
            ResolvedSourceFile {
                path: "b.bin".to_string(),
                authoritative_size: Some(12),
                source_sha256: Some(SHA_TWO.to_string()),
            },
            ResolvedSourceFile {
                path: "a.bin".to_string(),
                authoritative_size: Some(8),
                source_sha256: Some(SHA_ONE.to_string()),
            },
        ],
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &exact_other_order),
        Ok(())
    );
    let missing_size = ResolvedSourceIdentity {
        files: vec![
            ResolvedSourceFile {
                authoritative_size: None,
                ..exact_other_order.files[1].clone()
            },
            exact_other_order.files[0].clone(),
        ],
        ..exact_other_order.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &missing_size),
        Err(SourceIdentityMismatch::MissingAuthoritativeSize)
    );
    let wrong_size = ResolvedSourceIdentity {
        files: vec![
            ResolvedSourceFile {
                authoritative_size: Some(9),
                ..exact_other_order.files[1].clone()
            },
            exact_other_order.files[0].clone(),
        ],
        ..exact_other_order.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &wrong_size),
        Err(SourceIdentityMismatch::Size)
    );
    let missing_digest = ResolvedSourceIdentity {
        files: vec![
            ResolvedSourceFile {
                source_sha256: None,
                ..exact_other_order.files[1].clone()
            },
            exact_other_order.files[0].clone(),
        ],
        ..exact_other_order.clone()
    };
    assert_eq!(
        turbospark_catalog::matches_exact_source(&expected, &missing_digest),
        Err(SourceIdentityMismatch::MissingPinnedDigest)
    );
}

#[test]
fn source_resolution_rejects_mutated_digest_and_unavailable_size() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let body = metadata(
        "owner/model",
        REVISION,
        vec![listed_file(
            "weights/model.safetensors",
            Some(8),
            Some(SHA_TWO),
        )],
    );
    let (endpoint, server) = fixture_server(vec![
        FixtureResponse::Json(body.clone()),
        FixtureResponse::Json(body),
        FixtureResponse::Json(metadata(
            "owner/model",
            REVISION,
            vec![listed_file("weights/model.safetensors", None, None)],
        )),
        FixtureResponse::Head(None),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_source_fixture".to_string()));
    let hub = hub_for(&client);

    let digest_pin = source_group(
        "owner/model",
        REVISION,
        vec![source_file(
            "weights/model.safetensors",
            Some(8),
            Some(SHA_ONE),
        )],
    );
    let digest_error = hub
        .resolve_source_identity(&digest_pin)
        .expect_err("different live digest must fail closed");
    assert!(matches!(
        digest_error,
        turbospark_catalog::HubError::InvalidResponse { .. }
    ));

    let size_pin = source_group(
        "owner/model",
        REVISION,
        vec![source_file(
            "weights/model.safetensors",
            Some(9),
            Some(SHA_TWO),
        )],
    );
    let size_error = hub
        .resolve_source_identity(&size_pin)
        .expect_err("different live size must fail closed");
    assert!(matches!(
        size_error,
        turbospark_catalog::HubError::InvalidResponse { .. }
    ));

    let no_size_pin = source_group(
        "owner/model",
        REVISION,
        vec![source_file("weights/model.safetensors", None, None)],
    );
    let unavailable_error = hub
        .resolve_source_identity(&no_size_pin)
        .expect_err("missing authoritative size must fail closed");
    assert!(matches!(
        unavailable_error,
        turbospark_catalog::HubError::InvalidResponse { .. }
    ));
    assert_eq!(server.join().expect("fixture server joins").len(), 4);
}

#[test]
fn source_resolution_rejects_missing_pinned_digest_and_wrong_live_identity() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let path = "weights/model.safetensors";
    let requested = source_group(
        "owner/model",
        REVISION,
        vec![source_file(path, None, Some(SHA_ONE))],
    );
    let (endpoint, server) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/model",
            REVISION,
            vec![listed_file(path, Some(16 * 1024 * 1024 + 1), None)],
        )),
        FixtureResponse::Json(metadata(
            "owner/other",
            REVISION,
            vec![listed_file(path, Some(8), Some(SHA_ONE))],
        )),
        FixtureResponse::Json(metadata(
            "owner/model",
            &"f".repeat(40),
            vec![listed_file(path, Some(8), Some(SHA_ONE))],
        )),
        FixtureResponse::Json(metadata(
            "owner/model",
            REVISION,
            vec![serde_json::json!({
                "rfilename": path,
                "size": 8,
                "lfs": { "sha256": "not-a-sha256" },
            })],
        )),
        FixtureResponse::Json(metadata(
            "owner/model",
            REVISION,
            vec![listed_file(
                "weights/other.safetensors",
                Some(8),
                Some(SHA_ONE),
            )],
        )),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_source_fixture".to_string()));
    let hub = hub_for(&client);

    for _ in 0..5 {
        assert!(matches!(
            hub.resolve_source_identity(&requested),
            Err(turbospark_catalog::HubError::InvalidResponse { .. })
        ));
    }
    assert_eq!(server.join().expect("fixture server joins").len(), 5);
}

#[test]
fn source_resolution_rejects_unpinned_revisions_unsafe_paths_and_duplicate_paths() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, server) = fixture_server(Vec::new());
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_source_fixture".to_string()));
    let hub = hub_for(&client);

    let floating = source_group(
        "owner/model",
        "main",
        vec![source_file("weights/model.safetensors", None, None)],
    );
    assert_eq!(
        hub.resolve_source_identity(&floating),
        Err(turbospark_catalog::HubError::InvalidRequest {
            field: "revision",
            rule: "immutable_commit_revision",
        })
    );
    let unsafe_path = source_group(
        "owner/model",
        REVISION,
        vec![source_file("../model.safetensors", None, None)],
    );
    assert_eq!(
        hub.resolve_source_identity(&unsafe_path),
        Err(turbospark_catalog::HubError::InvalidRequest {
            field: "path",
            rule: "safe_relative_path",
        })
    );
    let duplicate_path = source_group(
        "owner/model",
        REVISION,
        vec![
            source_file("weights/model.safetensors", None, None),
            source_file("weights/model.safetensors", None, None),
        ],
    );
    assert_eq!(
        hub.resolve_source_identity(&duplicate_path),
        Err(turbospark_catalog::HubError::InvalidRequest {
            field: "path",
            rule: "unique_file_path",
        })
    );
    let invalid_digest = source_group(
        "owner/model",
        REVISION,
        vec![source_file(
            "weights/model.safetensors",
            None,
            Some("not-a-sha256"),
        )],
    );
    assert_eq!(
        hub.resolve_source_identity(&invalid_digest),
        Err(turbospark_catalog::HubError::InvalidRequest {
            field: "expected_sha256",
            rule: "sha256_digest_shape",
        })
    );
    assert!(server.join().expect("fixture server joins").is_empty());
}
