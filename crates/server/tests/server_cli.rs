use std::process::Command;

#[test]
fn tailnet_bind_without_api_key_exits_with_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-server"))
        .args(["--model", "/nonexistent", "--bind", "tailnet"])
        .env_remove("TURBOSPARK_API_KEY")
        .output()
        .expect("turbospark-server should execute");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--bind tailnet requires --api-key KEY or a non-empty TURBOSPARK_API_KEY"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn tailnet_bind_with_empty_env_key_exits_with_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-server"))
        .args(["--model", "/nonexistent", "--bind", "tailnet"])
        .env("TURBOSPARK_API_KEY", "")
        .output()
        .expect("turbospark-server should execute");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--bind tailnet requires --api-key KEY or a non-empty TURBOSPARK_API_KEY"),
        "unexpected stderr: {stderr}"
    );
}

/// A home whose library T7 is listed but unplugged, with a text install and
/// a music install recorded in it, removed when the test ends. Debug builds
/// take the volumes folder and mount table from the `TURBOSPARK_TEST_*`
/// overrides; release builds ignore them by design.
#[cfg(all(target_os = "macos", debug_assertions))]
struct UnpluggedLibrary {
    base: std::path::PathBuf,
}

#[cfg(all(target_os = "macos", debug_assertions))]
impl UnpluggedLibrary {
    fn new(tag: &str) -> Self {
        let base = std::env::temp_dir().join(format!(
            "turbospark-server-unplugged-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("home")).unwrap();
        std::fs::create_dir_all(base.join("Volumes")).unwrap();
        // Canonical, because the temp dir is behind the /var symlink and the
        // location rules compare canonical paths.
        let base = base.canonicalize().unwrap();
        std::fs::write(
            base.join("mounts.txt"),
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n",
        )
        .unwrap();
        let id = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";
        let root = base.join("Volumes").join("T7").join("TurboSpark");
        let list = serde_json::json!({
            "format": 1,
            "internal_is_default": true,
            "libraries": [{
                "id": id,
                "name": "T7",
                "volume_relative_path": "TurboSpark",
                "last_mount_path": base.join("Volumes").join("T7"),
                "default_for_downloads": false,
                "added_on": "2026-10-15"
            }],
        });
        std::fs::write(
            base.join("home").join("libraries.json"),
            serde_json::to_vec(&list).unwrap(),
        )
        .unwrap();
        let row = |alias: &str, modality: &str, relative: &str| {
            serde_json::json!({
                "alias": alias,
                "repo": format!("owner/{alias}"),
                "revision": "main",
                "path": root.join(relative),
                "family": "qwen3",
                "install_bytes": 4,
                "installed_on": "2026-10-15",
                "status": "runs",
                "kind": null,
                "modality": modality,
                "library": id,
                "relative_path": relative,
            })
        };
        let rows = [
            row("qq-away", "text", "text/qq-away.gturbo"),
            row("qq-song", "audio", "audio/qq-song.gturbo"),
        ];
        std::fs::write(
            base.join("home").join("installed.json"),
            serde_json::to_vec_pretty(&rows).unwrap(),
        )
        .unwrap();
        Self { base }
    }

    fn serve(&self, args: &[&str]) -> std::process::Output {
        use std::os::unix::fs::MetadataExt;
        // A home device unlike the temp folder's, as on a Mac whose external
        // drives are other disks.
        let device = std::fs::metadata(&self.base)
            .unwrap()
            .dev()
            .wrapping_add(1);
        Command::new(env!("CARGO_BIN_EXE_turbospark-server"))
            .args(args)
            .env("TURBOSPARK_HOME", self.base.join("home"))
            .env("TURBOSPARK_TEST_MOUNT_TABLE", self.base.join("mounts.txt"))
            .env("TURBOSPARK_TEST_VOLUMES_ROOT", self.base.join("Volumes"))
            .env("TURBOSPARK_TEST_HOME_DEVICE", device.to_string())
            .env_remove("TURBOSPARK_API_KEY")
            .output()
            .expect("turbospark-server should execute")
    }
}

#[cfg(all(target_os = "macos", debug_assertions))]
impl Drop for UnpluggedLibrary {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

/// Serving a model whose library is not connected exits naming the drive to
/// connect (10.7), for `--model` and for an audio model, before anything is
/// opened or bound.
#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn serving_a_model_on_an_unplugged_library_names_the_drive_to_connect() {
    let home = UnpluggedLibrary::new("serve");
    for (args, alias) in [
        (&["--model", "qq-away", "--port", "0"][..], "qq-away"),
        (&["--music-model", "qq-song", "--port", "0"][..], "qq-song"),
    ] {
        let output = home.serve(args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{args:?}: {stderr}");
        let expected = format!(
            "Model {alias} is on T7, which is not connected. Connect the drive and try again."
        );
        assert!(stderr.contains(&expected), "{args:?}: {stderr}");
    }
}
