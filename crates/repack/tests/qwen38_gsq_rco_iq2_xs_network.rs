//! Network gate for the immutable dense GSQ-RCO IQ2_XS artifact.
//!
//! ```sh
//! cargo test -p turbospark-repack --test qwen38_gsq_rco_iq2_xs_network --release -- --ignored --nocapture
//! ```

use turbospark_repack::{fetch_gguf_header, HttpRangeSource};

const REVISION: &str = "b71b542000a34ed0eb1e6a3a92f906e593728c65";
const URL: &str = "https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/b71b542000a34ed0eb1e6a3a92f906e593728c65/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf";
const BYTES: u64 = 8_422_841_472;
const SHA256: &str = "f0ae5006da0ce6225935339e4e989369f94de95d2263cf969519f8420c9ae02c";

fn header(name: &str) -> String {
    let response = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("client")
        .head(URL)
        .send()
        .expect("HEAD pinned IQ2_XS artifact");
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim_matches('"').to_string())
        .unwrap_or_else(|| panic!("HEAD {URL}: no {name} header"))
}

#[test]
#[ignore = "network: verifies pinned HTTP metadata and reads the 8.4 GB artifact header"]
fn pinned_artifact_metadata_and_dense_roles_are_unchanged() {
    assert_eq!(header("x-repo-commit"), REVISION);
    assert_eq!(header("x-linked-size").parse::<u64>().expect("size"), BYTES);
    assert_eq!(header("x-linked-etag"), SHA256);

    let source = HttpRangeSource::new(URL);
    let gguf = fetch_gguf_header(&source).expect("GGUF header");
    assert_eq!(gguf.architecture(), Some("qwen35"));
    let ty = |name: &str| gguf.tensors[name].ggml_type;
    assert_eq!(ty("token_embd.weight"), 29, "IQ1_M embedding");
    assert_eq!(ty("output.weight"), 23, "IQ4_XS output");
    assert_eq!(ty("blk.0.attn_qkv.weight"), 21, "IQ3_S attention");
    assert_eq!(ty("blk.45.ffn_down.weight"), 10, "Q2_K resident FFN");
}
