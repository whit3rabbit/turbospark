use super::*;
use crate::npz_fixture::{entry, npy, zip, Entry};

fn write(bytes: &[u8], tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbospark-npz-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("a.npz");
    std::fs::write(&p, bytes).unwrap();
    p
}

fn open_err(entries: &[Entry], tag: &str) -> String {
    match NpzArchive::open(&write(&zip(entries), tag)) {
        Ok(_) => panic!("{tag}: opened"),
        Err(e) => e.0,
    }
}

fn sample() -> Vec<Entry> {
    let codes: Vec<u8> = [1u32, 0xDEAD_BEEF, 7, 8, 9, 10]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    vec![
        entry("codes.npy", npy("<u4", "(2, 3)", &codes, false)),
        entry(
            "scales.npy",
            npy("<f2", "(2,)", &[0x00, 0x3C, 0x00, 0xC0], false),
        ),
        entry("signs.npy", npy("|i1", "(4,)", &[1, 0xFF, 1, 0xFF], false)),
        entry(
            "map.npy",
            npy("<i2", "(2, 2)", &[3, 0, 0, 0, 1, 0, 0, 0], false),
        ),
    ]
}

#[test]
fn crc32_matches_the_zip_check_value() {
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b""), 0);
}

#[test]
fn reads_savez_and_plain_layouts() {
    for zip64_local in [true, false] {
        let e: Vec<Entry> = sample()
            .into_iter()
            .map(|mut e| {
                e.force_zip64 = zip64_local;
                e
            })
            .collect();
        let mut a = NpzArchive::open(&write(&zip(&e), &format!("ok{zip64_local}"))).unwrap();
        assert_eq!(
            a.names().collect::<Vec<_>>(),
            ["codes", "map", "scales", "signs"]
        );
        let c = a.read("codes").unwrap();
        c.expect(NpyDtype::U32, &[2, 3]).unwrap();
        assert_eq!(c.to_u32().unwrap(), [1, 0xDEAD_BEEF, 7, 8, 9, 10]);
        assert_eq!(
            a.read("scales").unwrap().to_f16_bits().unwrap(),
            [0x3C00, 0xC000]
        );
        assert_eq!(a.read("signs").unwrap().to_i8().unwrap(), [1, -1, 1, -1]);
        assert_eq!(a.read("map").unwrap().to_i16().unwrap(), [3, 0, 1, 0]);
        assert!(
            a.read("codes").unwrap().to_f32().is_err(),
            "dtype is checked on read"
        );
        assert!(a.read("missing").is_err());
    }
}

#[test]
fn refuses_compressed_encrypted_and_descriptor_members() {
    let mut e = sample();
    e[1].method = 8;
    assert!(open_err(&e, "deflate").contains("compression method 8"));
    let mut e = sample();
    e[0].flags = 1;
    assert!(open_err(&e, "enc").contains("encrypted"));
    let mut e = sample();
    e[2].flags = 8;
    assert!(open_err(&e, "desc").contains("data-descriptor"));
}

#[test]
fn refuses_disagreeing_local_headers_and_duplicates() {
    let mut e = sample();
    e[0].local_name = Some("other.npy".into());
    assert!(open_err(&e, "lname").contains("name disagrees"));
    for zip64_local in [true, false] {
        let mut e = sample();
        e[3].force_zip64 = zip64_local;
        e[3].local_size = Some(e[3].data.len() as u64 + 1);
        assert!(open_err(&e, &format!("lsize{zip64_local}")).contains("sizes disagree"));
    }
    let mut e = sample();
    e[2].name = "codes.npy".into();
    e[2].local_name = Some("codes.npy".into());
    assert!(open_err(&e, "dup").contains("duplicate"));
    let mut e = sample();
    e[0].name = "../x.npy".into();
    e[0].local_name = Some("../x.npy".into());
    assert!(open_err(&e, "path").contains("refused"));
    let mut e = sample();
    e[0].name = "x.bin".into();
    e[0].local_name = Some("x.bin".into());
    assert!(open_err(&e, "ext").contains("not a .npy"));
}

#[test]
fn crc_mismatch_is_caught_on_read() {
    let mut e = sample();
    e[1].crc = Some(0x1234_5678);
    let mut a = NpzArchive::open(&write(&zip(&e), "crc")).unwrap();
    assert!(a.read("scales").unwrap_err().0.contains("CRC"));
    assert!(a.read("codes").is_ok());
}

#[test]
fn refuses_zip64_truncation_overlap_and_trailing_bytes() {
    let good = zip(&sample());
    // A zip64 locator right before the end record.
    let eocd = good.len() - 22;
    let mut z64 = good[..eocd].to_vec();
    z64.extend_from_slice(&0x0706_4B50u32.to_le_bytes());
    z64.extend_from_slice(&[0; 16]);
    z64.extend_from_slice(&good[eocd..]);
    // The central directory offset now points 20 bytes early of the record, so
    // either check may fire; both are refusals.
    assert!(NpzArchive::open(&write(&z64, "z64")).is_err());
    // Truncated: the end record is gone.
    assert!(NpzArchive::open(&write(&good[..good.len() - 5], "trunc")).is_err());
    // Bytes after the end record (not a comment).
    let mut tail = good.clone();
    tail.push(0);
    assert!(NpzArchive::open(&write(&tail, "tail")).is_err());
    // Two central entries pointing at the same local header.
    let mut ov = good.clone();
    let cd_off = u32_at(&ov, eocd + 16) as usize;
    let mut i = cd_off;
    for _ in 0..2 {
        let fl = u16_at(&ov, i + 28) as usize;
        if i != cd_off {
            ov[i + 42..i + 46].copy_from_slice(&0u32.to_le_bytes());
        }
        i += 46 + fl;
    }
    let msg = NpzArchive::open(&write(&ov, "overlap")).unwrap_err().0;
    assert!(msg.contains("disagree") || msg.contains("overlap"), "{msg}");
}

/// Member `b`'s central entry points at a valid copy of its local record that
/// sits inside member `a`'s data: every per-member check passes, so only the
/// overlap check can refuse it.
#[test]
fn refuses_a_member_hidden_inside_another() {
    let b = entry("b.npy", npy("|u1", "(3,)", &[1, 2, 3], false));
    let b_only = zip(std::slice::from_ref(&b));
    let local_b = &b_only[..30 + 5 + 20 + b.data.len()];
    let a = entry(
        "a.npy",
        npy("|u1", &format!("({},)", local_b.len()), local_b, false),
    );
    let mut z = zip(&[a, b]);
    let inner = z.windows(local_b.len()).position(|w| w == local_b).unwrap();
    assert!(inner > 0, "the copy inside a comes first");
    // Sanity: unpatched, the archive opens and b reads.
    assert!(NpzArchive::open(&write(&z, "hidden-ok"))
        .unwrap()
        .read("b")
        .is_ok());
    let eocd = z.len() - 22;
    let cd_off = u32_at(&z, eocd + 16) as usize;
    let second = cd_off + 46 + u16_at(&z, cd_off + 28) as usize;
    z[second + 42..second + 46].copy_from_slice(&(inner as u32).to_le_bytes());
    assert!(open_err_bytes(&z, "hidden").contains("overlaps"));
}

fn open_err_bytes(z: &[u8], tag: &str) -> String {
    NpzArchive::open(&write(z, tag)).unwrap_err().0
}

#[test]
fn npy_header_rules() {
    assert!(NpyArray::parse(&npy("<f4", "(1,)", &[0; 4], true))
        .unwrap_err()
        .0
        .contains("fortran"));
    assert!(NpyArray::parse(&npy(">f4", "(1,)", &[0; 4], false))
        .unwrap_err()
        .0
        .contains("refused"));
    assert!(NpyArray::parse(&npy("<f8", "(1,)", &[0; 8], false))
        .unwrap_err()
        .0
        .contains("refused"));
    assert!(NpyArray::parse(&npy("<f4", "(2,)", &[0; 4], false))
        .unwrap_err()
        .0
        .contains("data bytes"));
    let scalar = NpyArray::parse(&npy("<i4", "()", &7i32.to_le_bytes(), false)).unwrap();
    assert_eq!((scalar.shape.len(), scalar.to_i32().unwrap()), (0, vec![7]));
    let big = npy("<u4", "(4294967296, 4294967296, 4294967296)", &[], false);
    assert!(NpyArray::parse(&big).unwrap_err().0.contains("overflows"));
}

/// The real 5.4 GB release's small files, when present
/// (`~/models/lamq-probe/ink54`, see `docs/LAMQ.md`).
#[test]
#[ignore = "needs the LAMQ probe files"]
fn reads_the_real_release_luts_and_embed() {
    let dir = PathBuf::from(std::env::var("HOME").unwrap()).join("models/lamq-probe/ink54");
    let mut luts = NpzArchive::open(&dir.join("luts.npz")).unwrap();
    luts.read("ht2")
        .unwrap()
        .expect(NpyDtype::F16, &[512, 4])
        .unwrap();
    luts.read("ht3")
        .unwrap()
        .expect(NpyDtype::F16, &[1024, 2])
        .unwrap();
    for r in ["rtr1", "rtr2", "rtr3", "rtr4"] {
        luts.read(r)
            .unwrap()
            .expect(NpyDtype::F16, &[4096])
            .unwrap();
    }
    let mut e = NpzArchive::open(&dir.join("embed_packed.npz")).unwrap();
    assert_eq!(e.names().count(), 26);
    e.read("order")
        .unwrap()
        .expect(NpyDtype::I32, &[201_024])
        .unwrap();
    e.read("b0_codes")
        .unwrap()
        .expect(NpyDtype::U8, &[1_769_472])
        .unwrap();
    e.read("b7_codes")
        .unwrap()
        .expect(NpyDtype::U32, &[69_952, 72])
        .unwrap();
}
