//! Test-only builders for `.npz` archives (stored members, `np.savez`'s
//! local-header layout by default), shared by the npz and LAMQ intake tests.

use super::npz::crc32;

pub(crate) fn npy(descr: &str, shape: &str, data: &[u8], fortran: bool) -> Vec<u8> {
    let mut h = format!(
        "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
        if fortran { "True" } else { "False" }
    );
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(h.len() as u16).to_le_bytes());
    b.extend_from_slice(h.as_bytes());
    b.extend_from_slice(data);
    b
}

/// One member as the writer should emit it; fields can be corrupted per test.
#[derive(Clone)]
pub(crate) struct Entry {
    pub(crate) name: String,
    pub(crate) data: Vec<u8>,
    pub(crate) method: u16,
    pub(crate) flags: u16,
    /// `np.savez` layout: 0xFFFFFFFF local sizes plus a zip64 extra.
    pub(crate) force_zip64: bool,
    pub(crate) crc: Option<u32>,
    pub(crate) local_name: Option<String>,
    pub(crate) local_size: Option<u64>,
}

pub(crate) fn entry(name: &str, data: Vec<u8>) -> Entry {
    Entry {
        name: name.into(),
        data,
        method: 0,
        flags: 0,
        force_zip64: true,
        crc: None,
        local_name: None,
        local_size: None,
    }
}

pub(crate) fn zip(entries: &[Entry]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cd = Vec::new();
    for e in entries {
        let off = out.len() as u32;
        let crc = e.crc.unwrap_or_else(|| crc32(&e.data));
        let size = e.data.len() as u32;
        let lname = e.local_name.clone().unwrap_or_else(|| e.name.clone());
        let lsize = e.local_size.unwrap_or(u64::from(size));
        out.extend_from_slice(&0x0403_4B50u32.to_le_bytes());
        out.extend_from_slice(&45u16.to_le_bytes());
        out.extend_from_slice(&e.flags.to_le_bytes());
        out.extend_from_slice(&e.method.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&crc.to_le_bytes());
        let extra: Vec<u8> = if e.force_zip64 {
            out.extend_from_slice(&[0xFF; 8]);
            let mut x = vec![1, 0, 16, 0];
            x.extend_from_slice(&lsize.to_le_bytes());
            x.extend_from_slice(&lsize.to_le_bytes());
            x
        } else {
            out.extend_from_slice(&(lsize as u32).to_le_bytes());
            out.extend_from_slice(&(lsize as u32).to_le_bytes());
            Vec::new()
        };
        out.extend_from_slice(&(lname.len() as u16).to_le_bytes());
        out.extend_from_slice(&(extra.len() as u16).to_le_bytes());
        out.extend_from_slice(lname.as_bytes());
        out.extend_from_slice(&extra);
        out.extend_from_slice(&e.data);

        cd.extend_from_slice(&0x0201_4B50u32.to_le_bytes());
        cd.extend_from_slice(&813u16.to_le_bytes());
        cd.extend_from_slice(&45u16.to_le_bytes());
        cd.extend_from_slice(&e.flags.to_le_bytes());
        cd.extend_from_slice(&e.method.to_le_bytes());
        cd.extend_from_slice(&[0; 4]);
        cd.extend_from_slice(&crc.to_le_bytes());
        cd.extend_from_slice(&size.to_le_bytes());
        cd.extend_from_slice(&size.to_le_bytes());
        cd.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&[0; 8]); // extra, comment, disk, internal attrs
        cd.extend_from_slice(&[0; 4]); // external attrs
        cd.extend_from_slice(&off.to_le_bytes());
        cd.extend_from_slice(e.name.as_bytes());
    }
    let cd_off = out.len() as u32;
    out.extend_from_slice(&cd);
    out.extend_from_slice(&0x0605_4B50u32.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(cd.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&[0; 2]);
    out
}
