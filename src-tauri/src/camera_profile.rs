//! DNG camera profiles (.dcp): the per-camera colour and look that Lightroom/ACR,
//! RawTherapee and ART give a RAW before any edit.
//!
//! A profile does two jobs, which RapidRAW keeps apart:
//!
//! - **Colour**: the ForwardMatrix (or ColorMatrix) blended for the shot's white
//!   balance, then the HueSatMap. With "Camera profiles" on in settings it replaces
//!   rawler's generic D65 matrix when a RAW is decoded (`raw_processing`), so every
//!   tone mapper and adjustment works on it. See [`Calibration`].
//! - **Look**: BaselineExposureOffset, the LookTable and the hue-preserving
//!   ProfileToneCurve. That is a display transform, offered as the "Profile" tone
//!   mapper next to Basic and AgX. See [`CameraProfile::render`] and its GPU copy in
//!   `shaders/camera_profile.wgsl`; the parity test at the bottom keeps them in step.
//!
//! Both follow the DNG SDK:
//!
//! ```text
//! decode: camera RGB -> WB-blended ForwardMatrix -> ProPhoto -> HueSatMap -> working Rec.709
//! render: working Rec.709 -> ProPhoto -> BaselineExposureOffset -> LookTable
//!           -> ProfileToneCurve (RGBTone) -> sRGB
//! ```
//!
//! Profiles are found by camera model in `<app data>/camera_profiles`, then in the
//! RawTherapee and ART profile folders.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use bytemuck::Zeroable;
use rawler::decoders::RawDecodeParams;
use rawler::rawimage::{RawImage, RawPhotometricInterpretation};
use rawler::rawsource::RawSource;

use crate::app_settings::AppSettings;
use crate::image_processing::AllAdjustments;

/// The `toneMapper` adjustment value that selects profile rendering.
pub const TONE_MAPPER_ID: &str = "profile";
/// `tonemapper_mode` in the shader.
pub const TONE_MAPPER_MODE: u32 = 2;

/// Tone curve samples on [0, 1], linearly interpolated by both renderers.
const TONE_SAMPLES: usize = 4096;

/// Folders searched for `<Make> <Model>*.dcp`, after the user's own folder.
const SYSTEM_PROFILE_DIRS: &[&str] = &[
    "/usr/share/rawtherapee/dcpprofiles",
    "/usr/share/ART/dcpprofiles",
    "/usr/local/share/rawtherapee/dcpprofiles",
];

type Mat3 = [[f64; 3]; 3];

const PROPHOTO_TO_XYZ_D50: Mat3 = [
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100],
];
const XYZ_D50_TO_D65_BRADFORD: Mat3 = [
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098],
];
const XYZ_D65_TO_REC709: Mat3 = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];
const D50_XY: (f64, f64) = (0.3457, 0.3585);

// ---------------------------------------------------------------------------
// Small 3x3 helpers

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

fn mul_vec(a: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

fn diag(v: [f64; 3]) -> Mat3 {
    [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
}

fn scale(a: &Mat3, s: f64) -> Mat3 {
    a.map(|row| row.map(|x| x * s))
}

fn lerp(a: &Mat3, b: &Mat3, w: f64) -> Mat3 {
    let mut r = *a;
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = w * a[i][j] + (1.0 - w) * b[i][j];
        }
    }
    r
}

fn invert(m: &Mat3) -> Option<Mat3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some(scale(
        &[
            [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
            [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
            [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
        ],
        1.0 / det,
    ))
}

fn xy_to_xyz((x, y): (f64, f64)) -> [f64; 3] {
    let y = y.max(1e-6);
    [x / y, 1.0, (1.0 - x - y) / y]
}

fn xyz_to_xy(v: [f64; 3]) -> (f64, f64) {
    let sum = v[0] + v[1] + v[2];
    if sum > 0.0 {
        (v[0] / sum, v[1] / sum)
    } else {
        D50_XY
    }
}

/// Bradford adaptation from white `from` to white `to` (DNG SDK `MapWhiteMatrix`).
fn map_white(from: (f64, f64), to: (f64, f64)) -> Mat3 {
    let w1 = mul_vec(&BRADFORD, xy_to_xyz(from));
    let w2 = mul_vec(&BRADFORD, xy_to_xyz(to));
    let ratio = [0, 1, 2].map(|i| {
        if w1[i] > 0.0 {
            (w2[i] / w1[i]).clamp(0.1, 10.0)
        } else {
            1.0
        }
    });
    mul(&mul(&invert(&BRADFORD).unwrap(), &diag(ratio)), &BRADFORD)
}

/// Correlated colour temperature of a white point, Robertson's method as in the DNG SDK.
fn xy_to_temperature((x, y): (f64, f64)) -> f64 {
    // (mired, u, v, isotherm slope)
    const TABLE: [(f64, f64, f64, f64); 31] = [
        (0.0, 0.18006, 0.26352, -0.24341),
        (10.0, 0.18066, 0.26589, -0.25479),
        (20.0, 0.18133, 0.26846, -0.26876),
        (30.0, 0.18208, 0.27119, -0.28539),
        (40.0, 0.18293, 0.27407, -0.30470),
        (50.0, 0.18388, 0.27709, -0.32675),
        (60.0, 0.18494, 0.28021, -0.35156),
        (70.0, 0.18611, 0.28342, -0.37915),
        (80.0, 0.18740, 0.28668, -0.40955),
        (90.0, 0.18880, 0.28997, -0.44278),
        (100.0, 0.19032, 0.29326, -0.47888),
        (125.0, 0.19462, 0.30141, -0.58204),
        (150.0, 0.19962, 0.30921, -0.70471),
        (175.0, 0.20525, 0.31647, -0.84901),
        (200.0, 0.21142, 0.32312, -1.0182),
        (225.0, 0.21807, 0.32909, -1.2168),
        (250.0, 0.22511, 0.33439, -1.4512),
        (275.0, 0.23247, 0.33904, -1.7298),
        (300.0, 0.24010, 0.34308, -2.0637),
        (325.0, 0.24792, 0.34655, -2.4681),
        (350.0, 0.25591, 0.34951, -2.9641),
        (375.0, 0.26400, 0.35200, -3.5814),
        (400.0, 0.27218, 0.35407, -4.3633),
        (425.0, 0.28039, 0.35577, -5.3762),
        (450.0, 0.28863, 0.35714, -6.7262),
        (475.0, 0.29685, 0.35823, -8.5955),
        (500.0, 0.30505, 0.35907, -11.324),
        (525.0, 0.31320, 0.35968, -15.628),
        (550.0, 0.32129, 0.36011, -23.325),
        (575.0, 0.32931, 0.36038, -40.770),
        (600.0, 0.33724, 0.36051, -116.45),
    ];
    let denom = 1.5 - x + 6.0 * y;
    let (u, v) = (2.0 * x / denom, 3.0 * y / denom);
    let mut last_dt = 0.0;
    for i in 1..TABLE.len() {
        let len = (1.0 + TABLE[i].3 * TABLE[i].3).sqrt();
        let (du, dv) = (1.0 / len, TABLE[i].3 / len);
        let mut dt = -(u - TABLE[i].1) * dv + (v - TABLE[i].2) * du;
        if dt <= 0.0 || i == TABLE.len() - 1 {
            dt = (-dt).max(0.0);
            let f = if i == 1 { 0.0 } else { dt / (last_dt + dt) };
            return 1.0e6 / (TABLE[i - 1].0 * f + TABLE[i].0 * (1.0 - f));
        }
        last_dt = dt;
    }
    unreachable!()
}

/// DNG SDK `IlluminantToTemperature`.
fn illuminant_temperature(illuminant: u16) -> f64 {
    match illuminant {
        3 | 17 => 2850.0,
        24 => 3200.0,
        23 => 5000.0,
        1 | 4 | 9 | 18 | 20 => 5500.0,
        10 | 19 | 21 => 6500.0,
        11 | 22 => 7500.0,
        12 => 6400.0,
        13 => 5050.0,
        2 | 14 => 4150.0,
        15 => 3525.0,
        16 => 2925.0,
        _ => 0.0,
    }
}

fn srgb_encode(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_decode(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

// ---------------------------------------------------------------------------
// Profile file

/// A HueSatMap or LookTable: (hue shift in degrees, saturation scale, value scale)
/// per (value, hue, saturation) division, the DNG storage order.
#[derive(Clone, Debug)]
pub struct HsvTable {
    hue: u32,
    sat: u32,
    val: u32,
    srgb: bool,
    data: Vec<[f32; 3]>,
}

impl HsvTable {
    fn new(dims: &[u32], data: &[f32], srgb: bool) -> Option<Self> {
        let (hue, sat, val) = (*dims.first()?, *dims.get(1)?, *dims.get(2)?.max(&1));
        let n = (hue * sat * val) as usize;
        if hue < 2 || sat < 2 || data.len() < n * 3 {
            return None;
        }
        let data = data.as_chunks::<3>().0[..n].to_vec();
        Some(Self {
            hue,
            sat,
            val,
            srgb,
            data,
        })
    }

    fn blend(a: &Self, b: &Self, w: f64) -> Self {
        if (a.hue, a.sat, a.val) != (b.hue, b.sat, b.val) {
            return a.clone();
        }
        let w = w as f32;
        let data = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| [0, 1, 2].map(|i| w * x[i] + (1.0 - w) * y[i]))
            .collect();
        Self { data, ..a.clone() }
    }

    fn entry(&self, h: u32, s: u32, v: u32) -> [f32; 3] {
        self.data[((v * self.hue + h) * self.sat + s) as usize]
    }

    /// DNG SDK `RefBaselineHueSatMap`, without its clipping of value (the float
    /// pipeline keeps highlights until the tone curve).
    fn apply(&self, c: [f32; 3]) -> [f32; 3] {
        let [mut h, mut s, mut v] = rgb_to_hsv(c.map(|x| x.max(0.0)));
        let h_scaled = h * self.hue as f32 / 6.0;
        let s_scaled = s * (self.sat - 1) as f32;
        let mut h0 = h_scaled as u32;
        let s0 = (s_scaled as u32).min(self.sat - 2);
        let mut h1 = h0 + 1;
        if h0 >= self.hue - 1 {
            h0 = self.hue - 1;
            h1 = 0;
        }
        let hf = h_scaled - h0 as f32;
        let sf = s_scaled - s0 as f32;
        let bilinear = |v: u32| -> [f32; 3] {
            let (e00, e01) = (self.entry(h0, s0, v), self.entry(h1, s0, v));
            let (e10, e11) = (self.entry(h0, s0 + 1, v), self.entry(h1, s0 + 1, v));
            [0, 1, 2].map(|i| {
                let lo = e00[i] + (e01[i] - e00[i]) * hf;
                let hi = e10[i] + (e11[i] - e10[i]) * hf;
                lo + (hi - lo) * sf
            })
        };
        let mut v_encoded = v;
        let m = if self.val < 2 {
            bilinear(0)
        } else {
            if self.srgb {
                v_encoded = srgb_encode(v.clamp(0.0, 1.0));
            }
            let v_scaled = v_encoded * (self.val - 1) as f32;
            let v0 = (v_scaled as u32).min(self.val - 2);
            let vf = v_scaled - v0 as f32;
            let (a, b) = (bilinear(v0), bilinear(v0 + 1));
            [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * vf)
        };
        h += m[0] * (6.0 / 360.0);
        s = (s * m[1]).min(1.0);
        v = if self.srgb {
            srgb_decode((v_encoded * m[2]).clamp(0.0, 1.0))
        } else {
            v * m[2]
        };
        hsv_to_rgb([h, s, v])
    }
}

/// DNG SDK `DNG_RGBtoHSV`: hue in [0, 6).
fn rgb_to_hsv([r, g, b]: [f32; 3]) -> [f32; 3] {
    let v = r.max(g).max(b);
    let gap = v - r.min(g).min(b);
    if gap <= 0.0 {
        return [0.0, 0.0, v];
    }
    let h = if r == v {
        let h = (g - b) / gap;
        if h < 0.0 { h + 6.0 } else { h }
    } else if g == v {
        2.0 + (b - r) / gap
    } else {
        4.0 + (r - g) / gap
    };
    [h, gap / v, v]
}

/// DNG SDK `DNG_HSVtoRGB`.
fn hsv_to_rgb([mut h, s, v]: [f32; 3]) -> [f32; 3] {
    if s <= 0.0 {
        return [v, v, v];
    }
    if h < 0.0 {
        h += 6.0;
    }
    if h >= 6.0 {
        h -= 6.0;
    }
    let i = (h as u32).min(5);
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// A parsed .dcp file.
#[derive(Debug)]
pub struct CameraProfile {
    /// ProfileName, else the file name.
    pub name: String,
    pub unique_camera_model: String,
    illuminants: [u16; 2],
    color_matrix: [Option<Mat3>; 2],
    forward_matrix: [Option<Mat3>; 2],
    hue_sat_map: [Option<HsvTable>; 2],
    look_table: Option<HsvTable>,
    /// ProfileToneCurve resampled to `TONE_SAMPLES` points, or None.
    tone_curve: Option<Vec<f32>>,
    /// Working linear Rec.709 to linear ProPhoto, including BaselineExposureOffset.
    working_to_prophoto: [[f32; 3]; 3],
    prophoto_to_working: [[f32; 3]; 3],
}

/// Minimal TIFF-structure reader for the DCP's single IFD.
struct Ifd<'a> {
    buf: &'a [u8],
    le: bool,
    entries: HashMap<u16, (u16, u32, u32)>, // tag -> (type, count, value offset)
}

impl<'a> Ifd<'a> {
    fn parse(buf: &'a [u8]) -> Result<Self, String> {
        let le = match buf.get(0..2) {
            Some(b"II") => true,
            Some(b"MM") => false,
            _ => return Err("not a DCP/TIFF file".into()),
        };
        let mut ifd = Ifd {
            buf,
            le,
            entries: HashMap::new(),
        };
        let magic = ifd.u16_at(2).ok_or("truncated")?;
        if magic != 0x4352 && magic != 42 {
            return Err(format!("unexpected magic {magic:#x}"));
        }
        let offset = ifd.u32_at(4).ok_or("truncated")? as usize;
        let count = ifd.u16_at(offset).ok_or("truncated")? as usize;
        for i in 0..count {
            let e = offset + 2 + i * 12;
            let (Some(tag), Some(typ), Some(n)) =
                (ifd.u16_at(e), ifd.u16_at(e + 2), ifd.u32_at(e + 4))
            else {
                return Err("truncated IFD".into());
            };
            let size = n as usize * type_size(typ);
            let value_offset = if size <= 4 {
                (e + 8) as u32
            } else {
                ifd.u32_at(e + 8).ok_or("truncated")?
            };
            ifd.entries.insert(tag, (typ, n, value_offset));
        }
        Ok(ifd)
    }

    fn u16_at(&self, o: usize) -> Option<u16> {
        let b: [u8; 2] = self.buf.get(o..o + 2)?.try_into().ok()?;
        Some(if self.le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    }

    fn u32_at(&self, o: usize) -> Option<u32> {
        let b: [u8; 4] = self.buf.get(o..o + 4)?.try_into().ok()?;
        Some(if self.le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    /// Any numeric tag as f64s.
    fn numbers(&self, tag: u16) -> Option<Vec<f64>> {
        let &(typ, n, o) = self.entries.get(&tag)?;
        let o = o as usize;
        (0..n as usize)
            .map(|i| match typ {
                3 => self.u16_at(o + i * 2).map(f64::from),
                4 => self.u32_at(o + i * 4).map(f64::from),
                5 => {
                    Some(self.u32_at(o + i * 8)? as f64 / self.u32_at(o + i * 8 + 4)?.max(1) as f64)
                }
                9 => self.u32_at(o + i * 4).map(|v| v as i32 as f64),
                10 => {
                    let d = self.u32_at(o + i * 8 + 4)? as i32;
                    Some(
                        self.u32_at(o + i * 8)? as i32 as f64 / if d == 0 { 1.0 } else { d as f64 },
                    )
                }
                11 => self.u32_at(o + i * 4).map(|v| f32::from_bits(v) as f64),
                12 => {
                    let b: [u8; 8] = self.buf.get(o + i * 8..o + i * 8 + 8)?.try_into().ok()?;
                    Some(if self.le {
                        f64::from_le_bytes(b)
                    } else {
                        f64::from_be_bytes(b)
                    })
                }
                _ => None,
            })
            .collect()
    }

    fn floats(&self, tag: u16) -> Option<Vec<f32>> {
        Some(self.numbers(tag)?.into_iter().map(|v| v as f32).collect())
    }

    fn u32s(&self, tag: u16) -> Option<Vec<u32>> {
        Some(self.numbers(tag)?.into_iter().map(|v| v as u32).collect())
    }

    fn matrix(&self, tag: u16) -> Option<Mat3> {
        let v = self.numbers(tag)?;
        (v.len() == 9).then(|| [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
    }

    fn string(&self, tag: u16) -> Option<String> {
        let &(typ, n, o) = self.entries.get(&tag)?;
        if typ != 2 && typ != 1 {
            return None;
        }
        let bytes = self.buf.get(o as usize..o as usize + n as usize)?;
        let s = String::from_utf8_lossy(bytes);
        Some(s.trim_end_matches('\0').trim().to_string())
    }
}

fn type_size(typ: u16) -> usize {
    match typ {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => 1,
    }
}

impl CameraProfile {
    pub fn parse(buf: &[u8]) -> Result<Self, String> {
        let ifd = Ifd::parse(buf)?;
        // DNG SDK `NormalizeColorMatrix`: PCS white just saturates the camera.
        let color_matrix = [50721, 50722].map(|tag| {
            ifd.matrix(tag).map(|m| {
                let max = mul_vec(&m, xy_to_xyz(D50_XY))
                    .into_iter()
                    .fold(f64::MIN, f64::max);
                if max > 0.0 { scale(&m, 1.0 / max) } else { m }
            })
        });
        if color_matrix[0].is_none() {
            return Err("profile has no ColorMatrix1".into());
        }
        // DNG SDK `NormalizeForwardMatrix`: camera (1,1,1) maps to the D50 white.
        let forward_matrix = [50964, 50965].map(|tag| {
            ifd.matrix(tag).map(|m| {
                let xyz = mul_vec(&m, [1.0; 3]);
                let d50 = xy_to_xyz(D50_XY);
                let s = [0, 1, 2].map(|i| if xyz[i] != 0.0 { d50[i] / xyz[i] } else { 1.0 });
                mul(&diag(s), &m)
            })
        });
        let hsm_dims = ifd.u32s(50937).unwrap_or_default();
        let hsm_srgb = ifd.u32s(51107).and_then(|v| v.first().copied()) == Some(1);
        let hue_sat_map = [50938, 50939].map(|tag| {
            ifd.floats(tag)
                .and_then(|d| HsvTable::new(&hsm_dims, &d, hsm_srgb))
        });
        let look_srgb = ifd.u32s(51108).and_then(|v| v.first().copied()) == Some(1);
        let look_table = ifd
            .u32s(50981)
            .zip(ifd.floats(50982))
            .and_then(|(dims, data)| HsvTable::new(&dims, &data, look_srgb));
        let tone_curve = ifd.numbers(50940).and_then(|v| resample_tone_curve(&v));
        let illuminant = |tag| ifd.u32s(tag).and_then(|v| v.first().copied()).unwrap_or(0) as u16;
        let baseline_exposure_offset = ifd
            .numbers(51109)
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);
        let prophoto_to_working = prophoto_to_working();
        let working_to_prophoto = scale(
            &invert(&prophoto_to_working).unwrap(),
            baseline_exposure_offset.exp2(),
        );
        Ok(Self {
            name: ifd.string(50936).unwrap_or_default(),
            unique_camera_model: ifd.string(50708).unwrap_or_default(),
            illuminants: [illuminant(50778), illuminant(50779)],
            color_matrix,
            forward_matrix,
            hue_sat_map,
            look_table,
            tone_curve,
            working_to_prophoto: to_f32(&working_to_prophoto),
            prophoto_to_working: to_f32(&prophoto_to_working),
        })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let mut profile = Self::parse(&std::fs::read(path).map_err(|e| e.to_string())?)?;
        if profile.name.is_empty() {
            profile.name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
        }
        Ok(profile)
    }

    /// Weight of the first calibration set at colour temperature `t` (inverse-temperature
    /// interpolation between the two calibration illuminants, as in the DNG SDK).
    fn weight1(&self, t: f64) -> f64 {
        let (t1, t2) = (
            illuminant_temperature(self.illuminants[0]),
            illuminant_temperature(self.illuminants[1]),
        );
        if self.color_matrix[1].is_none() || t1 <= 0.0 || t2 <= 0.0 || t1 == t2 {
            return 1.0;
        }
        let (lo, hi) = (t1.min(t2), t1.max(t2));
        let w_lo = if t <= lo {
            1.0
        } else if t >= hi {
            0.0
        } else {
            (1.0 / t - 1.0 / hi) / (1.0 / lo - 1.0 / hi)
        };
        if t1 < t2 { w_lo } else { 1.0 - w_lo }
    }

    fn blended(pair: &[Option<Mat3>; 2], w: f64) -> Option<Mat3> {
        match pair {
            [Some(a), Some(b)] => Some(lerp(a, b, w)),
            [Some(a), None] | [None, Some(a)] => Some(*a),
            [None, None] => None,
        }
    }

    fn xyz_to_camera(&self, white: (f64, f64)) -> Mat3 {
        Self::blended(&self.color_matrix, self.weight1(xy_to_temperature(white))).unwrap()
    }

    /// DNG SDK `NeutralToXY`: the white point whose camera response is `neutral`.
    fn neutral_to_xy(&self, neutral: [f64; 3]) -> (f64, f64) {
        let mut last = D50_XY;
        for pass in 0..30 {
            let Some(inv) = invert(&self.xyz_to_camera(last)) else {
                return last;
            };
            let mut next = xyz_to_xy(mul_vec(&inv, neutral));
            if (next.0 - last.0).abs() + (next.1 - last.1).abs() < 1e-7 {
                return next;
            }
            if pass == 29 {
                next = ((last.0 + next.0) * 0.5, (last.1 + next.1) * 0.5);
            }
            last = next;
        }
        last
    }

    /// DNG SDK `dng_color_spec`: camera RGB to XYZ D50 for a shot whose camera
    /// neutral is `neutral`, plus the weight of calibration set 1.
    fn camera_to_xyz(&self, neutral: [f64; 3]) -> Option<(Mat3, f64)> {
        let white = self.neutral_to_xy(neutral);
        let w = self.weight1(xy_to_temperature(white));
        let color_matrix = self.xyz_to_camera(white);
        let m = if let Some(forward) = Self::blended(&self.forward_matrix, w) {
            let cw = mul_vec(&color_matrix, xy_to_xyz(white));
            let max = cw.into_iter().fold(f64::MIN, f64::max);
            if max <= 0.0 {
                return None;
            }
            let cw = cw.map(|c| (c / max).clamp(0.001, 1.0));
            mul(&forward, &diag(cw.map(|c| 1.0 / c)))
        } else {
            let pcs_to_camera = mul(&color_matrix, &map_white(D50_XY, white));
            let max = mul_vec(&pcs_to_camera, xy_to_xyz(D50_XY))
                .into_iter()
                .fold(f64::MIN, f64::max);
            invert(&scale(&pcs_to_camera, 1.0 / max.max(1e-6)))?
        };
        Some((m, w))
    }
}

/// Natural cubic spline through the (x, y) pairs, sampled at `TONE_SAMPLES` points.
fn resample_tone_curve(pairs: &[f64]) -> Option<Vec<f32>> {
    let pts: Vec<(f64, f64)> = pairs
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[x, y]| (x, y))
        .collect();
    let n = pts.len();
    if n < 2 || pts.windows(2).any(|w| w[1].0 <= w[0].0) {
        return None;
    }
    // Second derivatives, zero at both ends.
    let mut m = vec![0.0; n];
    if n > 2 {
        let (mut c, mut d) = (vec![0.0; n], vec![0.0; n]);
        for i in 1..n - 1 {
            let (h0, h1) = (pts[i].0 - pts[i - 1].0, pts[i + 1].0 - pts[i].0);
            let rhs = 6.0 * ((pts[i + 1].1 - pts[i].1) / h1 - (pts[i].1 - pts[i - 1].1) / h0);
            let denom = 2.0 * (h0 + h1) - h0 * c[i - 1];
            c[i] = h1 / denom;
            d[i] = (rhs - h0 * d[i - 1]) / denom;
        }
        for i in (1..n - 1).rev() {
            m[i] = d[i] - c[i] * m[i + 1];
        }
    }
    let mut seg = 0;
    Some(
        (0..TONE_SAMPLES)
            .map(|i| {
                let x = (i as f64 / (TONE_SAMPLES - 1) as f64).clamp(pts[0].0, pts[n - 1].0);
                while seg < n - 2 && x > pts[seg + 1].0 {
                    seg += 1;
                }
                let (x0, y0, x1, y1) = (pts[seg].0, pts[seg].1, pts[seg + 1].0, pts[seg + 1].1);
                let h = x1 - x0;
                let (a, b) = ((x1 - x) / h, (x - x0) / h);
                let y = a * y0
                    + b * y1
                    + ((a * a * a - a) * m[seg] + (b * b * b - b) * m[seg + 1]) * h * h / 6.0;
                y.clamp(0.0, 1.0) as f32
            })
            .collect(),
    )
}

fn to_f32(m: &Mat3) -> [[f32; 3]; 3] {
    m.map(|row| row.map(|x| x as f32))
}

fn mat_vec_f32(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// Linear ProPhoto (D50) to the linear Rec.709 (D65) working space, rows normalised
/// so grey stays exactly grey.
fn prophoto_to_working() -> Mat3 {
    mul(
        &XYZ_D65_TO_REC709,
        &mul(&XYZ_D50_TO_D65_BRADFORD, &PROPHOTO_TO_XYZ_D50),
    )
    .map(|row| row.map(|x| x / row.iter().sum::<f64>()))
}

// ---------------------------------------------------------------------------
// Colour, applied when the RAW is decoded

/// A profile's colour for one shot: camera RGB to the working space, in place of
/// rawler's Calibrate step.
pub struct Calibration {
    camera_to_prophoto: [[f32; 3]; 3],
    hue_sat_map: Option<HsvTable>,
    prophoto_to_working: [[f32; 3]; 3],
}

impl Calibration {
    /// Camera RGB (not white balanced) to working linear Rec.709. As in rawler's
    /// Calibrate step, colours outside Rec.709 lose their negative components.
    pub fn apply(&self, camera: [f32; 3]) -> [f32; 3] {
        let mut c = mat_vec_f32(&self.camera_to_prophoto, camera);
        if let Some(t) = &self.hue_sat_map {
            c = t.apply(c);
        }
        mat_vec_f32(&self.prophoto_to_working, c).map(|x| x.max(0.0))
    }
}

impl CameraProfile {
    /// The colour of a shot that rawler white-balances with the camera-channel
    /// multipliers `wb`.
    pub fn calibration(&self, wb: [f32; 3]) -> Option<Calibration> {
        let wb = wb.map(|w| {
            if w.is_finite() && w > 0.0 {
                w as f64
            } else {
                1.0
            }
        });
        let neutral = wb.map(|w| 1.0 / w);
        let max = neutral.into_iter().fold(f64::MIN, f64::max);
        let (camera_to_xyz, w) = self.camera_to_xyz(neutral.map(|n| n / max))?;
        // The DNG SDK scales white to clip with its brightest camera channel, rawler
        // just multiplies by `wb`. Keep rawler's scale: a profile changes colour, not
        // exposure.
        let camera_to_prophoto = scale(
            &mul(&invert(&PROPHOTO_TO_XYZ_D50)?, &camera_to_xyz),
            1.0 / max,
        );
        let hue_sat_map = match &self.hue_sat_map {
            [Some(a), Some(b)] => Some(HsvTable::blend(a, b, w)),
            [Some(a), None] | [None, Some(a)] => Some(a.clone()),
            [None, None] => None,
        };
        Some(Calibration {
            camera_to_prophoto: to_f32(&camera_to_prophoto),
            hue_sat_map,
            prophoto_to_working: self.prophoto_to_working,
        })
    }
}

// ---------------------------------------------------------------------------
// Look, the "Profile" tone mapper

/// GPU header, laid out as `CameraProfile` in `shaders/camera_profile.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuHeader {
    input_matrix: [[f32; 4]; 3],
    output_matrix: [[f32; 4]; 3],
    look_table: [u32; 4],
    tone_curve: [u32; 4],
    flags: [u32; 4],
}

fn gpu_mat3(m: &[[f32; 3]; 3]) -> [[f32; 4]; 3] {
    // WGSL matrices are column-major.
    [0, 1, 2].map(|c| [m[0][c], m[1][c], m[2][c], 0.0])
}

impl CameraProfile {
    /// Working linear Rec.709 to sRGB-encoded display values. The CPU reference
    /// for the shader's `camera_profile_render`.
    pub fn render(&self, linear_rec709: [f32; 3]) -> [f32; 3] {
        let mut c = mat_vec_f32(&self.working_to_prophoto, linear_rec709);
        if let Some(t) = &self.look_table {
            c = t.apply(c);
        }
        c = match &self.tone_curve {
            Some(curve) => rgb_tone(curve, c),
            None => c.map(|x| x.clamp(0.0, 1.0)),
        };
        mat_vec_f32(&self.prophoto_to_working, c).map(|x| srgb_encode(x.clamp(0.0, 1.0)))
    }

    /// Storage-buffer contents for `shaders/camera_profile.wgsl`.
    pub fn gpu_bytes(&self) -> Vec<u8> {
        let mut data: Vec<[f32; 4]> = Vec::new();
        let look_table = match &self.look_table {
            Some(t) => {
                let offset = data.len() as u32;
                data.extend(t.data.iter().map(|e| [e[0], e[1], e[2], 0.0]));
                [t.hue, t.sat, t.val, offset]
            }
            None => [0; 4],
        };
        let tone_curve = match &self.tone_curve {
            Some(curve) => {
                let offset = data.len() as u32;
                data.extend(
                    curve
                        .chunks(4)
                        .map(|c| [0, 1, 2, 3].map(|i| c.get(i).copied().unwrap_or(0.0))),
                );
                [curve.len() as u32, offset, 0, 0]
            }
            None => [0; 4],
        };
        if data.is_empty() {
            data.push([0.0; 4]);
        }
        let look_srgb = self.look_table.as_ref().is_some_and(|t| t.srgb);
        let header = GpuHeader {
            input_matrix: gpu_mat3(&self.working_to_prophoto),
            output_matrix: gpu_mat3(&self.prophoto_to_working),
            look_table,
            tone_curve,
            flags: [1, look_srgb as u32, 0, 0],
        };
        let mut bytes = bytemuck::bytes_of(&header).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(&data));
        bytes
    }

    /// Bound when no profile is in use; the shader then falls back to Basic.
    pub fn disabled_gpu_bytes() -> Vec<u8> {
        let mut bytes = bytemuck::bytes_of(&GpuHeader::zeroed()).to_vec();
        bytes.extend_from_slice(&[0u8; 16]);
        bytes
    }
}

fn tone_lookup(curve: &[f32], x: f32) -> f32 {
    let pos = x.clamp(0.0, 1.0) * (curve.len() - 1) as f32;
    let i = (pos as usize).min(curve.len() - 2);
    let f = pos - i as f32;
    curve[i] + (curve[i + 1] - curve[i]) * f
}

/// DNG SDK `RefBaselineRGBTone`: the curve on the largest and smallest channel,
/// the middle one keeps its relative position, so hue is preserved.
fn rgb_tone(curve: &[f32], c: [f32; 3]) -> [f32; 3] {
    let c = c.map(|x| x.clamp(0.0, 1.0));
    let (hi, lo) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let (hi_t, lo_t) = (tone_lookup(curve, hi), tone_lookup(curve, lo));
    if hi - lo <= 0.0 {
        return [hi_t; 3];
    }
    c.map(|x| lo_t + (hi_t - lo_t) * (x - lo) / (hi - lo))
}

// ---------------------------------------------------------------------------
// Finding the profile for an image

static USER_PROFILE_DIR: OnceLock<PathBuf> = OnceLock::new();
static PROFILE_CACHE: LazyLock<Mutex<HashMap<PathBuf, Arc<CameraProfile>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static IMAGE_CACHE: LazyLock<Mutex<HashMap<PathBuf, Option<Arc<CameraProfile>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The app's own profile folder (`<app data>/camera_profiles`), searched first.
pub fn set_user_profile_dir(dir: PathBuf) {
    let _ = std::fs::create_dir_all(&dir);
    let _ = USER_PROFILE_DIR.set(dir);
}

fn search_dirs() -> Vec<PathBuf> {
    USER_PROFILE_DIR
        .get()
        .cloned()
        .into_iter()
        .chain(SYSTEM_PROFILE_DIRS.iter().map(PathBuf::from))
        .collect()
}

/// First `<make> <model>.dcp` or `<make> <model> <anything>.dcp` (case-insensitive).
fn find_profile_file(make_model: &str) -> Option<PathBuf> {
    let key = make_model.to_lowercase();
    for dir in search_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut files: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        files.sort();
        for path in files {
            if !path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("dcp"))
            {
                continue;
            }
            let stem = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            if stem == key || stem.starts_with(&format!("{key} ")) {
                return Some(path);
            }
        }
    }
    None
}

fn load_cached(path: &Path) -> Option<Arc<CameraProfile>> {
    if let Some(profile) = PROFILE_CACHE.lock().unwrap().get(path) {
        return Some(profile.clone());
    }
    match CameraProfile::load(path) {
        Ok(profile) => {
            let profile = Arc::new(profile);
            PROFILE_CACHE
                .lock()
                .unwrap()
                .insert(path.to_path_buf(), profile.clone());
            Some(profile)
        }
        Err(e) => {
            log::warn!("Camera profile '{}' unreadable: {}", path.display(), e);
            None
        }
    }
}

/// The profile for this RAW's camera. Only three-colour mosaic sensors qualify:
/// they are what a profile's 3x3 matrices describe.
pub fn for_raw(raw: &RawImage) -> Option<Arc<CameraProfile>> {
    let RawPhotometricInterpretation::Cfa(config) = &raw.photometric else {
        return None;
    };
    if raw.cpp != 1 || !config.cfa.is_rgb() {
        return None;
    }
    load_cached(&find_profile_file(&format!(
        "{} {}",
        raw.clean_make, raw.clean_model
    ))?)
}

fn resolve(source: &Path) -> Result<Option<Arc<CameraProfile>>, String> {
    if !crate::formats::is_raw_file(source) {
        return Ok(None);
    }
    let raw_source = RawSource::new(source).map_err(|e| e.to_string())?;
    let decoder = rawler::get_decoder(&raw_source).map_err(|e| e.to_string())?;
    let raw = decoder
        .raw_image(&raw_source, &RawDecodeParams::default(), true)
        .map_err(|e| e.to_string())?;
    Ok(for_raw(&raw))
}

/// The profile a RAW (or a virtual copy of one) is decoded with under `settings`.
pub fn for_image(path: &str, settings: &AppSettings) -> Option<Arc<CameraProfile>> {
    if !settings.use_camera_profiles.unwrap_or(false) {
        return None;
    }
    let (source, _) = crate::file_management::parse_virtual_path(path);
    if let Some(hit) = IMAGE_CACHE.lock().unwrap().get(&source) {
        return hit.clone();
    }
    let profile = resolve(&source).unwrap_or_else(|e| {
        log::warn!(
            "Camera profile lookup failed for '{}': {}",
            source.display(),
            e
        );
        None
    });
    let mut cache = IMAGE_CACHE.lock().unwrap();
    if cache.len() > 512 {
        cache.clear();
    }
    cache.insert(source, profile.clone());
    profile
}

/// The profile a render of `path` needs: only the "Profile" tone mapper uses one.
pub fn for_render(
    path: &str,
    adjustments: &AllAdjustments,
    settings: &AppSettings,
) -> Option<Arc<CameraProfile>> {
    let global = &adjustments.global;
    if global.tonemapper_mode != TONE_MAPPER_MODE || global.is_raw_image != 1 {
        return None;
    }
    for_image(path, settings)
}

/// Name of the camera profile this image is decoded with, if any. The editor
/// only offers the "Profile" tone mapper when there is one.
#[tauri::command]
pub async fn get_camera_profile_name(path: String, app_handle: tauri::AppHandle) -> Option<String> {
    tauri::async_runtime::spawn_blocking(move || {
        let settings = crate::app_settings::load_settings(app_handle).unwrap_or_default();
        for_image(&path, &settings).map(|p| p.name.clone())
    })
    .await
    .ok()
    .flatten()
}

/// Harness (examples/camera_profile.rs): decodes `path` the way the editor does
/// with camera profiles on and renders it with the "Profile" tone mapper at
/// default settings.
#[doc(hidden)]
pub fn dev_render(path: &str, long_edge: u32) -> Result<(String, image::RgbImage), String> {
    let settings = AppSettings {
        use_camera_profiles: Some(true),
        ..Default::default()
    };
    let profile = for_image(path, &settings).ok_or("no camera profile for this image")?;
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let image =
        crate::image_loader::load_base_image_from_bytes(&bytes, path, false, &settings, None)
            .map_err(|e| e.to_string())?
            .resize(long_edge, long_edge, image::imageops::FilterType::Triangle)
            .to_rgb32f();
    let out = image::RgbImage::from_fn(image.width(), image.height(), |x, y| {
        let c = profile.render(image.get_pixel(x, y).0);
        image::Rgb(c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
    });
    Ok((profile.name.clone(), out))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RT_A7III: &str = "/usr/share/rawtherapee/dcpprofiles/SONY ILCE-7M3.dcp";

    fn rt_profile() -> Option<CameraProfile> {
        Path::new(RT_A7III)
            .exists()
            .then(|| CameraProfile::load(Path::new(RT_A7III)).unwrap())
    }

    /// Rawler's white-balance multipliers for a shot lit by `white` (green = 1).
    fn wb_under(profile: &CameraProfile, white: (f64, f64)) -> [f32; 3] {
        let neutral = mul_vec(&profile.xyz_to_camera(white), xy_to_xyz(white));
        neutral.map(|n| (neutral[1] / n) as f32)
    }

    #[test]
    fn temperature_of_standard_illuminants() {
        assert!((xy_to_temperature((0.31271, 0.32902)) - 6504.0).abs() < 15.0);
        assert!((xy_to_temperature((0.44757, 0.40745)) - 2856.0).abs() < 15.0);
        assert!((xy_to_temperature(D50_XY) - 5003.0).abs() < 15.0);
    }

    #[test]
    fn hsv_round_trip() {
        for c in [
            [0.2, 0.5, 0.9],
            [0.9, 0.1, 0.4],
            [0.3, 0.3, 0.3],
            [0.0, 1.0, 0.0],
        ] {
            let back = hsv_to_rgb(rgb_to_hsv(c));
            for i in 0..3 {
                assert!((back[i] - c[i]).abs() < 1e-6, "{c:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn tone_curve_spline_reproduces_a_line() {
        let curve = resample_tone_curve(&[0.0, 0.0, 0.5, 0.5, 1.0, 1.0]).unwrap();
        for (i, y) in curve.iter().enumerate() {
            assert!((y - i as f32 / (TONE_SAMPLES - 1) as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn rgb_tone_keeps_hue() {
        let curve: Vec<f32> = (0..TONE_SAMPLES)
            .map(|i| ((i as f32) / 4095.0).sqrt())
            .collect();
        let [r, g, b] = rgb_tone(&curve, [0.6, 0.3, 0.1]);
        let before = rgb_to_hsv([0.6, 0.3, 0.1])[0];
        assert!((rgb_to_hsv([r, g, b])[0] - before).abs() < 1e-3);
        assert!((r - 0.6f32.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn parses_rawtherapee_a7iii() {
        let Some(profile) = rt_profile() else { return };
        assert_eq!(profile.unique_camera_model, "Sony ILCE-7M3");
        assert_eq!(profile.illuminants, [17, 21]);
        assert!(profile.forward_matrix[0].is_some() && profile.forward_matrix[1].is_some());
        let hsm = profile.hue_sat_map[0].as_ref().unwrap();
        assert_eq!((hsm.hue, hsm.sat, hsm.val), (90, 30, 1));
        let look = profile.look_table.as_ref().unwrap();
        assert_eq!(
            (look.hue, look.sat, look.val, look.srgb),
            (90, 30, 30, true)
        );
        let curve = profile.tone_curve.as_ref().unwrap();
        assert!(
            curve.windows(2).all(|w| w[1] >= w[0]),
            "tone curve not monotonic"
        );
    }

    #[test]
    fn weights_follow_the_shot_white() {
        let Some(profile) = rt_profile() else { return };
        assert!((profile.weight1(2850.0) - 1.0).abs() < 1e-9);
        assert!(profile.weight1(6500.0).abs() < 1e-9);
        let mid = profile.weight1(4000.0);
        assert!(mid > 0.0 && mid < 1.0);
    }

    /// Whatever the light and however rawler scales its multipliers, a grey the
    /// camera saw decodes to the grey (and level) rawler's own matrix gives it, and
    /// the look keeps it grey.
    #[test]
    fn neutral_stays_neutral() {
        let Some(profile) = rt_profile() else { return };
        for white in [(0.44757, 0.40745), (0.3457, 0.3585), (0.31271, 0.32902)] {
            for wb_scale in [1.0, 0.5] {
                let wb = wb_under(&profile, white).map(|w| w * wb_scale);
                let calibration = profile.calibration(wb).unwrap();
                let grey = calibration.apply(wb.map(|w| 0.18 / w));
                for c in grey {
                    assert!((c - 0.18).abs() < 0.002, "{white:?} x{wb_scale}: {grey:?}");
                }
                let out = profile.render(grey);
                assert!(
                    (out[0] - out[1]).abs() < 0.01 && (out[2] - out[1]).abs() < 0.01,
                    "{white:?}: {out:?}"
                );
            }
        }
    }

    /// The shader and this file must render the same colours.
    #[test]
    fn gpu_matches_cpu() {
        let Some(profile) = rt_profile() else { return };
        let mut inputs: Vec<[f32; 4]> = Vec::new();
        let levels = [0.0, 0.004, 0.02, 0.07, 0.18, 0.4, 0.75, 1.0, 1.6];
        for r in levels {
            for g in levels {
                for b in levels {
                    inputs.push([r, g, b, 0.0]);
                }
            }
        }
        let Some(gpu) = test_gpu::render(&profile, &inputs) else {
            eprintln!("no GPU adapter, skipping parity test");
            return;
        };
        let mut worst = 0.0f32;
        for (i, c) in inputs.iter().enumerate() {
            let cpu = profile.render([c[0], c[1], c[2]]);
            for k in 0..3 {
                worst = worst.max((cpu[k] - gpu[i][k]).abs());
            }
        }
        assert!(worst < 2.0 / 255.0, "GPU and CPU differ by {worst}");
        eprintln!("GPU vs CPU worst difference: {worst}");
    }

    /// The main shader, with this file appended, must still compile.
    #[test]
    fn main_shader_compiles() {
        let Some((device, _queue)) = test_gpu::device() else {
            eprintln!("no GPU adapter, skipping shader test");
            return;
        };
        for (source, precision) in [
            (crate::gpu_processing::main_shader_source(), 0.0),
            (crate::gpu_processing::high_precision_shader_source(), 1.0),
        ] {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let _ = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("HIGH_PRECISION_OUTPUT", precision)],
                    ..Default::default()
                },
                cache: None,
            });
            if let Some(e) = pollster::block_on(scope.pop()) {
                panic!("{e}");
            }
        }
    }

    mod test_gpu {
        use super::super::CameraProfile;
        use wgpu::util::DeviceExt;

        pub fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
            pollster::block_on(adapter.request_device(&Default::default())).ok()
        }

        pub fn render(profile: &CameraProfile, inputs: &[[f32; 4]]) -> Option<Vec<[f32; 4]>> {
            let (device, queue) = device()?;
            let source = format!(
                "{}\n{}",
                include_str!("shaders/camera_profile.wgsl"),
                r#"
@group(0) @binding(0) var<storage, read> test_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> test_out: array<vec4<f32>>;
@compute @workgroup_size(64)
fn test_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&test_in)) { return; }
    test_out[id.x] = vec4<f32>(camera_profile_render(test_in[id.x].rgb), 1.0);
}
"#
            )
            .replace("@binding(14)", "@binding(2)");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("test_main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let storage = |contents: &[u8], usage| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents,
                    usage,
                })
            };
            let input = storage(bytemuck::cast_slice(inputs), wgpu::BufferUsages::STORAGE);
            let output = storage(
                &vec![0u8; inputs.len() * 16],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let profile_buf = storage(&profile.gpu_bytes(), wgpu::BufferUsages::STORAGE);
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (inputs.len() * 16) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: profile_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (inputs.len() * 16) as u64);
            queue.submit([encoder.finish()]);
            readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .ok()?;
            let data = readback.slice(..).get_mapped_range();
            Some(bytemuck::cast_slice(&data).to_vec())
        }
    }
}
