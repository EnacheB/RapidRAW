//! Face-aware auto exposure.
//!
//! Finds faces with YuNet (a 230 KB ONNX model embedded in the binary, so it
//! works offline and needs no download), meters the skin between the eyes and
//! the mouth, and solves for the exposure that puts it at a pleasant display
//! brightness through the same exposure -> highlights -> tone mapper chain the
//! shader runs. If that exposure would blow out the rest of the frame (the
//! backlit-portrait case), the global exposure is held back to a highlight
//! budget and each face gets the shortfall back from a feathered radial mask,
//! the way a phone's HDR pipeline or a careful editor would do it.

use std::sync::{Mutex, OnceLock};

use image::{DynamicImage, GenericImageView, Rgb32FImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{Value, json};

use crate::camera_profile::{luma, srgb_decode, srgb_encode};
use crate::image_processing::{apply_cpu_agx_tonemap, downscale_f32_image};

static YUNET_MODEL: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");
static YUNET_SESSION: OnceLock<Option<Mutex<Session>>> = OnceLock::new();

const YUNET_INPUT: usize = 640;
const DETECTION_SCORE: f32 = 0.7;
const NMS_IOU: f32 = 0.3;

/// Display luma (sRGB-encoded, Rec.709 weights) the metered skin is brought to.
const FACE_TARGET: f32 = 0.62;
/// Extra frame area allowed to clip beyond what the capture already clipped.
const CLIP_BUDGET: f32 = 0.01;
/// The most a face is lifted locally over the global exposure, in EV.
const MAX_FACE_LIFT: f32 = 1.75;
/// Below this the shortfall isn't worth a mask.
const MIN_FACE_LIFT: f32 = 0.2;
/// Highlight recovery levels tried, in shader units (slider / 120).
const HIGHLIGHT_LEVELS: [f32; 5] = [0.0, 0.15, 0.3, 0.45, 0.6];

const EXPOSURE_SLIDER_SCALE: f32 = 0.8;
const HIGHLIGHTS_SLIDER_SCALE: f32 = 120.0;

pub const AUTO_FACE_MASK_PREFIX: &str = "auto-face-lift";

#[derive(Clone)]
pub enum ToneMapper {
    Basic,
    Agx,
    Profile(std::sync::Arc<crate::camera_profile::CameraProfile>),
}

impl ToneMapper {
    /// The tone mapper the shader uses for `mode` on the image at `path`.
    pub fn new(mode: u32, path: &str, settings: &crate::app_settings::AppSettings) -> Self {
        match mode {
            1 => Self::Agx,
            crate::camera_profile::TONE_MAPPER_MODE => {
                crate::camera_profile::for_image(path, settings)
                    .map(Self::Profile)
                    .unwrap_or(Self::Basic)
            }
            _ => Self::Basic,
        }
    }

    /// Scene-linear channel value that renders as (near) white.
    fn clip_level(&self) -> f32 {
        match self {
            Self::Basic | Self::Profile(_) => 1.0,
            Self::Agx => 3.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Face {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    /// Right eye, left eye, nose tip, right mouth corner, left mouth corner.
    pub landmarks: [[f32; 2]; 5],
}

impl Face {
    fn iou(&self, o: &Face) -> f32 {
        let ix = (self.x + self.w).min(o.x + o.w) - self.x.max(o.x);
        let iy = (self.y + self.h).min(o.y + o.h) - self.y.max(o.y);
        if ix <= 0.0 || iy <= 0.0 {
            return 0.0;
        }
        let inter = ix * iy;
        inter / (self.w * self.h + o.w * o.h - inter)
    }

    fn scaled(&self, s: f32) -> Face {
        let mut f = self.clone();
        f.x *= s;
        f.y *= s;
        f.w *= s;
        f.h *= s;
        for p in f.landmarks.iter_mut() {
            p[0] *= s;
            p[1] *= s;
        }
        f
    }
}

#[derive(Clone, Debug)]
pub struct MeteredFace {
    pub face: Face,
    /// Mean scene-linear RGB of the mid-tone skin between the eyes and the mouth.
    pub skin: [f32; 3],
    /// Exposure (EV) that puts this face at the target on its own.
    pub ev: f32,
    /// Local lift given by its mask, in EV.
    pub lift: f32,
}

#[derive(Clone, Debug)]
pub struct FaceExposure {
    pub faces: Vec<MeteredFace>,
    pub exposure_ev: f32,
    /// Shader units (slider / 120), <= 0.
    pub highlights: f32,
    pub image_width: u32,
    pub image_height: u32,
}

fn yunet() -> Option<&'static Mutex<Session>> {
    YUNET_SESSION
        .get_or_init(|| {
            let _ = ort::init().with_name("AI").commit();
            match Session::builder().and_then(|b| b.commit_from_memory(YUNET_MODEL)) {
                Ok(session) => Some(Mutex::new(session)),
                Err(e) => {
                    log::warn!("Face detector unavailable: {}", e);
                    None
                }
            }
        })
        .as_ref()
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Exposure and the smooth (detail-free) part of `apply_highlights_adjustment`.
#[inline]
fn expose(c: [f32; 3], ev: f32, highlights: f32) -> [f32; 3] {
    let g = ev.exp2();
    let c = [c[0] * g, c[1] * g, c[2] * g];
    if highlights >= 0.0 {
        return c;
    }
    let l = luma([c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)]);
    const PIVOT: f32 = 0.06;
    if l <= PIVOT {
        return c;
    }
    let k = -highlights;
    let delta = l - PIVOT;
    let compressed = delta / (1.0 + k * 2.2 * (delta / (1.0 + delta * 0.35)));
    let target = l + (PIVOT + compressed - l) * smoothstep(PIVOT, PIVOT + 0.22, l);
    let r = target / l;
    let mut out = [c[0] * r, c[1] * r, c[2] * r];
    if l > 1.0 {
        let blowout = smoothstep(1.0, 3.5, l) * k * 0.40;
        for v in out.iter_mut() {
            *v += (target - *v) * blowout;
        }
    }
    out
}

/// The shader's default tone mapping for a linear buffer, in place, to display values.
fn tone_map(buf: &mut Rgb32FImage, tm: &ToneMapper, is_raw: bool) {
    match tm {
        ToneMapper::Profile(profile) if is_raw => {
            for p in buf.pixels_mut() {
                p.0 = profile.render(p.0);
            }
        }
        ToneMapper::Agx => {
            let mut img = DynamicImage::ImageRgb32F(std::mem::take(buf));
            apply_cpu_agx_tonemap(&mut img);
            *buf = img.into_rgb32f();
        }
        ToneMapper::Basic | ToneMapper::Profile(_) => {
            for v in buf.iter_mut() {
                let s = srgb_encode(*v);
                *v = if is_raw {
                    let s = s.powf(1.0 / 1.1);
                    let curve = s * s * (3.0 - 2.0 * s);
                    s + (curve - s) * 0.75
                } else {
                    s
                };
            }
        }
    }
}

fn display_luma(c: [f32; 3], ev: f32, highlights: f32, tm: &ToneMapper, is_raw: bool) -> f32 {
    let e = expose(c, ev, highlights);
    let mut px = Rgb32FImage::from_raw(1, 1, e.to_vec()).unwrap();
    tone_map(&mut px, tm, is_raw);
    luma([px[(0, 0)][0], px[(0, 0)][1], px[(0, 0)][2]])
}

/// Exposure that renders `skin` at FACE_TARGET, by bisection (the chain is monotonic).
fn solve_face_ev(skin: [f32; 3], highlights: f32, tm: &ToneMapper, is_raw: bool) -> f32 {
    let (mut lo, mut hi) = (-4.0f32, 6.0f32);
    for _ in 0..28 {
        let mid = 0.5 * (lo + hi);
        if display_luma(skin, mid, highlights, tm, is_raw) < FACE_TARGET {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn clip_fraction(samples: &[[f32; 3]], ev: f32, highlights: f32, clip: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let n = samples
        .iter()
        .filter(|&&c| {
            let e = expose(c, ev, highlights);
            e[0].max(e[1]).max(e[2]) >= clip * 0.999
        })
        .count();
    n as f32 / samples.len() as f32
}

fn linear_rgb32f(image: &DynamicImage, max_dim: u32, is_raw: bool) -> Rgb32FImage {
    let mut img = downscale_f32_image(image, max_dim, max_dim).to_rgb32f();
    if !is_raw {
        for v in img.iter_mut() {
            *v = srgb_decode(*v);
        }
    }
    img
}

fn run_yunet(session: &Mutex<Session>, lin: &Rgb32FImage, gain: f32) -> Vec<Face> {
    let (w, h) = lin.dimensions();
    let mut input = Array4::<f32>::zeros((1, 3, YUNET_INPUT, YUNET_INPUT));
    for (x, y, p) in lin.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        if x >= YUNET_INPUT || y >= YUNET_INPUT {
            continue;
        }
        // YuNet wants BGR in 0..255.
        input[[0, 0, y, x]] = srgb_encode(p[2] * gain) * 255.0;
        input[[0, 1, y, x]] = srgb_encode(p[1] * gain) * 255.0;
        input[[0, 2, y, x]] = srgb_encode(p[0] * gain) * 255.0;
    }

    let result = (|| -> ort::Result<Vec<Face>> {
        let tensor = Tensor::from_array(input)?;
        let mut sess = session.lock().unwrap();
        let outputs = sess.run(ort::inputs![tensor])?;
        let mut faces = Vec::new();
        for stride in [8usize, 16, 32] {
            let get = |name: &str| -> ort::Result<Vec<f32>> {
                let (_, data) = outputs[format!("{}_{}", name, stride).as_str()]
                    .try_extract_tensor::<f32>()?;
                Ok(data.to_vec())
            };
            let (cls, obj, bbox, kps) = (get("cls")?, get("obj")?, get("bbox")?, get("kps")?);
            let cols = YUNET_INPUT / stride;
            let s = stride as f32;
            for i in 0..cls.len() {
                let score = (cls[i].clamp(0.0, 1.0) * obj[i].clamp(0.0, 1.0)).sqrt();
                if score < DETECTION_SCORE {
                    continue;
                }
                let (r, c) = ((i / cols) as f32, (i % cols) as f32);
                let b = &bbox[i * 4..i * 4 + 4];
                let (cx, cy) = ((c + b[0]) * s, (r + b[1]) * s);
                let (bw, bh) = (b[2].exp() * s, b[3].exp() * s);
                let k = &kps[i * 10..i * 10 + 10];
                let mut landmarks = [[0.0f32; 2]; 5];
                for (j, lm) in landmarks.iter_mut().enumerate() {
                    *lm = [(k[2 * j] + c) * s, (k[2 * j + 1] + r) * s];
                }
                faces.push(Face {
                    x: cx - bw / 2.0,
                    y: cy - bh / 2.0,
                    w: bw,
                    h: bh,
                    score,
                    landmarks,
                });
            }
        }
        Ok(faces)
    })();

    match result {
        Ok(faces) => faces
            .into_iter()
            .filter(|f| f.x + f.w > 0.0 && f.y + f.h > 0.0 && f.x < w as f32 && f.y < h as f32)
            .collect(),
        Err(e) => {
            log::warn!("Face detection failed: {}", e);
            Vec::new()
        }
    }
}

fn nms(mut faces: Vec<Face>) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for f in faces {
        if kept.iter().all(|k| k.iou(&f) < NMS_IOU) {
            kept.push(f);
        }
    }
    kept
}

/// Faces in `image`'s pixel coordinates. `image` is the pipeline's input:
/// scene-linear for raws, sRGB-encoded otherwise.
pub fn detect_faces(image: &DynamicImage, is_raw: bool) -> Vec<Face> {
    let Some(session) = yunet() else {
        return Vec::new();
    };
    let (w, h) = image.dimensions();
    let lin = linear_rgb32f(image, YUNET_INPUT as u32, is_raw);
    let back = w as f32 / lin.width() as f32;

    // Normalise so the bright end sits near white, then try again two stops
    // brighter: faces in a backlit frame are often too dark to be found.
    let mut lumas: Vec<f32> = lin.pixels().map(|p| luma([p[0], p[1], p[2]])).collect();
    lumas.sort_by(f32::total_cmp);
    let p99 = lumas[(lumas.len() as f32 * 0.99) as usize].max(1e-4);
    let gain = (0.9 / p99).clamp(0.5, 16.0);

    let mut all = run_yunet(session, &lin, gain);
    all.extend(run_yunet(session, &lin, gain * 4.0));
    let faces: Vec<Face> = nms(all).into_iter().map(|f| f.scaled(back)).collect();

    // Keep the faces that matter: not tiny, and not far smaller than the main one.
    let long = w.max(h) as f32;
    let biggest = faces.iter().map(|f| f.w).fold(0.0, f32::max);
    faces
        .into_iter()
        .filter(|f| f.w >= long * 0.015 && f.w >= biggest * 0.3)
        .take(6)
        .collect()
}

/// Mid-tone skin colour between the eyes and the mouth, or None if the patch is too small.
fn meter_skin(lin: &Rgb32FImage, face: &Face) -> Option<[f32; 3]> {
    let [re, le, _nose, rm, lm] = face.landmarks;
    let x0 = re[0].min(le[0]).min(rm[0]).min(lm[0]);
    let x1 = re[0].max(le[0]).max(rm[0]).max(lm[0]);
    let eye_y = 0.5 * (re[1] + le[1]);
    let mouth_y = 0.5 * (rm[1] + lm[1]);
    let y0 = eye_y + 0.2 * (mouth_y - eye_y);
    let y1 = mouth_y;
    let (w, h) = lin.dimensions();
    let (x0, x1) = (x0.max(0.0) as u32, (x1 as u32).min(w.saturating_sub(1)));
    let (y0, y1) = (y0.max(0.0) as u32, (y1 as u32).min(h.saturating_sub(1)));
    if x1 <= x0 + 2 || y1 <= y0 + 2 {
        return None;
    }

    let mut px: Vec<[f32; 3]> = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let p = lin.get_pixel(x, y);
            px.push([p[0].max(0.0), p[1].max(0.0), p[2].max(0.0)]);
        }
    }
    // Drop nostrils, shadow edges and specular shine.
    px.sort_by(|a, b| luma(*a).total_cmp(&luma(*b)));
    let (lo, hi) = (px.len() / 5, px.len() * 4 / 5);
    let mid = &px[lo..hi.max(lo + 1)];
    let n = mid.len() as f32;
    let sum = mid.iter().fold([0.0f32; 3], |s, c| [s[0] + c[0], s[1] + c[1], s[2] + c[2]]);
    Some([sum[0] / n, sum[1] / n, sum[2] / n])
}

pub fn analyze(image: &DynamicImage, is_raw: bool, tm: &ToneMapper) -> Option<FaceExposure> {
    let faces = detect_faces(image, is_raw);
    if faces.is_empty() {
        return None;
    }
    let (w, h) = image.dimensions();

    let meter = linear_rgb32f(image, 1600, is_raw);
    let ms = meter.width() as f32 / w as f32;
    let mut metered: Vec<(Face, [f32; 3])> = faces
        .into_iter()
        .filter_map(|f| meter_skin(&meter, &f.scaled(ms)).map(|skin| (f, skin)))
        // Skin is warm under any sane white balance; this drops faces found in
        // stonework, foliage and book covers.
        .filter(|(_, s)| s[0] >= s[1] * 0.85 && s[0] >= s[2] * 0.95)
        .collect();
    if metered.is_empty() {
        return None;
    }
    metered.sort_by(|a, b| (b.0.w * b.0.h).total_cmp(&(a.0.w * a.0.h)));

    // Pixels outside the faces decide how far the frame can be pushed.
    let small = linear_rgb32f(image, 512, is_raw);
    let ss = small.width() as f32 / w as f32;
    let boxes: Vec<Face> = metered.iter().map(|(f, _)| f.scaled(ss)).collect();
    let samples: Vec<[f32; 3]> = small
        .enumerate_pixels()
        .filter(|(x, y, _)| {
            let (x, y) = (*x as f32, *y as f32);
            !boxes.iter().any(|b| x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h)
        })
        .map(|(_, _, p)| [p[0], p[1], p[2]])
        .collect();

    let clip = tm.clip_level();
    let allowed = clip_fraction(&samples, 0.0, 0.0, clip) + CLIP_BUDGET;

    // The main face (area-weighted towards the biggest) sets the frame.
    let weight_sum: f32 = metered.iter().map(|(f, _)| f.w * f.h).sum();
    let mut skin = [0.0f32; 3];
    for (f, s) in &metered {
        let wgt = f.w * f.h / weight_sum;
        for i in 0..3 {
            skin[i] += s[i] * wgt;
        }
    }

    // Least highlight recovery that lets the face reach the target without
    // blowing the frame; if none does, the strongest and a local lift.
    let mut choice = None;
    for &hl in HIGHLIGHT_LEVELS.iter() {
        let ev_face = solve_face_ev(skin, -hl, tm, is_raw);
        let ok = clip_fraction(&samples, ev_face, -hl, clip) <= allowed;
        choice = Some((hl, ev_face));
        if ok {
            break;
        }
    }
    let (hl, ev_face) = choice.unwrap();
    let highlights = -hl;

    let mut exposure_ev = ev_face;
    if clip_fraction(&samples, ev_face, highlights, clip) > allowed {
        let (mut lo, mut hi) = (ev_face - MAX_FACE_LIFT, ev_face);
        if clip_fraction(&samples, lo, highlights, clip) > allowed {
            hi = lo;
        }
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if clip_fraction(&samples, mid, highlights, clip) <= allowed {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        exposure_ev = lo;
    }
    // Recovery that isn't needed at the final exposure only flattens the image.
    let highlights = if clip_fraction(&samples, exposure_ev, 0.0, clip) <= allowed {
        0.0
    } else {
        highlights
    };

    let faces = metered
        .into_iter()
        .map(|(face, skin)| {
            let ev = solve_face_ev(skin, highlights, tm, is_raw);
            let lift = (ev - exposure_ev).clamp(0.0, MAX_FACE_LIFT);
            let lift = if lift < MIN_FACE_LIFT { 0.0 } else { lift };
            MeteredFace { face, skin, ev, lift }
        })
        .collect();

    Some(FaceExposure {
        faces,
        exposure_ev,
        highlights,
        image_width: w,
        image_height: h,
    })
}

/// Feathered ellipse over the head, tilted with the eye line, in image pixels.
pub fn face_ellipse(face: &Face) -> (f32, f32, f32, f32, f32) {
    let [re, le, ..] = face.landmarks;
    let rotation = (le[1] - re[1]).atan2(le[0] - re[0]).to_degrees();
    let cx = face.x + face.w * 0.5;
    let cy = face.y + face.h * 0.55;
    (cx, cy, face.w * 1.0, face.h * 1.15, rotation)
}

fn face_lift_mask(i: usize, face: &MeteredFace) -> Value {
    let (cx, cy, rx, ry, rotation) = face_ellipse(&face.face);
    json!({
        "id": format!("{}-{}", AUTO_FACE_MASK_PREFIX, i),
        "name": if i == 0 { "Face lift".to_string() } else { format!("Face lift {}", i + 1) },
        "visible": true,
        "invert": false,
        "opacity": 100,
        "adjustments": { "exposure": face.lift * EXPOSURE_SLIDER_SCALE },
        "subMasks": [{
            "id": format!("{}-{}-radial", AUTO_FACE_MASK_PREFIX, i),
            "type": "radial",
            "name": "Radial",
            "visible": true,
            "invert": false,
            "opacity": 100,
            "mode": "additive",
            "parameters": {
                "centerX": cx,
                "centerY": cy,
                "radiusX": rx,
                "radiusY": ry,
                "rotation": rotation,
                "feather": 0.65
            }
        }]
    })
}

/// Overrides the generic auto adjustments with face-driven exposure. Adds
/// `faceLiftMasks` (to be merged into `masks` by the caller, replacing earlier
/// auto face masks) and `autoFaces` (count, for the UI).
pub fn apply_to_auto_json(result: &FaceExposure, auto: &mut Value) {
    let Some(map) = auto.as_object_mut() else {
        return;
    };
    map.insert("exposure".into(), json!(result.exposure_ev * EXPOSURE_SLIDER_SCALE));
    map.insert("highlights".into(), json!(result.highlights * HIGHLIGHTS_SLIDER_SCALE));
    // These were derived from a histogram metered for the whole frame; with the
    // face deciding brightness they'd only push it off again.
    for key in ["brightness", "whites", "blacks", "shadows"] {
        map.insert(key.into(), json!(0.0));
    }
    let masks: Vec<Value> = result
        .faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.lift > 0.0)
        .map(|(i, f)| face_lift_mask(i, f))
        .collect();
    map.insert("faceLiftMasks".into(), Value::Array(masks));
    map.insert("autoFaces".into(), json!(result.faces.len()));
}

/// Replaces earlier auto face-lift masks in a sidecar's adjustments with `new_masks`.
pub fn merge_face_masks(adjustments: &mut Value, new_masks: &[Value]) {
    let Some(map) = adjustments.as_object_mut() else {
        return;
    };
    let mut masks: Vec<Value> = map
        .get("masks")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|m| {
            !m.get("id")
                .and_then(|id| id.as_str())
                .is_some_and(|id| id.starts_with(AUTO_FACE_MASK_PREFIX))
        })
        .collect();
    masks.extend(new_masks.iter().cloned());
    map.insert("masks".into(), Value::Array(masks));
}

/// Approximate render of the auto result (exposure, face masks, smooth
/// highlights, tone mapper), for tuning without the GPU pipeline.
pub fn preview(
    image: &DynamicImage,
    is_raw: bool,
    tm: &ToneMapper,
    result: Option<&FaceExposure>,
    max_dim: u32,
) -> Rgb32FImage {
    let (w, _) = image.dimensions();
    let mut lin = linear_rgb32f(image, max_dim, is_raw);
    let s = lin.width() as f32 / w as f32;
    if let Some(r) = result {
        let ellipses: Vec<((f32, f32, f32, f32, f32), f32)> = r
            .faces
            .iter()
            .map(|f| {
                let (cx, cy, rx, ry, rot) = face_ellipse(&f.face);
                ((cx * s, cy * s, rx * s, ry * s, rot.to_radians()), f.lift)
            })
            .collect();
        for (x, y, p) in lin.enumerate_pixels_mut() {
            let mut ev = r.exposure_ev;
            for &((cx, cy, rx, ry, rot), lift) in &ellipses {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let (rdx, rdy) = (dx * rot.cos() + dy * rot.sin(), -dx * rot.sin() + dy * rot.cos());
                let d = ((rdx / rx).powi(2) + (rdy / ry).powi(2)).sqrt();
                let inner = 1.0 - 0.65;
                ev += lift * (1.0 - (d - inner) / (1.0 - inner)).clamp(0.0, 1.0);
            }
            let e = expose([p[0], p[1], p[2]], ev, r.highlights);
            *p = image::Rgb(e);
        }
    }
    tone_map(&mut lin, tm, is_raw);
    lin
}

/// Tuning harness (examples/face_auto.rs): loads `path` the way the editor
/// does (with camera profiles on), runs the auto analysis and writes a
/// before/after preview PNG.
#[doc(hidden)]
pub fn dev_report(path: &str, out_png: &str, mode: u32) -> Result<Value, String> {
    let settings = crate::app_settings::AppSettings {
        use_camera_profiles: Some(true),
        ..Default::default()
    };
    let tm = &ToneMapper::new(mode, path, &settings);
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let image = crate::image_loader::load_base_image_from_bytes(&bytes, path, false, &settings, None)
        .map_err(|e| e.to_string())?;
    let is_raw = crate::formats::is_raw_file(path);

    let t0 = std::time::Instant::now();
    let mut auto = crate::image_processing::auto_results_to_json(
        &crate::image_processing::perform_auto_analysis(&image),
    );
    let result = analyze(&image, is_raw, tm);
    let elapsed = t0.elapsed().as_millis();
    if let Some(r) = &result {
        apply_to_auto_json(r, &mut auto);
    }

    let before = preview(&image, is_raw, tm, None, 900);
    let empty = FaceExposure {
        faces: vec![],
        exposure_ev: 0.0,
        highlights: 0.0,
        image_width: 0,
        image_height: 0,
    };
    let after = preview(&image, is_raw, tm, Some(result.as_ref().unwrap_or(&empty)), 900);
    let (pw, ph) = before.dimensions();
    let mut out = image::RgbImage::new(pw * 2 + 8, ph);
    for (x, y, p) in before.enumerate_pixels() {
        out.put_pixel(x, y, image::Rgb(p.0.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)));
    }
    for (x, y, p) in after.enumerate_pixels() {
        out.put_pixel(x + pw + 8, y, image::Rgb(p.0.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)));
    }
    if let Some(r) = &result {
        let s = pw as f32 / image.width() as f32;
        for f in &r.faces {
            let b = f.face.scaled(s);
            let (x0, y0) = (b.x.max(0.0) as u32, b.y.max(0.0) as u32);
            let (x1, y1) = (((b.x + b.w) as u32).min(pw - 1), ((b.y + b.h) as u32).min(ph - 1));
            for x in x0..=x1 {
                out.put_pixel(x, y0, image::Rgb([0, 255, 0]));
                out.put_pixel(x, y1, image::Rgb([0, 255, 0]));
            }
            for y in y0..=y1 {
                out.put_pixel(x0, y, image::Rgb([0, 255, 0]));
                out.put_pixel(x1, y, image::Rgb([0, 255, 0]));
            }
        }
    }
    out.save(out_png).map_err(|e| e.to_string())?;

    Ok(json!({
        "path": path,
        "ms": elapsed,
        "faces": result.as_ref().map(|r| r.faces.iter().map(|f| json!({
            "box": [f.face.x.round(), f.face.y.round(), f.face.w.round(), f.face.h.round()],
            "score": f.face.score,
            "skinLuma": luma(f.skin),
            "ev": f.ev,
            "lift": f.lift,
        })).collect::<Vec<_>>()),
        "exposureEv": result.as_ref().map(|r| r.exposure_ev),
        "highlights": result.as_ref().map(|r| r.highlights),
        "auto": auto,
    }))
}
