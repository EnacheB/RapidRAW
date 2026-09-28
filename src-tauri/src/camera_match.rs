//! Match camera: sets Exposure and Shadows so the camera profile's rendering of a
//! RAW has the same tones as the camera's own JPEG preview embedded in the file.
//!
//! The profile is the camera's fixed look; what the camera JPEG adds on top is
//! per shot (its metering and shadow lift), so it becomes slider values here.
//! Brightness histograms are compared rather than pixels, so crop, orientation
//! and lens correction don't matter (RawTherapee's auto-matched curve works the
//! same way). Where the JPEG clipped there is nothing to match, so that part is
//! left out.

use image::DynamicImage;
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::camera_profile::CameraProfile;

const EXPOSURE_SLIDER_SCALE: f32 = 0.8;
const SHADOWS_SLIDER_SCALE: f32 = 120.0;
/// Pixels of the RAW rendered per candidate; enough for stable percentiles.
const SAMPLES: usize = 4096;
/// Percentiles compared, skipping the noisy deepest blacks.
const Q_LO: f32 = 0.02;
const QUANTILES: usize = 96;

#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub exposure_ev: f32,
    /// Shader units (slider / 120).
    pub shadows: f32,
    /// Mean L* difference left over, for the harness.
    pub residual: f32,
}

#[inline]
fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[inline]
fn srgb_decode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// CIE L* of sRGB-encoded display values.
fn lightness(srgb: [f32; 3]) -> f32 {
    let y = luma(srgb.map(srgb_decode));
    let f = if y > 0.008856 { y.cbrt() } else { 7.787 * y + 16.0 / 116.0 };
    116.0 * f - 16.0
}

/// Exposure, then the Shadows slider as the shader's `apply_tonal_adjustments`
/// runs it, with the pixel as its own neighbourhood (the detail term is 1).
fn sliders(c: [f32; 3], ev: f32, sh: f32) -> [f32; 3] {
    let g = ev.exp2();
    let rgb = c.map(|v| v * g);
    if sh == 0.0 {
        return rgb;
    }
    let l = luma(rgb.map(|v| v.max(0.0))).max(1e-4);
    let t = l.powf(0.4545);
    let lift = sh * t * (1.0 - t).max(0.0).powf(4.5);
    let amount = lift.max(0.0);
    let curved = (t + lift).max(0.0);
    let stretch_focus = 1.0 - smoothstep(0.1, 0.85, t);
    let fin = (0.2 + (curved - 0.2) * (1.0 + amount * stretch_focus * 4.4)).max(0.0);
    let ratio = fin.powf(2.2) / l;
    let out = rgb.map(|v| v * ratio);
    if ratio > 1.0 {
        let y = luma(out);
        let boost = ((ratio - 1.0) * 0.05).clamp(0.0, 0.12);
        return out.map(|v| v + (y - v) * boost);
    }
    out
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn percentiles(mut l: Vec<f32>) -> Vec<f32> {
    l.sort_by(f32::total_cmp);
    let n = l.len() - 1;
    (0..QUANTILES)
        .map(|i| {
            let q = Q_LO + (1.0 - Q_LO) * i as f32 / QUANTILES as f32;
            l[(q * n as f32) as usize]
        })
        .collect()
}

/// The camera's JPEG preview inside the RAW.
pub fn embedded_jpeg(path: &str) -> Result<DynamicImage, String> {
    let (source, _) = crate::file_management::parse_virtual_path(path);
    let bytes = std::fs::read(&source).map_err(|e| e.to_string())?;
    crate::image_loader::largest_tiff_jpeg_preview(&bytes)
        .ok_or_else(|| "This RAW has no embedded camera JPEG to match".to_string())
}

/// Fits Exposure and Shadows so `profile` renders `image` (scene-linear working
/// space, as decoded) with the same tones as `reference` (the camera JPEG).
pub fn fit(image: &DynamicImage, reference: &DynamicImage, profile: &CameraProfile) -> Match {
    let reference = reference.thumbnail(400, 400).to_rgb32f();
    let clipped = reference
        .pixels()
        .filter(|p| p.0.iter().any(|&v| v >= 0.98))
        .count() as f32
        / reference.pixels().len() as f32;
    let target = percentiles(reference.pixels().map(|p| lightness(p.0)).collect());
    // Percentiles the JPEG clipped carry no information about the exposure.
    let usable = QUANTILES
        .min(((1.0 - clipped - 0.01 - Q_LO) / (1.0 - Q_LO) * QUANTILES as f32).max(8.0) as usize);

    let small = crate::image_processing::downscale_f32_image(image, 400, 400).to_rgb32f();
    let step = (small.pixels().len() / SAMPLES).max(1);
    let samples: Vec<[f32; 3]> = small.pixels().step_by(step).map(|p| p.0).collect();

    let loss = |ev: f32, sh: f32| -> f32 {
        let rendered = samples
            .iter()
            .map(|&c| lightness(profile.render(sliders(c, ev, sh))))
            .collect();
        let got = percentiles(rendered);
        got[..usable]
            .iter()
            .zip(&target[..usable])
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / usable as f32
    };
    let search = |evs: Vec<f32>, shs: Vec<f32>| -> Match {
        let grid: Vec<(f32, f32)> = shs
            .iter()
            .flat_map(|&sh| evs.iter().map(move |&ev| (ev, sh)))
            .collect();
        grid.into_par_iter()
            .map(|(ev, sh)| Match { exposure_ev: ev, shadows: sh, residual: loss(ev, sh) })
            .min_by(|a, b| a.residual.total_cmp(&b.residual))
            .unwrap()
    };
    let range = |lo: f32, hi: f32, step: f32| -> Vec<f32> {
        (0..=((hi - lo) / step).round() as usize)
            .map(|i| lo + i as f32 * step)
            .collect()
    };

    // Shadows from -50 to +100 on the slider; coarse grid, then refine.
    let (sh_lo, sh_hi) = (-50.0 / SHADOWS_SLIDER_SCALE, 100.0 / SHADOWS_SLIDER_SCALE);
    let sh_step = 10.0 / SHADOWS_SLIDER_SCALE;
    let coarse = search(range(-3.0, 4.0, 0.1), range(sh_lo, sh_hi, sh_step));
    let fine_sh = range(-sh_step, sh_step, sh_step / 5.0)
        .into_iter()
        .map(|d| (coarse.shadows + d).clamp(sh_lo, sh_hi))
        .collect();
    search(range(coarse.exposure_ev - 0.1, coarse.exposure_ev + 0.1, 0.01), fine_sh)
}

/// Slider values for `m`. The fit assumes the other tone sliders are neutral and
/// the profile is the tone mapper, so it sets those too.
pub fn to_adjustments(m: &Match) -> Value {
    json!({
        "toneMapper": crate::camera_profile::TONE_MAPPER_ID,
        "exposure": (m.exposure_ev * EXPOSURE_SLIDER_SCALE * 100.0).round() / 100.0,
        "shadows": (m.shadows * SHADOWS_SLIDER_SCALE).round(),
        "brightness": 0.0,
        "contrast": 0.0,
        "highlights": 0.0,
        "whites": 0.0,
        "blacks": 0.0,
    })
}

fn profile_for(path: &str, settings: &crate::app_settings::AppSettings) -> Result<std::sync::Arc<CameraProfile>, String> {
    crate::camera_profile::for_image(path, settings).ok_or_else(|| {
        "No camera profile for this photo (Settings: Camera Profiles must be on)".to_string()
    })
}

#[tauri::command]
pub fn match_camera(
    state: tauri::State<crate::AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Value, String> {
    let (image, path, is_raw) = {
        let guard = state.original_image.lock().unwrap();
        let loaded = guard.as_ref().ok_or("No image loaded")?;
        (loaded.image.clone(), loaded.path.clone(), loaded.is_raw)
    };
    if !is_raw {
        return Err("Match camera needs a RAW file".into());
    }
    let settings = crate::app_settings::load_settings(app_handle).unwrap_or_default();
    let profile = profile_for(&path, &settings)?;
    let reference = embedded_jpeg(&path)?;
    let m = fit(&image, &reference, &profile);
    log::info!("Match camera {}: {:?}", path, m);
    Ok(to_adjustments(&m))
}

/// Tuning harness (examples/camera_match.rs): fits `path` as the editor would.
#[doc(hidden)]
pub fn dev_fit(path: &str) -> Result<Match, String> {
    let settings = crate::app_settings::AppSettings {
        use_camera_profiles: Some(true),
        ..Default::default()
    };
    let profile = profile_for(path, &settings)?;
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let image = crate::image_loader::load_base_image_from_bytes(&bytes, path, false, &settings, None)
        .map_err(|e| e.to_string())?;
    let reference = embedded_jpeg(path)?;
    Ok(fit(&image, &reference, &profile))
}
