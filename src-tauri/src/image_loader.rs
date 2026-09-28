use crate::Cursor;
use crate::app_settings::{AppSettings, load_settings};
use crate::app_state::{AppState, LoadedImage};
use crate::brush_patches::{Image, Rendered};
use crate::exif_processing;
use crate::file_management::{parse_virtual_path, read_file_mapped};
use crate::formats::is_raw_file;
use crate::image_processing::ImageMetadata;
use crate::image_processing::{
    apply_orientation, apply_srgb_to_linear, remove_raw_artifacts_and_enhance,
};
use crate::mask_generation::{MaskDefinition, SubMask, generate_mask_bitmap};
use crate::white_balance::WhiteBalance;
use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose};
use exif::{Reader as ExifReader, Tag};
use image::{DynamicImage, GenericImageView, ImageReader, imageops};
use rawler::Orientation;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::panic;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

#[derive(serde::Serialize)]
pub struct LoadImageResult {
    pub width: u32,
    pub height: u32,
    pub metadata: ImageMetadata,
    pub exif: HashMap<String, String>,
    pub is_raw: bool,
    pub as_shot_white_balance: WhiteBalance,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PatchMaskInfo {
    id: String,
    name: String,
    #[serde(default)]
    invert: bool,
    #[serde(default)]
    sub_masks: Vec<SubMask>,
}

fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            let x = i as f32 / 255.0;
            *v = if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            };
        }
        lut
    })
}

pub fn load_and_composite(
    base_image: &[u8],
    path: &str,
    adjustments: &Value,
    use_fast_raw_dev: bool,
    settings: &AppSettings,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let base_image =
        load_base_image_from_bytes(base_image, path, use_fast_raw_dev, settings, cancel_token)?;
    composite_patches_on_image(&base_image, adjustments)
}

pub fn load_base_image_from_bytes(
    bytes: &[u8],
    path_for_ext_check: &str,
    use_fast_raw_dev: bool,
    settings: &AppSettings,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let highlight_compression = settings.raw_highlight_compression.unwrap_or(2.5);
    let linear_mode = settings.linear_raw_mode.clone();
    let use_camera_profiles = settings.use_camera_profiles.unwrap_or(false);
    let color_nr_setting = settings.raw_preprocessing_color_nr.unwrap_or(0.5);
    let color_nr_amount = if color_nr_setting <= 0.0 {
        0.0
    } else {
        let x = color_nr_setting.clamp(0.01, 1.0);
        (12.0 / x - 10.0).max(0.1)
    };
    let sharpening_amount = settings.raw_preprocessing_sharpening.unwrap_or(0.35);
    let apply_to_non_raws = settings.apply_preprocessing_to_non_raws.unwrap_or(false);

    crate::exif_processing::persist_exif_if_missing(
        Path::new(path_for_ext_check),
        path_for_ext_check,
        bytes,
    );

    if is_raw_file(path_for_ext_check)
        && !use_fast_raw_dev
        && settings.use_apple_raw9.unwrap_or(false)
    {
        if let Some((tracker, generation)) = &cancel_token
            && tracker.load(Ordering::SeqCst) != *generation
        {
            return Err(anyhow!("Load cancelled"));
        }

        match crate::apple_raw::develop_raw9(
            bytes,
            path_for_ext_check,
            &crate::apple_raw::Raw9Options::for_loading(),
        ) {
            Ok(image) => return Ok(image),
            Err(e) => log::warn!(
                "Apple RAW 9 unavailable for '{}', falling back to rawler: {}",
                path_for_ext_check,
                e
            ),
        }
    }

    if is_raw_file(path_for_ext_check) {
        match panic::catch_unwind(move || {
            crate::raw_processing::develop_raw_image(
                bytes,
                use_fast_raw_dev,
                highlight_compression,
                linear_mode,
                use_camera_profiles,
                cancel_token,
            )
        }) {
            Ok(Ok(mut image)) => {
                if !use_fast_raw_dev && (color_nr_amount > 0.0 || sharpening_amount > 0.0) {
                    let start = Instant::now();
                    remove_raw_artifacts_and_enhance(
                        &mut image,
                        color_nr_amount,
                        sharpening_amount,
                    );
                    let duration = start.elapsed();
                    log::info!(
                        "Raw enhancing for '{}' took {:?}",
                        path_for_ext_check,
                        duration
                    );
                }
                Ok(image)
            }
            Ok(Err(e)) => {
                let classified = classify_raw_develop_error(path_for_ext_check, e);

                if classified.to_string().contains("Load cancelled") {
                    return Err(classified);
                }

                log::warn!(
                    "Error developing RAW file '{}': {}",
                    path_for_ext_check,
                    classified
                );
                if let Some(preview) = safe_embedded_preview_fallback(bytes, path_for_ext_check) {
                    log::warn!(
                        "Using embedded preview fallback for '{}' ({}x{})",
                        path_for_ext_check,
                        preview.width(),
                        preview.height()
                    );

                    return Ok(linearize_embedded_preview(preview));
                }
                Err(classified)
            }
            Err(_) => {
                log::error!("Panic while processing RAW file: {}", path_for_ext_check);
                if let Some(preview) = safe_embedded_preview_fallback(bytes, path_for_ext_check) {
                    log::warn!(
                        "Using embedded preview fallback for '{}' after RAW decoder panic ({}x{})",
                        path_for_ext_check,
                        preview.width(),
                        preview.height()
                    );

                    return Ok(linearize_embedded_preview(preview));
                }
                Err(anyhow!(
                    "Failed to process RAW file: {}",
                    path_for_ext_check
                ))
            }
        }
    } else {
        let mut image = load_image_with_orientation(bytes, cancel_token)?;

        if apply_to_non_raws
            && !use_fast_raw_dev
            && (color_nr_amount > 0.0 || sharpening_amount > 0.0)
        {
            let start = Instant::now();
            remove_raw_artifacts_and_enhance(&mut image, color_nr_amount, sharpening_amount);
            let duration = start.elapsed();
            log::info!(
                "Enhancing non-RAW '{}' took {:?}",
                path_for_ext_check,
                duration
            );
        }

        Ok(image)
    }
}

fn classify_raw_develop_error(path: &str, err: anyhow::Error) -> anyhow::Error {
    let error_text = err.to_string();
    let lowered = error_text.to_ascii_lowercase();
    let unsupported_compression =
        lowered.contains("nef compression") && lowered.contains("not supported");

    if unsupported_compression {
        return anyhow!(
            "Unsupported RAW compression format for '{}'. Original error: {}",
            path,
            error_text
        );
    }

    err
}

pub(crate) fn largest_tiff_jpeg_preview(buf: &[u8]) -> Option<DynamicImage> {
    let le = match buf.get(..4)? {
        [0x49, 0x49, 0x2A, 0x00] => true,
        [0x4D, 0x4D, 0x00, 0x2A] => false,
        _ => return None,
    };
    let rd16 = |o: usize| -> Option<u64> {
        let b: [u8; 2] = buf.get(o..o + 2)?.try_into().ok()?;
        Some(if le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        } as u64)
    };
    let rd32 = |o: usize| -> Option<u64> {
        let b: [u8; 4] = buf.get(o..o + 4)?.try_into().ok()?;
        Some(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        } as u64)
    };

    let mut candidates: Vec<(u64, u64)> = Vec::new();
    let mut queue: Vec<u64> = vec![rd32(4)?];
    let mut seen = HashMap::new();

    while let Some(ifd) = queue.pop() {
        if seen.insert(ifd, ()).is_some() || seen.len() > 64 {
            continue;
        }
        let Some(n) = rd16(ifd as usize) else {
            continue;
        };

        let mut compression: u64 = 0;
        let mut strip: Option<(u64, u64)> = None;
        let mut old_jpeg: Option<(u64, u64)> = None;

        for i in 0..n {
            let e = ifd as usize + 2 + (i as usize) * 12;
            let (Some(tag), Some(count), Some(val)) = (rd16(e), rd32(e + 4), rd32(e + 8)) else {
                continue;
            };
            match tag {
                259 => compression = val,
                273 if count == 1 => strip = Some((val, strip.map_or(0, |s| s.1))),
                279 if count == 1 => strip = strip.map(|s| (s.0, val)).or(Some((0, val))),
                513 => old_jpeg = Some((val, old_jpeg.map_or(0, |s| s.1))),
                514 => old_jpeg = old_jpeg.map(|s| (s.0, val)).or(Some((0, val))),
                330 => {
                    if count == 1 {
                        queue.push(val);
                    } else {
                        for j in 0..count.min(8) {
                            if let Some(p) = rd32(val as usize + (j as usize) * 4) {
                                queue.push(p);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if matches!(compression, 6 | 7)
            && let Some(s) = strip
        {
            candidates.push(s);
        }
        if let Some(oj) = old_jpeg {
            candidates.push(oj);
        }
        if let Some(next) = rd32(ifd as usize + 2 + (n as usize) * 12)
            && next != 0
        {
            queue.push(next);
        }
    }

    candidates.sort_by_key(|&(_, len)| std::cmp::Reverse(len));

    for (off, len) in candidates {
        if let Some(bytes) = buf.get(off as usize..(off + len) as usize)
            && let Ok(img) = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg)
        {
            return Some(img);
        }
    }

    None
}

fn embedded_preview_fallback(bytes: &[u8]) -> Option<DynamicImage> {
    let Some(img) = largest_tiff_jpeg_preview(bytes) else {
        return crate::raw_processing::extract_embedded_preview(bytes);
    };

    let orientation = ExifReader::new()
        .read_from_container(&mut Cursor::new(bytes))
        .ok()
        .and_then(|exif| {
            exif.get_field(Tag::Orientation, exif::In::PRIMARY)?
                .value
                .get_uint(0)
        });

    Some(match orientation {
        Some(o) if o > 1 => apply_orientation(img, Orientation::from_u16(o as u16)),
        _ => img,
    })
}

pub fn safe_embedded_preview_fallback(bytes: &[u8], path: &str) -> Option<DynamicImage> {
    match panic::catch_unwind(panic::AssertUnwindSafe(|| embedded_preview_fallback(bytes))) {
        Ok(preview) => preview,
        Err(_) => {
            log::warn!("Embedded RAW preview extraction panicked for '{}'", path);
            None
        }
    }
}

fn linearize_embedded_preview(preview: DynamicImage) -> DynamicImage {
    let preview = DynamicImage::ImageRgb32F(preview.to_rgb32f());
    let mut linear_preview = apply_srgb_to_linear(preview).into_rgb32f();
    for pixel in linear_preview.pixels_mut() {
        pixel[0] *= 0.4;
        pixel[1] *= 0.4;
        pixel[2] *= 0.4;
    }
    DynamicImage::ImageRgb32F(linear_preview)
}

pub fn load_image_with_orientation(
    bytes: &[u8],
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let check_cancel = || -> Result<()> {
        if let Some((tracker, generation)) = &cancel_token
            && tracker.load(Ordering::SeqCst) != *generation
        {
            return Err(anyhow!("Load cancelled"));
        }
        Ok(())
    };

    let cursor = Cursor::new(bytes);
    let mut reader = ImageReader::new(cursor.clone())
        .with_guessed_format()
        .context("Failed to guess image format")?;

    reader.no_limits();

    check_cancel()?;

    let image = reader.decode().context("Failed to decode image")?;
    check_cancel()?;

    let oriented_image = {
        let exif_reader = ExifReader::new();
        if let Ok(exif) = exif_reader.read_from_container(&mut cursor.clone()) {
            if let Some(orientation) = exif
                .get_field(Tag::Orientation, exif::In::PRIMARY)
                .and_then(|f| f.value.get_uint(0))
            {
                check_cancel()?;
                apply_orientation(image, Orientation::from_u16(orientation as u16))
            } else {
                image
            }
        } else {
            image
        }
    };

    Ok(DynamicImage::ImageRgb32F(oriented_image.to_rgb32f()))
}

/// AI inpainting models take sRGB-encoded 0–1 images, but a RAW's base image is
/// linear light and can exceed 1.0. `encode_highlights` keeps values up to `knee`
/// as they are and rolls everything above it smoothly into (`knee`, 1), with slope
/// 1 at the knee; `decode_highlights` inverts it, so bright areas a model fills
/// come back as bright as their surroundings.
pub fn encode_highlights(x: f32, knee: f32) -> f32 {
    if x <= knee {
        return x;
    }
    let t = (x - knee) / (1.0 - knee);
    knee + (1.0 - knee) * t / (1.0 + t)
}

pub fn decode_highlights(y: f32, knee: f32) -> f32 {
    if y <= knee {
        return y;
    }
    // Cap just below 1, where the roll-off reaches infinity.
    let s = ((y - knee) / (1.0 - knee)).min(0.99);
    knee + (1.0 - knee) * s / (1.0 - s)
}

pub fn composite_patches_on_image(
    base_image: &DynamicImage,
    current_adjustments: &Value,
) -> Result<DynamicImage> {
    let patches_val = match current_adjustments.get("aiPatches") {
        Some(val) => val,
        None => return Ok(base_image.clone()),
    };

    let patches_arr = match patches_val.as_array() {
        Some(arr) if !arr.is_empty() => arr,
        _ => return Ok(base_image.clone()),
    };

    let visible_patches: Vec<&Value> = patches_arr
        .par_iter()
        .filter(|patch_obj| {
            let is_visible = patch_obj
                .get("visible")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            if !is_visible {
                return false;
            }
            patch_obj.get("patchData").is_some_and(|data| {
                data.get("sourceOffset").is_some()
                    || patch_obj
                        .get("subMasks")
                        .is_some_and(crate::brush_patches::is_brush_patch)
                    || data
                        .get("color")
                        .and_then(|color| color.as_str())
                        .is_some_and(|s| !s.is_empty())
            })
        })
        .collect();

    if visible_patches.is_empty() {
        return Ok(base_image.clone());
    }

    let (base_w, base_h) = base_image.dimensions();

    /// A patch ready to apply. Only AI inpainting stores pixels; the other tools
    /// are rendered from their recipe on the float image as it is when they apply.
    enum DecodedPatch<'a> {
        /// AI inpainting.
        Stored(Rendered),
        /// Heal or Clone: the spot under `mask` copies from `source_offset` away.
        Spot {
            origin: (usize, usize),
            mask: image::GrayImage,
            source_offset: (i32, i32),
            heal: bool,
        },
        /// Liquify or Retouch, which make their own mask from the strokes.
        Brush {
            sub_masks: &'a Value,
            linear: bool,
        },
    }

    let lut = srgb_to_linear_lut();
    let decoded_patches: Result<Vec<DecodedPatch>> = visible_patches
        .par_iter()
        .map(|patch_obj| {
            let patch_data = patch_obj.get("patchData").context("Missing patchData")?;
            let offset_x = patch_data
                .get("offsetX")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let offset_y = patch_data
                .get("offsetY")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let is_cropped = offset_x.is_some() && offset_y.is_some();

            let is_srgb_encoded = patch_data
                .get("isSrgbEncoded")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Patches from before Liquify and Retouch were recipes also carry stored
            // pixels; those are ignored and re-rendered from the strokes.
            if let Some(sub_masks) = patch_obj
                .get("subMasks")
                .filter(|sm| crate::brush_patches::is_brush_patch(sm))
            {
                return Ok(DecodedPatch::Brush {
                    sub_masks,
                    linear: is_srgb_encoded,
                });
            }

            let mask_bitmap = if let Some(mask_b64) = patch_data
                .get("mask")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                let mask_bytes = general_purpose::STANDARD.decode(mask_b64)?;
                let mask_img = image::load_from_memory(&mask_bytes)?.to_luma8();
                if !is_cropped && (mask_img.width() != base_w || mask_img.height() != base_h) {
                    imageops::resize(&mask_img, base_w, base_h, imageops::FilterType::Lanczos3)
                } else {
                    mask_img
                }
            } else {
                let patch_info: PatchMaskInfo = serde_json::from_value((*patch_obj).clone())
                    .context("Failed to deserialize patch info for mask generation")?;

                let mask_def = MaskDefinition {
                    id: patch_info.id,
                    name: patch_info.name,
                    visible: true,
                    invert: patch_info.invert,
                    opacity: 100.0,
                    adjustments: Value::Null,
                    sub_masks: patch_info.sub_masks,
                };

                let orientation_steps = current_adjustments
                    .get("orientationSteps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u8;
                let (trans_w, trans_h) = if orientation_steps % 2 == 1 {
                    (base_h, base_w)
                } else {
                    (base_w, base_h)
                };

                let mut gen_mask =
                    generate_mask_bitmap(&mask_def, trans_w, trans_h, 1.0, (0.0, 0.0), None)
                        .context("Failed to generate mask from sub_masks for compositing")?;

                gen_mask =
                    crate::image_processing::inverse_transform_mask(gen_mask, current_adjustments);

                if let (Some(ox), Some(oy)) = (offset_x, offset_y) {
                    let w = patch_data
                        .get("width")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .unwrap_or(base_w);
                    let h = patch_data
                        .get("height")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .unwrap_or(base_h);
                    let crop_w = w.min(base_w.saturating_sub(ox));
                    let crop_h = h.min(base_h.saturating_sub(oy));
                    gen_mask = imageops::crop_imm(&gen_mask, ox, oy, crop_w, crop_h).to_image();
                }
                gen_mask
            };

            let origin = (offset_x.unwrap_or(0) as usize, offset_y.unwrap_or(0) as usize);
            if let Some(offset) = patch_data.get("sourceOffset") {
                return Ok(DecodedPatch::Spot {
                    origin,
                    mask: mask_bitmap,
                    source_offset: serde_json::from_value(offset.clone())
                        .context("Invalid sourceOffset")?,
                    heal: patch_data.get("heal").and_then(|v| v.as_bool()).unwrap_or(false),
                });
            }

            let color_b64 = patch_data
                .get("color")
                .and_then(|v| v.as_str())
                .context("Missing color data")?;
            let color_bytes = general_purpose::STANDARD.decode(color_b64)?;
            let color = image::load_from_memory(&color_bytes)?.to_rgb8();
            let (patch_w, patch_h) = color.dimensions();
            let color = if !is_cropped && (base_w != patch_w || base_h != patch_h) {
                imageops::resize(&color, base_w, base_h, imageops::FilterType::Lanczos3)
            } else {
                color
            };
            let highlight_knee = patch_data
                .get("highlightKnee")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32);
            let colors = color
                .pixels()
                .map(|p| {
                    if !is_srgb_encoded {
                        return p.0.map(|v| v as f32 / 255.0);
                    }
                    p.0.map(|v| match highlight_knee {
                        Some(knee) => decode_highlights(lut[v as usize], knee),
                        None => lut[v as usize],
                    })
                })
                .collect();
            Ok(DecodedPatch::Stored(Rendered {
                origin,
                mask: mask_bitmap,
                colors,
            }))
        })
        .collect();

    let decoded_patches = decoded_patches?;

    // Patches apply in order, each on top of the ones before it, in linear float RGBA.
    let mut image = base_image.to_rgba32f();
    let width = base_w as usize;

    for patch in decoded_patches {
        let source = Image {
            pixels: image.as_raw(),
            width,
            height: base_h as usize,
        };
        let rendered = match patch {
            DecodedPatch::Stored(rendered) => Some(rendered),
            DecodedPatch::Spot {
                origin,
                mask,
                source_offset,
                heal,
            } => Some(crate::heal_source::render_spot(
                &source,
                mask,
                origin,
                source_offset,
                heal,
            )),
            DecodedPatch::Brush { sub_masks, linear } => {
                crate::brush_patches::render(&source, sub_masks, linear)
            }
        };
        let Some(Rendered {
            origin,
            mask,
            colors,
        }) = rendered
        else {
            continue;
        };

        let patch_w = mask.width() as usize;
        let crop_w = patch_w.min(width.saturating_sub(origin.0));
        let crop_h = (mask.height() as usize).min((base_h as usize).saturating_sub(origin.1));
        if crop_w == 0 || crop_h == 0 {
            continue;
        }

        let mask = mask.as_raw();
        image
            .par_chunks_mut(width * 4)
            .skip(origin.1)
            .take(crop_h)
            .enumerate()
            .for_each(|(py, row)| {
                for px in 0..crop_w {
                    let i = py * patch_w + px;
                    let alpha = mask[i] as f32 / 255.0;
                    if alpha > 0.0 {
                        let dst = &mut row[(origin.0 + px) * 4..(origin.0 + px) * 4 + 3];
                        for c in 0..3 {
                            dst[c] = colors[i][c] * alpha + dst[c] * (1.0 - alpha);
                        }
                    }
                }
            });
    }

    Ok(DynamicImage::ImageRgba32F(image))
}

#[tauri::command]
pub fn is_image_cached(path: String, state: tauri::State<'_, AppState>) -> bool {
    let (source_path, _) = parse_virtual_path(&path);
    let source_path_str = source_path.to_string_lossy().to_string();
    state
        .decoded_image_cache
        .lock()
        .unwrap()
        .get(&source_path_str)
        .is_some()
}

#[tauri::command]
pub async fn load_image(
    path: String,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<LoadImageResult, String> {
    let my_generation = state.load_image_generation.fetch_add(1, Ordering::SeqCst) + 1;
    let generation_tracker = state.load_image_generation.clone();
    let cancel_token = Some((generation_tracker.clone(), my_generation));

    {
        *state
            .original_image
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *state
            .cached_preview
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *state
            .gpu_image_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *state
            .full_warped_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        crate::cache_utils::clear_preview_stage_caches(&state);

        state
            .mask_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        state
            .patch_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        state
            .geometry_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        *state
            .denoise_result
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *state.hdr_result.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *state
            .panorama_result
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
    }

    let (source_path, sidecar_path) = parse_virtual_path(&path);
    let source_path_str = source_path.to_string_lossy().to_string();

    let metadata: ImageMetadata = crate::exif_processing::load_sidecar(&sidecar_path);

    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let path_clone = source_path_str.clone();

    let cached_data = state
        .decoded_image_cache
        .lock()
        .unwrap()
        .get(&source_path_str);

    let (pristine_arc, exif_data) = if let Some((cached_img, cached_exif)) = cached_data {
        (cached_img, cached_exif)
    } else {
        if crate::file_management::is_cloud_placeholder(&source_path) {
            return Err(format!(
                "'{}' is stored in iCloud and hasn't been downloaded yet. Download it in Finder, then try again.",
                source_path_str
            ));
        }

        let (pristine_img, exif_data_loaded) = tokio::task::spawn_blocking(move || {
            if generation_tracker.load(Ordering::SeqCst) != my_generation {
                return Err("Load cancelled".to_string());
            }

            let result: Result<(DynamicImage, HashMap<String, String>), String> =
                (|| match read_file_mapped(Path::new(&path_clone)) {
                    Ok(mmap) => {
                        if generation_tracker.load(Ordering::SeqCst) != my_generation {
                            return Err("Load cancelled".to_string());
                        }

                        let img = load_base_image_from_bytes(
                            &mmap,
                            &path_clone,
                            false,
                            &settings,
                            cancel_token.clone(),
                        )
                        .map_err(|e| e.to_string())?;
                        let exif = exif_processing::read_exif_data(&path_clone, &mmap);
                        Ok((img, exif))
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to memory-map file '{}': {}. Falling back to standard read.",
                            path_clone,
                            e
                        );
                        let bytes = fs::read(&path_clone).map_err(|io_err| {
                            format!("Fallback read failed for {}: {}", path_clone, io_err)
                        })?;

                        if generation_tracker.load(Ordering::SeqCst) != my_generation {
                            return Err("Load cancelled".to_string());
                        }

                        let img = load_base_image_from_bytes(
                            &bytes,
                            &path_clone,
                            false,
                            &settings,
                            cancel_token.clone(),
                        )
                        .map_err(|e| e.to_string())?;
                        let exif = exif_processing::read_exif_data(&path_clone, &bytes);
                        Ok((img, exif))
                    }
                })();
            result
        })
        .await
        .map_err(|e| e.to_string())??;

        let arc_img = Arc::new(pristine_img);

        state.decoded_image_cache.lock().unwrap().insert(
            source_path_str.clone(),
            arc_img.clone(),
            exif_data_loaded.clone(),
        );

        (arc_img, exif_data_loaded)
    };

    if state.load_image_generation.load(Ordering::SeqCst) != my_generation {
        return Err("Load cancelled".to_string());
    }

    let is_raw = is_raw_file(&source_path_str);

    if state.load_image_generation.load(Ordering::SeqCst) != my_generation {
        return Err("Load cancelled".to_string());
    }

    let (orig_width, orig_height) = pristine_arc.dimensions();
    let as_shot_white_balance = crate::white_balance::as_shot_white_balance(&source_path_str);

    *state.original_image.lock().unwrap() = Some(LoadedImage {
        path,
        image: pristine_arc,
        is_raw,
        as_shot_white_balance,
    });

    Ok(LoadImageResult {
        width: orig_width,
        height: orig_height,
        metadata,
        exif: exif_data,
        is_raw,
        as_shot_white_balance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, ImageBuffer, Luma, Rgba};

    #[test]
    fn clone_spot_is_rendered_from_the_float_image() {
        // Left half at 3.0, right half at 2.0 (both above display white). A cloned
        // spot on the left copying from the right must come out at 2.0, unclipped.
        let base = DynamicImage::ImageRgba32F(ImageBuffer::from_fn(40, 20, |x, _| {
            let v = if x < 20 { 3.0 } else { 2.0 };
            Rgba([v, v, v, 1.0])
        }));
        let mut png = Cursor::new(Vec::new());
        GrayImage::from_pixel(4, 4, Luma([255]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let adjustments = serde_json::json!({ "aiPatches": [{
            "id": "spot",
            "visible": true,
            "patchData": {
                "mask": general_purpose::STANDARD.encode(png.get_ref()),
                "offsetX": 5, "offsetY": 8, "width": 4, "height": 4,
                "sourceOffset": [25, 0], "heal": false
            }
        }]});
        let out = composite_patches_on_image(&base, &adjustments).unwrap().to_rgba32f();
        assert_eq!(out.get_pixel(6, 9)[0], 2.0);
        assert_eq!(out.get_pixel(2, 9)[0], 3.0);
    }

    #[test]
    fn retouch_keeps_highlights_above_white() {
        // A retouch stroke over a bright area (3.0) with a darker edge must smooth
        // in float, not clip to 1.0 as the old 8-bit patch did.
        let base = DynamicImage::ImageRgba32F(ImageBuffer::from_fn(60, 60, |x, _| {
            let v = if x < 40 { 3.0 } else { 0.5 };
            Rgba([v, v, v, 1.0])
        }));
        let adjustments = serde_json::json!({ "aiPatches": [{
            "id": "retouch",
            "visible": true,
            "patchData": { "isSrgbEncoded": true },
            "subMasks": [{ "type": "retouch", "parameters": { "intensity": 100,
                "lines": [{ "brushSize": 20, "feather": 0.5, "points": [{ "x": 20, "y": 30 }] }] } }]
        }]});
        let out = composite_patches_on_image(&base, &adjustments).unwrap().to_rgba32f();
        assert!((out.get_pixel(20, 30)[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn ai_patch_highlights_survive_8_bit_storage() {
        // What an AI patch goes through: rolled off, sRGB-encoded to 8 bits for the
        // model and PNG, then decoded. 2.0 must come back within 0.1 EV (8 bits
        // leave few codes above the knee), and values under the knee must match
        // plain 8-bit sRGB.
        let knee = 0.8;
        let store = |x: f32| {
            let y = encode_highlights(x, knee);
            let e = if y <= 0.003_130_8 { y * 12.92 } else { 1.055 * y.powf(1.0 / 2.4) - 0.055 };
            decode_highlights(srgb_to_linear_lut()[(e * 255.0).round() as usize], knee)
        };
        assert!((store(2.0) / 2.0).log2().abs() < 0.1, "{}", store(2.0));
        assert!((store(0.18) - 0.18).abs() < 0.002);
    }
}
