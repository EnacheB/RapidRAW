use std::io::Cursor;

use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage, RgbaImage};
use rayon::prelude::*;
use serde_json::Value;

use crate::ai_connector;
use crate::ai_processing;
use crate::app_settings::load_settings;
use crate::app_state::AppState;
use crate::image_loader::{composite_patches_on_image, encode_highlights};
use crate::image_processing::apply_linear_to_srgb;
use crate::mask_generation::{AiPatchDefinition, MaskDefinition, generate_mask_bitmap};
use crate::resolve_warped_image_for_masks;

/// Where AI inpainting starts rolling off a RAW's highlights to fit them into the
/// model's 0–1 range (`image_loader::encode_highlights`).
const HIGHLIGHT_KNEE: f32 = 0.8;

fn is_raw(state: &tauri::State<'_, AppState>) -> bool {
    let guard = state.original_image.lock().unwrap();
    guard.as_ref().map(|img| img.is_raw).unwrap_or(false)
}

/// The image with every other patch applied, sRGB-encoded in 0–1 for AI models
/// and the Heal/Clone source search. A RAW's highlights are rolled off above
/// `HIGHLIGHT_KNEE` rather than clipped. Also returns whether the image is a RAW.
fn prepare_source_image(
    patch_id: &str,
    current_adjustments: &Value,
    state: &tauri::State<'_, AppState>,
) -> Result<(DynamicImage, bool), String> {
    let mut source_image_adjustments = current_adjustments.clone();
    if let Some(patches) = source_image_adjustments
        .get_mut("aiPatches")
        .and_then(|v| v.as_array_mut())
    {
        patches.retain(|p| p.get("id").and_then(|id| id.as_str()) != Some(patch_id));
    }

    let is_raw = is_raw(state);

    let (base_image, _) = crate::get_original_image(state)?;
    let composited = composite_patches_on_image(&base_image, &source_image_adjustments)
        .map_err(|e| format!("Failed to prepare source image: {}", e))?;

    let source_image = if is_raw {
        let mut linear = composited.into_rgba32f();
        linear.par_chunks_mut(4).for_each(|p| {
            for v in &mut p[..3] {
                *v = encode_highlights(*v, HIGHLIGHT_KNEE);
            }
        });
        apply_linear_to_srgb(DynamicImage::ImageRgba32F(linear))
    } else {
        composited
    };

    Ok((source_image, is_raw))
}

fn calculate_mask_bounds(
    mask_bitmap: &image::GrayImage,
) -> Result<(usize, usize, usize, usize), String> {
    let (img_w, img_h) = mask_bitmap.dimensions();
    let mask_raw = mask_bitmap.as_raw();
    let img_w_usize = img_w as usize;
    let img_h_usize = img_h as usize;

    let mut min_y = img_h_usize;
    let mut max_y = 0;

    for y in 0..img_h_usize {
        let row_start = y * img_w_usize;
        if mask_raw[row_start..row_start + img_w_usize]
            .iter()
            .any(|&p| p > 0)
        {
            min_y = y;
            break;
        }
    }

    if min_y == img_h_usize {
        return Err("Mask is empty.".to_string());
    }

    for y in (min_y..img_h_usize).rev() {
        let row_start = y * img_w_usize;
        if mask_raw[row_start..row_start + img_w_usize]
            .iter()
            .any(|&p| p > 0)
        {
            max_y = y;
            break;
        }
    }

    let mut min_x = img_w_usize;
    let mut max_x = 0;
    for y in min_y..=max_y {
        let row_start = y * img_w_usize;
        let row = &mask_raw[row_start..row_start + img_w_usize];
        if let Some(first) = row.iter().position(|&p| p > 0)
            && first < min_x
        {
            min_x = first;
        }
        if let Some(last) = row.iter().rposition(|&p| p > 0)
            && last > max_x
        {
            max_x = last;
        }
    }

    Ok((min_x, max_x, min_y, max_y))
}

/// Stored pixels for an AI patch, lossless. For a RAW (`is_srgb`) they are
/// sRGB-encoded with highlights rolled off above `HIGHLIGHT_KNEE`, which
/// `composite_patches_on_image` undoes.
fn encode_patch_result(
    color_image: &RgbImage,
    mask_image: &image::GrayImage,
    offset: (u32, u32),
    is_srgb: bool,
) -> Result<Value, String> {
    let png = |image: &DynamicImage| -> Result<String, String> {
        let mut buf = Cursor::new(Vec::with_capacity(32768));
        image
            .write_to(&mut buf, image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        Ok(general_purpose::STANDARD.encode(buf.get_ref()))
    };
    let mut result = serde_json::json!({
        "color": png(&DynamicImage::ImageRgb8(color_image.clone()))?,
        "mask": png(&DynamicImage::ImageLuma8(mask_image.clone()))?,
        "offsetX": offset.0,
        "offsetY": offset.1,
        "width": color_image.width(),
        "height": color_image.height(),
        "isSrgbEncoded": is_srgb
    });
    if is_srgb {
        result["highlightKnee"] = HIGHLIGHT_KNEE.into();
    }
    Ok(result)
}

#[tauri::command]
pub async fn generate_manual_cleanup_patch(
    patch_definition: AiPatchDefinition,
    current_adjustments: Value,
    source_point: Option<(f64, f64)>,
    _task_id: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let (source_image, _) =
        prepare_source_image(&patch_definition.id, &current_adjustments, &state)?;
    let (img_w, img_h) = source_image.dimensions();

    let orientation_steps = current_adjustments
        .get("orientationSteps")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u8;
    let (trans_w, trans_h) = if orientation_steps % 2 == 1 {
        (img_h, img_w)
    } else {
        (img_w, img_h)
    };

    let mask_def_for_generation = MaskDefinition {
        id: patch_definition.id.clone(),
        name: patch_definition.name.clone(),
        visible: patch_definition.visible,
        invert: patch_definition.invert,
        opacity: 100.0,
        adjustments: serde_json::Value::Null,
        sub_masks: patch_definition.sub_masks.clone(),
    };

    let warped_image = resolve_warped_image_for_masks(
        &state,
        &current_adjustments,
        std::slice::from_ref(&mask_def_for_generation),
    );

    let mask_bitmap = generate_mask_bitmap(
        &mask_def_for_generation,
        trans_w,
        trans_h,
        1.0,
        (0.0, 0.0),
        warped_image.as_deref(),
    )
    .ok_or("Failed to generate mask bitmap for manual cleanup")?;

    let mask_bitmap =
        crate::image_processing::inverse_transform_mask(mask_bitmap, &current_adjustments);

    let (min_x, max_x, min_y, max_y) = calculate_mask_bounds(&mask_bitmap)?;

    let center_x = (min_x + max_x) as f64 / 2.0;
    let center_y = (min_y + max_y) as f64 / 2.0;

    let min_x_u32 = min_x as u32;
    let min_y_u32 = min_y as u32;
    let crop_w = (max_x - min_x + 1) as u32;
    let crop_h = (max_y - min_y + 1) as u32;

    let sub_masks_val = serde_json::to_value(&patch_definition.sub_masks).unwrap_or(Value::Null);
    let mut is_heal = false;
    if let Some(arr) = sub_masks_val.as_array() {
        for sm in arr {
            if let Some(t) = sm.get("type").and_then(|v| v.as_str())
                && t.eq_ignore_ascii_case("heal")
            {
                is_heal = true;
                break;
            }
        }
    }
    if !is_heal && patch_definition.name.to_lowercase().contains("heal") {
        is_heal = true;
    }

    // Without a source point, pick one and return it (in the frontend's transformed
    // coordinates) so it is stored with the edit and shown on the canvas.
    let (offset_x, offset_y, picked_source) = match source_point {
        Some((x, y)) => {
            let (ux, uy) = crate::image_processing::inverse_transform_point(
                x,
                y,
                trans_w as f64,
                trans_h as f64,
                &current_adjustments,
            );
            let offset_x = (ux - center_x).round() as i32;
            let offset_y = (uy - center_y).round() as i32;
            (offset_x, offset_y, None)
        }
        None => {
            let (offset_x, offset_y) = crate::heal_source::find_source_offset(
                &source_image,
                &mask_bitmap,
                (min_x, max_x, min_y, max_y),
                is_heal,
            )
            .ok_or_else(|| {
                let modifier = if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" };
                format!("Couldn't find a clean area nearby to copy from. {modifier}+click to choose one.")
            })?;
            let picked = crate::image_processing::transform_point(
                center_x + offset_x as f64,
                center_y + offset_y as f64,
                trans_w as f64,
                trans_h as f64,
                &current_adjustments,
            );
            (offset_x, offset_y, Some(picked))
        }
    };

    // Store the recipe, not pixels: composite_patches_on_image renders the spot
    // from the float image (heal_source::render_spot), so nothing is clipped or
    // quantised.
    let output_mask =
        image::imageops::crop_imm(&mask_bitmap, min_x_u32, min_y_u32, crop_w, crop_h).to_image();
    let mut mask_png = Cursor::new(Vec::new());
    output_mask
        .write_to(&mut mask_png, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    let mut result = serde_json::json!({
        "mask": general_purpose::STANDARD.encode(mask_png.get_ref()),
        "offsetX": min_x_u32,
        "offsetY": min_y_u32,
        "width": crop_w,
        "height": crop_h,
        "sourceOffset": [offset_x, offset_y],
        "heal": is_heal,
    });
    if let Some((x, y)) = picked_source {
        result["sourcePoint"] = serde_json::json!([x, y]);
    }
    Ok(result.to_string())
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn invoke_generative_replace_with_mask_def(
    path: String,
    patch_definition: AiPatchDefinition,
    current_adjustments: Value,
    use_fast_inpaint: bool,
    token: Option<String>,
    task_id: Option<String>,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let _guard = task_id
        .as_ref()
        .map(|id| crate::app_state::AiTaskGuard::new(&state.active_ai_tasks, id.clone()));
    let cancel_flag = _guard.as_ref().map(|g| &g.token);

    if cancel_flag.as_ref().is_some_and(|t| t.is_cancelled()) {
        return Err("Task cancelled".to_string());
    }

    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let (source_image, is_raw) =
        prepare_source_image(&patch_definition.id, &current_adjustments, &state)?;
    let (img_w, img_h) = source_image.dimensions();
    let img_w_usize = img_w as usize;
    let img_h_usize = img_h as usize;

    let orientation_steps = current_adjustments
        .get("orientationSteps")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u8;
    let (trans_w, trans_h) = if orientation_steps % 2 == 1 {
        (img_h, img_w)
    } else {
        (img_w, img_h)
    };

    let mask_def_for_generation = MaskDefinition {
        id: patch_definition.id.clone(),
        name: patch_definition.name.clone(),
        visible: patch_definition.visible,
        invert: patch_definition.invert,
        opacity: 100.0,
        adjustments: serde_json::Value::Null,
        sub_masks: patch_definition.sub_masks.clone(),
    };

    let warped_image = resolve_warped_image_for_masks(
        &state,
        &current_adjustments,
        std::slice::from_ref(&mask_def_for_generation),
    );

    let mask_bitmap = generate_mask_bitmap(
        &mask_def_for_generation,
        trans_w,
        trans_h,
        1.0,
        (0.0, 0.0),
        warped_image.as_deref(),
    )
    .ok_or("Failed to generate mask bitmap for AI replace")?;

    let mask_bitmap =
        crate::image_processing::inverse_transform_mask(mask_bitmap, &current_adjustments);

    let (min_x, max_x, min_y, max_y) = calculate_mask_bounds(&mask_bitmap)?;

    if cancel_flag.as_ref().is_some_and(|t| t.is_cancelled()) {
        return Err("Task cancelled".to_string());
    }

    let patch_rgba = if use_fast_inpaint {
        let lama_model = ai_processing::get_or_init_lama_model(
            &app_handle,
            &state.ai_state,
            &state.ai_init_lock,
        )
        .await
        .map_err(|e| e.to_string())?;

        if cancel_flag.as_ref().is_some_and(|t| t.is_cancelled()) {
            return Err("Task cancelled".to_string());
        }

        ai_processing::run_lama_inpainting(&source_image, &mask_bitmap, &lama_model)
            .map_err(|e| e.to_string())?
    } else if settings.ai_provider.as_deref() == Some("cloud")
        && let Some(auth_token) = token
    {
        let tight_w = max_x - min_x + 1;
        let tight_h = max_y - min_y + 1;

        let pad_x = tight_w / 2;
        let pad_y = tight_h / 2;

        let padded_min_x = min_x.saturating_sub(pad_x);
        let padded_min_y = min_y.saturating_sub(pad_y);
        let padded_max_x = (max_x + pad_x).min(img_w_usize - 1);
        let padded_max_y = (max_y + pad_y).min(img_h_usize - 1);

        let crop_w = (padded_max_x - padded_min_x + 1) as u32;
        let crop_h = (padded_max_y - padded_min_y + 1) as u32;

        let src_crop = image::imageops::crop_imm(
            &source_image,
            padded_min_x as u32,
            padded_min_y as u32,
            crop_w,
            crop_h,
        )
        .to_image();

        let mask_crop = image::imageops::crop_imm(
            &mask_bitmap,
            padded_min_x as u32,
            padded_min_y as u32,
            crop_w,
            crop_h,
        )
        .to_image();

        let max_pixels = 1_500_000_f32;
        let current_pixels = (crop_w * crop_h) as f32;

        let mut ai_w = crop_w;
        let mut ai_h = crop_h;
        if current_pixels > max_pixels {
            let scale = (max_pixels / current_pixels).sqrt();
            ai_w = (crop_w as f32 * scale) as u32;
            ai_h = (crop_h as f32 * scale) as u32;
        }

        ai_w = (ai_w / 16) * 16;
        ai_h = (ai_h / 16) * 16;

        let resize_needed = ai_w != crop_w || ai_h != crop_h;

        let (final_src_crop, final_mask_crop) = if resize_needed {
            (
                image::imageops::resize(
                    &src_crop,
                    ai_w,
                    ai_h,
                    image::imageops::FilterType::Lanczos3,
                ),
                image::imageops::resize(
                    &mask_crop,
                    ai_w,
                    ai_h,
                    image::imageops::FilterType::Triangle,
                ),
            )
        } else {
            (src_crop, mask_crop)
        };

        let mut rgba_mask = RgbaImage::new(ai_w, ai_h);
        for (src_val, dst_chunk) in final_mask_crop.as_raw().iter().zip(rgba_mask.chunks_mut(4)) {
            let intensity = *src_val;
            dst_chunk[0] = intensity;
            dst_chunk[1] = intensity;
            dst_chunk[2] = intensity;
            dst_chunk[3] = 255;
        }

        let base_url = "https://www.getrapidraw.com/api";

        let dyn_src_crop = DynamicImage::ImageRgba8(final_src_crop);
        let dyn_rgba_mask = DynamicImage::ImageRgba8(rgba_mask);

        let cloud_fut = ai_connector::process_cloud_inpainting(
            base_url,
            &dyn_src_crop,
            &dyn_rgba_mask,
            patch_definition.prompt,
            &auth_token,
        );

        let generated_ai_patch = if let Some(token) = cancel_flag.as_ref() {
            tokio::select! {
                res = cloud_fut => res.map_err(|e| e.to_string())?,
                _ = token.wait_for_cancel() => {
                    return Err("Task cancelled".to_string());
                }
            }
        } else {
            cloud_fut.await.map_err(|e| e.to_string())?
        };

        let generated_ai_patch_rgba = generated_ai_patch.to_rgba8();
        let restored_ai_patch = if resize_needed {
            image::imageops::resize(
                &generated_ai_patch_rgba,
                crop_w,
                crop_h,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            generated_ai_patch_rgba
        };

        let mut full_canvas = RgbaImage::new(img_w, img_h);
        image::imageops::overlay(
            &mut full_canvas,
            &restored_ai_patch,
            padded_min_x as i64,
            padded_min_y as i64,
        );

        full_canvas
    } else if settings.ai_provider.as_deref() == Some("ai-connector")
        && let Some(address) = settings.ai_connector_address
    {
        let base_url = format!("http://{}", address);

        let mut rgba_mask = RgbaImage::new(img_w, img_h);
        for (src_val, dst_chunk) in mask_bitmap.as_raw().iter().zip(rgba_mask.chunks_mut(4)) {
            let intensity = *src_val;
            dst_chunk[0] = intensity;
            dst_chunk[1] = intensity;
            dst_chunk[2] = intensity;
            dst_chunk[3] = 255;
        }
        let mask_image_dynamic = DynamicImage::ImageRgba8(rgba_mask);

        let (real_path_buf, _) = crate::file_management::parse_virtual_path(&path);
        let real_path_str = real_path_buf.to_string_lossy().to_string();

        let inpaint_fut = ai_connector::process_inpainting(
            &base_url,
            &real_path_str,
            &source_image,
            &mask_image_dynamic,
            patch_definition.prompt,
            None,
        );

        if let Some(token) = cancel_flag.as_ref() {
            tokio::select! {
                res = inpaint_fut => res.map_err(|e| e.to_string())?,
                _ = token.wait_for_cancel() => {
                    return Err("Task cancelled".to_string());
                }
            }
        } else {
            inpaint_fut.await.map_err(|e| e.to_string())?
        }
    } else {
        return Err(
            "No generative backend configured or connection invalid. Please check your AI settings."
                .to_string(),
        );
    };

    if cancel_flag.as_ref().is_some_and(|t| t.is_cancelled()) {
        return Err("Task cancelled".to_string());
    }

    let (patch_w, patch_h) = patch_rgba.dimensions();
    let final_patch = if patch_w != img_w || patch_h != img_h {
        image::imageops::resize(
            &patch_rgba,
            img_w,
            img_h,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        patch_rgba.clone()
    };

    let min_x_u32 = min_x as u32;
    let min_y_u32 = min_y as u32;
    let crop_w = (max_x - min_x + 1) as u32;
    let crop_h = (max_y - min_y + 1) as u32;

    let mut color_image = RgbImage::new(crop_w, crop_h);

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px_x = x as u32;
            let px_y = y as u32;
            let mask_value = mask_bitmap.get_pixel(px_x, px_y)[0];

            let out_x = px_x - min_x_u32;
            let out_y = px_y - min_y_u32;

            // Only masked pixels are ever composited; the rest stay black, which
            // compresses to almost nothing.
            if mask_value > 0 {
                let px = final_patch.get_pixel(px_x, px_y);
                color_image.put_pixel(out_x, out_y, Rgb([px[0], px[1], px[2]]));
            }
        }
    }

    let output_mask =
        image::imageops::crop_imm(&mask_bitmap, min_x_u32, min_y_u32, crop_w, crop_h).to_image();

    encode_patch_result(&color_image, &output_mask, (min_x_u32, min_y_u32), is_raw)
        .map(|v| v.to_string())
}

/// Liquify and Retouch are rendered from their strokes whenever patches are
/// composited (`brush_patches`), so their patch stores no pixels, only whether
/// the image is linear light. The commands stay so every brush tool finishes the
/// same way in the frontend.
#[tauri::command]
pub async fn generate_liquify_patch(state: tauri::State<'_, AppState>) -> Result<String, String> {
    Ok(serde_json::json!({ "isSrgbEncoded": is_raw(&state) }).to_string())
}

#[tauri::command]
pub async fn generate_retouch_patch(state: tauri::State<'_, AppState>) -> Result<String, String> {
    generate_liquify_patch(state).await
}
