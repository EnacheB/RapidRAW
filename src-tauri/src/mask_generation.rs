use crate::ai_processing::AiDepthMaskParameters;
use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, GenericImageView, GrayImage, ImageFormat, Luma, Rgba, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::f32::consts::PI;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::sync::Arc;

use crate::app_state::AppState;
use crate::image_processing::apply_geometry_warp;
use crate::{get_cached_full_warped_image, get_cached_full_warped_image_for_path};

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub enum SubMaskMode {
    Additive,
    Subtractive,
    Intersect,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct SubMask {
    pub id: String,
    #[serde(rename = "type")]
    pub mask_type: String,
    pub visible: bool,
    #[serde(default)]
    pub invert: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub mode: SubMaskMode,
    pub parameters: Value,
}

fn default_opacity() -> f32 {
    100.0
}

/// A sub-mask's parameters as `T`, or why they don't fit.
fn parse_params<T: serde::de::DeserializeOwned>(params: &Value) -> Result<T, String> {
    serde_json::from_value(params.clone()).map_err(|e| e.to_string())
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct MaskDefinition {
    pub id: String,
    pub name: String,
    pub visible: bool,
    pub invert: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub adjustments: Value,
    pub sub_masks: Vec<SubMask>,
}

/// The masks in `adjustments`. A mask that doesn't parse is logged and left out
/// on its own. The mask bitmaps and the mask adjustments both come from this
/// list, so they stay paired.
pub fn parse_masks(adjustments: &Value) -> Vec<MaskDefinition> {
    let Some(masks) = adjustments.get("masks").and_then(|m| m.as_array()) else {
        return Vec::new();
    };
    masks
        .iter()
        .filter_map(|m| {
            serde_json::from_value(m.clone())
                .map_err(|e| {
                    let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    log::error!("mask '{name}' left out: {e}");
                })
                .ok()
        })
        .collect()
}

impl MaskDefinition {
    pub fn requires_warped_image(&self) -> bool {
        self.sub_masks
            .iter()
            .any(|sm| sm.mask_type == "color" || sm.mask_type == "luminance")
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct PatchData {
    pub color: String,
    pub mask: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct AiPatchDefinition {
    pub id: String,
    pub name: String,
    pub visible: bool,
    pub invert: bool,
    pub prompt: String,
    #[serde(default)]
    pub patch_data: Option<PatchData>,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub sub_masks: Vec<SubMask>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct RadialMaskParameters {
    center_x: f64,
    center_y: f64,
    radius_x: f64,
    radius_y: f64,
    rotation: f32,
    feather: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct LinearMaskParameters {
    start_x: f64,
    start_y: f64,
    end_x: f64,
    end_y: f64,
    #[serde(default = "default_range")]
    range: f32,
}

fn default_range() -> f32 {
    50.0
}

impl Default for LinearMaskParameters {
    fn default() -> Self {
        Self {
            start_x: 0.0,
            start_y: 0.0,
            end_x: 0.0,
            end_y: 0.0,
            range: default_range(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct BrushLine {
    tool: String,
    brush_size: f32,
    points: Vec<Point>,
    #[serde(default = "default_brush_feather")]
    feather: f32,
}

fn default_brush_feather() -> f32 {
    0.5
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct BrushMaskParameters {
    #[serde(default)]
    lines: Vec<BrushLine>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct FlowLine {
    tool: String,
    brush_size: f32,
    points: Vec<Point>,
    #[serde(default = "default_brush_feather")]
    feather: f32,
    #[serde(default = "default_line_flow")]
    flow: f32,
}

fn default_line_flow() -> f32 {
    10.0
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct FlowMaskParameters {
    #[serde(default)]
    lines: Vec<FlowLine>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct ParametricMaskParameters {
    // The picked pixel in the uncropped image; see `target`.
    #[serde(default)]
    target_x: Option<f64>,
    #[serde(default)]
    target_y: Option<f64>,
    #[serde(default = "default_tolerance")]
    tolerance: f32,
    #[serde(default)]
    grow: f32,
    #[serde(default)]
    feather: f32,
    #[serde(default)]
    rotation: f32,
    #[serde(default)]
    flip_horizontal: bool,
    #[serde(default)]
    flip_vertical: bool,
    #[serde(default)]
    orientation_steps: u8,
}

fn default_tolerance() -> f32 {
    20.0
}

impl ParametricMaskParameters {
    /// The picked pixel, or None until one is picked (the editor stores -10000 then).
    fn target(&self) -> Option<(f64, f64)> {
        match (self.target_x, self.target_y) {
            (Some(x), Some(y)) if x >= 0.0 && y >= 0.0 => Some((x, y)),
            _ => None,
        }
    }
}

impl Default for ParametricMaskParameters {
    fn default() -> Self {
        Self {
            target_x: None,
            target_y: None,
            tolerance: default_tolerance(),
            grow: 0.0,
            feather: 35.0,
            rotation: 0.0,
            flip_horizontal: false,
            flip_vertical: false,
            orientation_steps: 0,
        }
    }
}

fn grayscale_dilate(image: &GrayImage, k: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return image.clone();
    }
    let w = width as usize;
    let h = height as usize;
    let r = k as i32;
    let src = image.as_raw();

    let mut temp = vec![0u8; w * h];
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let mut max_val = 0;
            let start = (x as i32 - r).max(0) as usize;
            let end = (x as i32 + r).min((w - 1) as i32) as usize;
            for xi in start..=end {
                max_val = max_val.max(src[row_offset + xi]);
            }
            temp[row_offset + x] = max_val;
        }
    }

    for x in 0..w {
        for y in 0..h {
            let mut max_val = 0;
            let start = (y as i32 - r).max(0) as usize;
            let end = (y as i32 + r).min((h - 1) as i32) as usize;
            for yi in start..=end {
                max_val = max_val.max(temp[yi * w + x]);
            }
            out[y * w + x] = max_val;
        }
    }

    GrayImage::from_raw(width, height, out).unwrap()
}

fn grayscale_erode(image: &GrayImage, k: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return image.clone();
    }
    let w = width as usize;
    let h = height as usize;
    let r = k as i32;
    let src = image.as_raw();

    let mut temp = vec![0u8; w * h];
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let mut min_val = 255;
            let start = (x as i32 - r).max(0) as usize;
            let end = (x as i32 + r).min((w - 1) as i32) as usize;
            for xi in start..=end {
                min_val = min_val.min(src[row_offset + xi]);
            }
            temp[row_offset + x] = min_val;
        }
    }

    for x in 0..w {
        for y in 0..h {
            let mut min_val = 255;
            let start = (y as i32 - r).max(0) as usize;
            let end = (y as i32 + r).min((h - 1) as i32) as usize;
            for yi in start..=end {
                min_val = min_val.min(temp[yi * w + x]);
            }
            out[y * w + x] = min_val;
        }
    }

    GrayImage::from_raw(width, height, out).unwrap()
}

fn apply_grow_and_feather(mask: &mut GrayImage, grow: f32, feather: f32, width: u32, height: u32) {
    let base_dimension = width.min(height) as f32;

    if grow.abs() > 0.01 {
        const MAX_GROW_PERCENTAGE: f32 = 0.01;
        let grow_pixels = (grow / 100.0) * base_dimension * MAX_GROW_PERCENTAGE;

        let amount = grow_pixels.abs().round() as u8;

        if amount > 0 {
            if grow_pixels > 0.0 {
                *mask = grayscale_dilate(mask, amount);
            } else {
                *mask = grayscale_erode(mask, amount);
            }
        }
    }

    if feather > 0.0 {
        const MAX_FEATHER_SIGMA_PERCENTAGE: f32 = 0.005;
        let sigma = (feather / 100.0) * base_dimension * MAX_FEATHER_SIGMA_PERCENTAGE;

        if sigma > 0.01 {
            *mask = imageproc::filter::gaussian_blur_f32(mask, sigma);
        }
    }
}

fn stroke_bounds(
    points: &[Point],
    width: u32,
    height: u32,
    radius: f32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Option<(u32, u32, u32, u32)> {
    if width == 0 || height == 0 || points.is_empty() {
        return None;
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let r_pad = radius.ceil() + 2.0;

    for p in points {
        let px = p.x as f32 * scale - crop_offset.0;
        let py = p.y as f32 * scale - crop_offset.1;

        min_x = min_x.min(px - r_pad);
        min_y = min_y.min(py - r_pad);
        max_x = max_x.max(px + r_pad);
        max_y = max_y.max(py + r_pad);
    }

    if max_x < 0.0 || max_y < 0.0 || min_x > (width - 1) as f32 || min_y > (height - 1) as f32 {
        return None;
    }

    let min_x = min_x.floor().max(0.0).min((width - 1) as f32) as u32;
    let min_y = min_y.floor().max(0.0).min((height - 1) as f32) as u32;
    let max_x = max_x.ceil().max(0.0).min((width - 1) as f32) as u32;
    let max_y = max_y.ceil().max(0.0).min((height - 1) as f32) as u32;

    if min_x > max_x || min_y > max_y {
        None
    } else {
        Some((min_x, min_y, max_x, max_y))
    }
}

#[allow(clippy::too_many_arguments)]
fn render_stroke_layer_parallel(
    points: &[Point],
    radius: f32,
    feather: f32,
    scale: f32,
    crop_offset: (f32, f32),
    layer_offset: (f32, f32),
    bb_w: u32,
    bb_h: u32,
) -> GrayImage {
    let mut out_pixels = vec![0u8; (bb_w * bb_h) as usize];
    if points.is_empty() || radius <= 0.0 {
        return GrayImage::from_raw(bb_w, bb_h, out_pixels).unwrap();
    }

    struct Segment {
        x1: f32,
        y1: f32,
        dx: f32,
        dy: f32,
        len_sq: f32,
        bounds_left: i32,
        bounds_right: i32,
        bounds_top: i32,
        bounds_bottom: i32,
    }

    let mut segments = Vec::with_capacity(points.len().saturating_sub(1));
    for pair in points.windows(2) {
        let x1 = pair[0].x as f32 * scale - crop_offset.0 - layer_offset.0;
        let y1 = pair[0].y as f32 * scale - crop_offset.1 - layer_offset.1;
        let x2 = pair[1].x as f32 * scale - crop_offset.0 - layer_offset.0;
        let y2 = pair[1].y as f32 * scale - crop_offset.1 - layer_offset.1;

        let left = ((x1.min(x2) - radius).floor() as i32).max(0);
        let right = ((x1.max(x2) + radius).ceil() as i32).min(bb_w as i32 - 1);
        let top = ((y1.min(y2) - radius).floor() as i32).max(0);
        let bottom = ((y1.max(y2) + radius).ceil() as i32).min(bb_h as i32 - 1);

        if left > right || top > bottom {
            continue;
        }

        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;

        segments.push(Segment {
            x1,
            y1,
            dx,
            dy,
            len_sq,
            bounds_left: left,
            bounds_right: right,
            bounds_top: top,
            bounds_bottom: bottom,
        });
    }

    let mut single_point = None;
    if segments.is_empty() && !points.is_empty() {
        let x1 = points[0].x as f32 * scale - crop_offset.0 - layer_offset.0;
        let y1 = points[0].y as f32 * scale - crop_offset.1 - layer_offset.1;
        let left = ((x1 - radius).floor() as i32).max(0);
        let right = ((x1 + radius).ceil() as i32).min(bb_w as i32 - 1);
        let top = ((y1 - radius).floor() as i32).max(0);
        let bottom = ((y1 + radius).ceil() as i32).min(bb_h as i32 - 1);
        if left <= right && top <= bottom {
            single_point = Some((x1, y1, left, right, top, bottom));
        }
    }

    let feather_amount = feather.clamp(0.0, 1.0);
    let inner_radius = radius * (1.0 - feather_amount);
    let feather_range = (radius - inner_radius).max(0.01);
    let radius_sq = radius * radius;
    let inner_radius_sq = inner_radius * inner_radius;

    out_pixels
        .par_chunks_mut(bb_w as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let py = y as f32;
            let y_i32 = y as i32;

            let mut active_segments = Vec::new();
            for seg in &segments {
                if y_i32 >= seg.bounds_top && y_i32 <= seg.bounds_bottom {
                    active_segments.push(seg);
                }
            }

            let is_point_active = if let Some(pt) = &single_point {
                y_i32 >= pt.4 && y_i32 <= pt.5
            } else {
                false
            };

            if active_segments.is_empty() && !is_point_active {
                return;
            }

            for (x, pixel) in row.iter_mut().enumerate() {
                let px = x as f32;
                let x_i32 = x as i32;

                let mut min_dist_sq = radius_sq + 1.0;

                for seg in &active_segments {
                    if x_i32 >= seg.bounds_left && x_i32 <= seg.bounds_right {
                        let dist_sq = if seg.len_sq < 0.0001 {
                            (px - seg.x1) * (px - seg.x1) + (py - seg.y1) * (py - seg.y1)
                        } else {
                            let t = (((px - seg.x1) * seg.dx + (py - seg.y1) * seg.dy)
                                / seg.len_sq)
                                .clamp(0.0, 1.0);
                            let proj_x = seg.x1 + t * seg.dx;
                            let proj_y = seg.y1 + t * seg.dy;
                            (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)
                        };
                        if dist_sq < min_dist_sq {
                            min_dist_sq = dist_sq;
                        }
                    }
                }

                if is_point_active {
                    let pt = single_point.as_ref().unwrap();
                    if x_i32 >= pt.2 && x_i32 <= pt.3 {
                        let dist_sq = (px - pt.0) * (px - pt.0) + (py - pt.1) * (py - pt.1);
                        if dist_sq < min_dist_sq {
                            min_dist_sq = dist_sq;
                        }
                    }
                }

                if min_dist_sq <= radius_sq {
                    let intensity = if min_dist_sq <= inner_radius_sq {
                        1.0
                    } else {
                        let dist = min_dist_sq.sqrt();
                        let t = ((dist - inner_radius) / feather_range).clamp(0.0, 1.0);
                        1.0 - (t * t * (3.0 - 2.0 * t))
                    };
                    *pixel = (intensity * 255.0).round() as u8;
                }
            }
        });

    GrayImage::from_raw(bb_w, bb_h, out_pixels).unwrap()
}

fn generate_radial_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Result<GrayImage, String> {
    let params: RadialMaskParameters = parse_params(params_value)?;
    let mut mask = GrayImage::new(width, height);

    let center_x = (params.center_x as f32 * scale - crop_offset.0) as i32;
    let center_y = (params.center_y as f32 * scale - crop_offset.1) as i32;
    let radius_x = params.radius_x as f32 * scale;
    let radius_y = params.radius_y as f32 * scale;
    let rotation_rad = params.rotation * PI / 180.0;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center_x as f32;
            let dy = y as f32 - center_y as f32;

            let cos_rot = rotation_rad.cos();
            let sin_rot = rotation_rad.sin();

            let rot_dx = dx * cos_rot + dy * sin_rot;
            let rot_dy = -dx * sin_rot + dy * cos_rot;

            let norm_x = rot_dx / radius_x.max(0.01);
            let norm_y = rot_dy / radius_y.max(0.01);

            let dist = (norm_x.powi(2) + norm_y.powi(2)).sqrt();

            let inner_bound = 1.0 - params.feather.clamp(0.0, 1.0);
            let intensity = 1.0 - (dist - inner_bound) / (1.0 - inner_bound).max(0.01);
            let clamped_intensity = intensity.clamp(0.0, 1.0);

            mask.put_pixel(x, y, Luma([(clamped_intensity * 255.0) as u8]));
        }
    }

    Ok(mask)
}

fn generate_linear_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Result<GrayImage, String> {
    let params: LinearMaskParameters = parse_params(params_value)?;
    let mut mask = GrayImage::new(width, height);

    let start_x = params.start_x as f32 * scale - crop_offset.0;
    let start_y = params.start_y as f32 * scale - crop_offset.1;
    let end_x = params.end_x as f32 * scale - crop_offset.0;
    let end_y = params.end_y as f32 * scale - crop_offset.1;
    let range = params.range * scale;

    let line_vec_x = end_x - start_x;
    let line_vec_y = end_y - start_y;

    let len_sq = line_vec_x.powi(2) + line_vec_y.powi(2);

    if len_sq < 0.01 {
        return Ok(mask);
    }

    let perp_vec_x = -line_vec_y / len_sq.sqrt();
    let perp_vec_y = line_vec_x / len_sq.sqrt();

    let half_width = range.max(0.01);

    for y_u in 0..height {
        for x_u in 0..width {
            let x = x_u as f32;
            let y = y_u as f32;

            let pixel_vec_x = x - start_x;
            let pixel_vec_y = y - start_y;

            let dist_perp = pixel_vec_x * perp_vec_x + pixel_vec_y * perp_vec_y;

            let t = dist_perp / half_width;

            let intensity = 0.5 - t * 0.5;

            let clamped_intensity = intensity.clamp(0.0, 1.0);

            mask.put_pixel(x_u, y_u, Luma([(clamped_intensity * 255.0) as u8]));
        }
    }

    Ok(mask)
}

fn generate_brush_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Result<GrayImage, String> {
    let params: BrushMaskParameters = parse_params(params_value)?;
    let mut final_mask = GrayImage::new(width, height);

    for line in &params.lines {
        if line.points.is_empty() {
            continue;
        }

        let is_eraser = line.tool == "eraser";
        let radius = (line.brush_size * scale / 2.0).max(0.0);
        let feather = line.feather.clamp(0.0, 1.0);

        let Some((min_x, min_y, max_x, max_y)) =
            stroke_bounds(&line.points, width, height, radius, scale, crop_offset)
        else {
            continue;
        };

        let bb_w = max_x - min_x + 1;
        let bb_h = max_y - min_y + 1;
        let layer_offset = (min_x as f32, min_y as f32);

        let line_mask = render_stroke_layer_parallel(
            &line.points,
            radius,
            feather,
            scale,
            crop_offset,
            layer_offset,
            bb_w,
            bb_h,
        );

        for y in 0..bb_h {
            for x in 0..bb_w {
                let src_val = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
                if src_val <= 0.0 {
                    continue;
                }

                let abs_x = min_x + x;
                let abs_y = min_y + y;
                let dst_pixel = final_mask.get_pixel_mut(abs_x, abs_y);
                let dst_val = dst_pixel[0] as f32 / 255.0;

                let blended = if is_eraser {
                    dst_val * (1.0 - src_val)
                } else {
                    dst_val + src_val - dst_val * src_val
                };

                dst_pixel[0] = (blended.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }

    Ok(final_mask)
}

fn generate_flow_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Result<GrayImage, String> {
    let params: FlowMaskParameters = parse_params(params_value)?;
    let mut final_mask = GrayImage::new(width, height);

    for line in &params.lines {
        if line.points.is_empty() {
            continue;
        }

        let is_eraser = line.tool == "eraser";
        let flow_per_stroke = (line.flow.clamp(0.0, 100.0) / 100.0) * 255.0;
        let radius = (line.brush_size * scale / 2.0).max(0.0);
        let feather = line.feather.clamp(0.0, 1.0);

        let Some((min_x, min_y, max_x, max_y)) =
            stroke_bounds(&line.points, width, height, radius, scale, crop_offset)
        else {
            continue;
        };

        let bb_w = max_x - min_x + 1;
        let bb_h = max_y - min_y + 1;
        let layer_offset = (min_x as f32, min_y as f32);

        let line_mask = render_stroke_layer_parallel(
            &line.points,
            radius,
            feather,
            scale,
            crop_offset,
            layer_offset,
            bb_w,
            bb_h,
        );

        for y in 0..bb_h {
            for x in 0..bb_w {
                let stroke_pixel = line_mask.get_pixel(x, y)[0] as f32;
                if stroke_pixel <= 0.0 {
                    continue;
                }

                let abs_x = min_x + x;
                let abs_y = min_y + y;
                let pixel = final_mask.get_pixel_mut(abs_x, abs_y);

                let c_norm = pixel[0] as f32 / 255.0;
                let delta = ((stroke_pixel / 255.0) * flow_per_stroke).round();
                let d_norm = (delta / 255.0).clamp(0.0, 1.0);

                let next = if is_eraser {
                    c_norm * (1.0 - d_norm)
                } else {
                    c_norm + d_norm - c_norm * d_norm
                };

                pixel[0] = (next.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }

    Ok(final_mask)
}

pub struct TransformParams {
    pub rotation: f32,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub orientation_steps: u8,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
    pub crop_offset: (f32, f32),
}

fn generate_ai_bitmap_from_full_mask(
    full_mask_image: &GrayImage,
    tf: &TransformParams,
) -> GrayImage {
    let (full_mask_w, full_mask_h) = full_mask_image.dimensions();
    let mut final_mask = GrayImage::new(tf.width, tf.height);

    let angle_rad = tf.rotation.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let (coarse_rotated_w, coarse_rotated_h) = if tf.orientation_steps % 2 == 1 {
        (full_mask_h, full_mask_w)
    } else {
        (full_mask_w, full_mask_h)
    };

    let scaled_coarse_rotated_w = coarse_rotated_w as f32 * tf.scale;
    let scaled_coarse_rotated_h = coarse_rotated_h as f32 * tf.scale;
    let center_x = scaled_coarse_rotated_w / 2.0;
    let center_y = scaled_coarse_rotated_h / 2.0;

    for y_out in 0..tf.height {
        for x_out in 0..tf.width {
            let x_uncrop = x_out as f32 + tf.crop_offset.0;
            let y_uncrop = y_out as f32 + tf.crop_offset.1;

            let x_centered = x_uncrop - center_x;
            let y_centered = y_uncrop - center_y;

            let x_unrotated = x_centered * cos_a + y_centered * sin_a + center_x;
            let y_unrotated = -x_centered * sin_a + y_centered * cos_a + center_y;

            let x_unflipped = if tf.flip_horizontal {
                scaled_coarse_rotated_w - x_unrotated
            } else {
                x_unrotated
            };
            let y_unflipped = if tf.flip_vertical {
                scaled_coarse_rotated_h - y_unrotated
            } else {
                y_unrotated
            };

            let (x_unrotated_coarse, y_unrotated_coarse) = match tf.orientation_steps {
                0 => (x_unflipped, y_unflipped),
                1 => (y_unflipped, scaled_coarse_rotated_w - x_unflipped),
                2 => (
                    scaled_coarse_rotated_w - x_unflipped,
                    scaled_coarse_rotated_h - y_unflipped,
                ),
                3 => (scaled_coarse_rotated_h - y_unflipped, x_unflipped),
                _ => (x_unflipped, y_unflipped),
            };

            let x_src = x_unrotated_coarse / tf.scale;
            let y_src = y_unrotated_coarse / tf.scale;

            if x_src >= 0.0
                && x_src < full_mask_w as f32
                && y_src >= 0.0
                && y_src < full_mask_h as f32
            {
                let pixel = full_mask_image.get_pixel(x_src as u32, y_src as u32);
                final_mask.put_pixel(x_out, y_out, *pixel);
            }
        }
    }

    final_mask
}

pub fn generate_ai_bitmap_from_base64(
    data_url: &str,
    tf: &TransformParams,
) -> Result<GrayImage, String> {
    let b64_data = if let Some(idx) = data_url.find(',') {
        &data_url[idx + 1..]
    } else {
        data_url
    };

    let decoded_bytes = general_purpose::STANDARD
        .decode(b64_data)
        .map_err(|e| format!("mask data: {e}"))?;
    let full_mask_image = image::load_from_memory(&decoded_bytes)
        .map_err(|e| format!("mask data: {e}"))?
        .to_luma8();

    Ok(generate_ai_bitmap_from_full_mask(&full_mask_image, tf))
}

/// What every AI sub-mask stores for rendering: the mask over the whole image
/// (a PNG data URL, absent until the model has run), the orientation it was
/// made under, and grow/feather. The box or point a subject mask was picked
/// with only matters for making the mask, so it isn't needed here.
#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase", default)]
struct AiMaskData {
    mask_data_base64: Option<String>,
    rotation: Option<f32>,
    flip_horizontal: Option<bool>,
    flip_vertical: Option<bool>,
    orientation_steps: Option<u8>,
    grow: Option<f32>,
    feather: Option<f32>,
}

impl AiMaskData {
    /// The stored mask mapped onto the output, or None if it hasn't been made yet.
    fn bitmap(
        &self,
        width: u32,
        height: u32,
        scale: f32,
        crop_offset: (f32, f32),
    ) -> Result<Option<GrayImage>, String> {
        let Some(data_url) = &self.mask_data_base64 else {
            return Ok(None);
        };
        let tf = TransformParams {
            rotation: self.rotation.unwrap_or(0.0),
            flip_horizontal: self.flip_horizontal.unwrap_or(false),
            flip_vertical: self.flip_vertical.unwrap_or(false),
            orientation_steps: self.orientation_steps.unwrap_or(0),
            width,
            height,
            scale,
            crop_offset,
        };
        generate_ai_bitmap_from_base64(data_url, &tf).map(Some)
    }

    fn grow_and_feather(&self, mask: &mut GrayImage, width: u32, height: u32) {
        let (grow, feather) = (self.grow.unwrap_or(0.0), self.feather.unwrap_or(0.0));
        apply_grow_and_feather(mask, grow, feather, width, height);
    }
}

/// Sky, foreground, subject and quick-eraser masks: the stored mask, grown and feathered.
fn generate_ai_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Result<Option<GrayImage>, String> {
    let data: AiMaskData = parse_params(params_value)?;
    let Some(mut mask) = data.bitmap(width, height, scale, crop_offset)? else {
        return Ok(None);
    };
    data.grow_and_feather(&mut mask, width, height);
    Ok(Some(mask))
}

fn generate_ai_depth_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Result<Option<GrayImage>, String> {
    let params: AiDepthMaskParameters = parse_params(params_value)?;
    let data: AiMaskData = parse_params(params_value)?;
    let Some(depth_map) = data.bitmap(width, height, scale, crop_offset)? else {
        return Ok(None);
    };

    let (w, h) = depth_map.dimensions();
    let mut mask = GrayImage::new(w, h);

    fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    let min_fade = params.min_fade;
    let max_fade = params.max_fade;

    for (x, y, p) in depth_map.enumerate_pixels() {
        let val_pct = (p[0] as f32 / 255.0) * 100.0;

        let lower_bound = smoothstep(params.min_depth - min_fade, params.min_depth, val_pct);
        let upper_bound = 1.0 - smoothstep(params.max_depth, params.max_depth + max_fade, val_pct);
        let bandpass_weight = lower_bound * upper_bound;

        let depth_intensity = val_pct / 100.0;
        let final_intensity = bandpass_weight * depth_intensity;

        mask.put_pixel(x, y, Luma([(final_intensity * 255.0) as u8]));
    }

    if params.feather > 0.0 {
        mask = image::imageops::blur(&mask, params.feather * 0.1);
    }

    data.grow_and_feather(&mut mask, width, height);

    Ok(Some(mask))
}

/// The pixel in 0..255 units, as floats: no 8-bit steps, and tones above white
/// stay apart instead of all reading 255.
fn sample_rgb(image: &DynamicImage, x: u32, y: u32) -> [f32; 3] {
    match image {
        DynamicImage::ImageRgb32F(img) => img.get_pixel(x, y).0.map(|v| v * 255.0),
        DynamicImage::ImageRgba32F(img) => {
            let [r, g, b, _] = img.get_pixel(x, y).0;
            [r * 255.0, g * 255.0, b * 255.0]
        }
        _ => {
            let [r, g, b, _] = image.get_pixel(x, y).0;
            [r as f32, g as f32, b as f32]
        }
    }
}

fn generate_color_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&image::DynamicImage>,
) -> Result<Option<GrayImage>, String> {
    let params: ParametricMaskParameters = parse_params(params_value)?;
    let Some((target_x, target_y)) = params.target() else {
        return Ok(None);
    };
    let warped = warped_image.ok_or("no image to sample")?;
    let (full_w, full_h) = warped.dimensions();

    let target_x = target_x.round() as i32;
    let target_y = target_y.round() as i32;
    if target_x < 0 || target_y < 0 || target_x >= full_w as i32 || target_y >= full_h as i32 {
        return Err(format!("picked pixel ({target_x}, {target_y}) is outside the image"));
    }

    let [ref_r, ref_g, ref_b] = sample_rgb(warped, target_x as u32, target_y as u32);

    let mut mask = GrayImage::new(width, height);

    let angle_rad = params.rotation * PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let (coarse_rotated_w, coarse_rotated_h) = if params.orientation_steps % 2 == 1 {
        (full_h, full_w)
    } else {
        (full_w, full_h)
    };

    let scaled_coarse_rotated_w = coarse_rotated_w as f32 * scale;
    let scaled_coarse_rotated_h = coarse_rotated_h as f32 * scale;
    let center_x = scaled_coarse_rotated_w / 2.0;
    let center_y = scaled_coarse_rotated_h / 2.0;

    let tolerance_sq = (params.tolerance * 2.55).max(1.0).powi(2) * 3.0;
    let inv_scale = 1.0 / scale;

    for y_out in 0..height {
        let y_uncrop = y_out as f32 + crop_offset.1;
        let y_centered = y_uncrop - center_y;
        let y_sin = y_centered * sin_a;
        let y_cos = y_centered * cos_a;

        for x_out in 0..width {
            let x_uncrop = x_out as f32 + crop_offset.0;
            let x_centered = x_uncrop - center_x;

            let x_unrotated = x_centered * cos_a + y_sin + center_x;
            let y_unrotated = -x_centered * sin_a + y_cos + center_y;

            let x_unflipped = if params.flip_horizontal {
                scaled_coarse_rotated_w - x_unrotated
            } else {
                x_unrotated
            };
            let y_unflipped = if params.flip_vertical {
                scaled_coarse_rotated_h - y_unrotated
            } else {
                y_unrotated
            };

            let (x_unrotated_coarse, y_unrotated_coarse) = match params.orientation_steps {
                0 => (x_unflipped, y_unflipped),
                1 => (y_unflipped, scaled_coarse_rotated_w - x_unflipped),
                2 => (
                    scaled_coarse_rotated_w - x_unflipped,
                    scaled_coarse_rotated_h - y_unflipped,
                ),
                3 => (scaled_coarse_rotated_h - y_unflipped, x_unflipped),
                _ => (x_unflipped, y_unflipped),
            };

            if x_unrotated_coarse >= 0.0 && y_unrotated_coarse >= 0.0 {
                let x_src = (x_unrotated_coarse * inv_scale) as u32;
                let y_src = (y_unrotated_coarse * inv_scale) as u32;

                if x_src < full_w && y_src < full_h {
                    let [r, g, b] = sample_rgb(warped, x_src, y_src);
                    let dist_sq = (r - ref_r).powi(2) + (g - ref_g).powi(2) + (b - ref_b).powi(2);

                    if dist_sq <= tolerance_sq {
                        let intensity = 1.0 - (dist_sq.sqrt() / tolerance_sq.sqrt());
                        mask.put_pixel(x_out, y_out, Luma([(intensity * 255.0) as u8]));
                    }
                }
            }
        }
    }

    apply_grow_and_feather(&mut mask, params.grow, params.feather, width, height);
    Ok(Some(mask))
}

fn generate_luminance_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&image::DynamicImage>,
) -> Result<Option<GrayImage>, String> {
    let params: ParametricMaskParameters = parse_params(params_value)?;
    let Some((target_x, target_y)) = params.target() else {
        return Ok(None);
    };
    let warped = warped_image.ok_or("no image to sample")?;
    let (full_w, full_h) = warped.dimensions();

    let target_x = target_x.round() as i32;
    let target_y = target_y.round() as i32;
    if target_x < 0 || target_y < 0 || target_x >= full_w as i32 || target_y >= full_h as i32 {
        return Err(format!("picked pixel ({target_x}, {target_y}) is outside the image"));
    }

    let [r, g, b] = sample_rgb(warped, target_x as u32, target_y as u32);
    let ref_luma = 0.299 * r + 0.587 * g + 0.114 * b;

    let mut mask = GrayImage::new(width, height);

    let angle_rad = params.rotation * PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let (coarse_rotated_w, coarse_rotated_h) = if params.orientation_steps % 2 == 1 {
        (full_h, full_w)
    } else {
        (full_w, full_h)
    };

    let scaled_coarse_rotated_w = coarse_rotated_w as f32 * scale;
    let scaled_coarse_rotated_h = coarse_rotated_h as f32 * scale;
    let center_x = scaled_coarse_rotated_w / 2.0;
    let center_y = scaled_coarse_rotated_h / 2.0;

    let tolerance_val = (params.tolerance * 2.55).max(1.0);
    let inv_scale = 1.0 / scale;

    for y_out in 0..height {
        let y_uncrop = y_out as f32 + crop_offset.1;
        let y_centered = y_uncrop - center_y;
        let y_sin = y_centered * sin_a;
        let y_cos = y_centered * cos_a;

        for x_out in 0..width {
            let x_uncrop = x_out as f32 + crop_offset.0;
            let x_centered = x_uncrop - center_x;

            let x_unrotated = x_centered * cos_a + y_sin + center_x;
            let y_unrotated = -x_centered * sin_a + y_cos + center_y;

            let x_unflipped = if params.flip_horizontal {
                scaled_coarse_rotated_w - x_unrotated
            } else {
                x_unrotated
            };
            let y_unflipped = if params.flip_vertical {
                scaled_coarse_rotated_h - y_unrotated
            } else {
                y_unrotated
            };

            let (x_unrotated_coarse, y_unrotated_coarse) = match params.orientation_steps {
                0 => (x_unflipped, y_unflipped),
                1 => (y_unflipped, scaled_coarse_rotated_w - x_unflipped),
                2 => (
                    scaled_coarse_rotated_w - x_unflipped,
                    scaled_coarse_rotated_h - y_unflipped,
                ),
                3 => (scaled_coarse_rotated_h - y_unflipped, x_unflipped),
                _ => (x_unflipped, y_unflipped),
            };

            if x_unrotated_coarse >= 0.0 && y_unrotated_coarse >= 0.0 {
                let x_src = (x_unrotated_coarse * inv_scale) as u32;
                let y_src = (y_unrotated_coarse * inv_scale) as u32;

                if x_src < full_w && y_src < full_h {
                    let [r, g, b] = sample_rgb(warped, x_src, y_src);
                    let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                    let dist = (luma - ref_luma).abs();

                    if dist <= tolerance_val {
                        let intensity = 1.0 - (dist / tolerance_val);
                        mask.put_pixel(x_out, y_out, Luma([(intensity * 255.0) as u8]));
                    }
                }
            }
        }
    }

    apply_grow_and_feather(&mut mask, params.grow, params.feather, width, height);
    Ok(Some(mask))
}

fn generate_all_bitmap(width: u32, height: u32) -> GrayImage {
    GrayImage::from_pixel(width, height, Luma([255]))
}

/// The sub-mask's bitmap; None if it's hidden or not set up yet (an AI mask the
/// model hasn't made, a colour or luminance mask with no pixel picked). Err if its
/// parameters can't be used.
fn generate_sub_mask_bitmap(
    sub_mask: &SubMask,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&DynamicImage>,
) -> Result<Option<GrayImage>, String> {
    if !sub_mask.visible {
        return Ok(None);
    }

    let params = &sub_mask.parameters;
    match sub_mask.mask_type.as_str() {
        "radial" => generate_radial_bitmap(params, width, height, scale, crop_offset).map(Some),
        "linear" => generate_linear_bitmap(params, width, height, scale, crop_offset).map(Some),
        "brush" | "clone" | "heal" | "liquify" | "retouch" => {
            generate_brush_bitmap(params, width, height, scale, crop_offset).map(Some)
        }
        "flow" => generate_flow_bitmap(params, width, height, scale, crop_offset).map(Some),
        "color" => {
            generate_color_bitmap(params, width, height, scale, crop_offset, warped_image)
        }
        "luminance" => {
            generate_luminance_bitmap(params, width, height, scale, crop_offset, warped_image)
        }
        "ai-subject" | "ai-foreground" | "ai-sky" | "quick-eraser" => {
            generate_ai_bitmap(params, width, height, scale, crop_offset)
        }
        "ai-depth" => generate_ai_depth_bitmap(params, width, height, scale, crop_offset),
        "all" => Ok(Some(generate_all_bitmap(width, height))),
        other => Err(format!("unknown sub-mask type '{other}'")),
    }
}

/// The mask's bitmap, or None if the mask is hidden. Every visible mask gets one,
/// even if it selects nothing, so the bitmaps stay paired with the mask
/// adjustments, which skip only hidden masks. A sub-mask that can't be built is
/// logged and left out.
pub fn generate_mask_bitmap(
    mask_def: &MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&DynamicImage>,
) -> Option<GrayImage> {
    build_mask_bitmap(mask_def, width, height, scale, crop_offset, warped_image, false)
        .unwrap_or_default()
}

/// generate_mask_bitmap for the dev harness: a sub-mask that can't be built is an error.
pub fn generate_mask_bitmap_strict(
    mask_def: &MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&DynamicImage>,
) -> Result<Option<GrayImage>, String> {
    build_mask_bitmap(mask_def, width, height, scale, crop_offset, warped_image, true)
}

fn build_mask_bitmap(
    mask_def: &MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&DynamicImage>,
    strict: bool,
) -> Result<Option<GrayImage>, String> {
    if !mask_def.visible {
        return Ok(None);
    }

    let mut final_mask = GrayImage::new(width, height);
    if mask_def.sub_masks.is_empty() {
        // Nothing added yet (or the last part deleted): selects nothing, inverted or not.
        return Ok(Some(final_mask));
    }

    for sub_mask in &mask_def.sub_masks {
        let mut sub_bitmap = match generate_sub_mask_bitmap(
            sub_mask,
            width,
            height,
            scale,
            crop_offset,
            warped_image,
        ) {
            Ok(Some(bitmap)) => bitmap,
            Ok(None) => continue,
            Err(e) => {
                let e = format!("mask '{}', {} sub-mask: {e}", mask_def.name, sub_mask.mask_type);
                if strict {
                    return Err(e);
                }
                log::error!("{e}; left out");
                continue;
            }
        };

        if sub_mask.invert {
            for p in sub_bitmap.pixels_mut() {
                p[0] = 255 - p[0];
            }
        }

        let opacity_multiplier = (sub_mask.opacity / 100.0).clamp(0.0, 1.0);
        if opacity_multiplier < 1.0 {
            for pixel in sub_bitmap.pixels_mut() {
                pixel[0] = (pixel[0] as f32 * opacity_multiplier) as u8;
            }
        }

        match sub_mask.mode {
            SubMaskMode::Additive => {
                for (x, y, pixel) in final_mask.enumerate_pixels_mut() {
                    let sub_pixel = sub_bitmap.get_pixel(x, y);
                    pixel[0] = pixel[0].max(sub_pixel[0]);
                }
            }
            SubMaskMode::Subtractive => {
                for (x, y, pixel) in final_mask.enumerate_pixels_mut() {
                    let sub_pixel = sub_bitmap.get_pixel(x, y);
                    pixel[0] = pixel[0].saturating_sub(sub_pixel[0]);
                }
            }
            SubMaskMode::Intersect => {
                for (x, y, pixel) in final_mask.enumerate_pixels_mut() {
                    let sub_pixel = sub_bitmap.get_pixel(x, y);
                    pixel[0] = pixel[0].min(sub_pixel[0]);
                }
            }
        }
    }

    if mask_def.invert {
        for pixel in final_mask.pixels_mut() {
            pixel[0] = 255 - pixel[0];
        }
    }

    let opacity_multiplier = (mask_def.opacity / 100.0).clamp(0.0, 1.0);
    if opacity_multiplier < 1.0 {
        for pixel in final_mask.pixels_mut() {
            pixel[0] = (pixel[0] as f32 * opacity_multiplier) as u8;
        }
    }

    Ok(Some(final_mask))
}

#[tauri::command]
pub fn generate_mask_overlay(
    mut mask_def: serde_json::Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    mut js_adjustments: Option<serde_json::Value>,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    if let Some(ref mut adj) = js_adjustments {
        crate::adjustment_utils::hydrate_adjustments(&state, adj);
    }

    if let Some(sub_masks) = mask_def.get_mut("subMasks").and_then(|v| v.as_array_mut()) {
        let mut cache = state.patch_cache.lock().unwrap_or_else(|e| e.into_inner());
        crate::adjustment_utils::hydrate_sub_masks(sub_masks, &mut cache);
    }

    let parsed_mask_def: MaskDefinition = serde_json::from_value(mask_def)
        .map_err(|e| format!("Failed to parse hydrated mask_def: {}", e))?;

    let scaled_crop_offset = (crop_offset.0 * scale, crop_offset.1 * scale);

    let warped_image = js_adjustments.as_ref().and_then(|adj| {
        resolve_warped_image_for_masks(&state, adj, std::slice::from_ref(&parsed_mask_def))
    });

    if let Some(gray_mask) = generate_mask_bitmap(
        &parsed_mask_def,
        width,
        height,
        scale,
        scaled_crop_offset,
        warped_image.as_deref(),
    ) {
        let mut rgba_mask = RgbaImage::new(width, height);
        for (x, y, pixel) in gray_mask.enumerate_pixels() {
            let intensity = pixel[0];
            let alpha = (intensity as f32 * 0.5) as u8;
            rgba_mask.put_pixel(x, y, Rgba([255, 0, 0, alpha]));
        }

        let mut buf = Cursor::new(Vec::new());
        rgba_mask
            .write_to(&mut buf, ImageFormat::Png)
            .map_err(|e| e.to_string())?;

        let base64_str = general_purpose::STANDARD.encode(buf.get_ref());
        let data_url = format!("data:image/png;base64,{}", base64_str);

        Ok(data_url)
    } else {
        Ok("".to_string())
    }
}

pub fn resolve_warped_image_for_masks(
    state: &tauri::State<AppState>,
    adjustments: &serde_json::Value,
    masks: &[MaskDefinition],
) -> Option<Arc<DynamicImage>> {
    if masks.iter().any(|m| m.requires_warped_image()) {
        get_cached_full_warped_image(state, adjustments).ok()
    } else {
        None
    }
}

/// The image the AI models see and colour and luminance masks sample, before
/// geometry: the decoded photo, a RAW with the CPU default look. Unlike
/// apply_cpu_default_raw_processing, values above white are kept, so masks can
/// tell bright tones apart; the models' 8-bit input clips them as before.
pub fn build_full_source_image(base_image: &DynamicImage, is_raw: bool) -> Cow<'_, DynamicImage> {
    if !is_raw {
        return Cow::Borrowed(base_image);
    }
    let mut look = base_image.to_rgb32f();
    look.par_iter_mut()
        .for_each(|v| *v = crate::image_processing::cpu_default_raw_look(*v).max(0.0));
    Cow::Owned(DynamicImage::ImageRgb32F(look))
}

pub fn build_full_warped_image<'a>(
    base_image: &'a DynamicImage,
    is_raw: bool,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    apply_geometry_warp(build_full_source_image(base_image, is_raw), adjustments)
}

pub fn build_warped_image_for_masks<'a>(
    base_image: &'a DynamicImage,
    is_raw: bool,
    adjustments: &serde_json::Value,
    masks: &[MaskDefinition],
) -> Option<Cow<'a, DynamicImage>> {
    if masks.iter().any(|m| m.requires_warped_image()) {
        Some(build_full_warped_image(base_image, is_raw, adjustments))
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_cached_or_generate_mask(
    state: &tauri::State<AppState>,
    path: &str,
    def: &MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    adjustments: &serde_json::Value,
) -> Option<GrayImage> {
    let mut hasher = DefaultHasher::new();

    let mut def_for_hash = def.clone();
    def_for_hash.adjustments = serde_json::Value::Null;
    let def_json = serde_json::to_string(&def_for_hash).unwrap_or_default();
    def_json.hash(&mut hasher);
    path.hash(&mut hasher);

    if def.requires_warped_image() {
        crate::cache_utils::calculate_geometry_hash(adjustments).hash(&mut hasher);
    }

    width.hash(&mut hasher);
    height.hash(&mut hasher);
    scale.to_bits().hash(&mut hasher);
    crop_offset.0.to_bits().hash(&mut hasher);
    crop_offset.1.to_bits().hash(&mut hasher);

    let key = hasher.finish();

    {
        let cache = state.mask_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(img) = cache.get(&key) {
            return Some(img.clone());
        }
    }

    let warped_image = if def.requires_warped_image() {
        get_cached_full_warped_image_for_path(state, Some(path), adjustments).ok()
    } else {
        None
    };

    let generated = generate_mask_bitmap(
        def,
        width,
        height,
        scale,
        crop_offset,
        warped_image.as_deref(),
    );

    if let Some(img) = &generated {
        let mut cache = state.mask_cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() > 50 {
            cache.clear();
        }
        cache.insert(key, img.clone());
    }

    generated
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // The GPU pairs mask bitmaps with mask adjustments by position. Deleting a
    // mask's last part used to drop its bitmap but keep its adjustments, so the
    // next mask got them.
    #[test]
    fn every_visible_mask_gets_a_bitmap_paired_with_its_adjustments() {
        let radial = json!({"centerX": 5, "centerY": 5, "radiusX": 3, "radiusY": 3,
                            "rotation": 0, "feather": 0});
        let adjustments = json!({"masks": [
            {"id": "a", "name": "emptied", "visible": true, "invert": true,
             "adjustments": {}, "subMasks": []},
            {"id": "b", "name": "broken", "visible": true},
            {"id": "c", "name": "radial", "visible": true, "invert": false, "adjustments": {},
             "subMasks": [{"id": "c1", "type": "radial", "visible": true, "mode": "additive",
                           "parameters": radial}]},
        ]});

        let masks = parse_masks(&adjustments);
        let names: Vec<_> = masks.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["emptied", "radial"]);

        let bitmaps: Vec<_> = masks
            .iter()
            .filter_map(|m| generate_mask_bitmap(m, 10, 10, 1.0, (0.0, 0.0), None))
            .collect();
        let all = crate::image_processing::get_all_adjustments_from_json(
            &adjustments,
            true,
            crate::white_balance::WhiteBalance::reference(),
            None,
        );
        assert_eq!(bitmaps.len(), all.mask_count as usize);
        assert!(bitmaps[0].pixels().all(|p| p[0] == 0));
        assert_eq!(bitmaps[1].get_pixel(5, 5)[0], 255);
    }

    // The editor stores -10000 until a pixel is picked: no mask yet, not an error.
    #[test]
    fn unpicked_colour_and_luminance_masks_select_nothing() {
        let unpicked = json!({"targetX": -10000, "targetY": -10000, "tolerance": 20});
        let mask: MaskDefinition = serde_json::from_value(json!({
            "id": "m", "name": "picker", "visible": true, "invert": false, "adjustments": {},
            "subMasks": [
                {"id": "c", "type": "color", "visible": true, "mode": "additive",
                 "parameters": unpicked},
                {"id": "l", "type": "luminance", "visible": true, "mode": "additive",
                 "parameters": unpicked},
            ]
        }))
        .unwrap();
        let image = DynamicImage::new_rgb32f(10, 10);
        let bitmap = generate_mask_bitmap_strict(&mask, 10, 10, 1.0, (0.0, 0.0), Some(&image));
        assert!(bitmap.unwrap().unwrap().pixels().all(|p| p[0] == 0));
    }
}
