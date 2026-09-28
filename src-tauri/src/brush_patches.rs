//! Liquify and Retouch: brush tools rendered from their strokes when patches are
//! composited, on the float image, like Heal and Clone (`heal_source`). The edit
//! stores only the strokes, so nothing is clipped or quantised.

use image::GrayImage;
use rayon::prelude::*;
use serde_json::Value;

/// A rendered patch: a box at `origin` with a blend mask and linear RGB for every
/// pixel of the box, row by row.
pub struct Rendered {
    pub origin: (usize, usize),
    pub mask: GrayImage,
    pub colors: Vec<[f32; 3]>,
}

/// Interleaved RGBA float image.
pub struct Image<'a> {
    pub pixels: &'a [f32],
    pub width: usize,
    pub height: usize,
}

impl Image<'_> {
    pub fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        let i = (y * self.width + x) * 4;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }

    fn bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let clamp_x = |v: f32| (v as i32).clamp(0, self.width as i32 - 1) as usize;
        let clamp_y = |v: f32| (v as i32).clamp(0, self.height as i32 - 1) as usize;
        let (xa, xb, ya, yb) = (clamp_x(x0), clamp_x(x0 + 1.0), clamp_y(y0), clamp_y(y0 + 1.0));
        let (p00, p10, p01, p11) = (
            self.pixel(xa, ya),
            self.pixel(xb, ya),
            self.pixel(xa, yb),
            self.pixel(xb, yb),
        );
        std::array::from_fn(|c| {
            p00[c] * (1.0 - fx) * (1.0 - fy)
                + p10[c] * fx * (1.0 - fy)
                + p01[c] * (1.0 - fx) * fy
                + p11[c] * fx * fy
        })
    }

    /// The box around `segments`, grown by `margin`, clipped to the image.
    fn bounds(&self, segments: &[Segment], margin: f32) -> Option<Bounds> {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for s in segments {
            x0 = x0.min(s.a.0.min(s.b.0) - s.radius);
            y0 = y0.min(s.a.1.min(s.b.1) - s.radius);
            x1 = x1.max(s.a.0.max(s.b.0) + s.radius);
            y1 = y1.max(s.a.1.max(s.b.1) + s.radius);
        }
        if segments.is_empty() || self.width == 0 || self.height == 0 {
            return None;
        }
        let clip = |v: f32, len: usize| (v as i32).clamp(0, len as i32 - 1) as usize;
        let (x0, y0) = (clip(x0 - margin, self.width), clip(y0 - margin, self.height));
        let (x1, y1) = (clip(x1 + margin, self.width), clip(y1 + margin, self.height));
        Some(Bounds { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 })
    }
}

struct Bounds {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

/// One piece of a brush line; a single click is a segment of length 0.
#[derive(Clone, Copy)]
struct Segment {
    a: (f32, f32),
    b: (f32, f32),
    radius: f32,
    feather: f32,
    eraser: bool,
}

impl Segment {
    /// Brush weight at `p` (1 inside the hard core, a raised-cosine falloff over
    /// the feathered edge) and where along the segment `p` projects, or None
    /// outside the brush.
    fn weight(&self, p: (f32, f32)) -> Option<(f32, f32)> {
        let (dx, dy) = (self.b.0 - self.a.0, self.b.1 - self.a.1);
        let len_sq = dx * dx + dy * dy;
        let t = if len_sq == 0.0 {
            0.0
        } else {
            (((p.0 - self.a.0) * dx + (p.1 - self.a.1) * dy) / len_sq).clamp(0.0, 1.0)
        };
        let (ex, ey) = (p.0 - self.a.0 - t * dx, p.1 - self.a.1 - t * dy);
        let dist = (ex * ex + ey * ey).sqrt();
        if dist > self.radius {
            return None;
        }
        let inner = self.radius * (1.0 - self.feather.clamp(0.01, 1.0));
        if dist <= inner {
            return Some((1.0, t));
        }
        let c = 0.5 * (1.0 + ((dist - inner) / (self.radius - inner) * std::f32::consts::PI).cos());
        Some((c * c, t))
    }
}

/// The segments of a brush sub-mask's `parameters.lines`.
fn segments(params: &Value) -> Vec<Segment> {
    let mut out = Vec::new();
    for line in params.get("lines").and_then(Value::as_array).into_iter().flatten() {
        let radius = line.get("brushSize").and_then(Value::as_f64).unwrap_or(50.0) as f32 / 2.0;
        let feather = line.get("feather").and_then(Value::as_f64).unwrap_or(0.5) as f32;
        let eraser = line.get("tool").and_then(Value::as_str) == Some("eraser");
        let points: Vec<(f32, f32)> = line
            .get("points")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|p| Some((p.get("x")?.as_f64()? as f32, p.get("y")?.as_f64()? as f32)))
            .collect();
        let segment = |a, b| Segment { a, b, radius, feather, eraser };
        match points.as_slice() {
            [p] => out.push(segment(*p, *p)),
            _ => out.extend(points.windows(2).map(|w| segment(w[0], w[1]))),
        }
    }
    out
}

/// The sub-masks of `type`, with their parameters.
fn sub_masks_of<'a>(sub_masks: &'a Value, kind: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
    sub_masks
        .as_array()
        .into_iter()
        .flatten()
        .filter(move |sm| sm.get("type").and_then(Value::as_str) == Some(kind))
        .filter_map(|sm| sm.get("parameters"))
}

pub fn is_brush_patch(sub_masks: &Value) -> bool {
    ["liquify", "retouch"].iter().any(|kind| sub_masks_of(sub_masks, kind).next().is_some())
}

/// Renders a Liquify or Retouch patch, or returns None for any other kind.
/// `linear` says whether the image is linear light (a RAW) rather than sRGB-encoded.
pub fn render(image: &Image, sub_masks: &Value, linear: bool) -> Option<Rendered> {
    liquify(image, sub_masks).or_else(|| retouch(image, sub_masks, linear))
}

#[derive(Clone, Copy)]
enum Warp {
    Push,
    Pinch,
    Expand,
    Twirl,
    Eraser,
}

/// Warps the image along the strokes. Each pixel samples from where the strokes
/// displace it; the blend mask is the displacement in pixels, so sub-pixel moves
/// blend with the original.
fn liquify(image: &Image, sub_masks: &Value) -> Option<Rendered> {
    let mut strokes = Vec::new();
    for params in sub_masks_of(sub_masks, "liquify") {
        let force = (params.get("pressure").and_then(Value::as_f64).unwrap_or(50.0) as f32 / 100.0)
            .clamp(0.01, 1.0);
        let warp = match params.get("liquifyMode").and_then(Value::as_str) {
            Some("pinch") => Warp::Pinch,
            Some("expand") => Warp::Expand,
            Some("twirl") => Warp::Twirl,
            _ => Warp::Push,
        };
        strokes.extend(segments(params).into_iter().map(|s| {
            (s, force, if s.eraser { Warp::Eraser } else { warp })
        }));
    }
    let segments: Vec<Segment> = strokes.iter().map(|s| s.0).collect();
    let max_radius = segments.iter().map(|s| s.radius).fold(0.0, f32::max);
    let b = image.bounds(&segments, (max_radius * 2.5).clamp(50.0, 300.0))?;

    let mut colors = vec![[0.0f32; 3]; b.w * b.h];
    let mut mask = vec![0u8; b.w * b.h];
    colors.par_chunks_mut(b.w).zip(mask.par_chunks_mut(b.w)).enumerate().for_each(
        |(y, (color_row, mask_row))| {
            let py = (b.y + y) as f32;
            for x in 0..b.w {
                let px = (b.x + x) as f32;
                let (mut disp_x, mut disp_y) = (0.0f32, 0.0f32);
                for (s, force, warp) in &strokes {
                    let Some((falloff, t)) = s.weight((px, py)) else { continue };
                    let (dx, dy) = (s.b.0 - s.a.0, s.b.1 - s.a.1);
                    let step = (dx * dx + dy * dy).sqrt();
                    let step_factor =
                        if step < 0.001 { 0.45 } else { (step / s.radius).clamp(0.05, 0.6) };
                    // From the nearest point on the stroke to this pixel.
                    let (rx, ry) = (px - (s.a.0 + t * dx), py - (s.a.1 + t * dy));
                    let strength = falloff * force * step_factor;
                    match warp {
                        Warp::Push => {
                            disp_x -= dx * falloff * force * 0.6;
                            disp_y -= dy * falloff * force * 0.6;
                        }
                        Warp::Pinch => {
                            disp_x += rx * strength * 0.45;
                            disp_y += ry * strength * 0.45;
                        }
                        Warp::Expand => {
                            disp_x -= rx * strength * 0.30;
                            disp_y -= ry * strength * 0.30;
                        }
                        Warp::Twirl => {
                            let (sin, cos) = (strength * 1.4).sin_cos();
                            disp_x += (rx * cos - ry * sin) - rx;
                            disp_y += (rx * sin + ry * cos) - ry;
                        }
                        Warp::Eraser => {
                            let retain = 1.0 - (strength * 5.0).clamp(0.0, 1.0);
                            disp_x *= retain;
                            disp_y *= retain;
                        }
                    }
                }
                let disp_sq = disp_x * disp_x + disp_y * disp_y;
                if disp_sq > 0.001 {
                    color_row[x] = image.bilinear(px + disp_x, py + disp_y);
                    mask_row[x] = (disp_sq.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
                } else {
                    color_row[x] = image.pixel(b.x + x, b.y + y);
                }
            }
        },
    );
    Some(Rendered {
        origin: (b.x, b.y),
        mask: GrayImage::from_raw(b.w as u32, b.h as u32, mask)?,
        colors,
    })
}

/// Smooths skin under the strokes with an edge-preserving (bilateral) filter.
/// Neighbours are weighted by how alike they look, compared in sRGB-encoded
/// values scaled to 0–255, and averaged in the image's own values (linear light
/// for a RAW).
fn retouch(image: &Image, sub_masks: &Value, linear: bool) -> Option<Rendered> {
    let params: Vec<&Value> = sub_masks_of(sub_masks, "retouch").collect();
    let intensity = params
        .last()
        .and_then(|p| p.get("intensity"))
        .and_then(Value::as_f64)
        .unwrap_or(50.0) as f32;
    let segments: Vec<Segment> = params.iter().flat_map(|p| segments(p)).collect();
    let max_radius = segments.iter().map(|s| s.radius).fold(0.0, f32::max);
    let b = image.bounds(&segments, (max_radius * 3.0 + 30.0).ceil())?;

    // Strokes add to the mask and erasers subtract, in painting order.
    let mut mask = vec![0u8; b.w * b.h];
    for s in &segments {
        let Some(sb) = image.bounds(std::slice::from_ref(s), 1.0) else { continue };
        for y in sb.y - b.y..sb.y - b.y + sb.h {
            for x in sb.x - b.x..sb.x - b.x + sb.w {
                if let Some((w, _)) = s.weight(((b.x + x) as f32, (b.y + y) as f32)) {
                    let m = &mut mask[y * b.w + x];
                    let v = (w * 255.0) as u8;
                    *m = if s.eraser { m.saturating_sub(v) } else { m.saturating_add(v) };
                }
            }
        }
    }

    let original: Vec<[f32; 3]> = (0..b.w * b.h)
        .map(|i| image.pixel(b.x + i % b.w, b.y + i / b.w))
        .collect();
    let encode = |v: f32| {
        let v = v.max(0.0);
        255.0 * if !linear {
            v
        } else if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };
    let looks: Vec<[f32; 3]> = original.iter().map(|p| p.map(encode)).collect();

    let strength = (intensity / 100.0).clamp(0.05, 1.0);
    let sigma_s = (1.0 + strength * 6.0).clamp(1.2, 8.0);
    let sigma_c = (8.0 + strength * 20.0).clamp(8.0, 28.0);
    let r = (sigma_s * 2.0).ceil() as i32;
    let spatial: Vec<f32> = (-r..=r)
        .flat_map(|dy| (-r..=r).map(move |dx| (dx * dx + dy * dy) as f32))
        .map(|d| (-d / (2.0 * sigma_s * sigma_s)).exp())
        .collect();
    let blend = |m: u8| {
        let m = m as f32 / 255.0;
        m * m * (3.0 - 2.0 * m) * (0.3 + strength * 0.7)
    };

    let mut colors = original.clone();
    colors.par_chunks_mut(b.w).enumerate().for_each(|(y, row)| {
        for (x, out) in row.iter_mut().enumerate() {
            let i = y * b.w + x;
            if mask[i] == 0 {
                continue;
            }
            let centre = looks[i];
            let (mut sum_w, mut sum) = (0.0f32, [0.0f32; 3]);
            let (y0, y1) = ((y as i32 - r).max(0), (y as i32 + r).min(b.h as i32 - 1));
            let (x0, x1) = ((x as i32 - r).max(0), (x as i32 + r).min(b.w as i32 - 1));
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let j = ny as usize * b.w + nx as usize;
                    let d: f32 = (0..3).map(|c| (looks[j][c] - centre[c]).powi(2)).sum();
                    let w = spatial[((ny - y as i32 + r) * (2 * r + 1) + nx - x as i32 + r) as usize]
                        * (-d / (2.0 * sigma_c * sigma_c)).exp();
                    sum_w += w;
                    for c in 0..3 {
                        sum[c] += original[j][c] * w;
                    }
                }
            }
            let k = blend(mask[i]);
            *out = std::array::from_fn(|c| original[i][c] * (1.0 - k) + sum[c] / sum_w * k);
        }
    });
    Some(Rendered {
        origin: (b.x, b.y),
        mask: GrayImage::from_raw(b.w as u32, b.h as u32, mask)?,
        colors,
    })
}
