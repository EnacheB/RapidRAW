//! Automatic source selection for the Heal and Clone tools.
//!
//! Like Lightroom's spot removal, the user paints over a blemish and a nearby
//! region to copy from is picked for them. Candidate offsets lie on rings around
//! the painted area. Each candidate is scored on:
//! - how well the band just outside the painted area matches at the candidate,
//!   so the copy blends in and edges (lips, jawline, hair) continue through it;
//! - how much its interior differs from that band, so another blemish, a mole
//!   or a stray hair isn't copied in;
//! - its distance, so of two equally good candidates the closer one wins.

use image::{DynamicImage, GenericImageView, GrayImage};

use crate::brush_patches::{Image, Rendered};

/// Width in pixels of the band around the painted area that is compared.
const RING: i32 = 3;
const ANGLES: usize = 32;
/// Candidate distances grow geometrically from just past the painted area.
const RADIUS_STEPS: i32 = 10;
const RADIUS_GROWTH: f64 = 1.25;
/// Pixels compared per candidate; larger areas are subsampled.
const MAX_INNER_SAMPLES: usize = 3072;
const MAX_RING_SAMPLES: usize = 1024;
/// Noise floor for the texture normalisation, in sRGB (0..1)² summed over channels.
const VARIANCE_FLOOR: f64 = 3e-5;
/// Cost of moving one minimum ring radius further away.
const DISTANCE_COST: f64 = 0.08;

type Rgb = [f64; 3];

/// Returns the offset from the painted area to the chosen source, in image pixels,
/// or `None` if no candidate fits (for example, the area covers most of the image).
///
/// `bounds` is the painted area's inclusive bounding box `(min_x, max_x, min_y, max_y)`.
/// Pass `blends: true` for Heal, whose blending evens out brightness differences,
/// and `false` for Clone, which copies pixels verbatim.
pub fn find_source_offset(
    image: &DynamicImage,
    mask: &GrayImage,
    bounds: (usize, usize, usize, usize),
    blends: bool,
) -> Option<(i32, i32)> {
    let (min_x, max_x, min_y, max_y) = bounds;
    let (img_w, img_h) = (mask.width() as i32, mask.height() as i32);
    let painted = |x: i32, y: i32| mask.get_pixel(x as u32, y as u32)[0] > 0;

    // The painted area's bounding box grown by the ring, clipped to the image.
    let x0 = (min_x as i32 - RING).max(0);
    let y0 = (min_y as i32 - RING).max(0);
    let x1 = (max_x as i32 + RING).min(img_w - 1);
    let y1 = (max_y as i32 + RING).min(img_h - 1);
    let (bw, bh) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);

    // Split the box into painted pixels and the band around them (a square
    // dilation by RING, done separably).
    let mut near_row = vec![false; bw * bh];
    for y in 0..bh {
        for x in 0..bw {
            let (ix, iy) = (x0 + x as i32, y0 + y as i32);
            near_row[y * bw + x] =
                ((ix - RING).max(x0)..=(ix + RING).min(x1)).any(|sx| painted(sx, iy));
        }
    }
    let mut inner = Vec::new();
    let mut ring = Vec::new();
    let mut edge_len = 0usize;
    for y in 0..bh {
        for x in 0..bw {
            let (ix, iy) = (x0 + x as i32, y0 + y as i32);
            if painted(ix, iy) {
                inner.push((ix, iy));
                let on_edge = [(-1, 0), (1, 0), (0, -1), (0, 1)].iter().any(|&(ox, oy)| {
                    let (nx, ny) = (ix + ox, iy + oy);
                    nx < 0 || ny < 0 || nx >= img_w || ny >= img_h || !painted(nx, ny)
                });
                edge_len += on_edge as usize;
            } else {
                let near = (y.saturating_sub(RING as usize)..=(y + RING as usize).min(bh - 1))
                    .any(|sy| near_row[sy * bw + x]);
                if near {
                    ring.push((ix, iy));
                }
            }
        }
    }
    if inner.is_empty() || ring.is_empty() {
        return None;
    }

    // Typical thickness of the painted area: its radius for a dab, its width for a stroke.
    let thickness = 2.0 * inner.len() as f64 / edge_len.max(1) as f64;
    let min_radius = thickness + 2.0 * RING as f64;

    let inner = subsample(inner, MAX_INNER_SAMPLES);
    let ring = subsample(ring, MAX_RING_SAMPLES);

    let pixel = |x: i32, y: i32| -> Rgb {
        let p = image.get_pixel(x as u32, y as u32);
        [
            p[0] as f64 / 255.0,
            p[1] as f64 / 255.0,
            p[2] as f64 / 255.0,
        ]
    };
    let target_ring: Vec<Rgb> = ring.iter().map(|&(x, y)| pixel(x, y)).collect();
    let target_mean = mean(&target_ring);
    let target_var = variance(&target_ring, target_mean);
    let target_sd = target_var.sqrt();
    let brightness_weight = if blends { 0.5 } else { 2.0 };

    // Lower is better; None if the candidate leaves the image or overlaps the painted area.
    let score = |dx: i32, dy: i32| -> Option<f64> {
        if x0 + dx < 0 || y0 + dy < 0 || x1 + dx >= img_w || y1 + dy >= img_h {
            return None;
        }
        let boxes_overlap = (dx.unsigned_abs() as usize) < bw && (dy.unsigned_abs() as usize) < bh;
        if boxes_overlap
            && inner
                .iter()
                .chain(&ring)
                .any(|&(x, y)| painted(x + dx, y + dy))
        {
            return None;
        }

        let source_ring: Vec<Rgb> = ring.iter().map(|&(x, y)| pixel(x + dx, y + dy)).collect();
        let source_mean = mean(&source_ring);
        let structure = target_ring
            .iter()
            .zip(&source_ring)
            .map(|(t, s)| dist2(sub(*t, target_mean), sub(*s, source_mean)))
            .sum::<f64>()
            / ring.len() as f64;
        let ring_texture = (variance(&source_ring, source_mean).sqrt() - target_sd).powi(2);

        let source_inner: Vec<Rgb> = inner.iter().map(|&(x, y)| pixel(x + dx, y + dy)).collect();
        let inner_mean = mean(&source_inner);
        let inner_texture = (variance(&source_inner, inner_mean).sqrt() - target_sd).powi(2);
        let inner_blob = dist2(inner_mean, source_mean);

        let brightness = dist2(source_mean, target_mean);
        let distance = ((dx * dx + dy * dy) as f64).sqrt() / min_radius;

        Some(
            (structure
                + ring_texture
                + inner_texture
                + inner_blob
                + brightness_weight * brightness)
                / (target_var + VARIANCE_FLOOR)
                + DISTANCE_COST * distance,
        )
    };

    // Coarse search on rings, then a local refinement around the winner.
    let mut best: Option<((i32, i32), f64)> = None;
    for step in 0..RADIUS_STEPS {
        let radius = min_radius * RADIUS_GROWTH.powi(step);
        for a in 0..ANGLES {
            let angle = a as f64 * std::f64::consts::TAU / ANGLES as f64;
            let d = (
                (radius * angle.cos()).round() as i32,
                (radius * angle.sin()).round() as i32,
            );
            if let Some(cost) = score(d.0, d.1)
                && best.is_none_or(|(_, c)| cost < c)
            {
                best = Some((d, cost));
            }
        }
    }
    let (mut d, mut cost) = best?;
    let mut step = ((thickness / 4.0).round() as i32).max(1);
    loop {
        let neighbours = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];
        let better = neighbours
            .iter()
            .filter_map(|&(ox, oy)| {
                let n = (d.0 + ox * step, d.1 + oy * step);
                score(n.0, n.1).map(|c| (n, c))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|&(_, c)| c < cost);
        match better {
            Some((n, c)) => (d, cost) = (n, c),
            None if step > 1 => step /= 2,
            None => break,
        }
    }
    Some(d)
}

/// A Heal or Clone spot, rendered from the linear float `image` as it is when the
/// spot applies, so an edit stores only its recipe (mask and offset) and loses
/// nothing.
///
/// The spot's `mask` is placed at `origin` and copies from `offset` away. Clone
/// copies verbatim; Heal also solves Laplace's equation for the difference along the
/// spot's edge, so the copy takes on the surrounding brightness and colour.
pub fn render_spot(
    image: &Image,
    mask: GrayImage,
    origin: (usize, usize),
    offset: (i32, i32),
    heal: bool,
) -> Rendered {
    let pixel = |x: i32, y: i32| {
        image.pixel(
            x.clamp(0, image.width as i32 - 1) as usize,
            y.clamp(0, image.height as i32 - 1) as usize,
        )
    };
    let (mw, mh) = (mask.width() as usize, mask.height() as usize);
    let (ox, oy) = (origin.0 as i32, origin.1 as i32);
    let source = |x: usize, y: usize| pixel(ox + x as i32 + offset.0, oy + y as i32 + offset.1);

    if !heal {
        let colors = (0..mh)
            .flat_map(|y| (0..mw).map(move |x| (x, y)))
            .map(|(x, y)| source(x, y))
            .collect();
        return Rendered { origin, mask, colors };
    }

    // The difference image, on the mask's box plus a one-pixel border. Pixels outside
    // the painted area are the fixed boundary (target minus source); painted pixels
    // are solved by successive over-relaxation.
    let (bw, bh) = (mw + 2, mh + 2);
    let painted = |x: usize, y: usize| {
        (1..=mw).contains(&x) && (1..=mh).contains(&y) && mask.get_pixel(x as u32 - 1, y as u32 - 1)[0] > 0
    };
    let mut diff = vec![[0.0f32; 3]; bw * bh];
    let mut unknown = Vec::new();
    for y in 0..bh {
        for x in 0..bw {
            if painted(x, y) {
                unknown.push(y * bw + x);
                continue;
            }
            let (ix, iy) = (ox + x as i32 - 1, oy + y as i32 - 1);
            let (t, s) = (pixel(ix, iy), pixel(ix + offset.0, iy + offset.1));
            diff[y * bw + x] = [t[0] - s[0], t[1] - s[1], t[2] - s[2]];
        }
    }
    const OMEGA: f32 = 1.6;
    const ITERATIONS: usize = 400;
    for _ in 0..ITERATIONS {
        for &i in &unknown {
            for c in 0..3 {
                let sum = diff[i - bw][c] + diff[i + bw][c] + diff[i - 1][c] + diff[i + 1][c];
                diff[i][c] = (1.0 - OMEGA) * diff[i][c] + OMEGA * 0.25 * sum;
            }
        }
    }

    let colors = (0..mh)
        .flat_map(|y| (0..mw).map(move |x| (x, y)))
        .map(|(x, y)| {
            let s = source(x, y);
            let d = diff[(y + 1) * bw + x + 1];
            // Light can't be negative; there is no upper limit.
            [(s[0] + d[0]).max(0.0), (s[1] + d[1]).max(0.0), (s[2] + d[2]).max(0.0)]
        })
        .collect();
    Rendered { origin, mask, colors }
}

fn subsample(points: Vec<(i32, i32)>, max: usize) -> Vec<(i32, i32)> {
    if points.len() <= max {
        return points;
    }
    let stride = points.len().div_ceil(max);
    points.into_iter().step_by(stride).collect()
}

fn mean(pixels: &[Rgb]) -> Rgb {
    let n = pixels.len().max(1) as f64;
    let sum = pixels.iter().fold([0.0; 3], |acc, p| {
        [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
    });
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

fn variance(pixels: &[Rgb], mean: Rgb) -> f64 {
    pixels.iter().map(|p| dist2(*p, mean)).sum::<f64>() / pixels.len().max(1) as f64
}

fn sub(a: Rgb, b: Rgb) -> Rgb {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dist2(a: Rgb, b: Rgb) -> f64 {
    let d = sub(a, b);
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb as ImgRgb, RgbImage};

    /// Mid-grey "skin" with deterministic fine texture.
    fn skin(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 17) as u8;
            ImgRgb([170 + n, 130 + n, 115 + n])
        })
    }

    fn dab(img: &mut RgbImage, cx: i32, cy: i32, r: i32, colour: [u8; 3]) {
        for y in cy - r..=cy + r {
            for x in cx - r..=cx + r {
                if (x - cx).pow(2) + (y - cy).pow(2) <= r * r {
                    img.put_pixel(x as u32, y as u32, ImgRgb(colour));
                }
            }
        }
    }

    fn dab_mask(
        w: u32,
        h: u32,
        cx: i32,
        cy: i32,
        r: i32,
    ) -> (GrayImage, (usize, usize, usize, usize)) {
        let mask = GrayImage::from_fn(w, h, |x, y| {
            let inside = (x as i32 - cx).pow(2) + (y as i32 - cy).pow(2) <= r * r;
            Luma([if inside { 255 } else { 0 }])
        });
        let b = (
            (cx - r) as usize,
            (cx + r) as usize,
            (cy - r) as usize,
            (cy + r) as usize,
        );
        (mask, b)
    }

    #[test]
    fn source_does_not_overlap_the_blemish() {
        let mut img = skin(200, 200);
        dab(&mut img, 100, 100, 6, [200, 60, 60]);
        let (mask, bounds) = dab_mask(200, 200, 100, 100, 8);
        let (dx, dy) =
            find_source_offset(&DynamicImage::ImageRgb8(img), &mask, bounds, true).unwrap();
        assert!(
            dx * dx + dy * dy > 16 * 16,
            "offset ({dx}, {dy}) overlaps the painted area"
        );
    }

    #[test]
    fn avoids_copying_a_neighbouring_blemish() {
        let mut img = skin(200, 200);
        dab(&mut img, 100, 100, 6, [200, 60, 60]);
        // Blemishes on every side but the left, at the distance a naive pick would use.
        let others = [
            (128, 100),
            (100, 128),
            (100, 72),
            (120, 120),
            (120, 80),
            (80, 120),
            (80, 80),
        ];
        for (x, y) in others {
            dab(&mut img, x, y, 6, [120, 50, 40]);
        }
        let (mask, bounds) = dab_mask(200, 200, 100, 100, 8);
        let (dx, dy) =
            find_source_offset(&DynamicImage::ImageRgb8(img), &mask, bounds, true).unwrap();
        for (x, y) in others {
            let gap = (((100 + dx - x).pow(2) + (100 + dy - y).pow(2)) as f64).sqrt();
            assert!(
                gap > 8.0 + 6.0,
                "source ({dx}, {dy}) overlaps the blemish at ({x}, {y})"
            );
        }
    }

    #[test]
    fn follows_an_edge_through_the_blemish() {
        // Dark left half, light right half; the blemish sits on the boundary.
        let mut img = skin(200, 200);
        for y in 0..200 {
            for x in 0..100 {
                let p = img.get_pixel(x, y).0;
                img.put_pixel(x, y, ImgRgb([p[0] - 100, p[1] - 90, p[2] - 80]));
            }
        }
        dab(&mut img, 100, 100, 5, [230, 230, 230]);
        let (mask, bounds) = dab_mask(200, 200, 100, 100, 7);
        let (dx, dy) =
            find_source_offset(&DynamicImage::ImageRgb8(img), &mask, bounds, true).unwrap();
        assert_eq!(
            dx, 0,
            "source ({dx}, {dy}) doesn't continue the vertical edge"
        );
    }

    #[test]
    fn gives_up_when_the_whole_image_is_painted() {
        let img = skin(40, 40);
        let mask = GrayImage::from_pixel(40, 40, Luma([255]));
        assert_eq!(
            find_source_offset(&DynamicImage::ImageRgb8(img), &mask, (0, 39, 0, 39), true),
            None
        );
    }

    #[test]
    fn heal_keeps_highlights_above_white() {
        // A bright sky (linear 3.0, far above display white) with a dark spot; the
        // spot is healed from sky at 2.0, so the result must come out near 3.0, not
        // clipped to 1.0 as the old 8-bit patches were.
        let (w, h) = (64usize, 64usize);
        let mut image = vec![0.0f32; w * h * 4];
        for (i, p) in image.chunks_exact_mut(4).enumerate() {
            let x = i % w;
            let v = if x < 32 { 3.0 } else { 2.0 };
            p.copy_from_slice(&[v, v, v, 1.0]);
        }
        let mut mask = GrayImage::new(6, 6);
        for p in mask.pixels_mut() {
            *p = Luma([255]);
        }
        let image = Image {
            pixels: &image,
            width: w,
            height: h,
        };
        for heal in [false, true] {
            let out = render_spot(&image, mask.clone(), (10, 10), (30, 0), heal);
            let expected = if heal { 3.0 } else { 2.0 };
            for c in &out.colors {
                assert!((c[0] - expected).abs() < 0.05, "heal={heal}: got {c:?}");
            }
        }
    }
}
