//! Lens corrections the camera writes into its own raw files, applied at decode.
//!
//! Each vendor reader turns its maker-specific tags into one neutral form, a radial
//! `Distortion` curve; `correct` knows nothing about vendors. To support another
//! brand, add a reader to `READERS`.
//!
//! Only distortion: bodies that also embed vignetting and lateral CA data (Sony)
//! bake those into the raw pixels already when their in-camera compensation is on,
//! so applying them again would over-correct.

use rawler::formats::tiff::{GenericTiffReader, reader::TiffReader};
use rawler::tags::ExifTag;
use rayon::prelude::*;

/// Where to sample the uncorrected image: at radius `r` (a fraction of the
/// half-diagonal, 0 = centre, 1 = corner) the source point is `r * scale(r)`,
/// with `scale` linearly interpolated between knots.
pub struct Distortion {
    radii: Vec<f32>,
    scales: Vec<f32>,
}

impl Distortion {
    fn scale_at(&self, r: f32) -> f32 {
        let i = self.radii.partition_point(|&k| k <= r);
        if i == 0 {
            return self.scales[0];
        }
        if i == self.radii.len() {
            return self.scales[i - 1];
        }
        let t = (r - self.radii[i - 1]) / (self.radii[i] - self.radii[i - 1]);
        self.scales[i - 1] + (self.scales[i] - self.scales[i - 1]) * t
    }
}

const READERS: &[fn(&[u8]) -> Option<Distortion>] = &[sony];

pub fn read_distortion(file_bytes: &[u8]) -> Option<Distortion> {
    READERS.iter().find_map(|read| read(file_bytes))
}

/// Sony: SubIFD tag DistortionCorrParams (0x7037), n values at radii i/(n-1),
/// scale 1 + v/2^14. Reproduces the camera's distortion-corrected JPEGs to ~0.05 px.
fn sony(file_bytes: &[u8]) -> Option<Distortion> {
    let tiff = GenericTiffReader::new_with_buffer(file_bytes, 0, 0, None).ok()?;
    let entry = tiff
        .find_first_ifd_with_tag(ExifTag::DistortionCorrParams)?
        .get_entry(ExifTag::DistortionCorrParams)?;
    let n = entry.force_i16(0) as usize;
    if !(2..=16).contains(&n) || entry.value.count() != n + 1 {
        return None;
    }
    let scales: Vec<f32> = (1..=n)
        .map(|i| 1.0 + entry.force_i16(i) as f32 / 16384.0)
        .collect();
    if scales.iter().all(|&s| s == 1.0) {
        return None;
    }
    let radii = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
    Some(Distortion { radii, scales })
}

/// The camera's crop: the widest view whose border still samples inside the frame.
/// Pincushion binds at the corners, barrel at the midpoints of the short edges.
fn fit_zoom(cx: f32, cy: f32, inv_half_diagonal: f32, distortion: &Distortion) -> f32 {
    let limit = |dx: f32, dy: f32| {
        let scale = distortion.scale_at(dx.hypot(dy) * inv_half_diagonal);
        (cx / (dx * scale)).min(cy / (dy * scale))
    };
    let top = (0..=cx as u32).map(|x| limit(x as f32, cy));
    let side = (0..=cy as u32).map(|y| limit(cx, y as f32));
    top.chain(side).fold(f32::MAX, f32::min)
}

/// Resamples an interleaved RGB image so that straight lines are straight, zoomed
/// just enough that no empty border shows (as the camera's own JPEG does).
pub fn correct(pixels: &[f32], width: usize, height: usize, distortion: &Distortion) -> Vec<f32> {
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let inv_half_diagonal = 1.0 / cx.hypot(cy);
    let zoom = fit_zoom(cx, cy, inv_half_diagonal, distortion);
    let (max_x, max_y) = (width as f32 - 1.001, height as f32 - 1.001);

    let mut out = vec![0.0f32; pixels.len()];
    out.par_chunks_exact_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let dy = y as f32 + 0.5 - cy;
            for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
                let dx = x as f32 + 0.5 - cx;
                let scale = zoom * distortion.scale_at(dx.hypot(dy) * inv_half_diagonal);
                crate::image_processing::interpolate_pixel(
                    pixels,
                    width,
                    height,
                    (cx + dx * scale - 0.5).clamp(0.0, max_x),
                    (cy + dy * scale - 0.5).clamp(0.0, max_y),
                    pixel,
                );
            }
        });
    out
}
