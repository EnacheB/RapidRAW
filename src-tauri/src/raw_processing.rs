use crate::image_processing::apply_orientation;
use crate::white_balance::WhiteBalance;
use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageBuffer, Rgba};
use rawler::{
    decoders::{Decoder, Orientation, RawDecodeParams},
    imgop::{
        develop::{DemosaicAlgorithm, Intermediate, ProcessingStep, RawDevelop},
        xyz::Illuminant,
    },
    rawimage::{RawImage, RawPhotometricInterpretation},
    rawsource::RawSource,
};
use rayon::prelude::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub fn develop_raw_image(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    use_camera_profiles: bool,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let (developed_image, orientation) = develop_internal(
        file_bytes,
        fast_demosaic,
        highlight_compression,
        linear_mode,
        use_camera_profiles,
        cancel_token,
    )?;
    Ok(apply_orientation(developed_image, orientation))
}

fn metadata_orientation(decoder: &dyn Decoder, source: &RawSource) -> Result<Orientation> {
    let metadata = decoder.raw_metadata(source, &RawDecodeParams::default())?;
    Ok(metadata
        .exif
        .orientation
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal))
}

pub fn extract_embedded_preview(file_bytes: &[u8]) -> Option<DynamicImage> {
    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source).ok()?;
    let preview = decoder
        .full_image(&source, &RawDecodeParams::default())
        .ok()??;
    let orientation =
        metadata_orientation(decoder.as_ref(), &source).unwrap_or(Orientation::Normal);
    Some(apply_orientation(preview, orientation))
}

fn is_linear_raw_format(raw_image: &RawImage) -> bool {
    matches!(
        raw_image.photometric,
        RawPhotometricInterpretation::LinearRaw
    )
}

#[inline]
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(3.0)
    }
}

/// Where a channel counts as clipped, as a fraction of the white level.
const CLIP: f32 = 0.987;

/// Rebuilds clipped highlights in demosaiced camera RGB (not white balanced, 1.0
/// = white level) with "inpaint opposed", darktable's default method, which
/// RawTherapee also offers. Only channels at the white level change, and only
/// upwards: each gets the mean of the other two channels around it (white
/// balanced by `wb`), plus the average amount by which that estimate misses the
/// real value in unclipped pixels next to clipped ones, so a blown sky keeps its
/// tint instead of turning magenta.
fn reconstruct_highlights(pixels: &mut [[f32; 3]], width: usize, height: usize, wb: [f32; 3]) {
    // Bit c is set where channel c is clipped.
    let mask: Vec<u8> = pixels
        .par_iter()
        .map(|p| (0..3).fold(0, |m, c| m | (((p[c] >= CLIP) as u8) << c)))
        .collect();
    if !mask.par_iter().any(|&m| m != 0) {
        return;
    }
    for (i, p) in opposed_fill(pixels, &mask, width, height, wb) {
        pixels[i] = p;
    }
}

/// The new value of every pixel with a clipped channel, for `reconstruct_highlights`.
fn opposed_fill(
    pixels: &[[f32; 3]],
    mask: &[u8],
    width: usize,
    height: usize,
    wb: [f32; 3],
) -> Vec<(usize, [f32; 3])> {
    let clip = wb.map(|w| CLIP * w);
    // RawTherapee's floor for pixels that inform the tint (green is denser).
    let dark = [0.03 * clip[0], 0.125 * clip[1], 0.03 * clip[2]];
    let value = |x: usize, y: usize, c: usize| pixels[y * width + x][c].max(0.0) * wb[c];
    // Each channel estimated as the mean of the other two over the 3x3
    // neighbourhood, averaged as cube roots.
    let opposed = |x: usize, y: usize| -> [f32; 3] {
        let (mut mean, mut n) = ([0.0f32; 3], 0.0);
        for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                for (c, m) in mean.iter_mut().enumerate() {
                    *m += value(xx, yy, c);
                }
                n += 1.0;
            }
        }
        let root = mean.map(|m| (m / n).cbrt());
        [0, 1, 2].map(|c| (0.5 * (root[(c + 1) % 3] + root[(c + 2) % 3])).powi(3))
    };

    // The tint: how far the estimate is from the truth, per channel, in unclipped
    // pixels within 3 pixels of that channel clipping.
    let near = dilate(mask, width, height, 3);
    let (sum, count) = (0..height)
        .into_par_iter()
        .map(|y| {
            let (mut sum, mut count) = ([0.0f64; 3], [0u32; 3]);
            for x in 0..width {
                let near = near[y * width + x];
                if near == 0 {
                    continue;
                }
                let mut estimate = None;
                for c in 0..3 {
                    let v = value(x, y, c);
                    if near & (1 << c) != 0 && v > dark[c] && v < clip[c] {
                        let e = estimate.get_or_insert_with(|| opposed(x, y));
                        sum[c] += (v - e[c]) as f64;
                        count[c] += 1;
                    }
                }
            }
            (sum, count)
        })
        .reduce(
            || ([0.0; 3], [0; 3]),
            |a, b| {
                (
                    std::array::from_fn(|c| a.0[c] + b.0[c]),
                    std::array::from_fn(|c| a.1[c] + b.1[c]),
                )
            },
        );
    let tint: [f32; 3] = std::array::from_fn(|c| (sum[c] / count[c].max(1) as f64) as f32);

    (0..height)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..width).filter_map(move |x| {
                let i = y * width + x;
                if mask[i] == 0 {
                    return None;
                }
                let e = opposed(x, y);
                let clipped = |c: usize| mask[i] & (1 << c) != 0;
                let mut p = pixels[i];
                for c in (0..3).filter(|&c| clipped(c)) {
                    p[c] = p[c].max((e[c] + tint[c]) / wb[c]);
                }
                // With two or three channels clipped, the estimates lean on clipped
                // values, which are only lower bounds, so the one channel still read
                // (or, with all three clipped, the white balance) set the colour:
                // blown clouds and a blown sun came out pink. A clipped channel is
                // at least as bright as the channel still read, and a pixel with
                // nothing left to read is neutral.
                if mask[i].count_ones() >= 2 {
                    let read = (0..3).filter(|&c| !clipped(c));
                    let floor = if mask[i] == 0b111 {
                        (0..3).map(|c| p[c] * wb[c]).fold(0.0f32, f32::max)
                    } else {
                        read.map(|c| p[c] * wb[c]).fold(0.0f32, f32::max)
                    };
                    for c in (0..3).filter(|&c| clipped(c)) {
                        p[c] = p[c].max(floor / wb[c]);
                    }
                }
                Some((i, p))
            })
        })
        .collect()
}

/// `mask` OR-ed over a (2r+1)-pixel square around each pixel.
fn dilate(mask: &[u8], width: usize, height: usize, r: usize) -> Vec<u8> {
    let mut rows = vec![0u8; mask.len()];
    rows.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let src = &mask[y * width..(y + 1) * width];
        for (x, out) in row.iter_mut().enumerate() {
            *out = src[x.saturating_sub(r)..=(x + r).min(width - 1)].iter().fold(0, |a, b| a | b);
        }
    });
    let mut out = vec![0u8; mask.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for yy in y.saturating_sub(r)..=(y + r).min(height - 1) {
            for (o, v) in row.iter_mut().zip(&rows[yy * width..(yy + 1) * width]) {
                *o |= v;
            }
        }
    });
    out
}

fn develop_internal(
    file_bytes: &[u8],
    fast_demosaic: bool,
    _highlight_compression: f32,
    linear_mode: String,
    use_camera_profiles: bool,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<(DynamicImage, Orientation)> {
    let check_cancel = || -> Result<()> {
        if let Some((tracker, generation)) = &cancel_token
            && tracker.load(Ordering::SeqCst) != *generation
        {
            return Err(anyhow!("Load cancelled"));
        }
        Ok(())
    };

    check_cancel()?;

    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source)?;

    check_cancel()?;
    let mut raw_image: RawImage = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;

    let orientation = metadata_orientation(decoder.as_ref(), &source)?;

    let is_linear_format = is_linear_raw_format(&raw_image);

    let (apply_ungamma, apply_calibration) = match linear_mode.as_str() {
        "gamma" => (true, true),
        "skip_calib" => (false, false),
        "gamma_skip_calib" => (true, false),
        _ => (false, true),
    };

    let original_white_level = raw_image
        .whitelevel
        .0
        .first()
        .cloned()
        .unwrap_or(u16::MAX as u32) as f32;
    let original_black_level = raw_image
        .blacklevel
        .levels
        .first()
        .map(|r| r.as_f32())
        .unwrap_or(0.0);

    for level in raw_image.whitelevel.0.iter_mut() {
        *level = u32::MAX;
    }

    let mut developer = RawDevelop::default();

    if is_linear_format {
        developer.steps.retain(|&step| {
            step != ProcessingStep::SRgb
                && step != ProcessingStep::Demosaic
                && (apply_calibration || step != ProcessingStep::Calibrate)
        });
    } else if fast_demosaic {
        developer.demosaic_algorithm = DemosaicAlgorithm::Speed;
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    } else {
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    }

    raw_image.wb_coeffs =
        crate::multi_exposure::neutralize_wb_if_multiexposure(raw_image.wb_coeffs, file_bytes);

    // Colour is applied here rather than in rawler's Calibrate step, so highlights
    // can be rebuilt in camera RGB first: the camera's own profile when there is
    // one, else rawler's generic matrix.
    let [r, g, b, _] = raw_image.wb_coeffs;
    let wb = if r.is_nan() { [1.0; 3] } else { [r, g, b] };
    let calibration = if !apply_calibration {
        None
    } else {
        use_camera_profiles
            .then(|| crate::camera_profile::for_raw(&raw_image).and_then(|p| p.calibration(wb)))
            .flatten()
            .or_else(|| crate::camera_profile::Calibration::from_raw_matrix(&raw_image))
    };
    if calibration.is_some() {
        developer
            .steps
            .retain(|&step| step != ProcessingStep::Calibrate);
    }

    check_cancel()?;
    let mut developed_intermediate = developer.develop_intermediate(&raw_image)?;

    drop(raw_image);

    let denominator = (original_white_level - original_black_level).max(1.0);
    let rescale_factor = (u32::MAX as f32 - original_black_level) / denominator;

    let safe_highlight_compression = 1000.0;

    let clamp_limit = if fast_demosaic {
        1.0
    } else {
        safe_highlight_compression
    };

    let (width, height) = {
        let dim = developed_intermediate.dim();
        (dim.w as u32, dim.h as u32)
    };

    check_cancel()?;

    match &mut developed_intermediate {
        Intermediate::Monochrome(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut linear_val = *p * rescale_factor;
                if is_linear_format && apply_ungamma {
                    linear_val = srgb_to_linear(linear_val.max(0.0));
                }
                *p = linear_val.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::ThreeColor(pixels) => {
            pixels.data.par_iter_mut().for_each(|p| {
                for v in p.iter_mut() {
                    *v = (*v * rescale_factor).max(0.0);
                    if is_linear_format && apply_ungamma {
                        *v = srgb_to_linear(*v);
                    }
                }
            });
            if let Some(calibration) = &calibration {
                reconstruct_highlights(&mut pixels.data, width as usize, height as usize, wb);
                pixels.data.par_iter_mut().for_each(|p| {
                    *p = calibration.apply(*p).map(|v| v.min(clamp_limit));
                });
            } else {
                pixels.data.par_iter_mut().for_each(|p| {
                    *p = p.map(|v| v.min(clamp_limit));
                });
            }
        }
        Intermediate::FourColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                p.iter_mut().for_each(|c| {
                    let mut linear_val = *c * rescale_factor;
                    if is_linear_format && apply_ungamma {
                        linear_val = srgb_to_linear(linear_val.max(0.0));
                    }
                    *c = linear_val.clamp(0.0, clamp_limit);
                });
            });
        }
    }

    check_cancel()?;

    if let Intermediate::ThreeColor(pixels) = &mut developed_intermediate
        && let Some(distortion) = crate::embedded_lens::read_distortion(file_bytes)
    {
        let corrected = crate::embedded_lens::correct(
            pixels.data.as_flattened(),
            width as usize,
            height as usize,
            &distortion,
        );
        pixels.data = corrected.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    }

    let dynamic_image = match developed_intermediate {
        Intermediate::ThreeColor(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p[0], p[1], p[2], 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        Intermediate::Monochrome(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p, p, p, 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        _ => {
            return Err(anyhow!("Unsupported intermediate format for conversion"));
        }
    };

    Ok((dynamic_image, orientation))
}

pub fn read_as_shot_white_balance(file_bytes: &[u8]) -> Option<WhiteBalance> {
    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source).ok()?;
    let raw_image = decoder
        .raw_image(&source, &RawDecodeParams::default(), true)
        .ok()?;
    if raw_image.cpp == 1 && !matches!(raw_image.photometric, RawPhotometricInterpretation::Cfa(_))
    {
        return None;
    }

    let wb_coeffs =
        crate::multi_exposure::neutralize_wb_if_multiexposure(raw_image.wb_coeffs, file_bytes);
    let neutral = if wb_coeffs[0].is_nan() {
        [1.0; 4]
    } else {
        wb_coeffs.map(|c| 1.0 / c)
    };

    let matrices = &raw_image.color_matrix;
    if let (Some(matrix_a), Some(matrix_d65)) =
        (matrices.get(&Illuminant::A), matrices.get(&Illuminant::D65))
    {
        return WhiteBalance::from_dual_illuminant_camera_neutral(matrix_a, matrix_d65, &neutral);
    }

    let color_matrix = matrices
        .get(&Illuminant::D65)
        .or_else(|| matrices.values().next())?;
    WhiteBalance::from_camera_neutral(color_matrix, &neutral)
}

pub fn get_fast_demosaic_scale_factor(
    file_bytes: &[u8],
    decoded_width: u32,
    decoded_height: u32,
) -> f32 {
    let source = RawSource::new_from_slice(file_bytes);
    if let Ok(decoder) = rawler::get_decoder(&source)
        && let Ok(raw_img) = decoder.raw_image(&source, &RawDecodeParams::default(), true)
    {
        let max_orig = (raw_img.width as f32).max(raw_img.height as f32);
        let max_comp = (decoded_width as f32).max(decoded_height as f32);
        if max_orig > 0.0 {
            let ratio = max_comp / max_orig;
            if ratio > 0.1 && ratio < 0.35 {
                return 0.25;
            } else if (0.35..0.75).contains(&ratio) {
                return 0.5;
            }
        }
    }
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opposed_rebuilds_only_clipped_channels() {
        // A neutral scene under white balance [2, 1, 1.5]: grey 0.6 around a bright
        // grey 1.3 square, where green (only) hits the white level. Unfixed, the
        // square is magenta; rebuilt, green rises towards red and blue, and nothing
        // else changes.
        let (w, h) = (24, 24);
        let wb = [2.0, 1.0, 1.5];
        let camera = |grey: f32| [grey / wb[0], (grey / wb[1]).min(1.0), grey / wb[2]];
        let mut pixels: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                camera(if (9..15).contains(&x) && (9..15).contains(&y) { 1.3 } else { 0.6 })
            })
            .collect();
        let before = pixels.clone();
        reconstruct_highlights(&mut pixels, w, h, wb);

        let centre = pixels[12 * w + 12];
        assert!(centre[1] > 1.2, "green not rebuilt: {centre:?}");
        assert_eq!([centre[0], centre[2]], [before[12 * w + 12][0], before[12 * w + 12][2]]);
        for (p, q) in pixels.iter().zip(&before) {
            if q[1] < CLIP {
                assert_eq!(p, q);
            }
        }
    }

    #[test]
    fn blown_white_stays_neutral() {
        // A white cloud under white balance [2.4, 1, 1.5], bright enough that green
        // and blue clip but red doesn't, and a sun where all three clip. They came
        // out pink: the clipped channels stayed below red.
        let (w, h) = (24, 24);
        let wb = [2.4, 1.0, 1.5];
        let camera = |grey: f32| [0, 1, 2].map(|c| (grey / wb[c]).min(1.0));
        let mut pixels: Vec<[f32; 3]> = (0..w * h)
            .map(|i| camera(if i % w < 12 { 2.2 } else { 3.0 }))
            .collect();
        reconstruct_highlights(&mut pixels, w, h, wb);

        for (name, p) in [("cloud", pixels[12 * w + 4]), ("sun", pixels[12 * w + 20])] {
            let balanced = [0, 1, 2].map(|c| p[c] * wb[c]);
            assert!(balanced[1] >= balanced[0] && balanced[2] >= balanced[0], "{name} {balanced:?}");
        }
        let sun = pixels[12 * w + 20];
        let balanced = [0, 1, 2].map(|c| sun[c] * wb[c]);
        assert!(balanced.iter().all(|&v| (v - balanced[0]).abs() < 1e-4), "sun {balanced:?}");
    }
}
