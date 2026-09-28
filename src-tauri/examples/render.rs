// cargo run --release --example render -- OUTDIR 'ADJUSTMENTS_JSON' IMAGE...
// Renders each image as the editor does (patches, geometry, masks, full GPU
// pipeline) to OUTDIR/<name>.png, e.g.
// render out '{"exposure":1,"shadows":50,"toneMapper":"profile"}' a.ARW
// ADJUSTMENTS_JSON may be @FILE, for adjustments too big for the command line (AI masks).
// Env: LONG_EDGE (default 1600; larger than the photo = full size), FORMAT=jpg
// for OUTDIR/<name>.jpg encoded as the editor exports it (JPEG_QUALITY, default 95).
fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(out_dir), Some(adjustments)) = (args.next(), args.next()) else {
        eprintln!("usage: render OUTDIR ADJUSTMENTS_JSON IMAGE...");
        std::process::exit(2);
    };
    let adjustments = match adjustments.strip_prefix('@') {
        Some(file) => std::fs::read_to_string(file).expect("adjustments file"),
        None => adjustments,
    };
    let adjustments: serde_json::Value =
        serde_json::from_str(&adjustments).expect("adjustments JSON");
    let env = |k: &str| std::env::var(k).ok();
    let long_edge = env("LONG_EDGE").map_or(1600, |v| v.parse().expect("LONG_EDGE"));
    let jpeg_quality = env("JPEG_QUALITY").map_or(95, |v| v.parse().expect("JPEG_QUALITY"));
    let jpeg = match env("FORMAT").as_deref() {
        None | Some("png") => false,
        Some("jpg") => true,
        Some(f) => panic!("FORMAT {f}: png or jpg"),
    };
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut failed = false;
    for path in args {
        let name = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        match rapidraw_lib::dev_render(&path, &adjustments, long_edge) {
            Ok(img) if jpeg => {
                let file = std::fs::File::create(format!("{out_dir}/{name}.jpg")).unwrap();
                let mut encoder =
                    image::codecs::jpeg::JpegEncoder::new_with_quality(file, jpeg_quality);
                encoder.encode_image(&img).unwrap();
            }
            Ok(img) => img.save(format!("{out_dir}/{name}.png")).unwrap(),
            Err(e) => {
                eprintln!("{path}: {e}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
