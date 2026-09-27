// cargo run --release --example camera_profile -- OUT_DIR IMAGE...
// Decodes each RAW with its camera profile and renders it with the "Profile" tone
// mapper at default settings.
fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = std::path::PathBuf::from(args.next().expect("OUT_DIR"));
    std::fs::create_dir_all(&out_dir).unwrap();
    for path in args {
        let name = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        match rapidraw_lib::camera_profile::dev_render(&path, 900) {
            Ok((profile, image)) => {
                image.save(out_dir.join(format!("{}.png", name))).unwrap();
                println!("{}: {}", path, profile);
            }
            Err(e) => eprintln!("{}: {}", path, e),
        }
    }
}
