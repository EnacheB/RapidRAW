// cargo run --release --example face_auto -- [--agx|--profile] OUT_DIR IMAGE...
fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match args.first().map(String::as_str) {
        Some("--agx") => 1,
        Some("--profile") => rapidraw_lib::camera_profile::TONE_MAPPER_MODE,
        _ => 0,
    };
    if mode != 0 {
        args.remove(0);
    }
    let out_dir = std::path::PathBuf::from(args.remove(0));
    std::fs::create_dir_all(&out_dir).unwrap();
    for path in args {
        let name = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        let png = out_dir.join(format!("{}.png", name));
        match rapidraw_lib::face_exposure::dev_report(&path, png.to_str().unwrap(), mode) {
            Ok(v) => println!("{}", v),
            Err(e) => eprintln!("{}: {}", path, e),
        }
    }
}
