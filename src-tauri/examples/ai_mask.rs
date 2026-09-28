// cargo run --release --example ai_mask -- sky|foreground|depth|subject:X1,Y1,X2,Y2 IMAGE OUT
// Writes the data URL the app stores in a sub-mask's maskDataBase64 to OUT,
// plus the mask itself to OUT.png for checking. Subject box is in image pixels.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [spec, image, out] = &args[..] else {
        eprintln!("usage: ai_mask sky|foreground|depth|subject:X1,Y1,X2,Y2 IMAGE OUT");
        std::process::exit(2);
    };
    let (kind, rect) = spec.split_once(':').unwrap_or((spec, "0,0,0,0"));
    let r: Vec<f64> = rect.split(',').map(|v| v.parse().expect("box number")).collect();
    let url = rapidraw_lib::dev_ai_mask(image, kind, (r[0], r[1], r[2], r[3])).unwrap();
    std::fs::write(out, &url).unwrap();
    use base64::Engine;
    let png = base64::engine::general_purpose::STANDARD.decode(&url[url.find(',').unwrap() + 1..]).unwrap();
    std::fs::write(format!("{out}.png"), png).unwrap();
}
