// cargo run --release --example camera_match -- IMAGE...
// Prints the Match camera slider values for each RAW as JSON lines.
fn main() {
    for path in std::env::args().skip(1) {
        match rapidraw_lib::camera_match::dev_fit(&path) {
            Ok(m) => println!(
                "{{\"path\":\"{}\",\"ev\":{},\"shadows\":{},\"residual\":{}}}",
                path, m.exposure_ev, m.shadows, m.residual
            ),
            Err(e) => eprintln!("{}: {}", path, e),
        }
    }
}
