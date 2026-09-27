// DNG camera profile look: the "profile" tone mapper. The profile's colour
// (matrices and HueSatMap) was already applied when the RAW was decoded.
// Mirrors `CameraProfile::render` in camera_profile.rs; the parity test there
// compiles this file on its own.

struct CameraProfile {
    // Working linear Rec.709 to linear ProPhoto, including BaselineExposureOffset.
    input_matrix: mat3x3<f32>,
    // Linear ProPhoto to working linear Rec.709.
    output_matrix: mat3x3<f32>,
    // Hue, saturation and value divisions (hue 0 = no table), first entry in `data`.
    look_table: vec4<u32>,
    // Sample count (0 = no curve), first vec4 in `data` (4 samples per vec4).
    tone_curve: vec4<u32>,
    // Enabled, look table sRGB-encoded.
    flags: vec4<u32>,
    data: array<vec4<f32>>,
}

@group(0) @binding(14) var<storage, read> camera_profile: CameraProfile;

fn cp_srgb_encode(v: f32) -> f32 {
    if (v <= 0.0031308) { return v * 12.92; }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn cp_srgb_decode(v: f32) -> f32 {
    if (v <= 0.04045) { return v / 12.92; }
    return pow((v + 0.055) / 1.055, 2.4);
}

// DNG SDK DNG_RGBtoHSV: hue in [0, 6).
fn cp_rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let v = max(c.r, max(c.g, c.b));
    let gap = v - min(c.r, min(c.g, c.b));
    if (gap <= 0.0) { return vec3<f32>(0.0, 0.0, v); }
    var h: f32;
    if (c.r == v) {
        h = (c.g - c.b) / gap;
        if (h < 0.0) { h += 6.0; }
    } else if (c.g == v) {
        h = 2.0 + (c.b - c.r) / gap;
    } else {
        h = 4.0 + (c.r - c.g) / gap;
    }
    return vec3<f32>(h, gap / v, v);
}

fn cp_hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    var h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;
    if (s <= 0.0) { return vec3<f32>(v); }
    if (h < 0.0) { h += 6.0; }
    if (h >= 6.0) { h -= 6.0; }
    let i = min(u32(h), 5u);
    let f = h - f32(i);
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    switch i {
        case 0u: { return vec3<f32>(v, t, p); }
        case 1u: { return vec3<f32>(q, v, p); }
        case 2u: { return vec3<f32>(p, v, t); }
        case 3u: { return vec3<f32>(p, q, v); }
        case 4u: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

fn cp_entry(t: vec4<u32>, h: u32, s: u32, v: u32) -> vec3<f32> {
    return camera_profile.data[t.w + (v * t.x + h) * t.y + s].xyz;
}

fn cp_bilinear(t: vec4<u32>, h0: u32, h1: u32, s0: u32, v: u32, hf: f32, sf: f32) -> vec3<f32> {
    let lo = mix(cp_entry(t, h0, s0, v), cp_entry(t, h1, s0, v), hf);
    let hi = mix(cp_entry(t, h0, s0 + 1u, v), cp_entry(t, h1, s0 + 1u, v), hf);
    return mix(lo, hi, sf);
}

// DNG SDK RefBaselineHueSatMap (here for the LookTable), without clipping value.
fn cp_apply_hsv_table(c: vec3<f32>, t: vec4<u32>, srgb: bool) -> vec3<f32> {
    let hsv = cp_rgb_to_hsv(max(c, vec3<f32>(0.0)));
    var h = hsv.x;
    var s = hsv.y;
    var v = hsv.z;
    let h_scaled = h * f32(t.x) / 6.0;
    let s_scaled = s * f32(t.y - 1u);
    var h0 = u32(h_scaled);
    let s0 = min(u32(s_scaled), t.y - 2u);
    var h1 = h0 + 1u;
    if (h0 >= t.x - 1u) {
        h0 = t.x - 1u;
        h1 = 0u;
    }
    let hf = h_scaled - f32(h0);
    let sf = s_scaled - f32(s0);
    var v_encoded = v;
    var m: vec3<f32>;
    if (t.z < 2u) {
        m = cp_bilinear(t, h0, h1, s0, 0u, hf, sf);
    } else {
        if (srgb) { v_encoded = cp_srgb_encode(clamp(v, 0.0, 1.0)); }
        let v_scaled = v_encoded * f32(t.z - 1u);
        let v0 = min(u32(v_scaled), t.z - 2u);
        let vf = v_scaled - f32(v0);
        m = mix(cp_bilinear(t, h0, h1, s0, v0, hf, sf), cp_bilinear(t, h0, h1, s0, v0 + 1u, hf, sf), vf);
    }
    h += m.x * (6.0 / 360.0);
    s = min(s * m.y, 1.0);
    if (srgb) {
        v = cp_srgb_decode(clamp(v_encoded * m.z, 0.0, 1.0));
    } else {
        v *= m.z;
    }
    return cp_hsv_to_rgb(vec3<f32>(h, s, v));
}

fn cp_tone_sample(i: u32) -> f32 {
    let v = camera_profile.data[camera_profile.tone_curve.y + i / 4u];
    return v[i % 4u];
}

fn cp_tone(x: f32) -> f32 {
    let n = camera_profile.tone_curve.x;
    let pos = clamp(x, 0.0, 1.0) * f32(n - 1u);
    let i = min(u32(pos), n - 2u);
    return mix(cp_tone_sample(i), cp_tone_sample(i + 1u), pos - f32(i));
}

// DNG SDK RefBaselineRGBTone: hue-preserving tone curve.
fn cp_rgb_tone(c_in: vec3<f32>) -> vec3<f32> {
    let c = clamp(c_in, vec3<f32>(0.0), vec3<f32>(1.0));
    let hi = max(c.r, max(c.g, c.b));
    let lo = min(c.r, min(c.g, c.b));
    let hi_t = cp_tone(hi);
    let lo_t = cp_tone(lo);
    if (hi - lo <= 0.0) { return vec3<f32>(hi_t); }
    return lo_t + (hi_t - lo_t) * (c - lo) / (hi - lo);
}

// Linear working Rec.709 to sRGB-encoded display values.
fn camera_profile_render(linear_rec709: vec3<f32>) -> vec3<f32> {
    var c = camera_profile.input_matrix * linear_rec709;
    if (camera_profile.look_table.x > 0u) {
        c = cp_apply_hsv_table(c, camera_profile.look_table, camera_profile.flags.y == 1u);
    }
    if (camera_profile.tone_curve.x > 0u) {
        c = cp_rgb_tone(c);
    } else {
        c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let out = clamp(camera_profile.output_matrix * c, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec3<f32>(cp_srgb_encode(out.r), cp_srgb_encode(out.g), cp_srgb_encode(out.b));
}
