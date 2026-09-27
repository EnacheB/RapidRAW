struct Transform {
    rect: vec4<f32>,
    clip: vec4<f32>,
    window: vec2<f32>,
    pixelated: f32,
    _pad: f32,
    bg_primary: vec4<f32>,
    bg_secondary: vec4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
    );
    let pos = uvs[id];

    let uv_x = transform.clip.x + pos.x * transform.clip.z;
    let uv_y = transform.clip.y + pos.y * transform.clip.w;

    let half_pixel_x = 0.5;
    let half_pixel_y = 0.5;
    let outset_x = (pos.x * 2.0 - 1.0) * half_pixel_x;
    let outset_y = (pos.y * 2.0 - 1.0) * half_pixel_y;

    let screen_x = uv_x + outset_x;
    let screen_y = uv_y + outset_y;

    let ndc_x = (screen_x / transform.window.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_y / transform.window.y) * 2.0;

    var out: VertexOutput;
    out.pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    out.uv = vec2<f32>(
        (uv_x - transform.rect.x) / transform.rect.z,
        (uv_y - transform.rect.y) / transform.rect.w
    );

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.uv.x < 0.0 || in.uv.x > 1.0 || in.uv.y < 0.0 || in.uv.y > 1.0) {
        return transform.bg_secondary;
    }

    let size = textureDimensions(tex);
    if (transform.pixelated > 0.5) {
        let texel = min(vec2<u32>(in.uv * vec2<f32>(size)), size - 1u);
        return textureLoad(tex, texel, 0);
    }

    // The texture is the whole photo at full resolution, with mips. Read the finest mip
    // that is at most twice the size on screen, where a screen pixel covers `footprint`
    // (1 to 2) texels. A bilinear tap averages a 1-texel box, so two taps per axis, set
    // (footprint - 1) / 2 texels apart from the centre, average the pixel's whole
    // footprint: zoomed-out views are neither blurry nor aliased, and 1:1 stays exact.
    let texels_per_pixel = max(f32(size.x) / transform.rect.z, f32(size.y) / transform.rect.w);
    let level = floor(log2(max(texels_per_pixel, 1.0)));
    let footprint = max(texels_per_pixel / exp2(level), 1.0);
    let offset = (footprint - 1.0) / (2.0 * footprint) / transform.rect.zw;
    var color = vec4<f32>(0.0);
    for (var y = -1.0; y <= 1.0; y += 2.0) {
        for (var x = -1.0; x <= 1.0; x += 2.0) {
            color += textureSampleLevel(tex, samp, in.uv + vec2<f32>(x, y) * offset, level);
        }
    }
    return color / 4.0;
}
