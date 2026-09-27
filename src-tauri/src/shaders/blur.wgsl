// The blurs the main pass reads, built once per uploaded image: a separable Gaussian,
// cut off at twice its sigma.

struct BlurParams {
    radius: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: BlurParams;

const F16_MAX = 65504.0;

fn load_clamped(coord: vec2<i32>) -> vec3<f32> {
    let last = vec2<i32>(textureDimensions(input_texture)) - 1;
    let color = textureLoad(input_texture, clamp(coord, vec2<i32>(0), last), 0).rgb;
    return clamp(color, vec3(0.0), vec3(F16_MAX));
}

fn gaussian(center: vec2<i32>, step: vec2<i32>) -> vec4<f32> {
    let radius = i32(params.radius);
    let sigma = f32(radius) / 2.0;
    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;
    for (var offset = -radius; offset <= radius; offset = offset + 1) {
        let x = f32(offset);
        let weight = exp(-(x * x) / (2.0 * sigma * sigma));
        total_color += load_clamped(center + step * offset) * weight;
        total_weight += weight;
    }
    return vec4<f32>(total_color / total_weight, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn horizontal_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output_texture))) {
        return;
    }
    textureStore(output_texture, id.xy, gaussian(vec2<i32>(id.xy), vec2<i32>(1, 0)));
}

@compute @workgroup_size(8, 8, 1)
fn vertical_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output_texture))) {
        return;
    }
    textureStore(output_texture, id.xy, gaussian(vec2<i32>(id.xy), vec2<i32>(0, 1)));
}
