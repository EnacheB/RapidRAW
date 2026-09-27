// Builds the next mip of the rendered photo: each texel is the mean of the 2x2 above it.

@group(0) @binding(0) var source_mip: texture_2d<f32>;
@group(0) @binding(1) var output_mip: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output_mip))) {
        return;
    }
    let last = vec2<i32>(textureDimensions(source_mip)) - 1;
    let origin = vec2<i32>(id.xy) * 2;
    var total = vec4<f32>(0.0);
    for (var y = 0; y < 2; y = y + 1) {
        for (var x = 0; x < 2; x = x + 1) {
            total += textureLoad(source_mip, min(origin + vec2<i32>(x, y), last), 0);
        }
    }
    textureStore(output_mip, id.xy, total / 4.0);
}
