struct Uniforms {
    mvp: mat4x4<f32>, 
    inv_mvp: mat4x4<f32>, 
}

struct Ray {
  start : vec3f,
  dir   : vec3f,
}


@group(0) @binding(0) var framebuffer : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> uniforms : Uniforms;

override WORKGROUP_SIZE_X : u32;
override WORKGROUP_SIZE_Y : u32;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn main(@builtin(global_invocation_id) invocation_id : vec3u) {
	let uv = vec2f(invocation_id.xy) / vec2f(textureDimensions(framebuffer).xy);
	let ndcXY = (uv - 0.5) * vec2(2, -2);
	var near = uniforms.inv_mvp * vec4f(ndcXY, 0.0, 1);
    var far = uniforms.inv_mvp * vec4f(ndcXY, 1, 1);
    near /= near.w;
    far /= far.w;
	let ray = Ray(near.xyz, normalize(far.xyz - near.xyz));
	textureStore(framebuffer, invocation_id.xy, vec4(abs(ray.dir), 1));
}