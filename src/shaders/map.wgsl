
@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_0: array<f32>;

// A parlor trick:
// All compute shaders with unary operations have essentially the same form,
// with just the operation changing. To avoid duplication, but still have syntactically
// valid WGSL, we define a function here with a sufficiently unique name.
// On the rust side, we have to do two things, in order:
// 1. Remove the function definition
// 2. Replace the function name with the actual operation name
// (if we don't do step 1, we'll also replace the name in the definition, which on my GPU causes a crash...)

fn replace_me_with_actual_operation(in: f32) -> f32 { discard; }

@compute @workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    // because of workgroup size, gidx is a multiple of 64. Our output array may not be,
    // so we need to make sure we don't go out of bounds. Such acesses are clamped by WGSL,
    // but will typically result in wrong results anyway.
    if(global_id.x >= arrayLength(&output_0)) {
        return;
    }
    output_0[gidx] = replace_me_with_actual_operation(input_0[gidx]);
}