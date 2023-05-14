@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_0: array<f32>;

// ndims, input_offset, chunk_size, input_strides, output_strides, output_shape
@group(0) @binding(2)
var<storage, read> strides_and_shape: array<u32>;

const preamble: u32 = 3u;

fn input_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble];
}

fn output_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] ];
}

fn output_shape(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 2u];
}

// A parlor trick:
// All compute shaders with unary operations have essentially the same form,
// with just the operation changing. To avoid duplication, but still have syntactically
// valid WGSL, we define a function here with a sufficiently unique name.
// On the rust side, we have to do two things, in order:
// 1. Remove the function definition
// 2. Replace the function name with the actual operation name
// (if we don't do step 1, we'll also replace the name in the definition, which on my GPU causes a crash...)

fn replace_me_with_actual_operation(in: f32) -> f32 { discard; }

// The identity functions is used for ravel, i.e. making a given tensor contiugous.
fn id(in: f32) -> f32 {
    return in;
}

// Find the index of the given output index in input_0.
fn input_index_of(output_i: u32) -> u32 {
    var input_i: u32 = strides_and_shape[1];
    
    // this works by transforming the output_i buffer index into a 
    // tensor coordinate. coord_i represents the i'th item in the tensor coordinate.
    // This is multiplied by the i'th stride of the input tensor, giving
    // us a buffer index in the input, input_i.
    for (var i: u32 = 0u; i < strides_and_shape[0]; i = i + 1u) {
        let len = output_shape(i);
        let stride = output_strides(i);
        let coord_i: u32 = output_i / stride % len;

        input_i += coord_i * input_strides(i);
    }

    return input_i;
}

@compute
@workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let fro = global_id.x * strides_and_shape[2];
    let to = fro + strides_and_shape[2];
    for (var gidx = fro; gidx < to; gidx = gidx + 1u) {
        // gidx is a multiple of workgroup size. Our output array may not be,
        // so we need to make sure we don't go out of bounds. Such acesses are clamped by WGSL,
        // but will typically result in wrong results anyway.
        if(gidx >= arrayLength(&output_0)) {
            return;
        }
        let index = input_index_of(gidx);
        output_0[gidx] = replace_me_with_actual_operation(input_0[index]);
    }
}