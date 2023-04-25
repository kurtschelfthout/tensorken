
@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_0: array<f32>;

// ndims, input_strides, output_strides, shape
@group(0) @binding(2)
var<storage, read> strides_and_shape: array<u32>;


fn input_strides(i: u32) -> u32 {
    return strides_and_shape[i+1u];
}

fn output_strides(i: u32) -> u32 {
    return strides_and_shape[i + 1u + strides_and_shape[0] ];
}

fn shape(i: u32) -> u32 {
    return strides_and_shape[i + 1u + strides_and_shape[0] * 2u];
}

// Find the index of the given output index in input_0.
fn input_index_of(output_i: u32) -> u32 {
    var input_i: u32 = 0u;
    
    for (var i: u32 = 0u; i < strides_and_shape[0]; i = i + 1u) {
        let len = shape(i);
        let stride = output_strides(i);
        let coord: u32 = output_i / stride % len;

        input_i += coord * input_strides(i);
    }

    return input_i;
}

@compute @workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    // because of workgroup size, gidx is a multiple of 64. Our output array may not be,
    // so we need to make sure we don't go out of bounds. Such acesses are clamped by WGSL,
    // but will typically result in wrong results anyway.
    if(global_id.x >= arrayLength(&output_0)) {
        return;
    }
    let index = input_index_of(gidx);
    output_0[gidx] = input_0[index];
}