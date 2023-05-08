@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<f32>;

// WebGPU afaik does not support structs with dynamic array lengths.
// So to encode our necessary strides and shape in one struct,
// we concatenate them all in one array, and the first element
// is the number of dimensions.
// ndims, offset_0, offset_1, strides_0, strides_1, output_strides, shape
// These are all elementwise ops, so shape must be identical for all
@group(0) @binding(3)
var<storage, read> strides_and_shape: array<u32>;

const preamble = 3u;

fn input_0_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble];
}

fn input_1_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] ];
}

fn output_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 2u];
}

fn shape(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 3u];
}

// Find the indexes of the element in input 0 and input 1.
fn input_index_of(index: u32) -> vec2<u32> {
    var index_0: u32 = strides_and_shape[1];
    var index_1: u32 = strides_and_shape[2];
    
    for (var i: u32 = 0u; i < strides_and_shape[0]; i = i + 1u) {
        let len = shape(i);
        let stride = output_strides(i);
        let coord: u32 = index / stride % len;

        index_0 += coord * input_0_strides(i);
        index_1 += coord * input_1_strides(i);
    }

    return vec2(index_0, index_1);
}

// Same parlor trick as in unary_ops.wgsl.
fn replace_me_with_actual_operation(in_1: f32, in_2: f32) -> f32 { discard; }

// extension to the parlor trick: infix operators are annoying to replace, 
// so we define counterparts for them here.

fn add(a: f32, b: f32) -> f32 { return a + b; }
fn sub(a: f32, b: f32) -> f32 { return a - b; }
fn mul(a: f32, b: f32) -> f32 { return a * b; }
fn div(a: f32, b: f32) -> f32 { return a / b; }
fn eq(a: f32, b: f32) -> f32 { return f32(a == b); }


@compute @workgroup_size(256)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    if(global_id.x >= arrayLength(&output_0)) {
        return;
    }
    let indexes = input_index_of(gidx);
    output_0[gidx] = replace_me_with_actual_operation(input_0[ indexes[0] ], input_1[ indexes[1] ]);
}
