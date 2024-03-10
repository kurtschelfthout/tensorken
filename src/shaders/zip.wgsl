alias element = f32;

@group(0) @binding(0)
var<storage, read> input_0: array<element>;

@group(0) @binding(1)
var<storage, read> input_1: array<element>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<element>;

// WebGPU afaik does not support structs with dynamic array lengths.
// So to encode our necessary strides and shape in one struct,
// we concatenate them all in one array, and the first element
// is the number of dimensions.
// ndims, offset_0, offset_1, chunk_size, strides_0, strides_1, output_strides, shape
// These are all elementwise ops, so shape must be identical for all.
@group(0) @binding(3)
var<storage, read> strides_and_shape: array<u32>;

const preamble = 4u;

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
fn input_index_of(output_i: u32) -> vec2<u32> {
    let ndims = strides_and_shape[0];
    let offset = vec2(strides_and_shape[1],strides_and_shape[2]);
    
    var input_i: vec2<u32> = offset;
    for (var i: u32 = 0u; i < ndims; i = i + 1u) {
        let len = shape(i);
        let stride = output_strides(i);
        let coord: u32 = output_i / stride % len;

        input_i += coord * vec2(input_0_strides(i), input_1_strides(i));
    }
    return input_i;
}

// Same parlor trick as in map.wgsl.
fn replace_me_with_actual_operation(in_1: element, in_2: element) -> element { discard; }

// extension to the parlor trick: infix operators are annoying to replace, 
// so we define counterparts for them here.
fn add(a: element, b: element) -> element { return a + b; }
fn sub(a: element, b: element) -> element { return a - b; }
fn mul(a: element, b: element) -> element { return a * b; }
fn div(a: element, b: element) -> element { return a / b; }
fn eq(a: element, b: element) -> element { return element(a == b); }
// pow is a builtin, no need to define it


@compute
@workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let fro = global_id.x * strides_and_shape[3];
    let to = fro + strides_and_shape[3];
    for (var gidx = fro; gidx < to; gidx = gidx + 1u) {
        if(gidx >= arrayLength(&output_0)) {
            return;
        }
        let indexes = input_index_of(gidx);
        output_0[gidx] = replace_me_with_actual_operation(input_0[ indexes[0] ], input_1[ indexes[1] ]);
    }
}
