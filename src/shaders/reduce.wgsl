
@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_0: array<f32>;

// ndims, input_offset, chunk_size, input_strides, output_strides, input_shape, reduced_strides,  reduced_shape, output_shape
@group(0) @binding(2)
var<storage, read> strides_and_shape: array<u32>;

const preamble: u32 = 3u;

fn input_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble];
}

fn output_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 1u];
}

fn input_shape(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 2u];
}

fn reduced_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 3u];
}

fn reduced_shape(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 4u];
}

fn output_shape(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 5u];
}

// Same parlor trick as in unary_ops.wgsl.
fn replace_me_with_actual_operation(in_1: f32, in_2: f32) -> f32 { discard; }
fn replace_me_with_actual_default() -> f32 { discard; }

// extension to the parlor trick: infix operators are annoying to replace, 
// so we define counterparts for them here.
fn sum(a: f32, b: f32) -> f32 { return a + b; }

// default values for the reduce
const MAX: f32 = -1.17549435082228750797e-38f;
const SUM: f32 = 0.0;

fn sizes() -> vec2<u32> {
    var size = vec2(1u, 1u);
    for (var i: u32 = 0u; i < strides_and_shape[0]; i += 1u) {
        size *= vec2(input_shape(i), output_shape(i));
    }
    return size;
}

fn input_index_of(output_i: u32) -> u32 {
    var input_i: u32 = strides_and_shape[1];
    
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
        if(gidx >= arrayLength(&output_0)) {
            return;
        }

        // Outline of approach - it's a bit awkward because we'd like to avoid synchronization, so
        // each thread only writes to a single output location in the output buffer. Furthermore, we
        // don't have a way to have an array to represent coordinates, so we need to inline all the calculations.
        
        // We should have multiple levels of parallelism here, but start with the first: each thread reduces one element in the
        // remaining tensor. 
        // The first step is to figure out to which element in the input buffer we're reducing.
        // We'll start one thread in the 'x' global id dimension for each element in the output buffer.
        // We translate that to the corresponding index in the input buffer.
        let target_input_idx = input_index_of(gidx);

        // We now have the target offset in the input tensor. Calculate the number of elements that need to
        // be reduced to the target element.
        // So e.g.
        // [2, 3].sum(0) --> reduce_size = 2
        // [2, 3].sum(1) --> reduce_size = 3
        let sizes = sizes();
        let reduce_size = sizes.x / sizes.y;

        // Now we can actually reduce.
        var acc = replace_me_with_actual_default();
        for (var rec_i = 0u; rec_i < reduce_size; rec_i += 1u) {

            // The reduced shape and strides here represent the virtual tensor to be reduced.
            // For example, if we're reducing [2, 3] to [2, 1], then the reduced shape is [1, 3].
            // The reduced strides are [3, 1]. The variable here will count up to 3, which we
            // interpret as a buffer index in the reduced virtual tensor. We calculate the tensor
            // coordinatee in the virtual reduced tensor from it, and then transform those coordinate
            // to the real buffer indices in the input tensor.
            var input_i = target_input_idx;
            for (var i = 0u; i < strides_and_shape[0]; i = i + 1u) {
                let len = reduced_shape(i);
                let stride = reduced_strides(i);
                let coord = rec_i / stride % len;
                input_i += coord * input_strides(i);
            }

            acc = replace_me_with_actual_operation(acc, input_0[input_i]);
        }

        output_0[gidx] = acc;
    }
}