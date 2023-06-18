
@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_0: array<f32>;

// ndims, input_offset, chunk_size, reduce_size, input_strides, output_strides, input_shape, reduced_strides,  reduced_shape, output_shape
@group(0) @binding(2)
var<storage, read> strides_and_shape: array<u32>;

// this is replaced with the actual size at shader creation stage.
const INTERMEDIATE_SIZE: u32 = 64u;
var<workgroup> intermediate: array<f32, INTERMEDIATE_SIZE>;

const preamble: u32 = 4u;

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

// Same parlor trick as in map.wgsl.
fn replace_me_with_actual_operation(in_1: f32, in_2: f32) -> f32 { discard; }
fn replace_me_with_actual_default() -> f32 { discard; }

// Same extension to parlor trick as in zip.wgsl.
fn sum(a: f32, b: f32) -> f32 { return a + b; }

// default values for the reduce
const MAX: f32 = -1.17549435082228750797e-38f;
const SUM: f32 = 0.0;

fn input_index_of(output_i: u32) -> u32 {
    let ndims = strides_and_shape[0];
    let offset = strides_and_shape[1];

    var input_i = offset;
    for (var i: u32 = 0u; i < ndims; i = i + 1u) {
        let len = output_shape(i);
        let stride = output_strides(i);
        let coord_i: u32 = output_i / stride % len;

        input_i += coord_i * input_strides(i);
    }

    return input_i;
}

fn reduce_index_of(offset: u32, reduce_i: u32) -> u32 {
    let ndims = strides_and_shape[0];

    var input_i = offset;
    for (var i = 0u; i < ndims; i = i + 1u) {
        let len = reduced_shape(i);
        let stride = reduced_strides(i);
        let coord = reduce_i / stride % len;

        input_i += coord * input_strides(i);
    }

    return input_i;
}

// this is replaced with the actual count at shader creation stage.
const REDUCE_THREADS: u32 = 64u;

// workgroup sizes will be sx,sy, 1, and are replaced at shader creation stage.
@compute
@workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // chunk_size is the number of output elements each thread reduces. It is only >1 if the output tensor's size
    // is larger than the max total workgroup size, as defined in WebGPU limits.
    let chunk_size = strides_and_shape[2];
    let fro = global_id.x * chunk_size;
    let to = fro + chunk_size;

    // reduce_size is the number of elements in the input that need to be reduced to one element in the output.
    // So e.g.
    // [2, 3].sum(&[0]) --> reduce_size = 2
    // [2, 3].sum(&[1]) --> reduce_size = 3
    // let sizes = sizes();
    // let input_size = strides_and_shape[ ];
    // let output_size 
    let reduce_size = strides_and_shape[3];

    let lidx = local_id.x;
    let lidy = local_id.y;

    // Loop over the chunk of output elements this thread is responsible for.
    for (var gidx = fro; gidx < to; gidx = gidx + 1u) {
        if(gidx >= arrayLength(&output_0)) {
            return;
        }

        // To avoid synchronization, each thread writes to separate locations in the output buffer.
        // The buffer index is given by gidx.
        // First we figure out what the set of input elements are that need to be reduced to output_0[gidx].
        // We do this by first calculating the offset in the input buffer of the first element that needs to be reduced to gidx:
        let reduce_offset = input_index_of(gidx);

        // Now we can reduce. The reduction step itself is separately parallelized, in the workgroup's y dimension.
        // It's the first part of a parallel prefix sum with just two levels. The result of the reduction is stored in the intermediate buffer.
        let intermediate_i = lidx * REDUCE_THREADS + lidy;
        intermediate[intermediate_i] = replace_me_with_actual_default();
        for (var reduce_i = lidy; reduce_i < reduce_size; reduce_i += REDUCE_THREADS) {

            // The reduced shape and strides here represent the virtual tensor to be reduced.
            // This reduced tensor contains a subset of the elements, exactly
            // the elements that need to be reduced to output_0[gidx].
            // For example, if we're reducing shape [2, 3] to [2, 1], then the reduced tensor's shape is [1, 3].
            // The reduced strides are always the contiguous strides of the reduced tensor shape, i.e. [3, 1].
            // That's because we're only using it to figure out the tensor index from the buffer index in the reduced (vritual)
            // tensor. The multiplication with the input_strides will then give us the buffer index in the input tensor.
            // The variable reduce_i will count up to reduce_size (3 in the example). We
            // interpret reduce_i as a buffer index in the reduced virtual tensor. We calculate the tensor
            // coordinate in the virtual reduced tensor from it, and then transform those coordinate
            // to the real buffer indices in the input tensor.
            var input_i = reduce_index_of(reduce_offset, reduce_i);
            intermediate[intermediate_i] = replace_me_with_actual_operation(intermediate[intermediate_i], input_0[input_i]);
        }

        // make sure all threads finished writing to the intermediate buffer.
        workgroupBarrier();

        // first y thread does final reduction. Could solve this with re-dispatch instead, not clear what is better.
        if (lidy == 0u) {
            var acc = intermediate[lidx * REDUCE_THREADS];
            for (var i = 1u; i < REDUCE_THREADS; i += 1u) {
                acc = replace_me_with_actual_operation(acc, intermediate[lidx * REDUCE_THREADS + i]);
            }
            output_0[gidx] = acc;
        }
    }
}