alias element = f32;

@group(0) @binding(0)
var<storage, read> input_0: array<element>;

@group(0) @binding(1)
var<storage, read_write> output_0: array<element>;

// ndims, input_offset, chunk_size, reduce_size, input_strides, output_strides, input_shape, reduced_strides,  reduced_shape, output_shape
@group(0) @binding(2)
var<storage, read> strides_and_shape: array<u32>;

// this is replaced with the actual size at shader creation stage.
const INTERMEDIATE_SIZE: u32 = 64u;
var<workgroup> intermediate: array<element, INTERMEDIATE_SIZE>;

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
fn replace_me_with_actual_operation(in_1: element, in_2: element) -> element { discard; }
fn replace_me_with_actual_default() -> element { discard; }

// Same extension to parlor trick as in zip.wgsl.
fn sum(a: element, b: element) -> element { return a + b; }

// default values for the reduce
const MAXF32: f32 = -0x1.fffffep+127f; //-1.17549435082228750797e-38f;
const SUMF32: f32 = f32();

const MAXI32: i32 = i32(-0x80000000);
const SUMI32: i32 = i32();

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

fn reduced_slice_index_of(offset: u32, reduced_slice_i: u32) -> u32 {
    let ndims = strides_and_shape[0];

    var input_i = offset;
    for (var i = 0u; i < ndims; i = i + 1u) {
        let len = reduced_shape(i);
        let stride = reduced_strides(i);
        let coord = reduced_slice_i / stride % len;

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

    // reduce_size is the number of elements in the input that need to be reduced to one element in the output,
    // aka the size of the reduced slice.
    // So e.g.
    // [2, 3].sum(&[0]) --> reduce_size = 2
    // [2, 3].sum(&[1]) --> reduce_size = 3
    // let sizes = sizes();
    // let input_size = strides_and_shape[ ];
    // let output_size 
    let reduced_slice_size = strides_and_shape[3];

    let lidx = local_id.x;
    let lidy = local_id.y;

    // Loop over the chunk of output elements this thread is responsible for.
    for (var gidx = fro; gidx < to; gidx = gidx + 1u) {
        if(gidx >= arrayLength(&output_0)) {
            return;
        }

        // To avoid synchronization, each thread writes to separate locations in the output buffer.
        // The output buffer index is given by gidx.
        // First we figure out what the slice of the input is that needs to be reduced to output_0[gidx].
        // We do this by first calculating the offset in the input buffer of the first element that needs to be reduced to gidx:
        let reduced_slice_offset = input_index_of(gidx);

        // Now we can reduce. The reduction step itself is separately parallelized, in the workgroup's y dimension.
        // It's the first part of a parallel prefix sum with just two levels. The result of the first level reduction is stored in the intermediate buffer.
        // Each thread in the workgroup has its own location in the intermediate buffer it writes to.
        let intermediate_i = lidx * REDUCE_THREADS + lidy;
        intermediate[intermediate_i] = replace_me_with_actual_default();
        for (var reduced_slice_i = lidy; reduced_slice_i < reduced_slice_size; reduced_slice_i += REDUCE_THREADS) {

            // The reduced shape and strides represent the slice to be reduced.
            // This reduced slice contains a subset of the elements, the elements that need to be reduced to output_0[gidx].
            // For example, if we're reducing shape [2, 3] to [2, 1], then the reduced slice's shape is [1, 3].
            // The reduced strides are always the contiguous strides of the reduced slice shape, i.e. [3, 1].
            // That's because we're only using it to figure out the tensor index from the buffer index in the reduced 
            // slice. The multiplication with the input_strides gives us the buffer index in the input.
            // The variable reduced_slice_i will count up to reduced_slice_size (3 in the example). We
            // interpret reduced_slice_i as a buffer index in the reduced slice (even though it has no separate buffer).
            // We calculate the tensor index in the reduced slice from it, and then transform that tensor index
            // to the buffer index in the input tensor.
            var input_i = reduced_slice_index_of(reduced_slice_offset, reduced_slice_i);
            intermediate[intermediate_i] = replace_me_with_actual_operation(intermediate[intermediate_i], input_0[input_i]);
        }

        // make sure all threads finished writing to the intermediate buffer.
        workgroupBarrier();

        // first y thread does final reduction. Could solve this with re-dispatch instead, not clear what is better.
        // also could add more reduction levels, dividing the number of threads each time.
        if (lidy == 0u) {
            var acc = intermediate[lidx * REDUCE_THREADS];
            for (var i = 1u; i < REDUCE_THREADS; i += 1u) {
                acc = replace_me_with_actual_operation(acc, intermediate[lidx * REDUCE_THREADS + i]);
            }
            output_0[gidx] = acc;
        }
    }
}