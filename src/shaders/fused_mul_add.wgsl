alias element = f32;

@group(0) @binding(0)
var<storage, read> input_0: array<element>;

@group(0) @binding(1)
var<storage, read> input_1: array<element>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<element>;

// ndims, offset_0, offset_1, chunk_size, reduce_size, strides_0, strides_1, output_strides, input_shape, 
// reduced_strides, reduced_shape, output_shape
@group(0) @binding(3)
var<storage, read> strides_and_shape: array<u32>;

const preamble = 5u;

fn strides_0(i: u32) -> u32 {
    return strides_and_shape[i + preamble];
}

fn strides_1(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 1u];
}

fn output_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 2u];
}

fn input_shape(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 3u];
}

fn reduced_strides(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 4u];
}

fn reduced_shape(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 5u];
}

fn output_shape(i: u32) -> u32 {
    return strides_and_shape[i + preamble + strides_and_shape[0] * 6u];
}

fn input_index_of(output_i: u32) -> vec2<u32> {
    var input_i = vec2(strides_and_shape[1], strides_and_shape[2]);
    
    for (var i: u32 = 0u; i < strides_and_shape[0]; i = i + 1u) {
        let len = output_shape(i);
        let stride = output_strides(i);
        let coord_i: u32 = output_i / stride % len;

        input_i += coord_i * vec2(strides_0(i), strides_1(i));
    }

    return input_i;
}

fn reduced_slice_index_of(offset: vec2<u32>, reduce_i: u32) -> vec2<u32> {
    var input_i = offset;

    for (var i = 0u; i < strides_and_shape[0]; i = i + 1u) {
        let len = reduced_shape(i);
        let stride = reduced_strides(i);
        let coord = reduce_i / stride % len;
        input_i += coord * vec2(strides_0(i), strides_1(i));
    }

    return input_i;
}

@compute
@workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chunk_size = strides_and_shape[3];
    let reduced_slice_size = strides_and_shape[4];
    let fro = global_id.x * chunk_size;
    let to = fro + chunk_size;
    
    for (var gidx = fro; gidx < to; gidx = gidx + 1u) {
        if(gidx >= arrayLength(&output_0)) {
            return;
        }

        let reduced_slice_offset = input_index_of(gidx);
        var acc = 0.0;
        for (var reduced_slice_i = 0u; reduced_slice_i < reduced_slice_size; reduced_slice_i += 1u) {
            var input_i = reduced_slice_index_of(reduced_slice_offset, reduced_slice_i);
            acc = fma(input_0[input_i.x], input_1[input_i.y], acc);
        }

        output_0[gidx] = acc;
    }
}