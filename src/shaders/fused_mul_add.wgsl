@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<f32>;

// ndims, offset_0, offset_1, chunk_size, strides_0, strides_1, output_strides, input_shape, 
// reduced_strides, reduced_shape, output_shape
@group(0) @binding(3)
var<storage, read> strides_and_shape: array<u32>;

const preamble = 4u;

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

fn sizes() -> vec2<u32> {
    var size = vec2(1u, 1u);
    for (var i: u32 = 0u; i < strides_and_shape[0]; i += 1u) {
        size *= vec2(input_shape(i), output_shape(i));
    }
    return size;
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

@compute
@workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chunk_size = strides_and_shape[3];
    let fro = global_id.x * chunk_size;
    let to = fro + chunk_size;
    for (var gidx = fro; gidx < to; gidx = gidx + 1u) {
        if(gidx >= arrayLength(&output_0)) {
            return;
        }

        let target_input_idx = input_index_of(gidx);

        let sizes = sizes();
        let reduce_size = sizes.x / sizes.y;

        var acc = 0.0;
        for (var rec_i = 0u; rec_i < reduce_size; rec_i += 1u) {

            var input_i = target_input_idx;
            for (var i = 0u; i < strides_and_shape[0]; i = i + 1u) {
                let len = reduced_shape(i);
                let stride = reduced_strides(i);
                let coord = rec_i / stride % len;
                input_i += coord * vec2(strides_0(i), strides_1(i));
            }

            acc = fma(input_0[input_i.x], input_1[input_i.y], acc);
        }

        output_0[gidx] = acc;
    }
}