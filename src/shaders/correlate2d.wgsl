alias element = f32;

@group(0) @binding(0)
var<storage, read> image: array<element>;

@group(0) @binding(1)
var<storage, read> kernel: array<element>;

@group(0) @binding(2)
var<storage, read_write> out: array<element>;

struct Parameters {
    // buffer shapes and strides
    im_shape: vec4i,
    im_stride: vec4i,
    ker_shape: vec4i,
    ker_stride: vec4i,
    out_shape: vec4i,
    out_stride: vec4i,

    // correlation settings
    corr_stride: vec2i,
    corr_dilation: vec2i,
    corr_fill: vec2i,
    corr_pad_start: vec2i,
    corr_pad_end: vec2i,

    // dispatch settings
    chunk_size: u32,
    im_offset: i32,
    ker_offset: i32,
}

@group(0) @binding(3)
var<uniform> params: Parameters;

// the contiguous strides. used for converting an flat index to a tensor index
fn kernel_mod() -> vec4i {
    let s = params.ker_shape;
    return vec4(s.y * s.z * s.w, s.z * s.w, s.w, 1);
}

@compute
@workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i0 = bitcast<i32>(global_id.x * params.chunk_size);
    let i1 = bitcast<i32>((global_id.x + 1) * params.chunk_size);

    let jm = kernel_mod();

    let corr_stride = vec4i(1, 1, params.corr_stride);
    let corr_dilation = vec4i(1, 1, params.corr_dilation);
    let corr_fill = vec4i(1, 1, params.corr_fill);
    let corr_pad = vec4i(0, 0, params.corr_pad_start);

    // loop over output elements
    for (var i = i0; i < i1; i = i + 1) {
        // [b, oc, oy, ox]
        let out_ti = vec4i(i) / params.out_stride % params.out_shape;

        let im_ti_start = vec4i(out_ti.x, 0, out_ti.zw) * corr_stride - corr_pad * corr_fill;
        let ker_ti_start = vec4i(out_ti.y, 0, 0, 0);

        var acc = 0.0;
        for (var j = 0; j < jm.x; j = j + 1) {
            // fma iteration index, [0, ic, iy, ix]
            let it = vec4i(j) / jm % params.ker_shape;

            let im_filled_ti = im_ti_start + it * corr_dilation;
            let ker_ti = ker_ti_start + it;

            if all(im_filled_ti % corr_fill == vec4i(0)) {
                let im_ti = im_filled_ti / corr_fill;

                if all(vec4i(0) <= im_ti) && all(im_ti < params.im_shape) {
                    let im_i = dot(im_ti, params.im_stride) + params.im_offset;
                    let ker_i = dot(ker_ti, params.ker_stride) + params.ker_offset;

                    acc = fma(image[im_i], kernel[ker_i], acc);
                }
            }
        }
        out[i] = acc;
    }
}