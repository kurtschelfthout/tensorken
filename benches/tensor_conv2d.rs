use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use tensorken::{ConvOpts, Cpu32, Wgpu32};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345u64);

    let mut group = c.benchmark_group("Tensor: conv2d");

    for ker_size in [3, 5] {
        for im_size in [64, 128, 256, 512, 1024, 2048] {
            let im_s = &[1, 1, im_size, im_size];
            let ker_s = &[1, 1, ker_size, ker_size];

            let im_gpu = Wgpu32::randn(im_s, &mut rng);
            let ker_gpu = Wgpu32::randn(ker_s, &mut rng);

            let im_cpu = Cpu32::from(&im_gpu);
            let ker_cpu = Cpu32::from(&ker_gpu);

            let im_array_buffer: Vec<f32> = im_cpu.ravel();
            let ker_array_buffer: Vec<f32> = ker_cpu.ravel();
            let im_array: Vec<&[f32]> = im_array_buffer.chunks_exact(im_size).collect();
            let ker_array: Vec<&[f32]> = ker_array_buffer.chunks_exact(ker_size).collect();

            let opts = ConvOpts {
                stride: [1, 1],
                padding: [(0, 0), (0, 0)],
            };

            // naive benchmark is rough lower bound on cpu time and is closer to gpu time for small sizes
            // group.bench_with_input(
            //     BenchmarkId::new(format!("cpu {ker_size}x{ker_size}"), im_size),
            //     &im_size,
            //     |b, _| b.iter(|| black_box(im_cpu.conv2d(&ker_cpu, opts).realize())),
            // );
            group.bench_with_input(
                BenchmarkId::new(format!("gpu {ker_size}x{ker_size}"), im_size),
                &im_size,
                |b, _| b.iter(|| black_box(im_gpu.conv2d(&ker_gpu, opts).realize())),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("naive {ker_size}x{ker_size}"), im_size),
                &im_size,
                |b, _| b.iter(|| black_box(naive_conv2d(&im_array[..], &ker_array[..]))),
            );
        }
    }
    group.finish();
}

fn naive_conv2d(im: &[&[f32]], ker: &[&[f32]]) -> Vec<Vec<f32>> {
    let ih = im.len();
    let iw = im[0].len();
    let kh = ker.len();
    let kw = ker[0].len();
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;

    let mut out = vec![vec![0.0; ow]; oh];

    for n in 0..oh {
        for m in 0..ow {
            for i in 0..kh {
                for j in 0..kw {
                    out[n][m] += ker[i][j] * im[n + i][m + j];
                }
            }
        }
    }

    out
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
