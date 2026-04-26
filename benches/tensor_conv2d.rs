use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use tensorken::{Cpu32, Wgpu32};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345u64);

    let mut group = c.benchmark_group("Tensor: conv2d");

    for ker_size in [3, 5] {
        for im_size in [64, 128, 256, 512, 1024] {
            let im_s = &[im_size, im_size];
            let ker_s = &[ker_size, ker_size];

            let im_gpu = Wgpu32::randn(im_s, &mut rng);
            let ker_gpu = Wgpu32::randn(ker_s, &mut rng);

            let im_cpu = Cpu32::from(&im_gpu);
            let ker_cpu = Cpu32::from(&ker_gpu);

            group.bench_with_input(
                BenchmarkId::new(format!("cpu {ker_size}x{ker_size}"), im_size),
                &im_size,
                |b, _| b.iter(|| black_box(im_cpu.conv2d(&ker_cpu).realize())),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("gpu {ker_size}x{ker_size}"), im_size),
                &im_size,
                |b, _| b.iter(|| black_box(im_gpu.conv2d(&ker_gpu).realize())),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
