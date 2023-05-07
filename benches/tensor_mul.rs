use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use tensorken::{tensor::Wgpu32, Diffable, DiffableExt};

// not much here on GPU that made much difference.
// - Using vec2 in the input/output index function didn't make much of a difference.
// - Tried optimizing for low number of dimensions by using vec2,3,4 instead of loop, but since dot product is not defined for vecN<u32> it didn't seem worth it.

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345u64);

    let mut group = c.benchmark_group("Tensor: mul");

    for size in [64, 128, 256, 512, 1024] {
        let square = &[size, size];

        let t1_gpu = Wgpu32::randn(square, &mut rng);
        let t1_gpu_nc = t1_gpu.reshape(&[size / 2, size * 2]).transpose(0, 1);

        let t2_gpu = Wgpu32::randn(square, &mut rng);
        let t2_gpu_nc = t2_gpu.reshape(&[size / 2, size * 2]).transpose(0, 1);

        let t1_cpu = t1_gpu.to_cpu();
        let t1_cpu_nc = t1_gpu_nc.to_cpu();
        let t2_cpu = t2_gpu.to_cpu();
        let t2_cpu_nc = t2_gpu_nc.to_cpu();

        group.bench_with_input(BenchmarkId::new("cpu contiguous", size), &size, |b, _| {
            b.iter(|| (black_box(&t1_cpu * &t2_cpu)))
        });
        group.bench_with_input(
            BenchmarkId::new("cpu non-contiguous", size),
            &size,
            |b, _| b.iter(|| (black_box(&t1_cpu_nc * &t2_cpu_nc))),
        );
        group.bench_with_input(BenchmarkId::new("gpu contiguous", size), &size, |b, _| {
            b.iter(|| (black_box(&t1_gpu * &t2_gpu)))
        });
        group.bench_with_input(
            BenchmarkId::new("gpu non-contiguous", size),
            &size,
            |b, _| b.iter(|| (black_box(&t1_gpu_nc * &t2_gpu_nc))),
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
