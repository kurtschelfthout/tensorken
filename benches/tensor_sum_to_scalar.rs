use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use tensorken::{tensor::Wgpu32, Diffable, DiffableExt};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345u64);

    let mut group = c.benchmark_group("Tensor: sum to scalar");

    let sizes = [64, 128, 256, 512, 1024, 2048];
    for size in sizes {
        let t1s = &[size, size];
        let t1_gpu = Wgpu32::randn(t1s, &mut rng);
        let t1_cpu = t1_gpu.to_cpu();

        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, _| {
            b.iter(|| (black_box(t1_cpu.sum(&[0, 1]))))
        });
        group.bench_with_input(BenchmarkId::new("gpu", size), &size, |b, _| {
            b.iter(|| (black_box(t1_gpu.sum(&[0, 1]))))
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
