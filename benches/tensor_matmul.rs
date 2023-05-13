use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use tensorken::tensor::Wgpu32;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345u64);

    let mut group = c.benchmark_group("Tensor: matmul");

    for size in [32, 64] {
        let t1s = &[size, size];
        let t1_gpu = Wgpu32::randn(t1s, &mut rng);
        let t2_gpu = Wgpu32::randn(t1s, &mut rng);
        let t1_cpu = t1_gpu.to_cpu();
        let t2_cpu = t2_gpu.to_cpu();

        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, _| {
            b.iter(|| (black_box(t1_cpu.matmul(&t2_cpu))))
        });
        group.bench_with_input(BenchmarkId::new("gpu", size), &size, |b, _| {
            b.iter(|| (black_box(t1_gpu.matmul(&t2_gpu))))
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
