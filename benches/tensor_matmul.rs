use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use tensorken::{tensor::Wgpu32, Cpu32, DiffableExt};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345u64);

    let mut group = c.benchmark_group("Tensor: matmul");

    for size in [64, 128, 256, 512, 1024] {
        let t1s = &[size, size];
        let t1_gpu = Wgpu32::randn(t1s, &mut rng);
        let t2_gpu = Wgpu32::randn(t1s, &mut rng);

        let t1_cpu = Cpu32::from(&t1_gpu);
        let t2_cpu = Cpu32::from(&t2_gpu);

        if size <= 256 {
            // cpu too slow - already takes 3 mins at 256.
            group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, _| {
                b.iter(|| (black_box(t1_cpu.matmul(&t2_cpu).realize())))
            });
        }
        group.bench_with_input(BenchmarkId::new("gpu", size), &size, |b, _| {
            b.iter(|| (black_box(t1_gpu.matmul(&t2_gpu).realize())))
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
