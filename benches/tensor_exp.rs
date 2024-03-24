use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use tensorken::{Cpu32, Wgpu32};

// general results for map operations on GPU:
// - opotimizing for contiguous memory layout doesn't make enough of a difference to warrent the complexity.
// - neither does switching to vec4<f32> from f32.
// - making the workgroup size = 64 from the max of 256 yields nice speedup - up to 40% for non-contiguous 1024x1024.

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345u64);

    let mut group = c.benchmark_group("Tensor: exp");

    for size in [64, 128, 256, 512, 1024] {
        let t1s = &[size, size];
        let t1_gpu = Wgpu32::randn(t1s, &mut rng);
        let t1_gpu_nc = t1_gpu.reshape(&[size / 2, size * 2]).transpose(0, 1);
        let t1_cpu = Cpu32::from(&t1_gpu);
        let t1_cpu_nc = Cpu32::from(&t1_gpu_nc);

        group.bench_with_input(BenchmarkId::new("cpu contiguous", size), &size, |b, _| {
            b.iter(|| (black_box(t1_cpu.exp().realize())))
        });
        group.bench_with_input(BenchmarkId::new("gpu contiguous", size), &size, |b, _| {
            b.iter(|| (black_box(t1_gpu.exp().realize())))
        });
        group.bench_with_input(
            BenchmarkId::new("cpu non-contiguous", size),
            &size,
            |b, _| b.iter(|| (black_box(t1_cpu_nc.exp().realize()))),
        );
        group.bench_with_input(
            BenchmarkId::new("gpu non-contiguous", size),
            &size,
            |b, _| b.iter(|| (black_box(t1_gpu_nc.exp().realize()))),
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
