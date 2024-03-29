use rand::{rngs::StdRng, SeedableRng};
use tensorken::{
    num::Float, Axes, Cpu32, CpuBool, CpuI32, CpuRawTensor, CpuRawTensorImpl, Diffable, IndexValue,
    Tensor, ToCpu, Wgpu32, WgpuRawTensor, WgpuRawTensorImpl,
};

fn assert_tensor_eq<T1, I1: ToCpu<Repr<f32> = T1>, T2, I2: ToCpu<Repr<f32> = T2>>(
    a: &Tensor<T1, f32, I1>,
    b: &Tensor<T2, f32, I2>,
) {
    let (a, b) = (a.ravel(), b.ravel());
    assert!(
        a.iter()
            .zip(b.iter())
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-2),
        "\r\nleft : {:?}\r\nright: {:?}",
        a,
        b
    )
}

fn assert_vec_eq(a: &[f32], b: &[f32]) {
    assert!(
        a.iter()
            .zip(b.iter())
            // big tolerance here - pow on the GPU is esp. sensitive to rounding errors
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || (a - b).abs() < 1.5),
        "\r\nleft : {:?}\r\nright: {:?}",
        a,
        b
    )
}

// a few functions that are "compile time" tests - to check that the
// TernsorLike traits are having the right effect.
fn fun<T, E: Float, I: Diffable<Repr<E> = T>>(
    t1: &Tensor<T, E, I>,
    t2: &Tensor<T, E, I>,
) -> Tensor<T, E, I> {
    let r1 = t1.exp(); // DiffTensor ops
    let r2 = t2.log();
    let r3 = t1 + r1; // &T + T
    let r4 = r2 - t2; // T - &T
    let r5 = t1 / t2; // T / T
    let r6 = r3 / r4.exp(); // T / T
    let r7 = t1 * t2; // &T * &T
    let r8 = &r6 + &r5;
    &r7 + r8 + &r7
}

#[test]
fn test_tensorlike() {
    let shape = &[2, 3];
    let t_wgpu = &Wgpu32::linspace(1.0, 6.0, 6_u8).reshape(shape);
    let t_cpu = &Cpu32::linspace(1.0, 6.0, 6_u8).reshape(shape);

    let r_cpu = fun(t_cpu, t_cpu);
    let r_gpu = fun(t_wgpu, t_wgpu);
    assert_tensor_eq(&r_cpu, &r_gpu);
}

#[test]
fn test_math_ops() {
    let shape = &[2, 3];
    let t1 = &Cpu32::linspace(1., 6., 6_u8).reshape(shape);
    let t2 = &Cpu32::linspace(6., 11., 6_u8).reshape(shape);
    math_ops(t1, t2);

    let t1 = &Wgpu32::linspace(1., 6., 6_u8).reshape(shape);
    let t2 = &Wgpu32::linspace(6., 11., 6_u8).reshape(shape);
    math_ops(t1, t2);
}

fn math_ops<T, I: ToCpu<Repr<f32> = T>>(t1: &Tensor<T, f32, I>, t2: &Tensor<T, f32, I>) {
    let r1 = t1.exp();
    assert_vec_eq(
        &r1.ravel(),
        &[
            2.7182817, 7.389056, 20.085537, 54.59815, 148.41316, 403.4288,
        ],
    );
    let r2 = t1.log();
    assert_vec_eq(
        &r2.ravel(),
        &[0.0, 0.69314724, 1.0986124, 1.3862945, 1.6094381, 1.7917596],
    );
    let r3 = t1 + t2;
    assert_eq!(r3.ravel(), vec![7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
    let r4 = t1 - t2;
    assert_eq!(r4.ravel(), vec![-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]);
    let r5 = t1 / t2;
    assert_vec_eq(
        &r5.ravel(),
        &[0.16666667, 0.2857143, 0.375, 0.44444445, 0.5, 0.54545456],
    );
    let r6 = t1 * t2;
    assert_eq!(r6.ravel(), vec![6.0, 14.0, 24.0, 36.0, 50.0, 66.0]);

    let r7 = t1.eq(t2);
    assert_eq!(r7.ravel(), vec![false; 6]);
}

#[test]
fn test_broadcasted_ops() {
    let t1s = &[1, 1, 2, 3];
    let t2s = &t1s[3..];

    let t1 = &Cpu32::linspace(1., 6., 6_u8).reshape(t1s);
    let t2 = &Cpu32::linspace(6., 8., 3_u8).reshape(t2s);
    let t3 = &Cpu32::linspace(1., 3., 3_u8).reshape(t2s);
    broadcasted_ops(t1, t2, t3);

    let t1 = &Wgpu32::linspace(1., 6., 6_u8).reshape(t1s);
    let t2 = &Wgpu32::linspace(6., 8., 3_u8).reshape(t2s);
    let t3 = &Wgpu32::linspace(1., 3., 3_u8).reshape(t2s);
    broadcasted_ops(t1, t2, t3);
}

fn broadcasted_ops<T, I: ToCpu<Repr<f32> = T>>(
    t1: &Tensor<T, f32, I>,
    t2: &Tensor<T, f32, I>,
    t3: &Tensor<T, f32, I>,
) {
    let (r3, r3r) = (t1 + t2, t2 + t1);
    assert_eq!(r3.ravel(), vec![7.0, 9.0, 11.0, 10.0, 12.0, 14.0]);
    assert_eq!(r3r.ravel(), vec![7.0, 9.0, 11.0, 10.0, 12.0, 14.0]);
    let (r4, r4r) = (t1 - t2, t2 - t1);
    assert_eq!(r4.ravel(), vec![-5.0, -5.0, -5.0, -2.0, -2.0, -2.0]);
    assert_eq!(r4r.ravel(), vec![5.0, 5.0, 5.0, 2.0, 2.0, 2.0]);
    let (r5, r5r) = (t1 / t2, t2 / t1);
    assert_vec_eq(
        &r5.ravel(),
        &[0.16666667, 0.2857143, 0.375, 0.6666667, 0.71428573, 0.75],
    );
    assert_eq!(r5r.ravel(), vec![6.0, 3.5, 2.6666667, 1.5, 1.4, 1.3333334]);
    let (r6, r6r) = (t1 * t2, t2 * t1);
    assert_eq!(r6.ravel(), vec![6.0, 14.0, 24.0, 24.0, 35.0, 48.0]);
    assert_eq!(r6r.ravel(), vec![6.0, 14.0, 24.0, 24.0, 35.0, 48.0]);

    let (r7, r7r) = (t1.pow(t2), t2.pow(t1));
    assert_vec_eq(
        &r7.ravel(),
        &[1.0, 128.0, 6561.0, 4096.0, 78125.0, 1679616.0],
    );
    assert_vec_eq(&r7r.ravel(), &[6.0, 49.0, 512.0, 1296.0, 16807.0, 262144.0]);

    let (r8, r8r) = (t1.eq(t3), t3.eq(t1));
    assert_eq!(r8.ravel(), vec![true, true, true, false, false, false]);
    assert_eq!(r8r.ravel(), vec![true, true, true, false, false, false]);
}

#[test]
fn test_reduce_ops() {
    let shape = &[2, 3];
    let t1 = &Cpu32::linspace(1., 6., 6_u8).reshape(shape);
    reduce_ops(t1);

    let t1 = &Wgpu32::linspace(1., 6., 6_u8).reshape(shape);
    reduce_ops(t1);
}

fn reduce_ops<T, I: ToCpu<Repr<f32> = T>>(t1: &Tensor<T, f32, I>) {
    let r1 = t1.sum(&[0]);
    assert_eq!(r1.ravel(), vec![5.0, 7.0, 9.0]);
    let r2 = t1.sum(&[1]);
    assert_eq!(r2.ravel(), vec![6.0, 15.0]);
    let r3 = t1.max(&[0]);
    assert_eq!(r3.ravel(), vec![4.0, 5.0, 6.0]);
    let r4 = t1.max(&[1]);
    assert_eq!(r4.ravel(), vec![3.0, 6.0]);
}

#[test]
fn test_reduce_ops_big() {
    let t1_gpu = &Wgpu32::linspace(1., 120., 4800_u16).reshape(&[150, 8, 4]);
    let t1_cpu = t1_gpu.to_cpu();

    let axes: [&[usize]; 7] = [&[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2], &[0, 1, 2]];
    for axis in axes {
        let r_gpu = t1_gpu.sum(axis);
        let r_cpu = t1_cpu.sum(axis);
        assert_vec_eq(&r_gpu.ravel(), &r_cpu.ravel());
    }
}

#[test]
fn test_movement_ops() {
    let shape = &[2, 3];
    let t1 = &Cpu32::linspace(1., 6., 6_u8).reshape(shape);
    movement_ops(t1);

    let t1 = &Wgpu32::linspace(1., 6., 6_u8).reshape(shape);
    movement_ops(t1);
}

fn movement_ops<T, I: ToCpu<Repr<f32> = T>>(t1: &Tensor<T, f32, I>) {
    let r1 = t1.reshape(&[3, 2]);
    assert_eq!(&[3, 2], r1.shape());
    assert_eq!(r1.ravel(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let r2 = t1.permute(&[1, 0]);
    assert_eq!(&[3, 2], r2.shape());
    assert_eq!(r2.ravel(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let r3 = t1.reshape(&[1, 2, 3]).expand(&[2, 2, 3]);
    assert_eq!(&[2, 2, 3], r3.shape());
}

#[test]
fn test_2x3_matmul_3x2() {
    let t1 = &Cpu32::linspace(1., 6., 6_u8).reshape(&[2, 3]);
    let t2 = &Cpu32::linspace(6., 1., 6_u8).reshape(&[3, 2]);

    do_2x3_matmul_3x2(t1, t2);

    let t1 = &Wgpu32::linspace(1., 6., 6_u8).reshape(&[2, 3]);
    let t2 = &Wgpu32::linspace(6., 1., 6_u8).reshape(&[3, 2]);
    do_2x3_matmul_3x2(t1, t2)
}

fn do_2x3_matmul_3x2<T, I: ToCpu<Repr<f32> = T>>(t1: &Tensor<T, f32, I>, t2: &Tensor<T, f32, I>) {
    let r1 = t1.matmul(t2);
    assert_eq!(r1.shape(), &[2, 2]);
    assert_eq!(r1.ravel(), vec![20.0, 14.0, 56.0, 41.0]);
}

#[test]
fn test_2x3x5_matmul_2x5x2() {
    let t1 = &Cpu32::linspace(0., 29., 30_u8).reshape(&[2, 3, 5]);
    let t2 = &Cpu32::linspace(0., 19., 20_u8).reshape(&[2, 5, 2]);
    do_2x3x5_matmul_2x5x2(t1, t2);

    let t1 = &Wgpu32::linspace(0., 29., 30_u8).reshape(&[2, 3, 5]);
    let t2 = &Wgpu32::linspace(0., 19., 20_u8).reshape(&[2, 5, 2]);
    do_2x3x5_matmul_2x5x2(t1, t2);
}

fn do_2x3x5_matmul_2x5x2<T, I: ToCpu<Repr<f32> = T>>(
    t1: &Tensor<T, f32, I>,
    t2: &Tensor<T, f32, I>,
) {
    let r1 = t1.matmul(t2);
    assert_eq!(r1.shape(), &[2, 3, 2]);
    assert_eq!(
        r1.ravel(),
        vec![
            60.0, 70.0, 160.0, 195.0, 260.0, 320.0, 1210.0, 1295.0, 1560.0, 1670.0, 1910.0, 2045.0
        ]
    );
}

#[test]
fn test_3x2x2x3_matmul_2x3x2() {
    // checked vs numpy
    let t1 = &Cpu32::linspace(1.0, 36.0, 36_u8).reshape(&[3, 2, 2, 3]);
    let t2 = &Cpu32::linspace(39.0, 72.0, 12_u8).reshape(&[2, 3, 2]);
    do_3x2x2x3_matmul_2x3x2(t1, t2);

    let t1 = &Wgpu32::linspace(1.0, 36.0, 36_u8).reshape(&[3, 2, 2, 3]);
    let t2 = &Wgpu32::linspace(39.0, 72.0, 12_u8).reshape(&[2, 3, 2]);
    do_3x2x2x3_matmul_2x3x2(t1, t2);
}

fn do_3x2x2x3_matmul_2x3x2<T, I: ToCpu<Repr<f32> = T>>(
    t1: &Tensor<T, f32, I>,
    t2: &Tensor<T, f32, I>,
) {
    let r = t1.matmul(t2);
    assert_eq!(r.shape(), &[3, 2, 2, 2]);
    assert_eq!(
        r.ravel(),
        vec![
            282., 300., 687., 732., 1524., 1596., 2091., 2190., 1902., 2028., 2307., 2460., 3792.,
            3972., 4359., 4566., 3522., 3756., 3927., 4188., 6060., 6348., 6627., 6942.
        ]
    );
}

#[test]
fn test_matmul_1d() {
    // test multiplying with 1d tensors works on the left and right
    let t1 = &Cpu32::new(
        &[4, 3],
        &[
            1., 2., 3., //
            4., 5., 6., //
            7., 8., 9., //
            10., 11., 12.,
        ],
    );
    let t2 = &Cpu32::new(&[3], &[1.0, 2.0, 3.0]);
    let t3 = &Cpu32::new(&[4], &[1.0, 2.0, 3.0, 4.0]);

    // 4x3 matmul 3x1 = 4x1
    let r = t1.matmul(t2);
    assert_eq!(vec![14., 32., 50., 68.], r.ravel());
    assert_eq!(r.shape(), &[4]);

    // 1x4 matmul 4x3 = 1x3
    let r = t3.matmul(t1);
    assert_eq!(vec![70.0, 80.0, 90.0], r.ravel());
    assert_eq!(r.shape(), &[3]);
}

#[test]
fn test_dot_1d() {
    let t1 = &Cpu32::new(&[3], &[1.0, 2.0, 3.0]);
    let t2 = &Cpu32::new(&[3], &[-1.0, -2.0, -3.0]);

    let r = t1.dot(t2);
    assert_eq!(vec![-14.0], r.ravel());
    assert_eq!(r.shape(), &[1]);

    let r = t2.dot(t1);
    assert_eq!(vec![-14.0], r.ravel());
    assert_eq!(r.shape(), &[1]);
}

#[test]
fn test_dot_2d() {
    let t1 = &Cpu32::eye(2);
    let t2 = &Cpu32::new(&[2, 2], &[4.0, -2.0, -3.0, 10.0]);

    let r = t1.dot(t2);
    assert_eq!(vec![4.0, -2.0, -3.0, 10.0], r.ravel());
    assert_eq!(r.shape(), &[2, 2]);

    let r = t2.dot(t1);
    assert_eq!(vec![4.0, -2.0, -3.0, 10.0], r.ravel());
    assert_eq!(r.shape(), &[2, 2]);
}

#[test]
fn test_dot_2dx1d() {
    let t1 = &Cpu32::linspace(1.0, 12.0, 12_u8).reshape(&[4, 3]);
    let t2 = &Cpu32::new(&[3], &[4.0, -2.0, -3.0]);

    // re: 4, 3
    // re: 1, 3
    // *:  4, 3
    // su: 4, 1
    let r = t1.dot(t2);
    assert_eq!(vec![-9.0, -12.0, -15.0, -18.0], r.ravel());
    assert_eq!(r.shape(), &[4]);
}

#[test]
fn test_dot_vs_matmul_nd_shape() {
    let a = Cpu32::ones(&[9, 5, 7, 4]);
    let c = Cpu32::ones(&[9, 5, 4, 3]);

    // re: 9, 5, 7, 1, 1, 1, 4
    // t :          9, 5, 3, 4
    // re: 1, 1, 1, 9, 5, 3, 4
    // * : 9, 5, 7, 9, 5, 3, 4
    // su: 9, 5, 7, 9, 5, 3, 1
    // re: 9, 5, 7, 9, 5, 3
    let r_dot = &a.dot(&c);
    assert_eq!(r_dot.shape(), &[9, 5, 7, 9, 5, 3]);

    let r = &a.matmul(&c);
    assert_eq!(r.shape(), &[9, 5, 7, 3]);
}

#[test]
fn test_dot_nd() {
    let fourbyfour = Cpu32::new(
        &[4, 4],
        &[
            1., 2., 3., 4., //
            3., 2., 1., 4., //
            5., 4., 6., 7., //
            11., 12., 13., 14.,
        ],
    );
    let threebyfourbytwo = Cpu32::new(
        &[3, 4, 2],
        &[
            2., 3., 11., 9., 32., 21., 28., 17., //
            2., 3., 1., 9., 3., 21., 28., 7., //
            2., 3., 1., 9., 3., 21., 28., 7.,
        ],
    );

    let matmul = fourbyfour.matmul(&threebyfourbytwo);
    assert_eq!(matmul.shape(), &[3, 4, 2]);

    let dot = fourbyfour.dot(&threebyfourbytwo);
    assert_eq!(dot.shape(), &[4, 3, 2]);
    assert_eq!(
        dot.ravel(),
        vec![
            232., 152., 125., 112., 125., 112., 172., 116., 123., 76., 123., 76., 442., 296., 228.,
            226., 228., 226., 962., 652., 465., 512., 465., 512.
        ]
    );
}

#[test]
fn test_matmul_big() {
    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }
    init();
    let mut rng = StdRng::seed_from_u64(42u64);
    let t1_gpu = &Wgpu32::randn(&[2000, 500], &mut rng);
    let t2_gpu = &Wgpu32::randn(&[500, 2000], &mut rng);

    let r_gpu = t1_gpu.matmul(t2_gpu);

    assert_eq!(r_gpu.shape(), &[2000, 2000]);

    // just take one row and column, too slow otherwise!
    let t1_cpu = &t1_gpu.to_cpu().crop(&[(0, 1), (0, 500)]);
    let t2_cpu = &t2_gpu.to_cpu().crop(&[(0, 500), (0, 1)]);
    let r_cpu = t1_cpu.matmul(t2_cpu);
    assert_eq!(r_cpu.shape(), &[1, 1]);

    assert_vec_eq(&r_cpu.ravel(), &r_gpu.to_cpu().ravel()[0..1]);
}

#[test]
fn test_eye() {
    for i in 1..8 {
        do_eye_test::<CpuRawTensor<f32>, CpuRawTensorImpl>(i);
        do_eye_test::<WgpuRawTensor<f32>, WgpuRawTensorImpl>(i);
    }
}

fn do_eye_test<T, I: ToCpu<Repr<f32> = T>>(dim: usize) {
    let t1: Tensor<T, f32, I> = Tensor::eye(dim);
    assert_eq!(t1.shape(), &[dim, dim]);
    let raveled = t1.ravel();
    assert_eq!(raveled.iter().filter(|&&x| x == 1.0).count(), dim);
    assert_eq!(
        raveled
            .chunks(dim)
            .map(|x| x.iter().sum::<f32>())
            .filter(|&x| x == 1.0)
            .count(),
        dim
    );
    assert_eq!(
        raveled.iter().filter(|&&x| x == 0.0).count(),
        dim * dim - dim
    );
}

#[test]
fn test_at() {
    let t1: Cpu32 = Tensor::eye(4);
    do_test_at(&t1);
    let t1: Wgpu32 = Tensor::eye(4);
    do_test_at(&t1);
}

fn do_test_at<T, I: ToCpu<Repr<f32> = T>>(t: &Tensor<T, f32, I>) {
    let s = t.at(1);
    assert_eq!(s.shape(), &[4]);
    assert_eq!(s.ravel(), vec![0.0, 1.0, 0.0, 0.0]);

    let s = s.at(1);
    assert_eq!(s.shape(), &[1]);

    let s = t.at(&[1]);
    assert_eq!(s.shape(), &[4]);
    assert_eq!(s.ravel(), vec![0.0, 1.0, 0.0, 0.0]);

    let s = t.at(&[1, 1]);
    assert_eq!(s.shape(), &[1]);
}

#[test]
fn test_randn() {
    let mut rng = StdRng::seed_from_u64(0u64);
    let t1 = Cpu32::randn(&[2, 3], &mut rng);
    assert_eq!(t1.shape(), &[2, 3]);
    assert_eq!(
        t1.ravel(),
        [0.712813, 0.85833144, -2.4362438, 0.16334426, -1.2750102, 1.287171]
    );
}

#[test]
fn test_concatenate() {
    let t1 = Cpu32::new(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let t2 = Cpu32::new(&[3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    let r = Cpu32::concatenate(&[&t1, &t2], 0);
    assert_eq!(r.shape(), &[5, 3]);
    assert_eq!(r.ravel(), [t1.ravel(), t2.ravel()].concat());

    let t1t = t1.transpose(1, 0);
    let t2t = t2.transpose(1, 0);
    let r = Cpu32::concatenate(&[&t1t, &t2t], 1);
    assert_eq!(r.shape(), &[3, 5]);
    assert_eq!(
        r.ravel(),
        [1.0, 4.0, 1.0, 4.0, 7.0, 2.0, 5.0, 2.0, 5.0, 8.0, 3.0, 6.0, 3.0, 6.0, 9.0]
    );
}

#[test]
fn test_squeeze_expand_dims() {
    let t1 = Cpu32::new(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let t2 = Cpu32::new(&[3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    let r1 = t1.expand_dims(0);
    let r2 = t2.expand_dims(1);
    assert_eq!(r1.shape(), &[1, 2, 3]);
    assert_eq!(r2.shape(), &[3, 1, 3]);

    let r3 = r1.squeeze(&Axes::All);
    let r4 = r1.squeeze(&Axes::Axis(0));
    assert_eq!(r3.shape(), &[2, 3]);
    assert_eq!(r4.shape(), &[2, 3]);
}

#[test]
fn test_cast() {
    let b = &CpuBool::new(&[2, 3], &[true, true, false, true, false, false]);
    let i: CpuI32 = b.cast();
    let f: Cpu32 = b.cast();
    assert_eq!(i.ravel(), vec![1, 1, 0, 1, 0, 0]);
    assert_eq!(f.ravel(), vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
}
