use tensorken::{
    raw_tensor::RawTensor, raw_tensor_cpu::CpuRawTensor, raw_tensor_wgpu::WgpuRawTensor,
    tensor::Tensor,
};

fn assert_tensor_eq<T1: RawTensor<Elem = f32>, T2: RawTensor<Elem = f32>>(
    a: &Tensor<T1>,
    b: &Tensor<T2>,
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
fn fun<'t, T: RawTensor>(t1: &'t Tensor<T>, t2: &'t Tensor<T>) -> Tensor<T> {
    let r1 = t1.exp();
    let r2 = t2.log();
    let r3 = t1 + r1; // &T + T
    let r4 = r2 - t2; // T - &T
    let r5 = t1 / t2; // T / T
    let r6 = r3 / r4.exp(); // T / T
    let r7 = t1 * t2; // &T * &T
    r6 + r5 + r7
}

#[test]
fn test_tensorlike() {
    let (shape, data) = (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t_wgpu = &Tensor::new_wgpu(shape, data);
    let t_cpu = &Tensor::new_cpu(shape, data);

    let r_cpu = fun(t_cpu, t_cpu);
    let r_gpu = fun(t_wgpu, t_wgpu);
    assert_tensor_eq(&r_cpu, &r_gpu);
}

#[test]
fn test_elementwise_ops() {
    let shape = &[2, 3];
    let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t2d = &[6.0, 7.0, 8.0, 9.0, 10.0, 12.0];
    let t1 = &Tensor::new_cpu(shape, t1d);
    let t2 = &Tensor::new_cpu(shape, t2d);
    elementwise_ops(t1, t2);

    let t1 = &Tensor::new_wgpu(shape, t1d);
    let t2 = &Tensor::new_wgpu(shape, t2d);
    elementwise_ops(t1, t2);
}

fn elementwise_ops<T: RawTensor<Elem = f32>>(t1: &Tensor<T>, t2: &Tensor<T>) {
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
    assert_eq!(r3.ravel(), vec![7.0, 9.0, 11.0, 13.0, 15.0, 18.0]);
    let r4 = t1 - t2;
    assert_eq!(r4.ravel(), vec![-5.0, -5.0, -5.0, -5.0, -5.0, -6.0]);
    let r5 = t1 / t2;
    assert_eq!(
        r5.ravel(),
        vec![0.16666667, 0.2857143, 0.375, 0.44444445, 0.5, 0.5]
    );
    let r6 = t1 * t2;
    assert_eq!(r6.ravel(), vec![6.0, 14.0, 24.0, 36.0, 50.0, 72.0]);

    let r7 = t1.eq(t2);
    assert_eq!(r7.ravel(), vec![0.0; 6]);
}

#[test]
fn test_broadcasted_ops() {
    let t1s = &[1, 1, 2, 3];
    let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t2s = &t1s[3..];
    let t2d = &[6.0, 7.0, 8.0];
    let t3d = &[1.0, 2.0, 3.0];

    let t1 = &Tensor::new_cpu(t1s, t1d);
    let t2 = &Tensor::new_cpu(t2s, t2d);
    let t3 = &Tensor::new_cpu(t2s, t3d);
    broadcasted_ops(t1, t2, t3);

    let t1 = &Tensor::new_wgpu(t1s, t1d);
    let t2 = &Tensor::new_wgpu(t2s, t2d);
    let t3 = &Tensor::new_wgpu(t2s, t3d);
    broadcasted_ops(t1, t2, t3);
}

fn broadcasted_ops<T: RawTensor<Elem = f32>>(t1: &Tensor<T>, t2: &Tensor<T>, t3: &Tensor<T>) {
    let (r3, r3r) = (t1 + t2, t2 + t1);
    assert_eq!(r3.ravel(), vec![7.0, 9.0, 11.0, 10.0, 12.0, 14.0]);
    assert_eq!(r3r.ravel(), vec![7.0, 9.0, 11.0, 10.0, 12.0, 14.0]);
    let (r4, r4r) = (t1 - t2, t2 - t1);
    assert_eq!(r4.ravel(), vec![-5.0, -5.0, -5.0, -2.0, -2.0, -2.0]);
    assert_eq!(r4r.ravel(), vec![5.0, 5.0, 5.0, 2.0, 2.0, 2.0]);
    let (r5, r5r) = (t1 / t2, t2 / t1);
    assert_eq!(
        r5.ravel(),
        vec![0.16666667, 0.2857143, 0.375, 0.6666667, 0.71428573, 0.75]
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
    assert_eq!(r8.ravel(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    assert_eq!(r8r.ravel(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_reduce_ops() {
    let shape = &[2, 3];
    let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t1 = &Tensor::new_cpu(shape, t1d);
    reduce_ops(t1);

    let t1 = &Tensor::new_wgpu(shape, t1d);
    reduce_ops(t1);
}

fn reduce_ops<T: RawTensor<Elem = f32>>(t1: &Tensor<T>) {
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
fn test_movement_ops() {
    let shape = &[2, 3];
    let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t1 = &Tensor::new_cpu(shape, t1d);
    movement_ops(t1);

    let t1 = &Tensor::new_wgpu(shape, t1d);
    movement_ops(t1);
}

fn movement_ops<T: RawTensor<Elem = f32>>(t1: &Tensor<T>) {
    let r1 = t1.reshape(&[3, 2]);
    assert_eq!(&[3, 2], r1.shape());
    assert_eq!(r1.ravel(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let r2 = t1.permute(&[1, 0]);
    assert_eq!(&[3, 2], r2.shape());
    assert_eq!(r2.ravel(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let r3 = t1.expand(&[2, 2, 3]);
    assert_eq!(&[2, 2, 3], r3.shape());
}

#[test]
fn test_2x3_dot_3x2() {
    let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t2d = &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let t1 = &Tensor::new_cpu(&[2, 3], t1d);
    let t2 = &Tensor::new_cpu(&[3, 2], t2d);

    do_2x3_dot_3x2(t1, t2);

    let t1 = &Tensor::new_wgpu(&[2, 3], t1d);
    let t2 = &Tensor::new_wgpu(&[3, 2], t2d);
    do_2x3_dot_3x2(t1, t2)
}

fn do_2x3_dot_3x2<T: RawTensor<Elem = f32>>(t1: &Tensor<T>, t2: &Tensor<T>) {
    let r1 = t1.matmul(t2);
    assert_eq!(r1.shape(), &[2, 2]);
    assert_eq!(r1.ravel(), vec![20.0, 14.0, 56.0, 41.0]);
}

#[test]
fn test_2x3x5_dot_2x5x2() {
    let t1d = (0..30).map(|x| x as f32).collect::<Vec<_>>();
    let t2d = (0..20).map(|x| x as f32).collect::<Vec<_>>();
    let t1 = &Tensor::new_cpu(&[2, 3, 5], &t1d);
    let t2 = &Tensor::new_cpu(&[2, 5, 2], &t2d);
    do_2x3x5_dot_2x5x2(t1, t2);

    let t1 = &Tensor::new_wgpu(&[2, 3, 5], &t1d);
    let t2 = &Tensor::new_wgpu(&[2, 5, 2], &t2d);
    do_2x3x5_dot_2x5x2(t1, t2);
}

fn do_2x3x5_dot_2x5x2<T: RawTensor<Elem = f32>>(t1: &Tensor<T>, t2: &Tensor<T>) {
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
fn test_eye() {
    for i in 1..8 {
        do_eye_test::<CpuRawTensor<f32>>(i);
        do_eye_test::<WgpuRawTensor<f32>>(i);
    }
}

fn do_eye_test<T: RawTensor<Elem = f32>>(dim: usize) {
    let t1 = Tensor::<T>::eye(dim);
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
