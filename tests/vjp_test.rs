use tensorken::{
    jacrev,
    num::{Elem, Float, Num, ZeroOne},
    value_and_grad1, value_and_grad2, vjpn, Cpu32, Diff, RealizedRawTensor, Shape, Tensor,
    TensorRev, Tensorken, Wgpu32,
};

use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

fn assert_vec_eq<E: Clone + Debug>(a: &[E], b: &[E])
where
    f32: From<E>,
{
    assert!(
        a.iter().zip(b.iter()).all(|(a, b)| {
            let a = f32::from(a.clone());
            let b = f32::from(b.clone());
            (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-2
        }),
        "\r\nleft : {a:?}\r\nright: {b:?}"
    );
}

/// Test that the first derivative of a function with a single input tensor is correct,
/// given the function to derive using vjp and the expected derivative function (symbolically derived).
#[allow(clippy::similar_names)]
fn test_df<Tsr: Diff>(
    f: impl Fn(&TensorRev<Tsr>) -> TensorRev<Tsr>,
    df: impl Fn(&Tensorken<Tsr>) -> Tensorken<Tsr>,
    ft: impl Fn(&Tensorken<Tsr>) -> Tensorken<Tsr>,
) where
    f32: From<Tsr::E>,
    Tsr::E: Num + Debug + From<u8>,
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    let at: Tensorken<Tsr> = Tensor::new(&[2, 4], &(1u8..9).map(Tsr::E::from).collect::<Vec<_>>());
    let (f_actual, df_actual) = value_and_grad1(f, &at);
    let f_expected = ft(&at);
    let df_expected = df(&at);
    assert_eq!(f_actual.shape(), f_expected.shape());
    assert_vec_eq(&f_actual.ravel(), &f_expected.ravel());
    assert_eq!(df_actual.shape(), df_expected.shape());
    assert_vec_eq(&df_actual.ravel(), &df_expected.ravel());
}

/// Test that the second derivative of a function with a single input tensor is correct,
/// given the function to derive using vjp and the expected derivative function (symbolically derived).
#[allow(clippy::similar_names)]
fn test_ddf<Tsr: Diff>(
    f: impl Fn(&<TensorRev<Tsr> as Diff>::Rev) -> <TensorRev<Tsr> as Diff>::Rev,
    ddf: impl Fn(&Tensorken<Tsr>) -> Tensorken<Tsr>,
    df: impl Fn(&Tensorken<Tsr>) -> Tensorken<Tsr>,
) where
    f32: From<Tsr::E>,
    Tsr::E: Num + Debug + From<u8>,
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    let at: Tensorken<Tsr> = Tensor::new(&[2, 2], &(1u8..5).map(Tsr::E::from).collect::<Vec<_>>());
    let (df_actual, ddf_actual) = value_and_grad1(|r| value_and_grad1(&f, r).1, &at);
    let df_expected = df(&at);
    let ddf_expected = ddf(&at);
    assert_eq!(df_actual.shape(), df_expected.shape());
    assert_vec_eq(&df_actual.ravel(), &df_expected.ravel());
    assert_eq!(ddf_actual.shape(), ddf_expected.shape());
    assert_vec_eq(&ddf_actual.ravel(), &ddf_expected.ravel());
}

/// Test that the first derivative of a function with two input tensors is correct.
#[allow(clippy::similar_names)]
fn test_df_2<Tsr: Diff<E = f32>>(
    f: impl Fn(&TensorRev<Tsr>, &TensorRev<Tsr>) -> TensorRev<Tsr>,
    ft: impl Fn(&Tensorken<Tsr>, &Tensorken<Tsr>) -> Tensorken<Tsr>,
    dfda: impl Fn(&Tensorken<Tsr>, &Tensorken<Tsr>) -> Tensorken<Tsr>,
    dfdb: impl Fn(&Tensorken<Tsr>, &Tensorken<Tsr>) -> Tensorken<Tsr>,
) where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    let a = &Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, -3.0, -2.0]);
    let b = &Tensor::new(&[2, 3], &[4.0, 5.0, 6.0, 7.0, -6.0, -5.0]);
    let (f_actual, (dfda_actual, dfdb_actual)) = value_and_grad2(f, a, b);
    let f_expected = ft(a, b);
    let dfda_expected = dfda(a, b);
    let dfdb_expected = dfdb(a, b);
    assert_eq!(f_actual.shape(), f_expected.shape());
    assert_vec_eq(&f_actual.ravel(), &f_expected.ravel());
    assert_eq!(dfda_actual.shape(), dfda_expected.shape());
    assert_vec_eq(&dfda_actual.ravel(), &dfda_expected.ravel());
    assert_eq!(dfdb_actual.shape(), dfdb_actual.shape());
    assert_vec_eq(&dfdb_actual.ravel(), &dfdb_expected.ravel());
}

#[test]
fn test_derivative_constant() {
    do_test_constant::<Cpu32>();
    do_test_constant::<Wgpu32>();
}

fn do_test_constant<Tsr: Diff>()
where
    f32: From<Tsr::E>,
    Tsr::E: Num + Debug + From<u8>,
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(|t| t.primal().lift_rev(), Tensor::zeros_like, Clone::clone);
    test_ddf::<Tsr>(
        |t| t.primal().primal().lift_rev().lift_rev(),
        Tensor::zeros_like,
        Tensor::zeros_like,
    );
}

fn f_id<T: Clone>(a: &T) -> T {
    a.clone()
}

#[test]
fn test_derivative_id() {
    do_test_id::<Cpu32>();
    do_test_id::<Wgpu32>();
}

fn do_test_id<Tsr: Diff>()
where
    f32: From<Tsr::E>,
    Tsr::E: Num + Debug + From<u8>,
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(f_id, Tensor::ones_like, f_id);
    test_ddf::<Tsr>(f_id, Tensor::zeros_like, Tensor::ones_like);
}

fn f_add<T: Diff>(a: &T) -> T
where
    for<'t> &'t T: Add<&'t T, Output = T>,
{
    a + a
}

#[test]
fn test_derivative_add() {
    do_test_add::<Cpu32>();
    do_test_add::<Wgpu32>();
}

fn do_test_add<Tsr: Diff<E = f32>>()
where
    f32: From<Tsr::E>,
    Tsr::E: Num + Debug + From<u8>,
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(f_add, |a| a.constant_like(2.0), f_add);
    test_ddf::<Tsr>(|a| a + a + a, Tensor::zeros_like, |a| a.constant_like(3.0));
    test_df_2::<Tsr>(
        |a, b| a + b,
        |a, b| a + b,
        |a, _| a.ones_like(),
        |_, b| b.ones_like(),
    );
}

fn f_mul<T: Diff>(a: &T) -> T
where
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    a * a
}

#[test]
fn test_derivative_mul() {
    do_test_mul::<Cpu32>();
    do_test_mul::<Wgpu32>();
}

fn do_test_mul<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(f_mul, |a| a.constant_like(2.0) * a, f_mul);
    test_ddf::<Tsr>(
        |a| a * a * a,
        |a| a.constant_like(6.0) * a,
        |a| a.constant_like(3.0) * a * a,
    );
    test_df_2::<Tsr>(
        |a, b| a * b,
        |a, b| a * b,
        |_, b| b.clone(),
        |a, _| a.clone(),
    );
}

#[allow(clippy::eq_op)]
fn f_sub<T: Diff>(a: &T) -> T
where
    for<'t> &'t T: Sub<&'t T, Output = T>,
{
    a - a
}

#[test]
fn test_derivative_sub() {
    do_test_sub::<Cpu32>();
    do_test_sub::<Wgpu32>();
}

#[allow(clippy::eq_op)]
fn do_test_sub<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(f_sub, Tensor::zeros_like, f_sub);
    test_ddf::<Tsr>(|a| a - a - a, Tensor::zeros_like, |a| -a.ones_like());
    test_df_2::<Tsr>(
        |a, b| a - b,
        |a, b| a - b,
        |a, _| a.ones_like(),
        |_, b| -b.ones_like(),
    );
}

fn f_div<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Num,
{
    a.ones_like() / a
}

#[test]
fn test_derivative_div() {
    do_test_div::<Cpu32>();
    do_test_div::<Wgpu32>();
}

fn do_test_div<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_div::<Tsr::Rev>,
        |a| -a.ones_like() / (a * a),
        |a| f_div::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        |a| a.ones_like() / a,
        |a| a.constant_like(2.0) / (a * a * a),
        |a| -a.ones_like() / (a * a),
    );
    test_df_2::<Tsr>(
        |a, b| a / b,
        |a, b| a / b,
        |_, b| b.ones_like() / b,
        |a, b| -a / (b * b),
    );
}

fn f_pow<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Float,
{
    a.pow(a)
}

#[test]
fn test_derivative_pow() {
    do_test_pow::<Cpu32>();
    do_test_pow::<Wgpu32>();
}

fn do_test_pow<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_pow::<Tsr::Rev>,
        |a| a.pow(a) * (a.log() + a.ones_like()),
        |a| f_pow::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        f_pow::<<Tsr::Rev as Diff>::Rev>,
        |a| {
            a.pow(a) * (a.log() + a.ones_like()).pow(&a.constant_like(2.0))
                + a.pow(&(a - a.ones_like()))
        },
        |a| a.pow(a) * (a.log() + a.ones_like()),
    );
    test_df_2::<Tsr>(
        |a, b| a.pow(b),
        Tensor::pow,
        |a, b| b * a.pow(&(b - b.ones_like())),
        |a, b| a.pow(b) * a.log(),
    );
}

#[test]
fn test_derivative_log() {
    do_test_log::<Cpu32>();
    do_test_log::<Wgpu32>();
}

fn do_test_log<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(|a| a.log(), |a| a.ones_like() / a, Tensor::log);
    test_ddf::<Tsr>(
        |a| a.log(),
        |a| -a.ones_like() / (a * a),
        |a| a.ones_like() / a,
    );
}

#[test]
fn test_derivative_exp() {
    do_test_exp::<Cpu32>();
    do_test_exp::<Wgpu32>();
}

fn do_test_exp<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(|a| a.exp(), Tensor::exp, Tensor::exp);
    test_ddf::<Tsr>(|a| a.exp(), Tensor::exp, Tensor::exp);
}

fn all_axes(shape: &[usize]) -> Vec<usize> {
    (0..shape.len()).collect()
}

fn f_sum<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Num,
{
    a.sum(&all_axes(a.shape()))
}

#[test]
fn test_derivative_sum() {
    do_test_sum::<Cpu32>();
    do_test_sum::<Wgpu32>();
}

fn do_test_sum<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(f_sum::<Tsr::Rev>, Tensor::ones_like, f_sum::<Tsr>);
    test_ddf::<Tsr>(
        |a| a.sum(&all_axes(a.shape())),
        Tensor::zeros_like,
        Tensor::ones_like,
    );
}

fn f_max<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Num + From<bool>,
{
    a.max(&all_axes(a.shape()))
}

#[test]
fn test_max() {
    do_test_max::<Cpu32>();
    do_test_max::<Wgpu32>();
}

fn do_test_max<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_max::<Tsr::Rev>,
        |a| a.max(&all_axes(a.shape())).expand(a.shape()).eq(a).cast(),
        |a| f_max::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        f_max::<<Tsr::Rev as Diff>::Rev>,
        |a| a.zeros_like(),
        |a| a.max(&all_axes(a.shape())).expand(a.shape()).eq(a).cast(),
    );
}

fn f_reshape<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Num,
{
    a.reshape(&[a.shape().size()])
}

#[test]
fn test_reshape() {
    do_test_reshape::<Cpu32>();
    do_test_reshape::<Wgpu32>();
}

fn do_test_reshape<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(f_reshape::<Tsr::Rev>, Tensor::ones_like, |a| {
        f_reshape::<Tsr>(a)
    });
    test_ddf::<Tsr>(
        |a| a.reshape(&[a.shape().size()]),
        Tensor::zeros_like,
        Tensor::ones_like,
    );
}

fn f_permute<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Elem,
{
    a.permute(&[1, 0])
}

#[test]
fn test_permute() {
    do_test_permute::<Cpu32>();
    do_test_permute::<Wgpu32>();
}

fn do_test_permute<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    // bit iffy - assumes at least 2 dimensions
    test_df::<Tsr>(f_permute::<Tsr::Rev>, Tensor::ones_like, |a| {
        f_permute::<Tsr>(a)
    });
    test_ddf::<Tsr>(
        |a| a.permute(&[1, 0]),
        Tensor::zeros_like,
        Tensor::ones_like,
    );
}

fn f_expand<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Num,
{
    a.reshape(&[1, 2, 4]).expand(&[4, 2, 4])
}

#[test]
fn test_expand() {
    do_test_expand::<Cpu32>();
    do_test_expand::<Wgpu32>();
}

fn do_test_expand<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_expand::<Tsr::Rev>,
        |a| a.constant_like(4.0),
        |a| f_expand::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        |a| a.reshape(&[1, 2, 2]).expand(&[4, 2, 2]),
        Tensor::zeros_like,
        |a| a.constant_like(4.0),
    );
}

fn f_crop<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: ZeroOne,
{
    a.crop(&[(0, 1), (1, 2)])
}

#[test]
fn test_crop() {
    do_test_crop::<Cpu32>();
    do_test_crop::<Wgpu32>();
}

fn do_test_crop<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_crop::<Tsr::Rev>,
        |a| a.ones_like().crop(&[(0, 1), (1, 2)]).pad(&[(0, 1), (1, 2)]),
        |a| f_crop::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        f_crop::<<Tsr::Rev as Diff>::Rev>,
        Tensor::zeros_like,
        // note: padding is different here from test_df, because test_ddf
        // tests with a [2,2] tensor only.
        |a| a.ones_like().crop(&[(0, 1), (1, 2)]).pad(&[(0, 1), (1, 0)]),
    );
}

fn f_pad<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: ZeroOne,
{
    a.pad(&[(1, 2), (3, 4)])
}

#[test]
fn test_pad() {
    do_test_pad::<Cpu32>();
    do_test_pad::<Wgpu32>();
}

fn do_test_pad<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(f_pad::<Tsr::Rev>, Tensor::ones_like, |a| f_pad::<Tsr>(a));
    test_ddf::<Tsr>(
        f_pad::<<Tsr::Rev as Diff>::Rev>,
        Tensor::zeros_like,
        Tensor::ones_like,
    );
}

fn f_matmul<T: Diff>(a: &Tensorken<T>, b: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Num,
{
    a.matmul(b)
}

#[test]
fn test_matmul() {
    do_test_matmul::<Cpu32>();
    do_test_matmul::<Wgpu32>();
}

fn do_test_matmul<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    let a: Tensorken<Tsr> = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b: Tensorken<Tsr> = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let (primals, pullback) = vjpn(|ab| f_matmul::<Tsr::Rev>(&ab[0], &ab[1]), &[&a, &b]);
    assert_eq!(primals.shape(), &[2, 2]);
    assert_eq!(primals.ravel(), &[58.0, 64.0, 139.0, 154.0]);

    let cotangent: Tensorken<Tsr> = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let grads = pullback.call(&cotangent);
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), a.shape());
    assert_eq!(grads[1].shape(), b.shape());
    assert_eq!(grads[0].ravel(), [23.0, 29.0, 35.0, 53.0, 67.0, 81.0,]);
    assert_eq!(grads[1].ravel(), [13.0, 18.0, 17.0, 24.0, 21.0, 30.0,]);
}

#[test]
fn test_jacrev() {
    do_test_jacrev::<Cpu32>();
    do_test_jacrev::<Wgpu32>();
}

fn f_pow2<T: Diff>(a: &Tensorken<T>) -> Tensorken<T>
where
    T::E: Float + From<u8>,
{
    a.pow(&a.constant_like(2u8.into()))
}

fn do_test_jacrev<Tsr: Diff<E = f32>>()
where
    Tsr::I: RealizedRawTensor<Repr<Tsr::E> = Tsr::T>,
{
    let a: Tensorken<Tsr> = Tensor::new(&[3], &[1.0, 2.0, 3.0]);
    let r = jacrev(f_pow2::<Tsr::Rev>, &a);
    assert_eq!(r.shape(), &[3, 3]);
    assert_vec_eq(
        &r.ravel(),
        &[
            2.0, 0.0, 0.0, //
            0.0, 4.0, 0.0, //
            0.0, 0.0, 6.0, //
        ],
    );
}
