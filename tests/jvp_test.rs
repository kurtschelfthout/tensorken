use tensorken::{
    jacfwd, jvpn, num::Num, value_and_diff1, value_and_diff2, Cpu32, Diff, Forward, ForwardImpl,
    Shape, Tensor, TensorBase, TensorFwd, ToCpu, Wgpu32,
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
    f: impl Fn(&TensorFwd<Tsr>) -> TensorFwd<Tsr>,
    df: impl Fn(&TensorBase<Tsr>) -> TensorBase<Tsr>,
    ft: impl Fn(&TensorBase<Tsr>) -> TensorBase<Tsr>,
) where
    f32: From<Tsr::E>,
    Tsr::E: Num + From<u8> + Debug,
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    let at: TensorBase<Tsr> = Tensor::new(&[2, 4], &(1u8..9).map(Tsr::E::from).collect::<Vec<_>>());
    let (f_actual, df_actual) = value_and_diff1(f, &at);
    let f_expected = ft(&at);
    let df_expected = df(&at);
    assert_eq!(f_actual.shape(), f_expected.shape(), "f shapes don't match");
    assert_vec_eq(&f_actual.ravel(), &f_expected.ravel());
    // println!("df_actual: {:?}", df_actual);
    // println!("df_expected: {:?}", df_expected);
    assert_eq!(
        df_actual.shape(),
        df_expected.shape(),
        "df shapes don't match"
    );
    assert_vec_eq(&df_actual.ravel(), &df_expected.ravel());
}

/// Test that the second derivative of a function with a single input tensor is correct,
/// given the function to derive using vjp and the expected derivative function (symbolically derived).
#[allow(clippy::similar_names)]
fn test_ddf<Tsr: Diff>(
    f: impl Fn(
        &Tensor<Forward<Forward<Tsr::T>>, Tsr::E, ForwardImpl<ForwardImpl<Tsr::I>>>,
    ) -> Tensor<Forward<Forward<Tsr::T>>, Tsr::E, ForwardImpl<ForwardImpl<Tsr::I>>>,
    ddf: impl Fn(&TensorBase<Tsr>) -> TensorBase<Tsr>,
    dft: impl Fn(&TensorBase<Tsr>) -> TensorBase<Tsr>,
) where
    f32: From<Tsr::E>,
    Tsr::E: Num + From<u8> + Debug,
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    let at: TensorBase<Tsr> = Tensor::new(&[2, 2], &(1u8..5).map(Tsr::E::from).collect::<Vec<_>>());
    let (df_actual, ddf_actual) = value_and_diff1(|r| value_and_diff1(&f, r).1, &at);
    let df_expected = dft(&at);
    let ddf_expected = ddf(&at);
    assert_eq!(df_actual.shape(), df_expected.shape());
    assert_vec_eq(&df_actual.ravel(), &df_expected.ravel());
    assert_eq!(ddf_actual.shape(), ddf_expected.shape());
    assert_vec_eq(&ddf_actual.ravel(), &ddf_expected.ravel());
}

/// Test that the first derivative of a function with two input tensors is correct.
#[allow(clippy::similar_names)]
fn test_df_2<Tsr: Diff<E = f32>>(
    f: impl Fn(&TensorFwd<Tsr>, &TensorFwd<Tsr>) -> TensorFwd<Tsr>,
    ft: impl Fn(&TensorBase<Tsr>, &TensorBase<Tsr>) -> TensorBase<Tsr>,
    dfda: impl Fn(&TensorBase<Tsr>, &TensorBase<Tsr>) -> TensorBase<Tsr>,
    dfdb: impl Fn(&TensorBase<Tsr>, &TensorBase<Tsr>) -> TensorBase<Tsr>,
) where
    Tsr::I: ToCpu<Repr<f32> = Tsr::T>,
{
    let a = &Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, -3.0, -2.0]);
    let b = &Tensor::new(&[2, 3], &[4.0, 5.0, 6.0, 7.0, -6.0, -5.0]);
    let (f_actual, (dfda_actual, dfdb_actual)) = value_and_diff2(f, a, b);
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
    Tsr::E: Num + From<u8> + Debug,
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
    f32: From<Tsr::E>,
{
    test_df::<Tsr>(|t| t.primal().lift_fwd(), Tensor::zeros_like, Clone::clone);
    test_ddf::<Tsr>(
        |t| t.primal().primal().lift_fwd().lift_fwd(),
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
    Tsr::E: Num + From<u8> + Debug,
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
    f32: From<Tsr::E>,
{
    test_df::<Tsr>(f_id, Tensor::ones_like, f_id);
    test_ddf::<Tsr>(f_id, Tensor::zeros_like, Tensor::ones_like);
}

fn f_add<T>(a: &T) -> T
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
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
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

fn f_mul<T>(a: &T) -> T
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
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
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
fn f_sub<T>(a: &T) -> T
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
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
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

fn f_div<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.ones_like() / a
}

#[test]
fn test_derivative_div() {
    do_test_div::<Cpu32>();
    do_test_div::<Wgpu32>();
}

fn do_test_div<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_div::<Tsr::Fwd>,
        |a| -a.ones_like() / (a * a),
        f_div::<Tsr>,
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

fn f_pow<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.pow(a)
}

#[test]
fn test_derivative_pow() {
    do_test_pow::<Cpu32>();
    do_test_pow::<Wgpu32>();
}

fn do_test_pow<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_pow::<Tsr::Fwd>,
        |a| a.pow(a) * (a.log() + a.ones_like()),
        f_pow::<Tsr>,
    );
    test_ddf::<Tsr>(
        f_pow::<<Tsr::Fwd as Diff>::Fwd>,
        |a| {
            a.pow(a) * (a.log() + a.ones_like()).pow(&a.constant_like(2.0))
                + a.pow(&(a - a.ones_like()))
        },
        |a| a.pow(a) * (a.log() + a.ones_like()),
    );
    // Because in forward mode both PowOp dfda and dfdb get calculated,
    // dfdb causes NaN. This could be solved by treating zero derivatives as
    // a special case (which would also be more efficient)
    // test_df_2::<Tsr>(
    //     |a, b| a.pow(b),
    //     Tensor::pow,
    //     |a, b| b * a.pow(&(b - b.ones_like())),
    //     |a, b| a.pow(b) * a.log(),
    // );
    let a: &TensorBase<Tsr> = &Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b: &TensorBase<Tsr> = &Tensor::new(&[2, 2], &[4.0, 5.0, 6.0, 7.0]);
    let (f_actual, (dfda_actual, dfdb_actual)) = value_and_diff2(|a, b| a.pow(b), a, b);
    let f_expected = a.pow(b);
    let dfda_expected = b * a.pow(&(b - b.ones_like()));
    let dfdb_expected = a.pow(b) * a.log();
    assert_eq!(f_actual.shape(), f_expected.shape());
    assert_vec_eq(&f_actual.ravel(), &f_expected.ravel());
    assert_eq!(dfda_actual.shape(), dfda_expected.shape());
    assert_vec_eq(&dfda_actual.ravel(), &dfda_expected.ravel());
    assert_eq!(dfdb_actual.shape(), dfdb_actual.shape());
    assert_vec_eq(&dfdb_actual.ravel(), &dfdb_expected.ravel());
}

#[test]
fn test_derivative_log() {
    do_test_log::<Cpu32>();
    do_test_log::<Wgpu32>();
}

fn do_test_log<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
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
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(|a| a.exp(), Tensor::exp, Tensor::exp);
    test_ddf::<Tsr>(|a| a.exp(), Tensor::exp, Tensor::exp);
}

fn all_axes(shape: &[usize]) -> Vec<usize> {
    (0..shape.len()).collect()
}

fn f_sum<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.sum(&all_axes(a.shape()))
}

#[test]
fn test_derivative_sum() {
    do_test_sum::<Cpu32>();
    do_test_sum::<Wgpu32>();
}

fn do_test_sum<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    fn df<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
        a.ones_like().sum(&all_axes(a.shape()))
    }
    fn ddf<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
        a.zeros_like().sum(&all_axes(a.shape()))
    }

    test_df::<Tsr>(f_sum::<Tsr::Fwd>, df::<Tsr>, f_sum::<Tsr>);
    test_ddf::<Tsr>(f_sum::<<Tsr::Fwd as Diff>::Fwd>, ddf::<Tsr>, df::<Tsr>);
}

fn f_max<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.max(&all_axes(a.shape()))
}

#[test]
fn test_max() {
    do_test_max::<Cpu32>();
    do_test_max::<Wgpu32>();
}

fn do_test_max<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_max::<Tsr::Fwd>,
        |a| a.max(&all_axes(a.shape())).expand(a.shape()).eq(a).cast(),
        f_max::<Tsr>,
    );
    test_ddf::<Tsr>(f_max::<<Tsr::Fwd as Diff>::Fwd>, Tensor::zeros_like, |a| {
        a.max(&all_axes(a.shape())).expand(a.shape()).eq(a).cast()
    });
}

fn f_reshape<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.reshape(&[a.shape().size()])
}

#[test]
fn test_reshape() {
    do_test_reshape::<Cpu32>();
    do_test_reshape::<Wgpu32>();
}

fn do_test_reshape<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_reshape::<Tsr::Fwd>,
        |a| f_reshape::<Tsr>(&a.ones_like()),
        |a| f_reshape::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        f_reshape::<<Tsr::Fwd as Diff>::Fwd>,
        |a| f_reshape::<Tsr>(&a.zeros_like()),
        |a| f_reshape::<Tsr>(&a.ones_like()),
    );
}

fn f_permute<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.permute(&[1, 0])
}

#[test]
fn test_permute() {
    do_test_permute::<Cpu32>();
    do_test_permute::<Wgpu32>();
}

fn do_test_permute<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    // bit iffy - assumes at least 2 dimensions
    test_df::<Tsr>(
        f_permute::<Tsr::Fwd>,
        |a| f_permute::<Tsr>(&a.ones_like()),
        |a| f_permute::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        |a| a.permute(&[1, 0]),
        |a| f_permute::<Tsr>(&a.zeros_like()),
        |a| f_permute::<Tsr>(&a.ones_like()),
    );
}

fn f_expand<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.reshape(&[1, 2, 4]).expand(&[4, 2, 4])
}

#[test]
fn test_expand() {
    do_test_expand::<Cpu32>();
    do_test_expand::<Wgpu32>();
}

fn do_test_expand<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_expand::<Tsr::Fwd>,
        |a| f_expand::<Tsr>(&a.ones_like()),
        |a| f_expand::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        |a| a.reshape(&[1, 2, 2]).expand(&[4, 2, 2]),
        |a| a.zeros_like().reshape(&[1, 2, 2]).expand(&[4, 2, 2]),
        |a| a.ones_like().reshape(&[1, 2, 2]).expand(&[4, 2, 2]),
    );
}

fn f_crop<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.crop(&[(0, 1), (1, 2)])
}

#[test]
fn test_crop() {
    do_test_crop::<Cpu32>();
    do_test_crop::<Wgpu32>();
}

fn do_test_crop<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_crop::<Tsr::Fwd>,
        |a| f_crop::<Tsr>(&a.ones_like()),
        |a| f_crop::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        f_crop::<<Tsr::Fwd as Diff>::Fwd>,
        |a| f_crop::<Tsr>(&a.zeros_like()),
        |a| f_crop::<Tsr>(&a.ones_like()),
    );
}

fn f_pad<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.pad(&[(1, 2), (3, 4)])
}

#[test]
fn test_pad() {
    do_test_pad::<Cpu32>();
    do_test_pad::<Wgpu32>();
}

fn do_test_pad<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    test_df::<Tsr>(
        f_pad::<Tsr::Fwd>,
        |a| f_pad::<Tsr>(&a.ones_like()),
        |a| f_pad::<Tsr>(a),
    );
    test_ddf::<Tsr>(
        f_pad::<<Tsr::Fwd as Diff>::Fwd>,
        |a| f_pad::<Tsr>(&a.zeros_like()),
        |a| f_pad::<Tsr>(&a.ones_like()),
    );
}

fn f_matmul<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>, b: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.matmul(b)
}

#[test]
fn test_matmul() {
    do_test_matmul::<Cpu32>();
    do_test_matmul::<Wgpu32>();
}

fn do_test_matmul<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    let a: Tensor<_, _, Tsr::I> = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b: Tensor<_, _, Tsr::I> = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let tangent_a = a.constant_like(2.0);
    let tangent_b = b.zeros_like();

    let (primals, df) = jvpn(
        |ab| f_matmul::<Tsr::Fwd>(&ab[0], &ab[1]),
        &[&a, &b],
        &[&tangent_a, &tangent_b],
    );
    assert_eq!(primals.shape(), &[2, 2]);
    assert_eq!(primals.ravel(), &[58.0, 64.0, 139.0, 154.0]);
    assert_eq!(df.ravel(), tangent_a.matmul(&b).ravel());

    let tangent_a: Tensor<_, _, Tsr::I> = b.zeros_like();
    let tangent_b: Tensor<_, _, Tsr::I> = b.constant_like(2.0);
    let (primals, df) = jvpn(
        |ab| f_matmul::<Tsr::Fwd>(&ab[0], &ab[1]),
        &[&a, &b],
        &[&tangent_a, &tangent_b],
    );
    assert_eq!(primals.shape(), &[2, 2]);
    assert_eq!(primals.ravel(), &[58.0, 64.0, 139.0, 154.0]);
    assert_eq!(df.ravel(), a.matmul(&tangent_b).ravel());
}

#[test]
fn test_jacfwd() {
    do_test_jacfwd::<Cpu32>();
    do_test_jacfwd::<Wgpu32>();
}

fn f_pow2<Tsr: Diff<E = f32>>(a: &TensorBase<Tsr>) -> TensorBase<Tsr> {
    a.pow(&a.constant_like(2u8.into()))
}

fn do_test_jacfwd<Tsr: Diff<E = f32>>()
where
    Tsr::I: ToCpu<Repr<Tsr::E> = Tsr::T>,
{
    let a: Tensor<_, _, Tsr::I> = Tensor::new(&[3], &[1.0, 2.0, 3.0]);
    let r = jacfwd(f_pow2::<Tsr::Fwd>, &a);
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
