use tensorken::{
    CpuRawTensor, Diffable, DiffableExt, RawTensor, Shape, WgpuRawTensor,
    {value_and_grad1, value_and_grad2, vjpn, Reverse}, {Tensor, TensorLike, TensorLikeRef},
};

use std::{fmt::Debug, ops::Add};

fn assert_vec_eq(a: &[f32], b: &[f32]) {
    assert!(
        a.iter()
            .zip(b.iter())
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-2),
        "\r\nleft : {a:?}\r\nright: {b:?}"
    );
}

/// Test that the first derivative of a function with a single input tensor is correct,
/// given the function to derive using vjp and the expected derivative function (symbolically derived).
#[allow(clippy::similar_names)]
fn test_df<'t, RT: 't + RawTensor<Elem = f32> + Clone + Debug, F, G, H>(f: F, df: G, ft: H)
where
    for<'a> F: Fn(&'a Reverse<'a, 't, Tensor<RT>>) -> Reverse<'a, 't, Tensor<RT>>,
    for<'a> G: Fn(&'a Tensor<RT>) -> Tensor<RT>, // G & H are identical, but if we want to pass closures,
    for<'a> H: Fn(&'a Tensor<RT>) -> Tensor<RT>, // we need to use two different types.
{
    let at: Tensor<RT> = Tensor::new(&[2, 4], &(1u8..9).map(f32::from).collect::<Vec<_>>());
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
fn test_ddf<RT: RawTensor<Elem = f32> + Clone + Debug, F, G, H>(f: F, ddf: G, dft: H)
where
    for<'a, 't, 'b, 'tt> F: Fn(
        &'a Reverse<'a, 't, Reverse<'b, 'tt, Tensor<RT>>>,
    ) -> Reverse<'a, 't, Reverse<'b, 'tt, Tensor<RT>>>,
    for<'a> G: Fn(&'a Tensor<RT>) -> Tensor<RT>,
    for<'a> H: Fn(&'a Tensor<RT>) -> Tensor<RT>,
{
    let at: Tensor<RT> = Tensor::new(&[2, 2], &(1u8..5).map(f32::from).collect::<Vec<_>>());
    let (df_actual, ddf_actual) = value_and_grad1(|r| value_and_grad1(&f, r).1, &at);
    let df_expected = dft(&at);
    let ddf_expected = ddf(&at);
    assert_eq!(df_actual.shape(), df_expected.shape());
    assert_vec_eq(&df_actual.ravel(), &df_expected.ravel());
    assert_eq!(ddf_actual.shape(), ddf_expected.shape());
    assert_vec_eq(&ddf_actual.ravel(), &ddf_expected.ravel());
}

/// Test that the first derivative of a function with two input tensors is correct.
#[allow(clippy::similar_names)]
fn test_df_2<RT: RawTensor<Elem = f32> + Clone + Debug, F, H, GA, GB>(
    f: F,
    ft: H,
    dfda: GA,
    dfdb: GB,
) where
    for<'a, 't> F: Fn(
        &'a Reverse<'a, 't, Tensor<RT>>,
        &'a Reverse<'a, 't, Tensor<RT>>,
    ) -> Reverse<'a, 't, Tensor<RT>>,
    for<'a> GA: Fn(&'a Tensor<RT>, &'a Tensor<RT>) -> Tensor<RT>,
    for<'a> GB: Fn(&'a Tensor<RT>, &'a Tensor<RT>) -> Tensor<RT>,
    for<'a> H: Fn(&'a Tensor<RT>, &'a Tensor<RT>) -> Tensor<RT>,
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
    do_test_constant::<CpuRawTensor<f32>>();
    do_test_constant::<WgpuRawTensor<f32>>();
}

fn do_test_constant<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(
        |t| Reverse::lift(t.primal()),
        Diffable::zeros_like,
        Clone::clone,
    );
    test_ddf::<RT, _, _, _>(
        |t| Reverse::lift(&Reverse::lift(t.primal().primal())),
        Diffable::zeros_like,
        Diffable::zeros_like,
    );
}

fn f_id<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.clone()
}

#[test]
fn test_derivative_id() {
    do_test_id::<CpuRawTensor<f32>>();
    do_test_id::<WgpuRawTensor<f32>>();
}

fn do_test_id<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| f_id(a), Diffable::ones_like, |a| f_id(a));
    test_ddf::<RT, _, _, _>(|a| f_id(a), Diffable::zeros_like, Diffable::ones_like);
}

fn f_add<'t, T>(a: &'t T) -> T
where
    &'t T: Add<&'t T, Output = T>,
{
    a + a
}

#[test]
fn test_derivative_add() {
    do_test_add::<CpuRawTensor<f32>>();
    do_test_add::<WgpuRawTensor<f32>>();
}

fn do_test_add<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    // let at: Tensor<CpuRawTensor<_>> = Tensor::new(&[2, 2], (1u8..5).map(f32::from).collect());
    // note: this doesn't work for some reason - needs explicit closure
    // let r = vjp1(f_add, &at);
    test_df::<RT, _, _, _>(|a| f_add(a), |a| a.constant_like(2.0), |a| f_add(a));
    test_ddf::<RT, _, _, _>(
        |a| a + a + a,
        Diffable::zeros_like,
        |a| a.constant_like(3.0),
    );
    test_df_2::<RT, _, _, _, _>(
        |a, b| a + b,
        |a, b| a + b,
        |a, _| a.ones_like(),
        |_, b| b.ones_like(),
    );
}

fn f_mul<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a * a
}

#[test]
fn test_derivative_mul() {
    do_test_mul::<CpuRawTensor<f32>>();
    do_test_mul::<WgpuRawTensor<f32>>();
}

fn do_test_mul<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| f_mul(a), |a| a.constant_like(2.0) * a, |a| f_mul(a));
    test_ddf::<RT, _, _, _>(
        |a| a * a * a,
        |a| a.constant_like(6.0) * a,
        |a| a.constant_like(3.0) * a * a,
    );
    test_df_2::<RT, _, _, _, _>(
        |a, b| a * b,
        |a, b| a * b,
        |_, b| b.clone(),
        |a, _| a.clone(),
    );
}

#[allow(clippy::eq_op)]
fn f_sub<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a - a
}

#[test]
fn test_derivative_sub() {
    do_test_sub::<CpuRawTensor<f32>>();
    do_test_sub::<WgpuRawTensor<f32>>();
}

#[allow(clippy::eq_op)]
fn do_test_sub<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| f_sub(a), Diffable::zeros_like, |a| f_sub(a));
    test_ddf::<RT, _, _, _>(|a| a - a - a, Diffable::zeros_like, |a| -a.ones_like());
    test_df_2::<RT, _, _, _, _>(
        |a, b| a - b,
        |a, b| a - b,
        |a, _| a.ones_like(),
        |_, b| -b.ones_like(),
    );
}

fn f_div<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.ones_like() / a
}

#[test]
fn test_derivative_div() {
    do_test_div::<CpuRawTensor<f32>>();
    do_test_div::<WgpuRawTensor<f32>>();
}

fn do_test_div<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| f_div(a), |a| -a.ones_like() / (a * a), |a| f_div(a));
    test_ddf::<RT, _, _, _>(
        |a| a.ones_like() / a,
        |a| a.constant_like(2.0) / (a * a * a),
        |a| -a.ones_like() / (a * a),
    );
    test_df_2::<RT, _, _, _, _>(
        |a, b| a / b,
        |a, b| a / b,
        |_, b| b.ones_like() / b,
        |a, b| -a / (b * b),
    );
}

fn f_pow<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.pow(a)
}

#[test]
fn test_derivative_pow() {
    do_test_pow::<CpuRawTensor<f32>>();
    do_test_pow::<WgpuRawTensor<f32>>();
}

fn do_test_pow<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(
        |a| f_pow(a),
        |a| a.pow(a) * (a.log() + a.ones_like()),
        |a| f_pow(a),
    );
    test_ddf::<RT, _, _, _>(
        |a| f_pow(a),
        |a| {
            a.pow(a) * (a.log() + a.ones_like()).pow(&a.constant_like(2.0))
                + a.pow(&(a - a.ones_like()))
        },
        |a| a.pow(a) * (a.log() + a.ones_like()),
    );
    test_df_2::<RT, _, _, _, _>(
        |a, b| a.pow(b),
        DiffableExt::pow,
        |a, b| b * a.pow(&(b - b.ones_like())),
        |a, b| a.pow(b) * a.log(),
    );
}

#[test]
fn test_derivative_log() {
    do_test_log::<CpuRawTensor<f32>>();
    do_test_log::<WgpuRawTensor<f32>>();
}

fn do_test_log<RT: RawTensor<Elem = f32> + Clone + Debug + 'static>() {
    test_df::<RT, _, _, _>(|a| a.log(), |a| a.ones_like() / a, Diffable::log);
    test_ddf::<RT, _, _, _>(
        |a| a.log(),
        |a| -a.ones_like() / (a * a),
        |a| a.ones_like() / a,
    );
}

#[test]
fn test_derivative_exp() {
    do_test_exp::<CpuRawTensor<f32>>();
    do_test_exp::<WgpuRawTensor<f32>>();
}

fn do_test_exp<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| a.exp(), Diffable::exp, Diffable::exp);
    test_ddf::<RT, _, _, _>(|a| a.exp(), Diffable::exp, Diffable::exp);
}

fn all_axes(shape: &[usize]) -> Vec<usize> {
    (0..shape.len()).collect::<Vec<_>>()
}

fn f_sum<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.sum(&all_axes(a.shape()))
}

#[test]
fn test_derivative_sum() {
    do_test_sum::<CpuRawTensor<f32>>();
    do_test_sum::<WgpuRawTensor<f32>>();
}

fn do_test_sum<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| f_sum(a), Diffable::ones_like, |a| f_sum(a));
    test_ddf::<RT, _, _, _>(
        |a| a.sum(&all_axes(a.shape())),
        Diffable::zeros_like,
        Diffable::ones_like,
    );
}

fn f_max<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.max(&all_axes(a.shape()))
}

#[test]
fn test_max() {
    do_test_max::<CpuRawTensor<f32>>();
    do_test_max::<WgpuRawTensor<f32>>();
}

fn do_test_max<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(
        |a| f_max(a),
        |a| a.max(&all_axes(a.shape())).expand(a.shape()).eq(a),
        |a| f_max(a),
    );
    // "Equality is not differentiable" because MaxOp uses eq...
    // test_ddf::<RT, _, _, _>(
    //     |a| a.max(&all_axes(a.shape())),
    //     |a| a.max(&all_axes(a.shape())).expand(a.shape()).eq(a),
    // );
}

fn f_reshape<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.reshape(&[a.shape().size()])
}

#[test]
fn test_reshape() {
    do_test_reshape::<CpuRawTensor<f32>>();
    do_test_reshape::<WgpuRawTensor<f32>>();
}

fn do_test_reshape<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| f_reshape(a), Diffable::ones_like, |a| f_reshape(a));
    test_ddf::<RT, _, _, _>(
        |a| a.reshape(&[a.shape().size()]),
        Diffable::zeros_like,
        Diffable::ones_like,
    );
}

fn f_permute<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.permute(&[1, 0])
}

#[test]
fn test_permute() {
    do_test_permute::<CpuRawTensor<f32>>();
    do_test_permute::<WgpuRawTensor<f32>>();
}

fn do_test_permute<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    // bit iffy - assumes at least 2 dimensions
    test_df::<RT, _, _, _>(|a| f_permute(a), Diffable::ones_like, |a| f_permute(a));
    test_ddf::<RT, _, _, _>(
        |a| a.permute(&[1, 0]),
        Diffable::zeros_like,
        Diffable::ones_like,
    );
}

fn f_expand<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.reshape(&[1, 2, 4]).expand(&[4, 2, 4])
}

#[test]
fn test_expand() {
    do_test_expand::<CpuRawTensor<f32>>();
    do_test_expand::<WgpuRawTensor<f32>>();
}

fn do_test_expand<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| f_expand(a), |a| a.constant_like(4.0), |a| f_expand(a));
    test_ddf::<RT, _, _, _>(
        |a| a.reshape(&[1, 2, 2]).expand(&[4, 2, 2]),
        Diffable::zeros_like,
        |a| a.constant_like(4.0),
    );
}

fn f_crop<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.crop(&[(0, 1), (1, 2)])
}

#[test]
fn test_crop() {
    do_test_crop::<CpuRawTensor<f32>>();
    do_test_crop::<WgpuRawTensor<f32>>();
}

fn do_test_crop<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(
        |a| f_crop(a),
        |a| a.ones_like().crop(&[(0, 1), (1, 2)]).pad(&[(0, 1), (1, 2)]),
        |a| f_crop(a),
    );
    test_ddf::<RT, _, _, _>(
        |a| f_crop(a),
        Diffable::zeros_like,
        // note: padding is different here from test_df, because test_ddf
        // tests with a [2,2] tensor only.
        |a| a.ones_like().crop(&[(0, 1), (1, 2)]).pad(&[(0, 1), (1, 0)]),
    );
}

fn f_pad<'t, T>(a: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.pad(&[(1, 2), (3, 4)])
}

#[test]
fn test_pad() {
    do_test_pad::<CpuRawTensor<f32>>();
    do_test_pad::<WgpuRawTensor<f32>>();
}

fn do_test_pad<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    test_df::<RT, _, _, _>(|a| f_pad(a), Diffable::ones_like, |a| f_pad(a));
    test_ddf::<RT, _, _, _>(|a| f_pad(a), Diffable::zeros_like, Diffable::ones_like);
}

fn f_matmul<'t, T>(a: &'t T, b: &'t T) -> T
where
    T: TensorLike<'t>,
    for<'s> &'s T: TensorLikeRef<T>,
{
    a.matmul(b)
}

#[test]
fn test_matmul() {
    do_test_matmul::<CpuRawTensor<f32>>();
    do_test_matmul::<WgpuRawTensor<f32>>();
}

fn do_test_matmul<RT: RawTensor<Elem = f32> + Clone + Debug>() {
    let a: Tensor<RT> = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b: Tensor<RT> = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let (primals, pullback) = vjpn(|ab| f_matmul(&ab[0], &ab[1]), &[&a, &b]);
    assert_eq!(primals.shape(), &[2, 2]);
    assert_eq!(primals.ravel(), &[58.0, 64.0, 139.0, 154.0]);

    let cotangent: Tensor<RT> = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let grads = pullback.call(&cotangent);
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), a.shape());
    assert_eq!(grads[1].shape(), b.shape());
    assert_eq!(grads[0].ravel(), [23.0, 29.0, 35.0, 53.0, 67.0, 81.0,]);
    assert_eq!(grads[1].ravel(), [13.0, 18.0, 17.0, 24.0, 21.0, 30.0,]);
}
