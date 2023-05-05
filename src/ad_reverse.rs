use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Add, Div, Index, Mul, Neg, Sub},
    ptr,
};

use crate::diffable_ops::{
    AddOp, BinaryOp, BinaryRevOp, Diffable, DivOp, EqOp, ExpOp, ExpandOp, LogOp, MaxOp, MulOp,
    PermuteOp, PowOp, ReshapeOp, SubOp, SumOp, UnaryOp, UnaryRevOp,
};

/// Reverse AD implementation.

enum TracedOp<'t, TTensor: 't> {
    Var,
    Unary(Box<dyn UnaryRevOp<TTensor> + 't>, usize),
    BinaryDA(Box<dyn BinaryRevOp<TTensor> + 't>, usize),
    BinaryDB(Box<dyn BinaryRevOp<TTensor> + 't>, usize),
    Binary(Box<dyn BinaryRevOp<TTensor> + 't>, usize, usize),
}

impl<T> Debug for TracedOp<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TracedOp::Var => write!(f, "Var"),
            TracedOp::Unary(_, i) => write!(f, "Unary(_, {i})"),
            TracedOp::Binary(_, i, j) => write!(f, "Binary(_, {i}, {j})"),
            TracedOp::BinaryDA(_, i) => write!(f, "BinaryL(_, {i})"),
            TracedOp::BinaryDB(_, i) => write!(f, "BinaryR(_, {i})"),
        }
    }
}

#[derive(Debug)]
pub struct Trace<'t, TTensor> {
    trace: RefCell<Vec<TracedOp<'t, TTensor>>>,
}

impl<'t, TTensor> Trace<'t, TTensor> {
    pub fn new() -> Self {
        Trace {
            trace: RefCell::new(vec![]),
        }
    }

    fn push_op(&self, primal: TTensor, node: TracedOp<'t, TTensor>) -> Reverse<'_, 't, TTensor> {
        let mut trace = self.trace.borrow_mut();
        let index = trace.len();
        trace.push(node);
        Reverse::Reverse(self, primal, index)
    }

    pub fn var(&self, primal: TTensor) -> Reverse<'_, 't, TTensor> {
        self.push_op(primal, TracedOp::Var)
    }
}

impl<T: Default> Default for Trace<'_, T> {
    fn default() -> Self {
        Trace::new()
    }
}

#[derive(Clone)] //needed for higher order derivatives
pub enum Reverse<'a, 't, T> {
    Lift(T),
    Reverse(&'a Trace<'t, T>, T, usize),
}

impl<T> Reverse<'_, '_, T> {
    pub fn lift(x: T) -> Self {
        Reverse::Lift(x)
    }

    fn into_primal(self) -> T {
        match self {
            Reverse::Lift(x) | Reverse::Reverse(_, x, _) => x,
        }
    }

    pub fn primal(&self) -> &T {
        match self {
            Reverse::Lift(x) | Reverse::Reverse(_, x, _) => x,
        }
    }
}

impl<T: Debug> Debug for Reverse<'_, '_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reverse::Lift(x) => write!(f, "Lift({x:?})"),
            Reverse::Reverse(_, x, i) => write!(f, "Reverse(_, {x:?}, {i})"),
        }
    }
}

impl<T: Default> Default for Reverse<'_, '_, T> {
    fn default() -> Self {
        Reverse::Lift(T::default())
    }
}

impl<T: PartialEq> PartialEq for Reverse<'_, '_, T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Lift(l0), Self::Lift(r0)) => l0 == r0,
            (Self::Reverse(_, l1, _), Self::Reverse(_, r1, _)) => l1 == r1,
            _ => false,
        }
    }
}

impl<'t, T: Diffable + Clone> Reverse<'_, 't, T> {
    fn unary<O: UnaryOp<T, Args = TArgs> + 't, TArgs>(&self, args: &TArgs) -> Self {
        let (op, primal) = O::f(self.primal(), args);
        match self {
            Reverse::Lift(_) => Reverse::Lift(primal),
            Reverse::Reverse(trace, _, tan) => {
                let op = TracedOp::Unary(Box::new(op), *tan);
                trace.push_op(primal, op)
            }
        }
    }

    fn binary<O: BinaryOp<T> + 't>(&self, rhs: &Self) -> Self {
        let (op, primal) = O::f(self.primal(), rhs.primal());
        match (self, rhs) {
            (Reverse::Lift(_), Reverse::Lift(_)) => Reverse::Lift(primal),
            (Reverse::Lift(_), Reverse::Reverse(trace, _, idx)) => {
                let op = TracedOp::BinaryDB(Box::new(op), *idx);
                trace.push_op(primal, op)
            }
            (Reverse::Reverse(trace, _, idx), Reverse::Lift(_)) => {
                let op = TracedOp::BinaryDA(Box::new(op), *idx);
                trace.push_op(primal, op)
            }
            (Reverse::Reverse(left_trace, _, left), Reverse::Reverse(right_trace, _, right)) => {
                assert!(ptr::eq(*left_trace, *right_trace), "traces must be the same - likely perturbation confusion. Are lifts in the right place?");
                let op = TracedOp::Binary(Box::new(op), *left, *right);
                left_trace.push_op(primal, op)
            }
        }
    }
}
impl<T: Clone + Diffable> Diffable for Reverse<'_, '_, T> {
    fn zeros_like(&self) -> Self {
        Reverse::lift(self.primal().zeros_like())
    }

    fn ones_like(&self) -> Self {
        Reverse::lift(self.primal().ones_like())
    }

    fn shape(&self) -> &[usize] {
        self.primal().shape()
    }

    fn add(&self, rhs: &Self) -> Self {
        self.binary::<AddOp>(rhs)
    }

    fn sub(&self, rhs: &Self) -> Self {
        self.binary::<SubOp>(rhs)
    }

    fn mul(&self, rhs: &Self) -> Self {
        self.binary::<MulOp<T>>(rhs)
    }

    fn div(&self, rhs: &Self) -> Self {
        self.binary::<DivOp<T>>(rhs)
    }

    fn pow(&self, rhs: &Self) -> Self {
        self.binary::<PowOp<T>>(rhs)
    }

    fn eq(&self, other: &Self) -> Self {
        self.binary::<EqOp>(other)
    }

    fn log(&self) -> Self {
        self.unary::<LogOp<T>, _>(&())
    }

    fn exp(&self) -> Self {
        self.unary::<ExpOp<T>, _>(&())
    }

    fn sum(&self, axes: &[usize]) -> Self {
        self.unary::<SumOp, _>(&axes.to_vec())
    }

    fn max(&self, axes: &[usize]) -> Self {
        self.unary::<MaxOp<T>, _>(&axes.to_vec())
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        self.unary::<ReshapeOp, _>(&shape.to_vec())
    }

    fn permute(&self, dims: &[usize]) -> Self {
        self.unary::<PermuteOp, _>(&dims.to_vec())
    }

    fn expand(&self, shape: &[usize]) -> Self {
        self.unary::<ExpandOp, _>(&shape.to_vec())
    }
}

macro_rules! impl_difftensor_reverse {
    ($op_trait:ident, $op_fn:ident) => {
        impl<'a, 't, T: Diffable + Clone> $op_trait for Reverse<'a, 't, T> {
            type Output = Self;

            fn $op_fn(self, rhs: Self) -> Self::Output {
                Diffable::$op_fn(&self, &rhs)
            }
        }

        impl<'a, 't, T: Diffable + Clone> $op_trait for &'a Reverse<'a, 't, T> {
            type Output = Reverse<'a, 't, T>;

            fn $op_fn(self, rhs: Self) -> Self::Output {
                Diffable::$op_fn(self, rhs)
            }
        }

        impl<'a, 't, T: Diffable + Clone> $op_trait<&'a Reverse<'a, 't, T>> for Reverse<'a, 't, T> {
            type Output = Self;

            fn $op_fn(self, rhs: &'a Reverse<'a, 't, T>) -> Self::Output {
                Diffable::$op_fn(&self, rhs)
            }
        }

        impl<'a, 't, T: Diffable + Clone> $op_trait<Reverse<'a, 't, T>> for &'a Reverse<'a, 't, T> {
            type Output = Reverse<'a, 't, T>;

            fn $op_fn(self, rhs: Reverse<'a, 't, T>) -> Self::Output {
                Diffable::$op_fn(self, &rhs)
            }
        }
    };
}

impl_difftensor_reverse!(Add, add);
impl_difftensor_reverse!(Sub, sub);
impl_difftensor_reverse!(Mul, mul);
impl_difftensor_reverse!(Div, div);

impl<'a, 't, T: Clone + Diffable> Neg for &'a Reverse<'a, 't, T> {
    type Output = Reverse<'a, 't, T>;

    fn neg(self) -> Self::Output {
        Reverse::lift(self.primal().zeros_like()).sub(self)
    }
}

impl<'a, 't, T: Clone + Diffable> Neg for Reverse<'a, 't, T> {
    type Output = Reverse<'a, 't, T>;

    fn neg(self) -> Self::Output {
        Reverse::lift(self.primal().zeros_like()).sub(self)
    }
}

#[derive(Debug)]
struct Grad<T> {
    grad: Option<Vec<T>>,
}

#[derive(Debug)]
struct Adjoints<T> {
    adjoints: Vec<Option<T>>,
}

impl<T: Diffable + Clone> Adjoints<T> {
    fn new(len: usize) -> Self {
        Adjoints {
            adjoints: vec![None; len],
        }
    }
    fn update(&mut self, idx: usize, dfda: T) {
        self.adjoints[idx] = self.adjoints[idx]
            .as_ref()
            .map(|c| c.add(&dfda))
            .or(Some(dfda));
    }
    fn pop(&mut self) {
        self.adjoints.pop();
    }
}

impl<T> Index<usize> for Adjoints<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        self.adjoints[idx].as_ref().unwrap()
    }
}

impl<T: Diffable + Clone> Grad<T> {
    pub fn of(result: &Reverse<T>) -> Self {
        match result {
            Reverse::Reverse(trace, primal, var) => {
                let trace = trace.trace.borrow();
                let mut adjoints = Adjoints::new(var + 1);
                adjoints.adjoints[*var] = Some(primal.ones_like());

                // backpropagate
                for i in (0..=*var).rev() {
                    if adjoints.adjoints[i].is_none() {
                        // no gradient to propagate - this node makes no contribution.
                        continue;
                    }
                    let node = &trace[i];
                    match node {
                        TracedOp::Var => {
                            // vars are always at the start of the trace (see vjp)
                            // and don't contribute to each other. So we can stop now.
                            break;
                        }
                        TracedOp::Unary(op, a) => adjoints.update(*a, op.df_dfda(&adjoints[i])),
                        TracedOp::Binary(op, a, b) => {
                            adjoints.update(*a, op.df_dfda(&adjoints[i]));
                            adjoints.update(*b, op.df_dfdb(&adjoints[i]));
                        }
                        TracedOp::BinaryDA(op, a) => {
                            adjoints.update(*a, op.df_dfda(&adjoints[i]));
                        }
                        TracedOp::BinaryDB(op, b) => {
                            adjoints.update(*b, op.df_dfdb(&adjoints[i]));
                        }
                    }
                    adjoints.pop();
                }

                Self {
                    grad: Some(
                        adjoints
                            .adjoints
                            .into_iter()
                            // zeros is correct, but we don't know the shape.
                            // Perhaps grad should contain options as well?
                            .map(|x| x.unwrap_or(primal.zeros_like()))
                            .collect(),
                    ),
                }
            }
            Reverse::Lift(_) => Self {
                // signal that the result was a constant, so we have no trace.
                grad: None,
            },
        }
    }

    pub fn get(&self, vars: &[&Reverse<T>]) -> Vec<T> {
        vars.iter()
            .map(|rev| match rev {
                Reverse::Reverse(_, p, var) => self
                    .grad
                    .as_ref()
                    .map_or(p.zeros_like(), |v| v[*var].clone()),
                Reverse::Lift(x) => x.zeros_like(),
            })
            .collect()
    }
}

fn wrap_slice<'a, 't, TTensor: Clone>(
    primal: &[&TTensor],
    trace: &'a Trace<'t, TTensor>,
) -> Vec<Reverse<'a, 't, TTensor>> {
    primal.iter().map(|&ati| trace.var(ati.clone())).collect()
}

#[allow(dead_code)]
pub fn vjp1<'t, TTensor: Diffable + Clone + 't, F>(f: F, at: &TTensor) -> (TTensor, TTensor)
where
    for<'a> F: Fn(&'a Reverse<'a, 't, TTensor>) -> Reverse<'a, 't, TTensor>,
{
    let trace = Trace::new();

    let owned_vars = wrap_slice(&[at], &trace);
    let vars: Vec<_> = owned_vars.iter().collect();

    let result = f(vars[0]);

    let grad = Grad::of(&result);
    (
        result.into_primal(),
        grad.get(&vars).into_iter().next().unwrap(),
    )
}

#[allow(dead_code)]
pub fn vjpn<'t, TTensor: Diffable + Clone + 't, F>(f: F, at: &[&TTensor]) -> (TTensor, Vec<TTensor>)
where
    for<'a> F: Fn(&'a [&'a Reverse<'a, 't, TTensor>]) -> Reverse<'a, 't, TTensor>,
{
    let trace = Trace::new();

    let owned_vars = wrap_slice(at, &trace);
    let vars: Vec<_> = owned_vars.iter().collect();

    let result = f(&vars);

    let grad = Grad::of(&result);
    (result.into_primal(), grad.get(&vars))
}

#[cfg(test)]
mod tests {

    use crate::{
        raw_tensor::RawTensor,
        raw_tensor_cpu::CpuRawTensor,
        raw_tensor_wgpu::WgpuRawTensor,
        shape::Shape,
        tensor::{Tensor, TensorLike, TensorLikeRef},
    };

    use super::*;

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
        let (f_actual, df_actual) = vjp1(f, &at);
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
        let (df_actual, ddf_actual) = vjp1(|r| vjp1(&f, r).1, &at);
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
        let a: &Tensor<RT> = &Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, -3.0, -2.0]);
        let b: &Tensor<RT> = &Tensor::new(&[2, 3], &[4.0, 5.0, 6.0, 7.0, -6.0, -5.0]);
        let (f_actual, df_actual) = vjpn(|es| f(es[0], es[1]), &[a, b]);
        let (dfda_actual, dfdb_actual) = (&df_actual[0], &df_actual[1]);
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
            |t| Reverse::lift(t.primal().clone()),
            Diffable::zeros_like,
            Clone::clone,
        );
        test_ddf::<RT, _, _, _>(
            |t| Reverse::lift(Reverse::lift(t.primal().primal().clone())),
            Diffable::zeros_like,
            Diffable::zeros_like,
        );
    }

    fn f_id<'t, T>(a: &'t T) -> T
    where
        T: TensorLike<'t>,
        &'t T: TensorLikeRef<T>,
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
        &'t T: TensorLikeRef<T>,
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
        &'t T: TensorLikeRef<T>,
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
        &'t T: TensorLikeRef<T>,
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
        &'t T: TensorLikeRef<T>,
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
            Diffable::pow,
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
        &'t T: TensorLikeRef<T>,
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
        &'t T: TensorLikeRef<T>,
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
        &'t T: TensorLikeRef<T>,
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
        &'t T: TensorLikeRef<T>,
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
        &'t T: TensorLikeRef<T>,
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
}
