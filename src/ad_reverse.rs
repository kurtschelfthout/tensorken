use std::{
    fmt::Debug,
    ops::{Add, Div, Index, Mul, Neg, Sub},
    ptr,
};

use crate::{
    ad_ops::{
        AddOp, BinaryDiffOp, BinaryOp, DivOp, ExpOp, LogOp, MulOp, PowOp, SubOp, UnaryDiffOp,
        UnaryOp,
    },
    ad_reverse_ops::{CropOp, ExpandOp, MaxOp, PadOp, PermuteOp, ReshapeOp, SumOp},
    ad_trace::{Trace, TracedOp},
    Diffable, DiffableExt, IndexValue, Shape,
};

/// Reverse AD implementation.

#[derive(Clone)] //needed for higher order derivatives
pub enum Reverse<'a, 't, T> {
    Lift(T),
    Reverse(&'a Trace<'t, T>, T, usize),
}

impl<T> Reverse<'_, '_, T> {
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

    fn try_get_adjoint_index(&self) -> Option<usize> {
        match self {
            Reverse::Reverse(_, _, i) => Some(*i),
            Reverse::Lift(_) => None,
        }
    }
}

impl<T: Clone> Reverse<'_, '_, T> {
    pub fn lift(x: &T) -> Self {
        Reverse::Lift(x.clone())
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

impl<T: PartialEq> PartialEq for Reverse<'_, '_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.primal() == other.primal()
    }
}

impl<'a, 't, T: Diffable> Reverse<'a, 't, T> {
    fn push_op(trace: &'a Trace<'t, T>, primal: T, op: TracedOp<'t, T>) -> Reverse<'a, 't, T> {
        let idx = trace.push_op(op);
        Reverse::Reverse(trace, primal, idx)
    }

    fn unary<Op: UnaryOp<T, Args = TArgs> + UnaryDiffOp<T> + 't, TArgs: ?Sized>(
        &self,
        args: &TArgs,
    ) -> Self {
        let (op, primal) = Op::f(self.primal(), args);
        match self {
            Reverse::Lift(_) => Reverse::Lift(primal),
            Reverse::Reverse(trace, _, tan) => {
                let op = TracedOp::Unary(Box::new(op), *tan);
                Self::push_op(trace, primal, op)
            }
        }
    }

    fn binary<Op: BinaryOp<T> + BinaryDiffOp<T> + 't>(&self, rhs: &Self) -> Self {
        let (op, primal) = Op::f(self.primal(), rhs.primal());
        match (self, rhs) {
            (Reverse::Lift(_), Reverse::Lift(_)) => Reverse::Lift(primal),
            (Reverse::Lift(_), Reverse::Reverse(trace, _, idx)) => {
                let op = TracedOp::BinaryDB(Box::new(op), *idx);
                Self::push_op(trace, primal, op)
            }
            (Reverse::Reverse(trace, _, idx), Reverse::Lift(_)) => {
                let op = TracedOp::BinaryDA(Box::new(op), *idx);
                Self::push_op(trace, primal, op)
            }
            (Reverse::Reverse(left_trace, _, left), Reverse::Reverse(right_trace, _, right)) => {
                assert!(ptr::eq(*left_trace, *right_trace), "traces must be the same - likely perturbation confusion. Are lifts in the right place?");
                let op = TracedOp::Binary(Box::new(op), *left, *right);
                Self::push_op(left_trace, primal, op)
            }
        }
    }
}

impl<T: Clone + Diffable> Diffable for Reverse<'_, '_, T> {
    type Elem = T::Elem;
    fn log(&self) -> Self {
        self.unary::<LogOp<T>, _>(&())
    }

    fn exp(&self) -> Self {
        self.unary::<ExpOp<T>, _>(&())
    }

    fn elementwise_add(&self, rhs: &Self) -> Self {
        self.binary::<AddOp>(rhs)
    }

    fn elementwise_sub(&self, rhs: &Self) -> Self {
        self.binary::<SubOp>(rhs)
    }

    fn elementwise_mul(&self, rhs: &Self) -> Self {
        self.binary::<MulOp<T>>(rhs)
    }

    fn elementwise_div(&self, rhs: &Self) -> Self {
        self.binary::<DivOp<T>>(rhs)
    }

    fn elementwise_pow(&self, rhs: &Self) -> Self {
        self.binary::<PowOp<T>>(rhs)
    }

    fn elementwise_eq(&self, other: &Self) -> Self {
        Reverse::Lift(self.primal().elementwise_eq(other.primal()))
    }

    fn sum(&self, axes: &[usize]) -> Self {
        self.unary::<SumOp, _>(axes)
    }

    fn max(&self, axes: &[usize]) -> Self {
        self.unary::<MaxOp<T>, _>(axes)
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        self.unary::<ReshapeOp, _>(shape)
    }

    fn permute(&self, dims: &[usize]) -> Self {
        self.unary::<PermuteOp, _>(dims)
    }

    fn expand(&self, shape: &[usize]) -> Self {
        self.unary::<ExpandOp, _>(shape)
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        self.unary::<PadOp, _>(padding)
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        self.unary::<CropOp, _>(limits)
    }

    fn shape(&self) -> &[usize] {
        self.primal().shape()
    }

    fn new(shape: &[usize], data: &[Self::Elem]) -> Self {
        Reverse::Lift(T::new(shape, data))
    }
}

crate::math_macros::impl_bin_op!(Add, add, Reverse<'a, 't, T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Sub, sub, Reverse<'a, 't, T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Mul, mul, Reverse<'a, 't, T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Div, div, Reverse<'a, 't, T: Diffable + Clone>);

crate::math_macros::impl_un_op!(Neg, neg, Reverse<'a, 't, T: Diffable + Clone>);

// somewhat wonky helper type to deal with optional adjoints
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
    fn update(&mut self, idx: usize, df: T) {
        self.adjoints[idx] = self.adjoints[idx]
            .as_ref()
            .map(|c| c.elementwise_add(&df))
            .or(Some(df));
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

/// `PullBack` is a function from a cotangent vector to a `Vec` of cotangent vectors.
/// Use `call` to access it.
pub struct PullBack<'t, T> {
    trace: Trace<'t, T>,
    index_result: Option<usize>,
    zero_primals: Vec<T>,
    // only used to assert the shape matches with cotangent
    primal_out_shape: Vec<usize>,
}

impl<T: Diffable + Clone> PullBack<'_, T> {
    fn reverse(&self, var: usize, adjoint: &T) -> Vec<T> {
        assert!(
            self.primal_out_shape == adjoint.shape(),
            "cotangent shape must match primal shape"
        );
        let trace = self.trace.borrow();
        let mut adjoints_acc = Adjoints::new(var + 1);
        adjoints_acc.adjoints[var] = Some(adjoint.clone());

        // backpropagate
        for i in (0..=var).rev() {
            // if none, there's no gradient to propagate - this node makes no contribution.
            if adjoints_acc.adjoints[i].is_some() {
                let node = &trace[i];
                match node {
                    TracedOp::Var => {
                        // vars are always at the start of the trace (see vjp)
                        // and don't contribute to each other. We can stop.
                        // We can't pop this adjoint, so we must break.
                        break;
                    }
                    TracedOp::Unary(op, a) => adjoints_acc.update(*a, op.df_dfda(&adjoints_acc[i])),
                    TracedOp::Binary(op, a, b) => {
                        adjoints_acc.update(*a, op.df_dfda(&adjoints_acc[i]));
                        adjoints_acc.update(*b, op.df_dfdb(&adjoints_acc[i]));
                    }
                    TracedOp::BinaryDA(op, a) => {
                        adjoints_acc.update(*a, op.df_dfda(&adjoints_acc[i]));
                    }
                    TracedOp::BinaryDB(op, b) => {
                        adjoints_acc.update(*b, op.df_dfdb(&adjoints_acc[i]));
                    }
                }
            }
            adjoints_acc.pop();
        }
        assert_eq!(
            adjoints_acc.adjoints.len(),
            self.zero_primals.len(),
            "adjoints length after propagation must match length of given zero primals"
        );
        adjoints_acc
            .adjoints
            .into_iter()
            .zip(self.zero_primals.iter())
            .map(|(x, z)| x.unwrap_or(z.clone()))
            .collect()
    }

    /// Takes a cotangent tensor with the same shape as the result of this `PullBack`'s originating vjp function,
    /// and returns a `Vec` of cotangent vectors with the same number and shapes as vjp's primals,
    /// representing the vector-Jacobian product of vjp's function evaluated at primals.
    pub fn call(&self, adjoint: &T) -> Vec<T>
    where
        T: Diffable + Clone,
    {
        match self.index_result {
            None => self.zero_primals.clone(),
            Some(var) => self.reverse(var, adjoint),
        }
    }
}

/// Compute a reverse-mode vector-Jacobian product of a function `f` evaluated at the given primals.
/// Returns a tuple of the result of `f` and a `PullBack` object. `PullBack.call` can be used to
/// compute the vector-Jacobian product of `f` at any cotangent.
pub fn vjpn<'b, 't, T: Diffable + Clone + 't, F>(f: F, at: &[&T]) -> (T, PullBack<'t, T>)
where
    for<'a> F: Fn(&'a [Reverse<'a, 't, T>]) -> Reverse<'a, 't, T>,
{
    let trace = Trace::new();
    let vars: Vec<_> = at
        .iter()
        .map(|&ati| {
            let index = trace.var();
            Reverse::Reverse(&trace, ati.clone(), index)
        })
        .collect();
    let result = f(&vars);

    let index_result = result.try_get_adjoint_index();
    let zero_primals: Vec<_> = at.iter().map(|&ati| ati.zeros_like()).collect();
    let primal_out_shape = result.shape().to_vec();
    (
        result.into_primal(),
        PullBack {
            trace,
            index_result,
            zero_primals,
            primal_out_shape,
        },
    )
}

/// Compute the result and the gradient of a function at the given primals.
pub fn value_and_gradn<'t, T: Diffable + Clone + 't, F>(f: F, at: &[&T]) -> (T, Vec<T>)
where
    for<'a> F: Fn(&'a [Reverse<'a, 't, T>]) -> Reverse<'a, 't, T>,
{
    let (primal, pullback) = vjpn(f, at);
    let tangents = pullback.call(&primal.ones_like());
    (primal, tangents)
}

/// Compute the result and the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn value_and_grad1<'t, T: Diffable + Clone + 't, F>(f: F, at: &T) -> (T, T)
where
    for<'a> F: Fn(&'a Reverse<'a, 't, T>) -> Reverse<'a, 't, T>,
{
    let (primal, tangents) = value_and_gradn(|s| f(&s[0]), &[at]);
    (primal, tangents.into_iter().next().unwrap())
}

/// Compute the result and the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc)]
pub fn value_and_grad2<'t, T: Diffable + Clone + 't, F>(f: F, at0: &T, at1: &T) -> (T, (T, T))
where
    for<'a> F: Fn(&'a Reverse<'a, 't, T>, &'a Reverse<'a, 't, T>) -> Reverse<'a, 't, T>,
{
    let (primal, tangents) = value_and_gradn(|s| f(&s[0], &s[1]), &[at0, at1]);
    let mut dr_iter = tangents.into_iter();
    (primal, (dr_iter.next().unwrap(), dr_iter.next().unwrap()))
}

/// Compute the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn grad1<'t, T: Diffable + Clone + 't, F>(f: F, at: &T) -> T
where
    for<'a> F: Fn(&'a Reverse<'a, 't, T>) -> Reverse<'a, 't, T>,
{
    value_and_grad1(f, at).1
}

/// Compute the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc)]
pub fn grad2<'t, T: Diffable + Clone + 't, F>(f: F, at0: &T, at1: &T) -> (T, T)
where
    for<'a> F: Fn(&'a Reverse<'a, 't, T>, &'a Reverse<'a, 't, T>) -> Reverse<'a, 't, T>,
{
    value_and_grad2(f, at0, at1).1
}

/// Jacobian of `f` evaluated row-by-row at `at` using reverse-mode AD.
#[allow(clippy::missing_panics_doc)]
pub fn jacrev<'b, 't, T: Diffable + Clone + 't, F>(f: F, at: &T) -> T
where
    for<'a> F: Fn(&'a Reverse<'a, 't, T>) -> Reverse<'a, 't, T>,
{
    let (primal, pullback) = vjpn(|s| f(&s[0]), &[at]);

    let mut s = vec![primal.shape().size()];
    s.extend(primal.shape());
    let i = T::eye(primal.shape().size()).reshape(&s);

    let mut tangents: Vec<T> = Vec::with_capacity(i.shape()[0]);
    for row_idx in 0..i.shape()[0] {
        let row = i.at(row_idx);
        let row_tangent = pullback.call(&row).into_iter().next().unwrap();
        tangents.push(row_tangent);
    }
    let t_refs = tangents.iter().collect::<Vec<_>>();
    T::stack(&t_refs[..], 0)
}
