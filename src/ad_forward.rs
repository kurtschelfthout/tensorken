use std::{
    fmt::Debug,
    ops::{Add, Div, Index, Mul, Neg, Sub},
    ptr,
};

use crate::{
    ad_forward_ops::{CropOp, ExpandOp, MaxOp, PadOp, PermuteOp, ReshapeOp, SumOp},
    ad_ops::{
        AddOp, BinaryDiffOp, BinaryOp, DivOp, EqOp, ExpOp, LogOp, MulOp, PowOp, SubOp, UnaryDiffOp,
        UnaryOp,
    },
    ad_trace::{Trace, TracedOp},
    Diffable, DiffableExt,
};

/// Forward AD implementation.

#[derive(Clone)] //needed for higher order derivatives
pub enum Forward<'a, 't, T> {
    Lift(T),
    Forward(&'a Trace<'t, T>, T, usize),
}

impl<T> Forward<'_, '_, T> {
    fn into_primal(self) -> T {
        match self {
            Forward::Lift(x) | Forward::Forward(_, x, _) => x,
        }
    }

    pub fn primal(&self) -> &T {
        match self {
            Forward::Lift(x) | Forward::Forward(_, x, _) => x,
        }
    }

    fn try_get_adjoint_index(&self) -> Option<usize> {
        match self {
            Forward::Forward(_, _, i) => Some(*i),
            Forward::Lift(_) => None,
        }
    }
}

impl<T: Clone> Forward<'_, '_, T> {
    pub fn lift(x: &T) -> Self {
        Forward::Lift(x.clone())
    }
}

impl<T: Debug> Debug for Forward<'_, '_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Forward::Lift(x) => write!(f, "Lift({x:?})"),
            Forward::Forward(_, x, i) => write!(f, "Forward(_, {x:?}, {i})"),
        }
    }
}

impl<T: PartialEq> PartialEq for Forward<'_, '_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.primal() == other.primal()
    }
}

impl<'a, 't, T: Diffable> Forward<'a, 't, T> {
    fn push_op(trace: &'a Trace<'t, T>, primal: T, op: TracedOp<'t, T>) -> Forward<'a, 't, T> {
        let idx = trace.push_op(op);
        Forward::Forward(trace, primal, idx)
    }

    fn unary<Op: UnaryOp<T, Args = TArgs> + UnaryDiffOp<T> + 't, TArgs: ?Sized>(
        &self,
        args: &TArgs,
    ) -> Self {
        let (op, primal) = Op::f(self.primal(), args);
        match self {
            Forward::Lift(_) => Forward::Lift(primal),
            Forward::Forward(trace, _, tan) => {
                let op = TracedOp::Unary(Box::new(op), *tan);
                Self::push_op(trace, primal, op)
            }
        }
    }

    fn binary<Op: BinaryOp<T> + BinaryDiffOp<T> + 't>(&self, rhs: &Self) -> Self {
        let (op, primal) = Op::f(self.primal(), rhs.primal());
        match (self, rhs) {
            (Forward::Lift(_), Forward::Lift(_)) => Forward::Lift(primal),
            (Forward::Lift(_), Forward::Forward(trace, _, idx)) => {
                let op = TracedOp::BinaryDB(Box::new(op), *idx);
                Self::push_op(trace, primal, op)
            }
            (Forward::Forward(trace, _, idx), Forward::Lift(_)) => {
                let op = TracedOp::BinaryDA(Box::new(op), *idx);
                Self::push_op(trace, primal, op)
            }
            (Forward::Forward(left_trace, _, left), Forward::Forward(right_trace, _, right)) => {
                assert!(ptr::eq(*left_trace, *right_trace), "traces must be the same - likely perturbation confusion. Are lifts in the right place?");
                let op = TracedOp::Binary(Box::new(op), *left, *right);
                Self::push_op(left_trace, primal, op)
            }
        }
    }
}

impl<T: Clone + Diffable> Diffable for Forward<'_, '_, T> {
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
        self.binary::<EqOp>(other)
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
        Forward::Lift(T::new(shape, data))
    }

    fn ravel(&self) -> Vec<Self::Elem> {
        self.primal().ravel()
    }
}

crate::math_macros::impl_bin_op!(Add, add, Forward<'a, 't, T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Sub, sub, Forward<'a, 't, T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Mul, mul, Forward<'a, 't, T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Div, div, Forward<'a, 't, T: Diffable + Clone>);

crate::math_macros::impl_un_op!(Neg, neg, Forward<'a, 't, T: Diffable + Clone>);

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
}

impl<T> Index<usize> for Adjoints<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        self.adjoints[idx].as_ref().unwrap()
    }
}

/// `PushForward` is a function from a tangent vector to a `Vec` of tangent vectors.
/// Use `call` to access it.
pub struct PushForward<'t, T> {
    trace: Trace<'t, T>,
    index_result: Option<usize>,
    zero_primal: T,
    // only used to assert the shape matches with tangent
    // primal_shape: Vec<usize>,
}

impl<T: Diffable + Clone> PushForward<'_, T> {
    fn forward(&self, var: usize, tangents: &[&T]) -> T {
        // assert!(
        //     self.primal_shape == tangent.shape(),
        //     "tangent shape must match primal shape"
        // );
        let trace = self.trace.borrow();
        let mut adjoints = Adjoints::new(var + 1);
        for (i, tangent) in tangents.iter().enumerate() {
            adjoints.adjoints[i] = Some((*tangent).clone());
        }

        // propagate
        for i in 0..=var {
            // if none, there's no gradient to propagate - this node makes no contribution.
            // if adjoints.adjoints[i].is_some() {
            let node = &trace[i];
            match node {
                TracedOp::Var => {
                    // vars are always at the start of the trace (see jvp)
                    // and don't contribute to each other. We can continue.
                    continue;
                }
                TracedOp::Unary(op, a) => adjoints.update(i, op.df_dfda(&adjoints[*a])),
                TracedOp::Binary(op, a, b) => {
                    adjoints.update(i, op.df_dfda(&adjoints[*a]));
                    adjoints.update(i, op.df_dfdb(&adjoints[*b]));
                }
                TracedOp::BinaryDA(op, a) => {
                    adjoints.update(i, op.df_dfda(&adjoints[*a]));
                }
                TracedOp::BinaryDB(op, b) => {
                    adjoints.update(i, op.df_dfdb(&adjoints[*b]));
                } // }
            }
        }
        adjoints
            .adjoints
            .swap_remove(var)
            .unwrap_or(self.zero_primal.clone())
    }

    /// Takes a cotangent tensor with the same shape as the result of this `PullBack`'s originating vjp function,
    /// and returns a `Vec` of cotangent vectors with the same number and shapes as vjp's primals,
    /// representing the vector-Jacobian product of vjp's function evaluated at primals.
    pub fn call(&self, tangents: &[&T]) -> T
    where
        T: Diffable + Clone,
    {
        match self.index_result {
            None => self.zero_primal.clone(),
            Some(var) => self.forward(var, tangents),
        }
    }
}

/// Compute a forward-mode Jacobian-vector product of a function `f` evaluated at the given primals.
/// Returns a tuple of the result of `f` and a `PushForward` object. `PushForward.call` can be used to
/// compute the Jacobian-vector product of `f` at any tangent.
pub fn jvpn<'b, 't, T: Diffable + Clone + 't, F>(f: F, at: &[&T]) -> (T, PushForward<'t, T>)
where
    for<'a> F: Fn(&'a [Forward<'a, 't, T>]) -> Forward<'a, 't, T>,
{
    let trace = Trace::new();
    let vars: Vec<_> = at
        .iter()
        .map(|&ati| {
            let index = trace.var();
            Forward::Forward(&trace, ati.clone(), index)
        })
        .collect();
    let result = f(&vars);

    let index_result = result.try_get_adjoint_index();
    let zero_primal = result.primal().zeros_like();
    (
        result.into_primal(),
        PushForward {
            trace,
            index_result,
            zero_primal,
            // primal_shape,
        },
    )
}

/// Compute the result and the gradient of a function at the given primals.
pub fn value_and_diff_forwardn<'t, T: Diffable + Clone + 't, F>(f: F, at: &[&T]) -> (T, Vec<T>)
where
    for<'a> F: Fn(&'a [Forward<'a, 't, T>]) -> Forward<'a, 't, T>,
{
    let (primal, pushforward) = jvpn(f, at);

    let mut tangents = Vec::with_capacity(at.len());
    let args = at.iter().map(|&ati| ati.zeros_like()).collect::<Vec<_>>();
    for (i, tangent) in at.iter().enumerate() {
        let mut args: Vec<_> = args.iter().clone().collect();
        let one = tangent.ones_like();
        args[i] = &one;
        let tangent = pushforward.call(&args);
        tangents.push(tangent);
    }

    (primal, tangents)
}

/// Compute the result and the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn value_and_diff_forward1<'t, T: Diffable + Clone + 't, F>(f: F, at: &T) -> (T, T)
where
    for<'a> F: Fn(&'a Forward<'a, 't, T>) -> Forward<'a, 't, T>,
{
    let (primal, tangents) = value_and_diff_forwardn(|s| f(&s[0]), &[at]);
    (primal, tangents.into_iter().next().unwrap())
}

/// Compute the result and the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc)]
pub fn value_and_diff_forward2<'t, T: Diffable + Clone + 't, F>(
    f: F,
    at0: &T,
    at1: &T,
) -> (T, (T, T))
where
    for<'a> F: Fn(&'a Forward<'a, 't, T>, &'a Forward<'a, 't, T>) -> Forward<'a, 't, T>,
{
    let (primal, tangents) = value_and_diff_forwardn(|s| f(&s[0], &s[1]), &[at0, at1]);
    let mut dr_iter = tangents.into_iter();
    (primal, (dr_iter.next().unwrap(), dr_iter.next().unwrap()))
}

/// Compute the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn diff1<'t, T: Diffable + Clone + 't, F>(f: F, at: &T) -> T
where
    for<'a> F: Fn(&'a Forward<'a, 't, T>) -> Forward<'a, 't, T>,
{
    value_and_diff_forward1(f, at).1
}

/// Compute the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc)]
pub fn diff2<'t, T: Diffable + Clone + 't, F>(f: F, at0: &T, at1: &T) -> (T, T)
where
    for<'a> F: Fn(&'a Forward<'a, 't, T>, &'a Forward<'a, 't, T>) -> Forward<'a, 't, T>,
{
    value_and_diff_forward2(f, at0, at1).1
}

// /// Jacobian of `f` evaluated row-by-row at `at` using reverse-mode AD.
// #[allow(clippy::missing_panics_doc)]
// pub fn jacrevn<'b, 't, T: Diffable + Clone + 't, F>(f: F, at: &[&T]) -> T
// where
//     for<'a> F: Fn(&'a [Reverse<'a, 't, T>]) -> Reverse<'a, 't, T>,
// {
//     let (primal, pullback) = vjpn(f, at);
//     let mut s = vec![primal
//         .shape()
//         .iter()
//         .copied()
//         .reduce(|acc, e| acc * e)
//         .unwrap()];
//     s.extend(primal.shape());
//     let i = T::eye(primal.shape().size()).reshape(&s);
//     let mut tangents: Vec<T> = Vec::with_capacity(i.shape()[0]);
//     for row_idx in 0..i.shape()[0] {
//         let row = i.at(row_idx);
//         let row_tangent = pullback.call(&row).into_iter().next().unwrap();
//         tangents.push(row_tangent);
//     }
//     let t_refs = tangents.iter().collect::<Vec<_>>();
//     T::stack(&t_refs, 1)
// }

// /// Jacobian of `f` evaluated row-by-row at `at` using reverse-mode AD.
// pub fn jacrev1<'t, T: Diffable + Clone + 't, F>(f: F, at: &T) -> T
// where
//     for<'a> F: Fn(&'a Reverse<'a, 't, T>) -> Reverse<'a, 't, T>,
// {
//     jacrevn(|s| f(&s[0]), &[at])
// }

// /// Jacobian of `f` evaluated row-by-row at `at0, at1` using reverse-mode AD.
// pub fn jacrev2<'t, T: Diffable + Clone + 't, F>(f: F, at0: &T, at1: &T) -> T
// where
//     for<'a> F: Fn(&'a Reverse<'a, 't, T>, &'a Reverse<'a, 't, T>) -> Reverse<'a, 't, T>,
// {
//     jacrevn(|s| f(&s[0], &s[1]), &[at0, at1])
// }
