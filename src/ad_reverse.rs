use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Add, Div, Index, Mul, Neg, Sub},
    ptr,
};

use crate::{
    ad_reverse_ops::{
        AddOp, BinaryOp, BinaryRevOp, CropOp, DivOp, EqOp, ExpOp, ExpandOp, LogOp, MaxOp, MulOp,
        PadOp, PermuteOp, PowOp, ReshapeOp, SubOp, SumOp, UnaryOp, UnaryRevOp,
    },
    diffable::DiffableExt,
    Diffable,
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
    #[must_use]
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
        match (self, other) {
            (Self::Lift(l0), Self::Lift(r0)) => l0 == r0,
            (Self::Reverse(_, l1, _), Self::Reverse(_, r1, _)) => l1 == r1,
            _ => false,
        }
    }
}

impl<'t, T: Diffable> Reverse<'_, 't, T> {
    fn unary<O: UnaryOp<T, Args = TArgs> + 't, TArgs: ?Sized>(&self, args: &TArgs) -> Self {
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

    fn zeros_like(&self) -> Self {
        Reverse::Lift(self.primal().zeros_like())
    }

    fn ones_like(&self) -> Self {
        Reverse::Lift(self.primal().ones_like())
    }

    fn shape(&self) -> &[usize] {
        self.primal().shape()
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
    fn update(&mut self, idx: usize, dfda: T) {
        self.adjoints[idx] = self.adjoints[idx]
            .as_ref()
            .map(|c| c.elementwise_add(&dfda))
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
    fn reverse(&self, var: usize, cotangent: &T) -> Vec<T> {
        assert!(
            self.primal_out_shape == cotangent.shape(),
            "cotangent shape must match primal shape"
        );
        let trace = self.trace.trace.borrow();
        let mut adjoints = Adjoints::new(var + 1);
        adjoints.adjoints[var] = Some(cotangent.clone());

        // backpropagate
        for i in (0..=var).rev() {
            // if none, there's no gradient to propagate - this node makes no contribution.
            if adjoints.adjoints[i].is_some() {
                let node = &trace[i];
                match node {
                    TracedOp::Var => {
                        // vars are always at the start of the trace (see vjp)
                        // and don't contribute to each other. We can stop.
                        // We can't pop this adjoint, so we must break.
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
            }
            adjoints.pop();
        }
        assert_eq!(
            adjoints.adjoints.len(),
            self.zero_primals.len(),
            "adjoints length after propagation must match length of given zero primals"
        );
        adjoints
            .adjoints
            .into_iter()
            .zip(self.zero_primals.iter())
            .map(|(x, z)| x.unwrap_or(z.clone()))
            .collect()
    }

    /// Takes a cotangent tensor with the same shape as the result of this `PullBack` originating vjp function,
    /// and returns a `Vec` of cotangent vectors with the same number and shapes as vjp's primals,
    ///  representing the vector-Jacobian product of vjp's function evaluated at primals.
    pub fn call(&self, cotangent: &T) -> Vec<T>
    where
        T: Diffable + Clone,
    {
        match self.index_result {
            None => self.zero_primals.clone(),
            Some(var) => self.reverse(var, cotangent),
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
    let vars: Vec<_> = at.iter().map(|&ati| trace.var(ati.clone())).collect();
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
    let (primal, pb) = vjpn(f, at);
    let tangents = pb.call(&primal.ones_like());
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
