use std::{fmt::Debug, marker::PhantomData, ops::Index, rc::Rc};

use crate::{
    ad_ops::{
        AddOp, BinaryDiffOp, BinaryOp, DivOp, ExpOp, LogOp, MulOp, PowOp, SubOp, UnaryDiffOp,
        UnaryOp,
    },
    ad_ops_reverse::{CropOp, ExpandOp, MaxOp, PadOp, PermuteOp, ReshapeOp, SumOp},
    ad_trace::{Trace, TracedOp},
    num::{Bool, CastFrom, Elem, Float, Num},
    DiffableOps, Shape, Tensor,
};

/// Reverse AD implementation.

#[derive(Clone)] //needed for higher order derivatives
pub enum Reverse<T> {
    Lift(T),
    Reverse(Rc<Trace<T>>, T, usize),
}

impl<T> Reverse<T> {
    fn into_primal(self) -> T {
        match self {
            Self::Lift(x) | Self::Reverse(_, x, _) => x,
        }
    }

    pub fn primal(&self) -> &T {
        match self {
            Self::Lift(x) | Self::Reverse(_, x, _) => x,
        }
    }

    fn try_get_adjoint_index(&self) -> Option<usize> {
        match self {
            Self::Reverse(_, _, i) => Some(*i),
            Self::Lift(_) => None,
        }
    }
}

impl<T: Debug> Debug for Reverse<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lift(x) => write!(f, "Lift({x:?})"),
            Self::Reverse(_, x, i) => write!(f, "Reverse(_, {x:?}, {i})"),
        }
    }
}

impl<T: PartialEq> PartialEq for Reverse<T> {
    fn eq(&self, other: &Self) -> bool {
        self.primal() == other.primal()
    }
}

impl<T> Reverse<T> {
    fn push_op(trace: &Rc<Trace<T>>, primal: T, op: TracedOp<T>) -> Self {
        let idx = trace.push_op(op);
        Self::Reverse(Rc::clone(trace), primal, idx)
    }

    fn unary<Op: UnaryOp<T, Args = TArgs> + UnaryDiffOp<T> + 'static, TArgs: ?Sized>(
        &self,
        args: &TArgs,
    ) -> Self {
        let (primal, op) = Op::f(self.primal(), args);
        match self {
            Self::Lift(_) => Self::Lift(primal),
            Self::Reverse(trace, _, tan) => {
                let op = TracedOp::Unary(Box::new(op), *tan);
                Self::push_op(trace, primal, op)
            }
        }
    }

    fn binary<Op: BinaryOp<T> + BinaryDiffOp<T> + 'static>(&self, rhs: &Self) -> Self {
        let (primal, op) = Op::f(self.primal(), rhs.primal());
        match (self, rhs) {
            (Self::Lift(_), Self::Lift(_)) => Self::Lift(primal),
            (Self::Lift(_), Self::Reverse(trace, _, idx)) => {
                let op = TracedOp::BinaryDB(Box::new(op), *idx);
                Self::push_op(trace, primal, op)
            }
            (Self::Reverse(trace, _, idx), Self::Lift(_)) => {
                let op = TracedOp::BinaryDA(Box::new(op), *idx);
                Self::push_op(trace, primal, op)
            }
            (Self::Reverse(left_trace, _, left), Self::Reverse(right_trace, _, right)) => {
                assert!(Rc::ptr_eq(left_trace, right_trace), "traces must be the same - likely perturbation confusion. Are lifts in the right place?");
                let op = TracedOp::Binary(Box::new(op), *left, *right);
                Self::push_op(left_trace, primal, op)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReverseImpl<I>(PhantomData<I>);

impl<I: 'static + DiffableOps> DiffableOps for ReverseImpl<I> {
    type Repr<E: Clone> = Reverse<I::Repr<E>>;

    fn log<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.unary::<LogOp<I::Repr<E>, E, I>, _>(&())
    }

    fn exp<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.unary::<ExpOp<I::Repr<E>, E, I>, _>(&())
    }

    fn elementwise_add<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<AddOp<E, I>>(rhs)
    }

    fn elementwise_sub<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<SubOp<E, I>>(rhs)
    }

    fn elementwise_mul<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<MulOp<I::Repr<E>, E, I>>(rhs)
    }

    fn elementwise_div<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<DivOp<I::Repr<E>, E, I>>(rhs)
    }

    fn elementwise_pow<E: Float>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<PowOp<I::Repr<E>, E, I>>(rhs)
    }

    fn elementwise_eq<E: PartialEq + Elem>(
        lhs: &Self::Repr<E>,
        rhs: &Self::Repr<E>,
    ) -> Self::Repr<bool> {
        Reverse::Lift(I::elementwise_eq::<E>(lhs.primal(), rhs.primal()))
    }

    fn sum<E: Num>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        t.unary::<SumOp<E, I>, _>(axes)
    }

    fn max<E: Num + CastFrom<bool>>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        t.unary::<MaxOp<I::Repr<E>, E, I>, _>(axes)
    }

    fn reshape<E: Elem>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        t.unary::<ReshapeOp<E, I>, _>(shape)
    }

    fn permute<E: Elem>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E> {
        t.unary::<PermuteOp<E, I>, _>(permutation)
    }

    fn expand<E: Num>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        t.unary::<ExpandOp<E, I>, _>(shape)
    }

    fn pad<E: Bool>(t: &Self::Repr<E>, padding: &[(usize, usize)]) -> Self::Repr<E> {
        t.unary::<PadOp<E, I>, _>(padding)
    }

    fn crop<E: Bool>(t: &Self::Repr<E>, limits: &[(usize, usize)]) -> Self::Repr<E> {
        t.unary::<CropOp<E, I>, _>(limits)
    }

    fn new<E: Elem>(shape: &[usize], data: &[E]) -> Self::Repr<E> {
        Reverse::Lift(I::new::<E>(shape, data))
    }

    fn shape<E: Clone>(t: &Self::Repr<E>) -> &[usize] {
        I::shape(t.primal())
    }

    fn cast<EFro: Elem, ETo: CastFrom<EFro> + Elem>(t: &Self::Repr<EFro>) -> Self::Repr<ETo> {
        Reverse::Lift(I::cast(t.primal()))
    }
}

// somewhat wonky helper type to deal with optional adjoints
#[derive(Debug)]
struct Adjoints<T> {
    adjoints: Vec<Option<T>>,
}

impl<T: Clone> Adjoints<T> {
    fn new(len: usize) -> Self {
        Self {
            adjoints: vec![None; len],
        }
    }
    fn update<E: Num, I: DiffableOps<Repr<E> = T>>(&mut self, idx: usize, df: T) {
        self.adjoints[idx] = self.adjoints[idx]
            .as_ref()
            .map(|c| I::elementwise_add::<E>(c, &df))
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
pub struct PullBack<T, E: Clone, I: DiffableOps<Repr<E> = T>> {
    trace: Rc<Trace<T>>,
    index_result: Option<usize>,
    zero_primals: Vec<T>,
    // only used to assert the shape matches with cotangent
    primal_out_shape: Vec<usize>,
    phantom: PhantomData<(E, I)>,
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> PullBack<T, E, I> {
    fn reverse(&self, var: usize, adjoint: &T) -> Vec<T> {
        assert!(
            self.primal_out_shape == I::shape::<E>(adjoint),
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
                    TracedOp::Unary(op, a) => {
                        adjoints_acc.update::<E, I>(*a, op.dfda(&adjoints_acc[i]));
                    }
                    TracedOp::Binary(op, a, b) => {
                        adjoints_acc.update::<E, I>(*a, op.dfda(&adjoints_acc[i]));
                        adjoints_acc.update::<E, I>(*b, op.dfdb(&adjoints_acc[i]));
                    }
                    TracedOp::BinaryDA(op, a) => {
                        adjoints_acc.update::<E, I>(*a, op.dfda(&adjoints_acc[i]));
                    }
                    TracedOp::BinaryDB(op, b) => {
                        adjoints_acc.update::<E, I>(*b, op.dfdb(&adjoints_acc[i]));
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
    pub fn call(&self, cotangent: &Tensor<T, E, I>) -> Vec<Tensor<T, E, I>> {
        let ts = match self.index_result {
            None => self.zero_primals.clone(),
            Some(var) => self.reverse(var, &cotangent.0),
        };
        ts.into_iter().map(|t| Tensor(t, PhantomData)).collect()
    }
}

// pub fn vjp1<'b, 't, T: Diffable + Clone + 't, F>(f: F, at: &T) -> (T, PullBack<'t, T>)
// where
//     for<'a> F: Fn(&'a Reverse< T>) -> Reverse< T>,
// {
//     let trace = Trace::new();
//     let reverse = Reverse::Reverse(&trace, at.clone(), trace.var());
//     let result = f(&reverse);

//     let index_result = result.try_get_adjoint_index();
//     let zero_primals: Vec<_> = vec![at.zeros_like()];
//     let primal_out_shape = result.shape().to_vec();
//     (
//         result.into_primal(),
//         PullBack {
//             trace,
//             index_result,
//             zero_primals,
//             primal_out_shape,
//         },
//     )
// }

/// Compute a reverse-mode vector-Jacobian product of a function `f` evaluated at the given primals.
/// Returns a tuple of the result of `f` and a `PullBack` object. `PullBack.call` can be used to
/// compute the vector-Jacobian product of `f` at any cotangent.
pub fn vjpn<T: Clone, E: Num, I: 'static + DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &[&Tensor<T, E, I>],
) -> (Tensor<T, E, I>, PullBack<T, E, I>)
where
    for<'a> F:
        Fn(&'a [Tensor<Reverse<T>, E, ReverseImpl<I>>]) -> Tensor<Reverse<T>, E, ReverseImpl<I>>,
{
    let trace = Rc::new(Trace::new());
    let vars: Vec<_> = at
        .iter()
        .map(|&ati| {
            let index = trace.var();
            Tensor(
                Reverse::Reverse(Rc::clone(&trace), ati.0.clone(), index),
                PhantomData,
            )
        })
        .collect();
    let result = f(&vars);

    let index_result = result.0.try_get_adjoint_index();
    let zero_primals: Vec<_> = at.iter().map(|&ati| ati.zeros_like().0).collect();
    let primal_out_shape = result.shape().to_vec();
    (
        Tensor(result.0.into_primal(), PhantomData),
        PullBack {
            trace: Rc::clone(&trace),
            index_result,
            zero_primals,
            primal_out_shape,
            phantom: PhantomData,
        },
    )
}

/// Compute the result and the gradient of a function at the given primals.
#[allow(clippy::type_complexity)]
pub fn value_and_gradn<T: Clone, E: Num, I: 'static + DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &[&Tensor<T, E, I>],
) -> (Tensor<T, E, I>, Vec<Tensor<T, E, I>>)
where
    for<'a> F:
        Fn(&'a [Tensor<Reverse<T>, E, ReverseImpl<I>>]) -> Tensor<Reverse<T>, E, ReverseImpl<I>>,
{
    let (primal, pullback) = vjpn(f, at);
    let tangents = pullback.call(&primal.ones_like());
    (primal, tangents)
}

/// Compute the result and the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn value_and_grad1<T: Clone, E: Num, I: 'static + DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &Tensor<T, E, I>,
) -> (Tensor<T, E, I>, Tensor<T, E, I>)
where
    for<'a> F:
        Fn(&'a Tensor<Reverse<T>, E, ReverseImpl<I>>) -> Tensor<Reverse<T>, E, ReverseImpl<I>>,
{
    let (primal, tangents) = value_and_gradn(|s| f(&s[0]), &[at]);
    (primal, tangents.into_iter().next().unwrap())
}

/// Compute the result and the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc, clippy::type_complexity)]
pub fn value_and_grad2<T: Clone, E: Num, I: 'static + DiffableOps<Repr<E> = T>, F>(
    f: F,
    at0: &Tensor<T, E, I>,
    at1: &Tensor<T, E, I>,
) -> (Tensor<T, E, I>, (Tensor<T, E, I>, Tensor<T, E, I>))
where
    for<'a> F: Fn(
        &'a Tensor<Reverse<T>, E, ReverseImpl<I>>,
        &'a Tensor<Reverse<T>, E, ReverseImpl<I>>,
    ) -> Tensor<Reverse<T>, E, ReverseImpl<I>>,
{
    let (primal, tangents) = value_and_gradn(|s| f(&s[0], &s[1]), &[at0, at1]);
    let mut dr_iter = tangents.into_iter();
    (primal, (dr_iter.next().unwrap(), dr_iter.next().unwrap()))
}

/// Compute the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn grad1<T: Clone, E: Num, I: 'static + DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &Tensor<T, E, I>,
) -> Tensor<T, E, I>
where
    for<'a> F:
        Fn(&'a Tensor<Reverse<T>, E, ReverseImpl<I>>) -> Tensor<Reverse<T>, E, ReverseImpl<I>>,
{
    value_and_grad1(f, at).1
}

/// Compute the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc)]
pub fn grad2<T: Clone, E: Num, I: 'static + DiffableOps<Repr<E> = T>, F>(
    f: F,
    at0: &Tensor<T, E, I>,
    at1: &Tensor<T, E, I>,
) -> (Tensor<T, E, I>, Tensor<T, E, I>)
where
    for<'a> F: Fn(
        &'a Tensor<Reverse<T>, E, ReverseImpl<I>>,
        &'a Tensor<Reverse<T>, E, ReverseImpl<I>>,
    ) -> Tensor<Reverse<T>, E, ReverseImpl<I>>,
{
    value_and_grad2(f, at0, at1).1
}

/// Jacobian of `f` evaluated row-by-row at `at` using reverse-mode AD.
#[allow(clippy::missing_panics_doc)]
pub fn jacrev<T: Clone, E: Num, I: 'static + DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &Tensor<T, E, I>,
) -> Tensor<T, E, I>
where
    for<'a> F:
        Fn(&'a Tensor<Reverse<T>, E, ReverseImpl<I>>) -> Tensor<Reverse<T>, E, ReverseImpl<I>>,
{
    let (primal, pullback) = vjpn(|s| f(&s[0]), &[at]);

    let mut s = vec![primal.shape().size()];
    s.extend(primal.shape());
    let i = Tensor::<T, E, I>::eye(primal.shape().size()).reshape(&s);

    let mut tangents: Vec<Tensor<T, E, I>> = Vec::with_capacity(i.shape()[0]);
    for row_idx in 0..i.shape()[0] {
        let row = i.at1(row_idx);
        let row_tangent = pullback.call(&row).into_iter().next().unwrap();
        tangents.push(row_tangent);
    }
    let t_refs: Vec<_> = tangents.iter().collect();
    Tensor::stack(&t_refs[..], 0)
}
