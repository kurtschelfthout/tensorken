use std::{fmt::Debug, marker::PhantomData};

use crate::{
    ad_ops::{
        AddOp, BinaryDiffOp, BinaryOp, DivOp, ExpOp, FlipOp, LogOp, MulOp, PowOp, SubOp,
        UnaryDiffOp, UnaryOp,
    },
    ad_ops_forward::{CropOp, ExpandOp, MaxOp, PadOp, PermuteOp, ReshapeOp, SumOp},
    num::{Bool, CastFrom, Elem, Float, Num},
    DiffableOps, Shape, Tensor,
};

/// Forward AD implementation.

#[derive(Clone)] //needed for higher order derivatives
pub enum Forward<T> {
    Lift(T),
    Forward(T, T),
}

impl<T> Forward<T> {
    pub fn primal(&self) -> &T {
        match self {
            Self::Lift(x) | Self::Forward(x, _) => x,
        }
    }
}

impl<T: Debug> Debug for Forward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lift(x) => write!(f, "Lift({x:?})"),
            Self::Forward(p, x) => write!(f, "Forward({p:?}, {x:?})"),
        }
    }
}

impl<T: PartialEq> PartialEq for Forward<T> {
    fn eq(&self, other: &Self) -> bool {
        self.primal() == other.primal()
    }
}

impl<T> Forward<T> {
    fn unary<Op: UnaryOp<T, Args = TArgs> + UnaryDiffOp<T>, TArgs: ?Sized>(
        &self,
        args: &TArgs,
    ) -> Self {
        let (primal, op) = Op::f(self.primal(), args);
        match self {
            Self::Lift(_) => Self::Lift(primal),
            Self::Forward(_, tan) => Self::Forward(primal, op.dfda(tan)),
        }
    }

    fn binary<Op: BinaryOp<T> + BinaryDiffOp<T>, E: Num, I: DiffableOps<Repr<E> = T>>(
        &self,
        rhs: &Self,
    ) -> Self {
        let (primal, op) = Op::f(self.primal(), rhs.primal());
        match (self, rhs) {
            (Self::Lift(_), Self::Lift(_)) => Self::Lift(primal),
            (Self::Lift(_), Self::Forward(_, tan)) => Self::Forward(primal, op.dfdb(tan)),
            (Self::Forward(_, tan), Self::Lift(_)) => Self::Forward(primal, op.dfda(tan)),
            (Self::Forward(_, left), Self::Forward(_, right)) => Self::Forward(
                primal,
                I::elementwise_add::<E>(&op.dfda(left), &op.dfdb(right)),
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForwardImpl<I>(PhantomData<I>);

impl<I: DiffableOps> DiffableOps for ForwardImpl<I> {
    type Repr<E: Clone> = Forward<I::Repr<E>>;

    fn log<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.unary::<LogOp<I::Repr<E>, E, I>, _>(&())
    }

    fn exp<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.unary::<ExpOp<I::Repr<E>, E, I>, _>(&())
    }

    fn elementwise_add<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<AddOp<E, I>, E, I>(rhs)
    }

    fn elementwise_sub<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<SubOp<E, I>, E, I>(rhs)
    }

    fn elementwise_mul<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<MulOp<I::Repr<E>, E, I>, E, I>(rhs)
    }

    fn elementwise_div<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<DivOp<I::Repr<E>, E, I>, E, I>(rhs)
    }

    fn elementwise_pow<E: Float>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.binary::<PowOp<I::Repr<E>, E, I>, E, I>(rhs)
    }

    fn elementwise_eq<E: PartialEq + Elem>(
        lhs: &Self::Repr<E>,
        rhs: &Self::Repr<E>,
    ) -> Self::Repr<bool> {
        Forward::Lift(I::elementwise_eq::<E>(lhs.primal(), rhs.primal()))
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

    fn flip<E: Bool>(t: &Self::Repr<E>, flips: &[bool]) -> Self::Repr<E> {
        t.unary::<FlipOp<E, I>, _>(flips)
    }

    fn new<E: Elem>(shape: &[usize], data: &[E]) -> Self::Repr<E> {
        Forward::Lift(I::new::<E>(shape, data))
    }

    fn shape<E: Clone>(t: &Self::Repr<E>) -> &[usize] {
        I::shape(t.primal())
    }

    fn cast<EFro: Elem, ETo: CastFrom<EFro> + Elem>(t: &Self::Repr<EFro>) -> Self::Repr<ETo> {
        Forward::Lift(I::cast(t.primal()))
    }
}

// /// Compute a forward-mode Jacobian-vector product of a function `f` evaluated at the given primals.
// /// Returns a tuple of the result of `f` and the tangent of `f`.
// pub fn jvp1<T: Diffable + Clone, F>(f: F, at: &T, tangent: &T) -> (T, T)
// where
//     for<'a> F: Fn(&'a Forward<T>) -> Forward<T>,
// {
//     let forward = Forward::Forward(at.clone(), tangent.clone());
//     let result = f(&forward);

//     match result {
//         Forward::Lift(p) => (p.clone(), p.zeros_like()),
//         Forward::Forward(p, t) => (p, t),
//     }
// }

/// Compute a forward-mode Jacobian-vector product of a function `f` evaluated at the given primals.
/// Returns a tuple of the result of `f` and the tangent of `f`.
pub fn jvpn<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &[&Tensor<T, E, I>],
    tangents: &[&Tensor<T, E, I>],
) -> (Tensor<T, E, I>, Tensor<T, E, I>)
where
    for<'a> F:
        Fn(&'a [Tensor<Forward<T>, E, ForwardImpl<I>>]) -> Tensor<Forward<T>, E, ForwardImpl<I>>,
{
    let vars: Vec<_> = at
        .iter()
        .zip(tangents.iter())
        .map(|(&ati, &tani)| {
            Tensor(
                Forward::Forward(ati.0.clone(), tani.0.clone()),
                PhantomData::<(E, ForwardImpl<I>)>,
            )
        })
        .collect();
    let result = f(&vars);

    match result.0 {
        Forward::Lift(p) => (
            Tensor(p.clone(), PhantomData),
            Tensor(I::zeros_like::<E>(&p), PhantomData),
        ),
        Forward::Forward(p, t) => (Tensor(p, PhantomData), Tensor(t, PhantomData)),
    }
}

// pub fn value_and_diff1_simple<T: Diffable + Clone, F>(f: F, at: &T) -> (T, T)
// where
//     for<'a> F: Fn(&'a Forward<T>) -> Forward<T>,
// {
//     jvp1(&f, at, &at.ones_like())
// }

/// Compute the result and the gradient of a function at the given primals.
#[allow(clippy::type_complexity)]
pub fn value_and_diffn<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &[&Tensor<T, E, I>],
) -> (Tensor<T, E, I>, Vec<Tensor<T, E, I>>)
where
    for<'a> F:
        Fn(&'a [Tensor<Forward<T>, E, ForwardImpl<I>>]) -> Tensor<Forward<T>, E, ForwardImpl<I>>,
{
    let args: Vec<_> = at.iter().map(|&ati| ati.zeros_like()).collect();

    let (primal, _) = jvpn(&f, at, &args.iter().collect::<Vec<_>>());
    let mut tangents = Vec::with_capacity(at.len());

    for (i, tangent) in at.iter().enumerate() {
        let mut args: Vec<_> = args.iter().clone().collect();
        let one = tangent.ones_like();
        args[i] = &one;
        let (_, tangent) = jvpn(&f, at, &args);
        tangents.push(tangent);
    }

    (primal, tangents)
}

/// Compute the result and the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn value_and_diff1<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &Tensor<T, E, I>,
) -> (Tensor<T, E, I>, Tensor<T, E, I>)
where
    for<'a> F:
        Fn(&'a Tensor<Forward<T>, E, ForwardImpl<I>>) -> Tensor<Forward<T>, E, ForwardImpl<I>>,
{
    let (primal, tangents) = value_and_diffn(|s| f(&s[0]), &[at]);
    (primal, tangents.into_iter().next().unwrap())
}

/// Compute the result and the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc, clippy::type_complexity)]
pub fn value_and_diff2<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, F>(
    f: F,
    at0: &Tensor<T, E, I>,
    at1: &Tensor<T, E, I>,
) -> (Tensor<T, E, I>, (Tensor<T, E, I>, Tensor<T, E, I>))
where
    for<'a> F: Fn(
        &'a Tensor<Forward<T>, E, ForwardImpl<I>>,
        &'a Tensor<Forward<T>, E, ForwardImpl<I>>,
    ) -> Tensor<Forward<T>, E, ForwardImpl<I>>,
{
    let (primal, tangents) = value_and_diffn(|s| f(&s[0], &s[1]), &[at0, at1]);
    let mut dr_iter = tangents.into_iter();
    (primal, (dr_iter.next().unwrap(), dr_iter.next().unwrap()))
}

/// Compute the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn diff1<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &Tensor<T, E, I>,
) -> Tensor<T, E, I>
where
    for<'a> F:
        Fn(&'a Tensor<Forward<T>, E, ForwardImpl<I>>) -> Tensor<Forward<T>, E, ForwardImpl<I>>,
{
    value_and_diff1(f, at).1
}

/// Compute the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc)]
pub fn diff2<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, F>(
    f: F,
    at0: &Tensor<T, E, I>,
    at1: &Tensor<T, E, I>,
) -> (Tensor<T, E, I>, Tensor<T, E, I>)
where
    for<'a> F: Fn(
        &'a Tensor<Forward<T>, E, ForwardImpl<I>>,
        &'a Tensor<Forward<T>, E, ForwardImpl<I>>,
    ) -> Tensor<Forward<T>, E, ForwardImpl<I>>,
{
    value_and_diff2(f, at0, at1).1
}

/// Jacobian of `f` evaluated column-by-column at `at` using forward-mode AD.
#[allow(clippy::missing_panics_doc)]
pub fn jacfwd<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, F>(
    f: F,
    at: &Tensor<T, E, I>,
) -> Tensor<T, E, I>
where
    for<'a> F:
        Fn(&'a Tensor<Forward<T>, E, ForwardImpl<I>>) -> Tensor<Forward<T>, E, ForwardImpl<I>>,
{
    let mut s = vec![at.shape().size()];
    s.extend(at.shape());
    let i = Tensor::eye(at.shape().size()).reshape(&s);

    let mut tangents: Vec<_> = Vec::with_capacity(i.shape()[1]);
    for col_idx in 0..i.shape()[1] {
        let col = i.at2(.., col_idx);
        let (_, col_tangent) = jvpn(|s| f(&s[0]), &[at], &[&col]);
        tangents.push(col_tangent);
    }
    let t_refs: Vec<_> = tangents.iter().collect();
    Tensor::stack(&t_refs, 1)
}
