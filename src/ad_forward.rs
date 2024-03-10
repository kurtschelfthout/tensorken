use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    ad_ops::{
        AddOp, BinaryDiffOp, BinaryOp, DivOp, ExpOp, LogOp, MulOp, PowOp, SubOp, UnaryDiffOp,
        UnaryOp,
    },
    ad_ops_forward::{CropOp, ExpandOp, MaxOp, PadOp, PermuteOp, ReshapeOp, SumOp},
    num::{Float, Num, ZeroOne},
    raw_tensor::CastInto,
    sl2, Axes, Diffable, DiffableExt, IndexValue, Shape,
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

impl<T: Clone> Forward<T> {
    pub fn lift(x: &T) -> Self {
        Self::Lift(x.clone())
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

impl<T: Diffable> Forward<T> {
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

    fn binary<Op: BinaryOp<T> + BinaryDiffOp<T>>(&self, rhs: &Self) -> Self
    where
        T::Elem: Num,
    {
        let (primal, op) = Op::f(self.primal(), rhs.primal());
        match (self, rhs) {
            (Self::Lift(_), Self::Lift(_)) => Self::Lift(primal),
            (Self::Lift(_), Self::Forward(_, tan)) => Self::Forward(primal, op.dfdb(tan)),
            (Self::Forward(_, tan), Self::Lift(_)) => Self::Forward(primal, op.dfda(tan)),
            (Self::Forward(_, left), Self::Forward(_, right)) => {
                Self::Forward(primal, op.dfda(left).elementwise_add(&op.dfdb(right)))
            }
        }
    }
}

impl<T: Clone + Diffable> Diffable for Forward<T>
where
    T::Elem: Num,
{
    type Elem = T::Elem;
    fn log(&self) -> Self
    where
        T::Elem: Float,
    {
        self.unary::<LogOp<T>, _>(&())
    }

    fn exp(&self) -> Self
    where
        T::Elem: Float,
    {
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

    fn elementwise_pow(&self, rhs: &Self) -> Self
    where
        T::Elem: Float,
    {
        self.binary::<PowOp<T>>(rhs)
    }

    fn elementwise_eq(&self, other: &Self) -> Self {
        Self::Lift(self.primal().elementwise_eq(other.primal()))
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
        Self::Lift(T::new(shape, data))
    }
}

// not derivable for the moment
impl<TFro: Diffable + CastInto<TTo>, TTo: Diffable> CastInto<Forward<TTo>> for Forward<TFro> {
    fn cast(&self) -> Forward<TTo> {
        Forward::Lift(self.primal().cast())
    }
}

crate::math_macros::impl_bin_op!(Add, add, Forward<T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Sub, sub, Forward<T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Mul, mul, Forward<T: Diffable + Clone>);
crate::math_macros::impl_bin_op!(Div, div, Forward<T: Diffable + Clone>);

crate::math_macros::impl_un_op!(Neg, neg, Forward<T: Diffable + Clone>);

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
pub fn jvpn<T: Diffable + Clone, F>(f: F, at: &[&T], tangents: &[&T]) -> (T, T)
where
    T::Elem: ZeroOne,
    for<'a> F: Fn(&'a [Forward<T>]) -> Forward<T>,
{
    let vars: Vec<_> = at
        .iter()
        .zip(tangents.iter())
        .map(|(&ati, &tani)| Forward::Forward(ati.clone(), tani.clone()))
        .collect();
    let result = f(&vars);

    match result {
        Forward::Lift(p) => (p.clone(), p.zeros_like()),
        Forward::Forward(p, t) => (p, t),
    }
}

// pub fn value_and_diff1_simple<T: Diffable + Clone, F>(f: F, at: &T) -> (T, T)
// where
//     for<'a> F: Fn(&'a Forward<T>) -> Forward<T>,
// {
//     jvp1(&f, at, &at.ones_like())
// }

/// Compute the result and the gradient of a function at the given primals.
pub fn value_and_diffn<T: Diffable + Clone, F>(f: F, at: &[&T]) -> (T, Vec<T>)
where
    T::Elem: ZeroOne,
    for<'a> F: Fn(&'a [Forward<T>]) -> Forward<T>,
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
pub fn value_and_diff1<T: Diffable + Clone, F>(f: F, at: &T) -> (T, T)
where
    T::Elem: ZeroOne,
    for<'a> F: Fn(&'a Forward<T>) -> Forward<T>,
{
    let (primal, tangents) = value_and_diffn(|s| f(&s[0]), &[at]);
    (primal, tangents.into_iter().next().unwrap())
}

/// Compute the result and the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc)]
pub fn value_and_diff2<T: Diffable + Clone, F>(f: F, at0: &T, at1: &T) -> (T, (T, T))
where
    T::Elem: ZeroOne,
    for<'a> F: Fn(&'a Forward<T>, &'a Forward<T>) -> Forward<T>,
{
    let (primal, tangents) = value_and_diffn(|s| f(&s[0], &s[1]), &[at0, at1]);
    let mut dr_iter = tangents.into_iter();
    (primal, (dr_iter.next().unwrap(), dr_iter.next().unwrap()))
}

/// Compute the gradient of a function at the given primal.
#[allow(clippy::missing_panics_doc)]
pub fn diff1<T: Diffable + Clone, F>(f: F, at: &T) -> T
where
    T::Elem: ZeroOne,
    for<'a> F: Fn(&'a Forward<T>) -> Forward<T>,
{
    value_and_diff1(f, at).1
}

/// Compute the gradient of a function at the given primals.
#[allow(clippy::missing_panics_doc)]
pub fn diff2<T: Diffable + Clone, F>(f: F, at0: &T, at1: &T) -> (T, T)
where
    T::Elem: ZeroOne,
    for<'a> F: Fn(&'a Forward<T>, &'a Forward<T>) -> Forward<T>,
{
    value_and_diff2(f, at0, at1).1
}

/// Jacobian of `f` evaluated column-by-column at `at` using forward-mode AD.
#[allow(clippy::missing_panics_doc)]
pub fn jacfwd<T: Diffable + Clone, F>(f: F, at: &T) -> T
where
    T::Elem: Num,
    for<'a> F: Fn(&'a Forward<T>) -> Forward<T>,
{
    let mut s = vec![at.shape().size()];
    s.extend(at.shape());
    let i = T::eye(at.shape().size()).reshape(&s);

    let mut tangents: Vec<T> = Vec::with_capacity(i.shape()[1]);
    for col_idx in 0..i.shape()[1] {
        let col = i.at(sl2(.., col_idx)).squeeze(Axes::Axis(1));
        let (_, col_tangent) = jvpn(|s| f(&s[0]), &[at], &[&col]);
        tangents.push(col_tangent);
    }
    let t_refs: Vec<_> = tangents.iter().collect();
    T::stack(&t_refs, 1)
}
