use std::marker::PhantomData;

use crate::{
    num::{Bool, Float, Num},
    DiffableOps,
};

/// A trait that represents the operation on the primal value, and
/// returns a `UnaryDiffOp`, which is the operation on the tangent values.
/// This design allows the derivative calculation to reuse results or inputs from the primal calculation.
pub trait UnaryOp<T> {
    type Args: ?Sized;
    fn f(a: &T, args: &Self::Args) -> (T, Self);
}

/// Same as `UnaryOp`, but for binary operations.
pub trait BinaryOp<T> {
    fn f(a: &T, b: &T) -> (T, Self);
}

/// Propagate the derivative of a unary operation.
pub trait UnaryDiffOp<T> {
    fn dfda(&self, d: &T) -> T;
}

/// Propagate the derivative of a binary operation.
pub trait BinaryDiffOp<T> {
    fn dfda(&self, d: &T) -> T;
    fn dfdb(&self, d: &T) -> T;
}

// The rest of this file are implementations of the above traits for element-wise operations.
// They are the same for forward and reverse mode, and so we can share them.
// Forward-mode specific ops are in ad_ops_forward, and reverse-mode specific ops are in ad_ops_reverse.

pub(crate) struct AddOp<E, I>(PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> BinaryOp<T> for AddOp<E, I> {
    fn f(a: &T, b: &T) -> (T, Self) {
        (I::elementwise_add::<E>(a, b), AddOp(PhantomData))
    }
}

impl<T: Clone, E: Clone, I: DiffableOps<Repr<E> = T>> BinaryDiffOp<T> for AddOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        d.clone()
    }

    fn dfdb(&self, d: &T) -> T {
        d.clone()
    }
}

pub(crate) struct MulOp<T, E, I>(T, T, PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> BinaryOp<T> for MulOp<T, E, I> {
    fn f(a: &T, b: &T) -> (T, Self) {
        (
            I::elementwise_mul::<E>(a, b),
            Self(a.clone(), b.clone(), PhantomData),
        )
    }
}

impl<T, E: Num, I: DiffableOps<Repr<E> = T>> BinaryDiffOp<T> for MulOp<T, E, I> {
    fn dfda(&self, d: &T) -> T {
        I::elementwise_mul::<E>(d, &self.1)
    }

    fn dfdb(&self, d: &T) -> T {
        I::elementwise_mul::<E>(d, &self.0)
    }
}

pub(crate) struct SubOp<E, I>(PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> BinaryOp<T> for SubOp<E, I> {
    fn f(a: &T, b: &T) -> (T, Self) {
        (I::elementwise_sub::<E>(a, b), SubOp(PhantomData))
    }
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> BinaryDiffOp<T> for SubOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        d.clone()
    }

    fn dfdb(&self, d: &T) -> T {
        I::neg::<E>(d)
    }
}

pub(crate) struct DivOp<T, E, I>(T, T, PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> BinaryOp<T> for DivOp<T, E, I> {
    fn f(a: &T, b: &T) -> (T, Self) {
        (
            I::elementwise_div::<E>(a, b),
            Self(a.clone(), b.clone(), PhantomData),
        )
    }
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> BinaryDiffOp<T> for DivOp<T, E, I> {
    fn dfda(&self, d: &T) -> T {
        I::elementwise_div::<E>(d, &self.1)
    }

    fn dfdb(&self, d: &T) -> T {
        let b_squared = I::elementwise_mul::<E>(&self.1, &self.1);
        let mul_div = I::elementwise_mul::<E>(d, &self.0);
        let mul_div = I::elementwise_div::<E>(&mul_div, &b_squared);
        I::neg::<E>(&mul_div)
    }
}

pub(crate) struct PowOp<T, E, I>(T, T, T, PhantomData<(E, I)>);

impl<T: Clone, E: Float, I: DiffableOps<Repr<E> = T>> BinaryOp<T> for PowOp<T, E, I> {
    fn f(a: &T, b: &T) -> (T, Self) {
        let r = I::elementwise_pow::<E>(a, b);
        (r.clone(), Self(a.clone(), b.clone(), r, PhantomData))
    }
}

impl<T: Clone, E: 'static + Float, I: DiffableOps<Repr<E> = T>> BinaryDiffOp<T> for PowOp<T, E, I> {
    fn dfda(&self, d: &T) -> T {
        I::elementwise_mul::<E>(
            d,
            &I::elementwise_mul::<E>(&self.1, &I::elementwise_div::<E>(&self.2, &self.0)),
        )
    }

    fn dfdb(&self, d: &T) -> T {
        I::elementwise_mul::<E>(d, &I::elementwise_mul::<E>(&I::log::<E>(&self.0), &self.2))
    }
}

pub(crate) struct LogOp<T, E, I>(T, PhantomData<(E, I)>);

impl<T: Clone, E: Float, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for LogOp<T, E, I> {
    type Args = ();
    fn f(a: &T, (): &Self::Args) -> (T, Self) {
        (I::log::<E>(a), LogOp(a.clone(), PhantomData))
    }
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for LogOp<T, E, I> {
    fn dfda(&self, d: &T) -> T {
        I::elementwise_div::<E>(d, &self.0)
    }
}

pub(crate) struct ExpOp<T, E, I>(T, PhantomData<(E, I)>);

impl<T: Clone, E: 'static + Float, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for ExpOp<T, E, I> {
    type Args = ();
    fn f(a: &T, (): &Self::Args) -> (T, Self) {
        let r = I::exp::<E>(a);
        (r.clone(), ExpOp(r, PhantomData))
    }
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for ExpOp<T, E, I> {
    fn dfda(&self, d: &T) -> T {
        I::elementwise_mul::<E>(d, &self.0)
    }
}

pub(crate) struct FlipOp<E, I>(Vec<bool>, PhantomData<(E, I)>);

impl<T: Clone, E: Bool, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for FlipOp<E, I> {
    type Args = [bool];
    fn f(a: &T, flips: &Self::Args) -> (T, Self) {
        let r = I::flip::<E>(a, flips);
        (r.clone(), FlipOp(flips.to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Bool, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for FlipOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::flip::<E>(d, &self.0)
    }
}
