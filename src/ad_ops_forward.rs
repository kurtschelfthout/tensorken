use std::marker::PhantomData;

use crate::{
    ad_ops::{UnaryDiffOp, UnaryOp},
    num::{Elem, Num, ZeroOne},
    Diffable,
};

pub(crate) struct SumOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: Diffable<Repr<E> = T>> UnaryOp<T> for SumOp<E, I> {
    type Args = [usize];
    fn f(a: &T, axes: &Self::Args) -> (T, Self) {
        let r = I::sum::<E>(a, axes);
        (r, SumOp(axes.to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Num, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for SumOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::sum::<E>(d, &self.0)
    }
}

pub(crate) struct MaxOp<T, E, I>(T, T, PhantomData<(T, E, I)>);

impl<T: Clone, E: Num + From<bool>, I: Diffable<Repr<E> = T>> UnaryOp<T> for MaxOp<T, E, I> {
    type Args = [usize];
    fn f(a: &T, axes: &Self::Args) -> (T, Self) {
        let r = I::max::<E>(a, axes);
        (r.clone(), MaxOp(a.clone(), r, PhantomData))
    }
}

impl<T: Clone, E: Num + From<bool>, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for MaxOp<T, E, I> {
    fn dfda(&self, d: &T) -> T {
        let max_is_1s = I::eq::<E>(&self.0, &I::expand::<E>(&self.1, I::shape::<E>(&self.0)));
        I::elementwise_mul::<E>(&I::cast::<bool, E>(&max_is_1s), d)
    }
}

pub(crate) struct ExpandOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: Diffable<Repr<E> = T>> UnaryOp<T> for ExpandOp<E, I> {
    type Args = [usize];
    fn f(a: &T, new_shape: &Self::Args) -> (T, Self) {
        let r = I::expand::<E>(a, new_shape);
        (r, ExpandOp(new_shape.to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Num, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for ExpandOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::expand::<E>(d, &self.0)
    }
}

pub(crate) struct ReshapeOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Elem, I: Diffable<Repr<E> = T>> UnaryOp<T> for ReshapeOp<E, I> {
    type Args = [usize];
    fn f(a: &T, new_shape: &Self::Args) -> (T, Self) {
        let r = I::reshape::<E>(a, new_shape);
        (r, Self(new_shape.to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Elem, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for ReshapeOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::reshape::<E>(d, &self.0)
    }
}

pub(crate) struct PermuteOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Elem, I: Diffable<Repr<E> = T>> UnaryOp<T> for PermuteOp<E, I> {
    type Args = [usize];
    fn f(a: &T, order: &Self::Args) -> (T, Self) {
        (
            I::permute::<E>(a, order),
            PermuteOp(order.to_vec(), PhantomData),
        )
    }
}

impl<T: Clone, E: Elem, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for PermuteOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::permute::<E>(d, &self.0)
    }
}

pub(crate) struct PadOp<E, I>(Vec<(usize, usize)>, PhantomData<(E, I)>);

impl<T: Clone, E: ZeroOne, I: Diffable<Repr<E> = T>> UnaryOp<T> for PadOp<E, I> {
    type Args = [(usize, usize)];
    fn f(a: &T, padding: &Self::Args) -> (T, Self) {
        let r = I::pad::<E>(a, padding);
        (r.clone(), PadOp(padding.to_vec(), PhantomData))
    }
}

impl<T: Clone, E: ZeroOne, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for PadOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::pad::<E>(d, &self.0)
    }
}

pub(crate) struct CropOp<E, I>(Vec<(usize, usize)>, PhantomData<(E, I)>);

impl<T: Clone, E: ZeroOne, I: Diffable<Repr<E> = T>> UnaryOp<T> for CropOp<E, I> {
    type Args = [(usize, usize)];
    fn f(a: &T, limits: &Self::Args) -> (T, Self) {
        let r = I::crop::<E>(a, limits);
        (r.clone(), CropOp(limits.to_vec(), PhantomData))
    }
}

impl<T: Clone, E: ZeroOne, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for CropOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::crop::<E>(d, &self.0)
    }
}
