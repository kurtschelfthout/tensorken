use std::marker::PhantomData;

use crate::{
    ad_ops::{UnaryDiffOp, UnaryOp},
    num::{Bool, CastFrom, Elem, Num},
    Diffable,
};

pub(crate) struct SumOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: Diffable<Repr<E> = T>> UnaryOp<T> for SumOp<E, I> {
    type Args = [usize];
    fn f(a: &T, axes: &Self::Args) -> (T, Self) {
        let r = I::sum::<E>(a, axes);
        (r, SumOp(I::shape::<E>(a).to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Num, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for SumOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::expand::<E>(d, &self.0)
    }
}

pub(crate) struct MaxOp<T, E, I>(T, T, PhantomData<(E, I)>);

impl<T: Clone, E: Num + CastFrom<bool>, I: Diffable<Repr<E> = T>> UnaryOp<T> for MaxOp<T, E, I> {
    type Args = [usize];
    fn f(a: &T, axes: &Self::Args) -> (T, Self) {
        let r = I::max::<E>(a, axes);
        (r.clone(), MaxOp(a.clone(), r, PhantomData))
    }
}

fn shape_to_axes(old_shape: &[usize], new_shape: &[usize]) -> Vec<usize> {
    assert!(
        old_shape.len() == new_shape.len(),
        "shape_to_axes: old_shape.len() != new_shape.len()"
    );
    old_shape
        .iter()
        .zip(new_shape.iter())
        .enumerate()
        .filter_map(|(i, (a, b))| if a == b { None } else { Some(i) })
        .collect()
}

impl<T: Clone, E: Num + CastFrom<bool>, I: Diffable<Repr<E> = T>> UnaryDiffOp<T>
    for MaxOp<T, E, I>
{
    fn dfda(&self, d: &T) -> T {
        let a_shape = I::shape::<E>(&self.0);
        let res_expanded = I::expand::<E>(&self.1, a_shape);
        let max_is_1s = I::eq::<E>(&self.0, &res_expanded);
        let max_is_1s = I::cast::<bool, E>(&max_is_1s);
        let div = I::sum::<E>(
            &max_is_1s,
            &shape_to_axes(I::shape::<E>(&max_is_1s), I::shape::<E>(d)),
        );
        let div = I::expand::<E>(&div, a_shape);
        let max_is_amount = I::elementwise_div::<E>(&max_is_1s, &div);
        let df_expanded = I::expand::<E>(d, a_shape);

        I::elementwise_mul::<E>(&max_is_amount, &df_expanded)
    }
}

pub(crate) struct ExpandOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: Diffable<Repr<E> = T>> UnaryOp<T> for ExpandOp<E, I> {
    type Args = [usize];
    fn f(a: &T, new_shape: &Self::Args) -> (T, Self) {
        let r = I::expand::<E>(a, new_shape);
        (r, ExpandOp(I::shape::<E>(a).to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Num, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for ExpandOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::sum::<E>(d, &shape_to_axes(I::shape::<E>(d), &self.0))
    }
}

pub(crate) struct ReshapeOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Elem, I: Diffable<Repr<E> = T>> UnaryOp<T> for ReshapeOp<E, I> {
    type Args = [usize];
    fn f(a: &T, new_shape: &Self::Args) -> (T, Self) {
        let r = I::reshape::<E>(a, new_shape);
        (r, Self(I::shape::<E>(a).to_vec(), PhantomData))
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
        (I::permute::<E>(a, order), Self(order.to_vec(), PhantomData))
    }
}

// like numpy argsort: returns the indices that would sort an array.
// Here only used to invert the permutation in the backward pass.
fn argsort(v: &[usize]) -> Vec<usize> {
    let mut v: Vec<_> = v.iter().enumerate().collect();
    v.sort_by_key(|&(_, k)| *k);
    v.into_iter().map(|(i, _)| i).collect()
}

impl<T: Clone, E: Elem, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for PermuteOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::permute::<E>(d, &argsort(&self.0))
    }
}

pub(crate) struct PadOp<E, I>(Vec<(usize, usize)>, PhantomData<(E, I)>);

impl<T: Clone, E: Bool, I: Diffable<Repr<E> = T>> UnaryOp<T> for PadOp<E, I> {
    type Args = [(usize, usize)];
    fn f(a: &T, padding: &Self::Args) -> (T, Self) {
        let r = I::pad::<E>(a, padding);
        let limits = padding
            .iter()
            .zip(I::shape::<E>(a))
            .map(|((pl, _), s)| (*pl, pl + s))
            .collect();
        (r, Self(limits, PhantomData))
    }
}

impl<T: Clone, E: Bool, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for PadOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::crop::<E>(d, &self.0)
    }
}

pub(crate) struct CropOp<E, I>(Vec<(usize, usize)>, PhantomData<(E, I)>);

impl<T: Clone, E: Bool, I: Diffable<Repr<E> = T>> UnaryOp<T> for CropOp<E, I> {
    type Args = [(usize, usize)];
    fn f(a: &T, limits: &Self::Args) -> (T, Self) {
        let r = I::crop::<E>(a, limits);
        let padding = limits
            .iter()
            .zip(I::shape::<E>(a))
            .map(|((l0, l1), s)| (*l0, s - l1))
            .collect();
        (r, Self(padding, PhantomData))
    }
}

impl<T: Clone, E: Bool, I: Diffable<Repr<E> = T>> UnaryDiffOp<T> for CropOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::pad::<E>(d, &self.0)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_argsort() {
        assert_eq!(argsort(&[0, 1]), [0, 1]);
        assert_eq!(argsort(&[1, 0]), [1, 0]);
        assert_eq!(argsort(&[2, 0, 1]), [1, 2, 0]);
        assert_eq!(argsort(&[0, 1, 2]), [0, 1, 2]);
    }
}
