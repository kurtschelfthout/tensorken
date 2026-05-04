use std::marker::PhantomData;

use crate::{
    ad_ops::{BinaryDiffOp, BinaryOp, UnaryDiffOp, UnaryOp},
    conv::CorrelateOpts,
    num::{Bool, CastFrom, Elem, Num},
    DiffableOps,
};

pub(crate) struct SumOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for SumOp<E, I> {
    type Args = [usize];
    fn f(a: &T, axes: &Self::Args) -> (T, Self) {
        let r = I::sum::<E>(a, axes);
        (r, SumOp(I::shape::<E>(a).to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for SumOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::expand::<E>(d, &self.0)
    }
}

pub(crate) struct MaxOp<T, E, I>(T, T, PhantomData<(E, I)>);

impl<T: Clone, E: Num + CastFrom<bool>, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for MaxOp<T, E, I> {
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

impl<T: Clone, E: Num + CastFrom<bool>, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T>
    for MaxOp<T, E, I>
{
    fn dfda(&self, d: &T) -> T {
        let a_shape = I::shape::<E>(&self.0);
        let res_expanded = I::expand::<E>(&self.1, a_shape);
        let max_is_1s = I::elementwise_eq::<E>(&self.0, &res_expanded);
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

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for ExpandOp<E, I> {
    type Args = [usize];
    fn f(a: &T, new_shape: &Self::Args) -> (T, Self) {
        let r = I::expand::<E>(a, new_shape);
        (r, ExpandOp(I::shape::<E>(a).to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for ExpandOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::sum::<E>(d, &shape_to_axes(I::shape::<E>(d), &self.0))
    }
}

pub(crate) struct ReshapeOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Elem, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for ReshapeOp<E, I> {
    type Args = [usize];
    fn f(a: &T, new_shape: &Self::Args) -> (T, Self) {
        let r = I::reshape::<E>(a, new_shape);
        (r, Self(I::shape::<E>(a).to_vec(), PhantomData))
    }
}

impl<T: Clone, E: Elem, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for ReshapeOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::reshape::<E>(d, &self.0)
    }
}

pub(crate) struct PermuteOp<E, I>(Vec<usize>, PhantomData<(E, I)>);

impl<T: Clone, E: Elem, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for PermuteOp<E, I> {
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

impl<T: Clone, E: Elem, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for PermuteOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::permute::<E>(d, &argsort(&self.0))
    }
}

pub(crate) struct PadOp<E, I>(Vec<(usize, usize)>, PhantomData<(E, I)>);

impl<T: Clone, E: Bool, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for PadOp<E, I> {
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

impl<T: Clone, E: Bool, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for PadOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::crop::<E>(d, &self.0)
    }
}

pub(crate) struct CropOp<E, I>(Vec<(usize, usize)>, PhantomData<(E, I)>);

impl<T: Clone, E: Bool, I: DiffableOps<Repr<E> = T>> UnaryOp<T> for CropOp<E, I> {
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

impl<T: Clone, E: Bool, I: DiffableOps<Repr<E> = T>> UnaryDiffOp<T> for CropOp<E, I> {
    fn dfda(&self, d: &T) -> T {
        I::pad::<E>(d, &self.0)
    }
}

pub(crate) struct CorrelateOp<T, E, I, const N: usize> {
    im: T,
    ker: T,
    opts: CorrelateOpts<N>,
    _phantom: PhantomData<(E, I)>,
}

impl<T, E, I, const N: usize> CorrelateOp<T, E, I, N> {
    fn flips() -> Vec<bool> {
        let mut res = vec![true; N + 2];
        res[0] = false;
        res[1] = false;
        res
    }

    fn permutes() -> Vec<usize> {
        let mut res: Vec<_> = (0..N + 2).collect();
        res.swap(0, 1);
        res
    }
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, const N: usize> BinaryOp<T>
    for CorrelateOp<T, E, I, N>
{
    type Args = CorrelateOpts<N>;

    fn f(im: &T, ker: &T, &args: &CorrelateOpts<N>) -> (T, Self) {
        let op = CorrelateOp {
            im: im.clone(),
            ker: ker.clone(),
            opts: args,
            _phantom: PhantomData,
        };
        (I::correlate::<E, N>(im, ker, args), op)
    }
}

impl<T: Clone, E: Num, I: DiffableOps<Repr<E> = T>, const N: usize> BinaryDiffOp<T>
    for CorrelateOp<T, E, I, N>
{
    //           [ B, iC, iH, iW] * [oC, iC, kH, kW] -> [ B, oC, oH, oW]
    // for f     [20,  3, 50, 60] * [ 8,  3,  5,  5] -> [20,  8, 46, 56]
    // for dfda  [20,  8, 54, 64] * [ 3,  8,  5,  5] -> [20,  3, 50, 60]
    //                  pad-^   permute-^  flip-^
    // for dfdb  [ 3, 20, 50, 60] * [ 8, 20, 46, 56] -> [ 3,  8,  5,  5]
    //       permute-^          permute-^           permute-^

    fn dfda(&self, d: &T) -> T {
        // wrt image. kernel is constant
        let ker_shape = I::shape::<E>(&self.ker);
        I::correlate::<E, N>(
            d,
            &I::permute::<E>(&I::flip::<E>(&self.ker, &Self::flips()), &Self::permutes()),
            self.opts.for_kernel_transpose(ker_shape),
        )
    }

    fn dfdb(&self, d: &T) -> T {
        // wrt kernel. image is constant
        let permutes = Self::permutes();
        let ker_shape = I::shape::<E>(&self.ker);
        I::permute::<E>(
            &I::correlate::<E, N>(
                &I::permute::<E>(&self.im, &permutes),
                &I::permute::<E>(d, &permutes),
                self.opts.for_image_transpose(ker_shape),
            ),
            &permutes,
        )
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
