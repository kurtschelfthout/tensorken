use std::{
    iter,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
};

use crate::{num::ZeroOne, Diffable, Tensor};

/// A variation of `Index` and `IndexMut`, that returns the output
/// by value. Sadly, we can't use the standard Index trait, because
/// it requires that the output be a reference. But we want to be able
/// to return new tensors, which we can't give a lifetime long enough so
/// they can be returned from the index method.
/// This means we also can't use the actual [] syntax :( I made the name
/// as short as I could think of.
pub trait IndexValue<Idx> {
    type Output;

    fn at(&self, index: Idx) -> Self::Output;
}

impl<T, E: ZeroOne, I: Diffable<Repr<E> = T>> IndexValue<usize> for Tensor<T, E, I> {
    type Output = Self;

    /// Slices the tensor at the given index, in the first dimension, and removes the dimension.
    /// E.g. if the tensor's shape is [2, 3, 4], then at(1) will return a tensor of shape [3, 4].
    fn at(&self, index: usize) -> Self::Output {
        let mut limits = self.shape().iter().map(|&n| (0, n)).collect::<Vec<_>>();
        limits[0] = (index, index + 1);
        let cropped = self.crop(&limits);
        if cropped.shape().len() > 1 {
            cropped.reshape(&self.shape()[1..])
        } else {
            cropped
        }
    }
}

impl<T, E: ZeroOne, I: Diffable<Repr<E> = T>, const N: usize> IndexValue<&[usize; N]>
    for Tensor<T, E, I>
{
    type Output = Self;

    /// Returns the tensor at the given index. There must be at most as many indices as dimensions.
    fn at(&self, index: &[usize; N]) -> Self::Output {
        let mut limits: Vec<_> = index.iter().map(|&i| (i, i + 1)).collect();
        let mut new_shape: Vec<_> = self.shape().iter().copied().skip(limits.len()).collect();
        if new_shape.is_empty() {
            new_shape.push(1);
        };
        limits.extend(
            self.shape()
                .iter()
                .skip(limits.len())
                .map(|s| (0, *s))
                .collect::<Vec<_>>(),
        );
        self.crop(&limits).reshape(&new_shape)
    }
}

/// Specifies where to slice from - the start of the axis, or the end.
/// In Python, the end is specified as a negative number, i.e. -1 means the last element.
/// -1 would be End(0) in this notation, which gives it a more pleasing symmetry.
#[derive(Clone, Copy, Debug)]
enum SliceFrom {
    Start(usize),
    End(usize),
}

/// Specifies what to slice along each axis. Any omitted axes at the end are not sliced.
#[derive(Default)]
pub struct Slice {
    axes: Vec<(SliceFrom, SliceFrom)>,
}

/// Overload for supported `Range`-likes.
pub trait SliceIdx<SliceArg> {
    #[must_use]
    fn idx(self, index: SliceArg) -> Self;
}

impl SliceIdx<Range<usize>> for Slice {
    fn idx(mut self, index: Range<usize>) -> Self {
        self.axes
            .push((SliceFrom::Start(index.start), SliceFrom::Start(index.end)));
        self
    }
}

impl SliceIdx<RangeFrom<usize>> for Slice {
    fn idx(mut self, index: RangeFrom<usize>) -> Self {
        self.axes
            .push((SliceFrom::Start(index.start), SliceFrom::End(0)));
        self
    }
}

impl SliceIdx<RangeTo<usize>> for Slice {
    fn idx(mut self, index: RangeTo<usize>) -> Self {
        self.axes
            .push((SliceFrom::Start(0), SliceFrom::Start(index.end)));
        self
    }
}

impl SliceIdx<RangeFull> for Slice {
    fn idx(mut self, _: RangeFull) -> Self {
        self.axes.push((SliceFrom::Start(0), SliceFrom::End(0)));
        self
    }
}

impl SliceIdx<usize> for Slice {
    fn idx(mut self, index: usize) -> Self {
        self.axes
            .push((SliceFrom::Start(index), SliceFrom::Start(index + 1)));
        self
    }
}

impl Slice {
    fn crop_limits(&self, shape: &[usize]) -> Vec<(usize, usize)> {
        let mut limits = Vec::with_capacity(shape.len());
        for (i, size) in shape.iter().enumerate() {
            match self.axes.get(i) {
                None => limits.push((0, *size)),
                Some((start, end)) => {
                    let s = match start {
                        SliceFrom::Start(start) => *std::cmp::min(start, size),
                        SliceFrom::End(start) => size.saturating_sub(*start),
                    };
                    let e = match end {
                        SliceFrom::Start(end) => *std::cmp::min(end, size),
                        SliceFrom::End(end) => size.saturating_sub(*end),
                    };
                    limits.push((s, e));
                }
            }
        }
        limits
    }
}

/// Create a new slice, which returns the whole tensor.
#[must_use]
pub fn sl() -> Slice {
    Slice::default()
}

/// Create a new slice along the first axis.
pub fn sl1<T>(index: T) -> Slice
where
    Slice: SliceIdx<T>,
{
    sl().idx(index)
}

/// Create a new slice along the first two axes.
pub fn sl2<T1, T2>(index1: T1, index2: T2) -> Slice
where
    Slice: SliceIdx<T1> + SliceIdx<T2>,
{
    sl1(index1).idx(index2)
}

/// Create a new slice along the first three axes.
pub fn sl3<T1, T2, T3>(index1: T1, index2: T2, index3: T3) -> Slice
where
    Slice: SliceIdx<T1> + SliceIdx<T2> + SliceIdx<T3>,
{
    sl2(index1, index2).idx(index3)
}

/// Create a new slice along the first four axes.
pub fn sl4<T1, T2, T3, T4>(index1: T1, index2: T2, index3: T3, index4: T4) -> Slice
where
    Slice: SliceIdx<T1> + SliceIdx<T2> + SliceIdx<T3> + SliceIdx<T4>,
{
    sl3(index1, index2, index3).idx(index4)
}

impl<E: ZeroOne, I: Diffable> IndexValue<Slice> for Tensor<I::Repr<E>, E, I> {
    type Output = Self;

    /// Slice the tensor.
    fn at(&self, index: Slice) -> Self::Output {
        let mut limits = index.crop_limits(self.shape());
        limits.extend(iter::repeat((0, 0)).take(self.shape().len() - limits.len()));
        self.crop(&limits)
    }
}

#[cfg(test)]
mod tests {

    use crate::{Cpu32, CpuI32};

    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_at_index() {
        let t = Cpu32::new(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(t.at(&[0, 0]).to_scalar(), 1.);
        assert_eq!(t.at(&[0, 1]).to_scalar(), 2.);
        assert_eq!(t.at(&[0, 2]).to_scalar(), 3.);
        assert_eq!(t.at(&[1, 0]).to_scalar(), 4.);
        assert_eq!(t.at(&[1, 1]).to_scalar(), 5.);
        assert_eq!(t.at(&[1, 2]).to_scalar(), 6.);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_at_slice() {
        let t = Cpu32::new(
            &[2, 3],
            &[
                1., 2., 3., //
                4., 5., 6.,
            ],
        );

        let r = t.at(sl2(0, ..));
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[1., 2., 3.]);

        let r = t.at(sl2(1, ..));
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[4., 5., 6.]);

        let r = t.at(sl2(.., 0));
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.ravel(), &[1., 4.]);

        let r = t.at(sl2(0..1, ..));
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[1., 2., 3.]);

        let r = t.at(sl1(1..2));
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[4., 5., 6.]);

        let r = t.at(sl2(.., 1..2));
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.ravel(), &[2., 5.]);
    }

    #[test]
    fn test_i32_tensor() {
        let t = CpuI32::new(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        let mut r = &t + &t;
        r = CpuI32::scalar(2) * &r;
        assert_eq!(r.ravel(), &[4, 8, 12, 16, 20, 24]);
    }

    // TODO this should work, but eq requires Num now
    // because broadcasting it requires expand, which requires sum
    // for diffing.
    // Elementwise eq would work.
    // Perhaps just need to separate add/mul/sum in a trait between ZeroOne and Num.
    // Then could also use add as or and mul as and for bools.
    // #[test]
    // fn test_bool_tensor() {
    //     let t = CpuBool::new(&[2, 3], &[true, true, false, false, true, true]);
    //     let r = &t.eq(&t);
    //     assert_eq!(r.ravel(), &[true, true, true, true, true, true]);
    // }
}
