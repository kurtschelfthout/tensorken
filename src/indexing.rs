use std::{
    cmp::max,
    iter,
    marker::PhantomData,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
};

use crate::{
    num::{Bool, CastFrom, Num},
    Axes, DiffableOps, Shape, Tensor,
};

/// A variation of `Index` and `IndexMut`, that returns the output
/// by value. Sadly, we can't use the standard Index trait, because
/// it requires that the output be a reference. But we want to be able
/// to return new tensors, which we can't give a lifetime long enough so
/// they can be returned from the index method.
/// This means we also can't use the actual [] syntax.

pub trait BasicIndex<Idx> {
    /// Basic indexing. Supports single indexing, slices, new axes, and ellipsis.
    /// Always non-copying.
    #[must_use]
    fn ix(&self, index: Idx) -> Self;
}

/// Following a proposal for numpy: <https://numpy.org/neps/nep-0021-advanced-indexing.html/>
/// There are two methods that differ in their handling of int tensor indexes.
pub trait FancyIndex<Idx> {
    /// Outer indexing. A straightforward generalization of slicing with int tensor indexes.
    /// Copying if tensor indexes are used.
    #[must_use]
    fn oix(&self, index: Idx) -> Self;

    /// Vectorized indexing. More powerful than oix, but also more complex.
    /// Copying if tensor indexes are used.
    #[must_use]
    fn vix(&self, index: Idx) -> Self;
}

/// A single index that specifies where to start counting from - the start of the axis, or the end.
/// In Python, the end is specified as a negative number, i.e. -1 means the last element.
/// -1 would be Tail(0) in this notation, which gives it a more pleasing symmetry.
#[derive(Clone, Copy, Debug)]
pub enum SingleIndex {
    Head(usize),
    Tail(usize),
}

#[derive(Clone)]
pub enum Fancy<I: DiffableOps> {
    Full,
    IntTensor(Tensor<I::Repr<i32>, i32, I>),
    // BoolTensor(Tensor<I::Repr<bool>, bool, I>),
}

pub enum IndexElement<I: DiffableOps> {
    // A single element in an axis.
    Single(SingleIndex),
    // A range of elements in an axis.
    Slice(SingleIndex, SingleIndex),
    NewAxis,
    Ellipsis,
    Fancy(Fancy<I>),
}

pub enum BasicIndexingWitness {}
pub enum AdvancedIndexingWitness {}

/// Specifies what to select along each axis.
#[must_use]
pub struct IndexSpec<I: DiffableOps, IndexingWitness> {
    axes: Vec<IndexElement<I>>,
    witness: PhantomData<IndexingWitness>,
}

impl<I: DiffableOps> IndexSpec<I, BasicIndexingWitness> {
    pub fn basic() -> Self {
        Self {
            axes: Vec::new(),
            witness: PhantomData,
        }
    }
}

impl<I: DiffableOps> IndexSpec<I, AdvancedIndexingWitness> {
    pub fn advanced() -> Self {
        Self {
            axes: Vec::new(),
            witness: PhantomData,
        }
    }
}

/// Build an [`IndexSpec`] by chaining calls to [`IndexSpecBuilder::idx`].
pub trait IndexSpecBuilder<Idx> {
    #[must_use]
    fn idx(self, index: Idx) -> Self;
}

pub trait AdvancedIndexSpecBuilder<Idx>: IndexSpecBuilder<Idx> {}

impl<I: DiffableOps, W> IndexSpecBuilder<Range<usize>> for IndexSpec<I, W> {
    fn idx(self, index: Range<usize>) -> Self {
        self.idx(SingleIndex::Head(index.start)..SingleIndex::Head(index.end))
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<Range<SingleIndex>> for IndexSpec<I, W> {
    fn idx(mut self, index: Range<SingleIndex>) -> Self {
        self.axes.push(IndexElement::Slice(index.start, index.end));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeFrom<usize>> for IndexSpec<I, W> {
    fn idx(self, index: RangeFrom<usize>) -> Self {
        self.idx(SingleIndex::Head(index.start)..SingleIndex::Tail(0))
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeFrom<SingleIndex>> for IndexSpec<I, W> {
    fn idx(mut self, index: RangeFrom<SingleIndex>) -> Self {
        self.axes
            .push(IndexElement::Slice(index.start, SingleIndex::Tail(0)));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeTo<usize>> for IndexSpec<I, W> {
    fn idx(self, index: RangeTo<usize>) -> Self {
        self.idx(SingleIndex::Head(0)..SingleIndex::Head(index.end))
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeTo<SingleIndex>> for IndexSpec<I, W> {
    fn idx(mut self, index: RangeTo<SingleIndex>) -> Self {
        self.axes
            .push(IndexElement::Slice(SingleIndex::Head(0), index.end));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeFull> for IndexSpec<I, W> {
    fn idx(self, _: RangeFull) -> Self {
        self.idx(SingleIndex::Head(0)..SingleIndex::Tail(0))
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<usize> for IndexSpec<I, W> {
    fn idx(mut self, index: usize) -> Self {
        self.axes
            .push(IndexElement::Single(SingleIndex::Head(index)));
        self
    }
}

#[must_use]
pub const fn hd(i: usize) -> SingleIndex {
    SingleIndex::Head(i)
}

#[must_use]
pub const fn tl(i: usize) -> SingleIndex {
    SingleIndex::Tail(i)
}

impl<I: DiffableOps, W> IndexSpecBuilder<SingleIndex> for IndexSpec<I, W> {
    fn idx(mut self, element: SingleIndex) -> Self {
        self.axes.push(IndexElement::Single(element));
        self
    }
}

pub struct NewAxis;

impl<I: DiffableOps, W> IndexSpecBuilder<NewAxis> for IndexSpec<I, W> {
    fn idx(mut self, _: NewAxis) -> Self {
        self.axes.push(IndexElement::NewAxis);
        self
    }
}

pub struct Ellipsis;

impl<I: DiffableOps, W> IndexSpecBuilder<Ellipsis> for IndexSpec<I, W> {
    fn idx(mut self, _: Ellipsis) -> Self {
        self.axes.push(IndexElement::Ellipsis);
        self
    }
}

impl SingleIndex {
    /// Given the index of the last element, return the index in the array.
    /// The min valid index is assumed to be zero.
    fn get_index(&self, max_valid_index: usize) -> usize {
        match self {
            SingleIndex::Head(i) => *std::cmp::min(i, &max_valid_index),
            SingleIndex::Tail(i) => max_valid_index.saturating_sub(*i),
        }
    }
}

struct BasicIndexResolution<'a, I: DiffableOps> {
    limits: Vec<(usize, usize)>,
    flips: Vec<bool>,
    shape: Vec<usize>,
    fancy: Vec<&'a Fancy<I>>,
}

impl<'a, I: DiffableOps> BasicIndexResolution<'a, I> {
    fn add_full_axis(&mut self, size: usize) {
        self.limits.push((0, size));
        self.flips.push(false);
        self.shape.push(size);
        self.fancy.push(&Fancy::Full);
    }

    fn add_fancy_axis(&mut self, size: usize, fancy_element: &'a Fancy<I>) {
        self.limits.push((0, size));
        self.flips.push(false);
        self.shape.push(size);
        self.fancy.push(fancy_element);
    }

    fn add_single(&mut self, s: usize) {
        // crop to the single element
        self.limits.push((s, s + 1));
        // no flip necessary
        self.flips.push(false);
        // no change to shape - this dimension is squeezed out.
    }

    fn add_slice(&mut self, s: usize, e: usize) {
        if e >= s {
            // if the range is increasing, we have s..e. Add the limits as is.
            self.limits.push((s, e));
            self.flips.push(false); // no need to flip
            self.shape.push(e - s); // the new shape is the size of the range
        } else {
            // if the range is decreasing, we have e..s+1. Add the limits in reverse order.
            self.limits.push((e, s + 1));
            self.flips.push(true); // flip the axis
            self.shape.push(s + 1 - e);
        }
        self.fancy.push(&Fancy::Full);
    }

    fn add_new_axis(&mut self) {
        // no limits or flips change because this is a new axis.
        self.shape.push(1);
        self.fancy.push(&Fancy::Full);
    }

    fn fixup_empty(&mut self) {
        // A hack, because tensorken doesn't currently deal with empty shapes well.
        // Perhaps we should return an enum Scalar/Tensor instead of a tensor as the
        // indexing result.
        if self.shape.is_empty() {
            self.shape.push(1);
            self.fancy.push(&Fancy::Full);
        }
    }
}

impl<W, I: DiffableOps> IndexSpec<I, W> {
    fn resolve_basic(&self, shape: &[usize]) -> BasicIndexResolution<I> {
        // could be more, but oh well.
        let new_shape_len = std::cmp::max(shape.len(), self.axes.len());
        let mut result = BasicIndexResolution {
            limits: Vec::with_capacity(shape.len()),
            flips: Vec::with_capacity(shape.len()),
            shape: Vec::with_capacity(new_shape_len),
            fancy: Vec::with_capacity(new_shape_len),
        };
        let axes_len = self
            .axes
            .iter()
            .filter(|a| !matches!(a, IndexElement::NewAxis))
            .count();
        // the index in the IndexSpec
        let mut idx_i = 0;
        // the index in the shape
        let mut shape_i = 0;

        while shape_i < shape.len() {
            let size = shape[shape_i];

            match self.axes.get(idx_i) {
                None => {
                    // if there are no more index elements, keep the axis as is.
                    // This is equivalent to adding implicit ELLIPSIS() at the end.
                    result.add_full_axis(size);
                    shape_i += 1;
                }
                Some(index_element) => match index_element {
                    IndexElement::Fancy(fancy_element) => {
                        result.add_fancy_axis(size, fancy_element);
                        idx_i += 1;
                        shape_i += 1;
                    }
                    IndexElement::Single(idx) => {
                        // translate the index to the actual index in the tensor
                        let s = idx.get_index(size - 1);
                        result.add_single(s);
                        idx_i += 1;
                        shape_i += 1;
                    }
                    IndexElement::Slice(start, end) => {
                        // Get the start index. Pass in size-1 because the last element has index == size-1.
                        let s = start.get_index(size - 1);
                        // Get the end index. Pass size, because the last element of a range is exclusive, so the max valid index is size.
                        let e = end.get_index(size);
                        result.add_slice(s, e);
                        idx_i += 1;
                        shape_i += 1;
                    }
                    IndexElement::NewAxis => {
                        // No limits or flips change because this is a new axis.
                        result.add_new_axis();
                        idx_i += 1;
                    }
                    IndexElement::Ellipsis => {
                        // Add a limit if there aren't enough remaining elements in the IndexSpec
                        let remaining_idx_elems = axes_len.saturating_sub(idx_i + 1);
                        let remaining_shape_dims = shape.len() - shape_i;
                        if remaining_idx_elems < remaining_shape_dims {
                            // The ellipsis need to do something in this axis. Keep axis as is.
                            result.add_full_axis(size);
                            shape_i += 1;
                            // don't increment idx_i, so the ellipsis can be used again.
                        } else {
                            // This was the last axis for which we need ellipsis. Move on to the next index element.
                            idx_i += 1;
                        }
                    }
                },
            }
        }
        // we may have new axes to add at the end.
        while idx_i < self.axes.len() {
            match self.axes.get(idx_i) {
                Some(IndexElement::NewAxis) => result.add_new_axis(),
                Some(IndexElement::Ellipsis) => (), // ignore
                _ => panic!("Invalid index spec."),
            }
            idx_i += 1;
        }

        result.fixup_empty();
        result
    }
}

impl<E: Bool, I: DiffableOps> BasicIndex<IndexSpec<I, BasicIndexingWitness>>
    for Tensor<I::Repr<E>, E, I>
{
    /// Index a tensor using outer indexing. See [sl] to build an [`IndexSpec`].
    /// In outer indexing, indexes that are int tensors are treated similarly to slices.
    /// Any new dimensions are added where the int tensor is. `oix` is more intuitive to
    /// understand, but `vix` is more powerful.
    fn ix(&self, index: IndexSpec<I, BasicIndexingWitness>) -> Self {
        // first do the basic indexing - no copy.
        let resolution = index.resolve_basic(self.shape());

        self.crop(&resolution.limits)
            .flip(&resolution.flips)
            .reshape(&resolution.shape)
    }
}

impl<I: DiffableOps> IndexSpecBuilder<Tensor<I::Repr<i32>, i32, I>>
    for IndexSpec<I, AdvancedIndexingWitness>
{
    fn idx(mut self, t: Tensor<I::Repr<i32>, i32, I>) -> Self {
        self.axes.push(IndexElement::Fancy(Fancy::IntTensor(t)));
        self
    }
}

impl<E: Num + CastFrom<bool>, I: DiffableOps> FancyIndex<IndexSpec<I, AdvancedIndexingWitness>>
    for Tensor<I::Repr<E>, E, I>
{
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn oix(&self, index: IndexSpec<I, AdvancedIndexingWitness>) -> Self {
        // first do the basic indexing - no copy.
        let resolution = index.resolve_basic(self.shape());
        let basic = self
            .crop(&resolution.limits)
            .flip(&resolution.flips)
            .reshape(&resolution.shape);

        // then any advanced indexing - always copy.
        let mut result = basic;
        // the dimension we're at in the result
        let mut dim_result = 0;
        for (fancy, size) in resolution.fancy.iter().zip(resolution.shape) {
            match fancy {
                Fancy::Full => dim_result += 1,
                Fancy::IntTensor(i) => {
                    let mut i_range_shape = vec![size];
                    let ones = vec![1; i.shape().ndims()];
                    i_range_shape.extend(&ones);
                    // shape is [size, 1, 1, ...]
                    let i_range: Tensor<I::Repr<i32>, i32, I> = Tensor::new(
                        &i_range_shape,
                        (0..i_range_shape.size() as i32)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                    // shape is [size, i.shape]
                    let i_one_hot = i.eq(&i_range).cast::<E>();
                    // make room in the result for the new dimensions by adding 1s after the current dimension, dim.
                    // We need as many as there are dimentsions in i.
                    let mut result_shape = result.shape().to_vec();
                    result_shape.splice((dim_result + 1)..=dim_result, ones);

                    // also reshape the i_one_hot to add any needed 1s after the current dimension. (Any needed 1s before are added by broadcasting)
                    let mut i_one_hot_shape = i_one_hot.shape().to_vec();
                    i_one_hot_shape.extend(vec![1; result.shape().ndims() - (dim_result + 1)]);

                    result = result.reshape(&result_shape);
                    let i_one_hot = i_one_hot.reshape(&i_one_hot_shape);
                    result = result * &i_one_hot;
                    result = result.sum(&[dim_result]).squeeze(&Axes::Axis(dim_result));
                    dim_result += i.shape().ndims();
                } // Fancy::BoolTensor(t) => todo!("bool tensor fancy indexing"),
            };
        }
        result
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn vix(&self, index: IndexSpec<I, AdvancedIndexingWitness>) -> Self {
        // first do the basic indexing - no copy.
        let resolution = index.resolve_basic(self.shape());
        let basic = self
            .crop(&resolution.limits)
            .flip(&resolution.flips)
            .reshape(&resolution.shape);

        // then any advanced indexing - always copy.
        let mut result = basic;

        // the dimension we're at in the result
        let mut dim = 0;
        let mut squeezed = 0;
        let orig_result_shape_ndims = result.shape().ndims();
        // the max number of dimensions seen in any index tensor so far
        let mut max_i_ndims = 0;

        for (fancy, size) in resolution.fancy.iter().zip(resolution.shape) {
            match fancy {
                Fancy::Full => dim += 1, //dim_result += 1,
                Fancy::IntTensor(i) => {
                    max_i_ndims = max(max_i_ndims, i.shape().ndims());
                    let mut i_range_shape = vec![1; i.shape().ndims()];
                    i_range_shape.push(size);
                    // shape is [1, 1, ..., size]
                    let i_range: Tensor<I::Repr<i32>, i32, I> = Tensor::new(
                        &i_range_shape,
                        (0..i_range_shape.size() as i32)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                    let mut i_shape = i.shape().to_vec();
                    i_shape.push(1);
                    // shape is [i.shape, size]
                    let i_one_hot = i.reshape(&i_shape).eq(&i_range).cast::<E>();

                    // reshape the i_one_hot to add any needed 1s before and after the size:
                    // [i.shape, 1, 1, ..., size, 1, 1, ...]
                    let mut i_one_hot_shape = i.shape().to_vec();
                    i_one_hot_shape.extend(iter::repeat(1).take(dim));
                    i_one_hot_shape.push(size);
                    i_one_hot_shape.extend(
                        iter::repeat(1)
                            .take(orig_result_shape_ndims.saturating_sub(dim + 1 + squeezed)),
                    );
                    let i_one_hot = i_one_hot.reshape(&i_one_hot_shape);

                    result = &result * &i_one_hot;
                    result = result
                        .sum(&[dim + max_i_ndims])
                        .squeeze(&Axes::Axis(dim + max_i_ndims));
                    squeezed += 1;
                    // we removed a dim, so don't update dim.
                } // Fancy::BoolTensor(t) => todo!("bool tensor fancy indexing"),
            };
        }
        result
    }
}

impl<E: Bool, I: DiffableOps> Tensor<I::Repr<E>, E, I> {
    /// Shorthand for outer indexing along the first axis.
    pub fn ix1<T>(&self, index: T) -> Self
    where
        IndexSpec<I, BasicIndexingWitness>: IndexSpecBuilder<T>,
    {
        self.ix(IndexSpec::basic().idx(index))
    }

    /// Shorthand for outer indexing along the first two axes.
    pub fn ix2<T1, T2>(&self, index1: T1, index2: T2) -> Self
    where
        IndexSpec<I, BasicIndexingWitness>: IndexSpecBuilder<T1> + IndexSpecBuilder<T2>,
    {
        self.ix(IndexSpec::basic().idx(index1).idx(index2))
    }

    /// Shorthand for outer indexing along the first three axes.
    pub fn ix3<T1, T2, T3>(&self, index1: T1, index2: T2, index3: T3) -> Self
    where
        IndexSpec<I, BasicIndexingWitness>:
            IndexSpecBuilder<T1> + IndexSpecBuilder<T2> + IndexSpecBuilder<T3>,
    {
        self.ix(IndexSpec::basic().idx(index1).idx(index2).idx(index3))
    }

    /// Shorthand for outer indexing along the first four axes.
    pub fn ix4<T1, T2, T3, T4>(&self, index1: T1, index2: T2, index3: T3, index4: T4) -> Self
    where
        IndexSpec<I, BasicIndexingWitness>: IndexSpecBuilder<T1>
            + IndexSpecBuilder<T2>
            + IndexSpecBuilder<T3>
            + IndexSpecBuilder<T4>,
    {
        self.ix(IndexSpec::basic()
            .idx(index1)
            .idx(index2)
            .idx(index3)
            .idx(index4))
    }
}

#[cfg(test)]
mod tests {

    use crate::CpuI32;

    use super::*;

    #[test]
    fn test_at_simple() {
        let t = CpuI32::new(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(t.ix2(0, 0).to_scalar(), 1);
        assert_eq!(t.ix2(0, 1).to_scalar(), 2);
        assert_eq!(t.ix2(0, 2).to_scalar(), 3);
        assert_eq!(t.ix2(1, 0).to_scalar(), 4);
        assert_eq!(t.ix2(1, 1).to_scalar(), 5);
        assert_eq!(t.ix2(1, 2).to_scalar(), 6);
    }

    #[test]
    fn test_at_single_index() {
        let t = CpuI32::new(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(t.ix2(hd(0), hd(0)).to_scalar(), 1);
        assert_eq!(t.ix2(hd(0), hd(1)).to_scalar(), 2);
        assert_eq!(t.ix2(hd(0), hd(2)).to_scalar(), 3);
        assert_eq!(t.ix2(hd(1), hd(0)).to_scalar(), 4);
        assert_eq!(t.ix2(hd(1), hd(1)).to_scalar(), 5);
        assert_eq!(t.ix2(hd(1), hd(2)).to_scalar(), 6);

        assert_eq!(t.ix2(tl(0), tl(0)).to_scalar(), 6);
        assert_eq!(t.ix2(tl(0), tl(1)).to_scalar(), 5);
        assert_eq!(t.ix2(tl(0), tl(2)).to_scalar(), 4);
        assert_eq!(t.ix2(tl(1), tl(0)).to_scalar(), 3);
        assert_eq!(t.ix2(tl(1), tl(1)).to_scalar(), 2);
        assert_eq!(t.ix2(tl(1), tl(2)).to_scalar(), 1);
    }

    #[test]
    fn test_at_range() {
        let t = CpuI32::new(
            &[2, 3],
            &[
                1, 2, 3, //
                4, 5, 6,
            ],
        );

        let r = t.ix2(0, ..);
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.ravel(), &[1, 2, 3]);

        let r = t.ix2(1, ..);
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.ravel(), &[4, 5, 6]);

        let r = t.ix2(.., 0);
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.ravel(), &[1, 4]);

        let r = t.ix2(0..1, ..);
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[1, 2, 3]);

        let r = t.ix1(1..2);
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[4, 5, 6]);

        let r = t.ix2(.., 1..2);
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.ravel(), &[2, 5]);
    }

    #[test]
    fn test_at_reverse_range() {
        let t = CpuI32::new(
            &[2, 3],
            &[
                1, 2, 3, //
                4, 5, 6,
            ],
        );

        let r = t.ix2(tl(0)..hd(0), ..);
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.ravel(), &[4, 5, 6, 1, 2, 3]);

        let r = t.ix2(.., tl(1)..hd(0));
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.ravel(), &[2, 1, 5, 4]);

        let r = t.ix2(tl(0)..hd(0), tl(0)..hd(0));
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.ravel(), &[6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_ellipsis() {
        let t = CpuI32::new(&[3, 2, 4], &(0..24).collect::<Vec<_>>());

        let r = t.ix2(1, Ellipsis);
        assert_eq!(r.shape(), &[2, 4]);
        assert_eq!(r.ravel(), (8..16).collect::<Vec<_>>());

        let r = t.ix2(Ellipsis, 1);
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.ravel(), vec![1, 5, 9, 13, 17, 21]);

        let r = t.ix3(0, Ellipsis, 1);
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.ravel(), vec![1, 5]);

        let r = t.ix4(0, Ellipsis, 1, 1);
        assert_eq!(r.shape(), &[1]);
        assert_eq!(r.ravel(), vec![5]);
    }

    #[test]
    fn test_new_axis() {
        let t = CpuI32::new(&[3, 2, 4], &(0..24).collect::<Vec<_>>());

        let r = t.ix2(1, NewAxis);
        assert_eq!(r.shape(), &[1, 2, 4]);

        let r = t.ix4(1, NewAxis, 1, ..4);
        assert_eq!(r.shape(), &[1, 4]);

        let r = t.ix2(Ellipsis, NewAxis);
        assert_eq!(r.shape(), &[3, 2, 4, 1]);

        let r = t.ix4(NewAxis, Ellipsis, NewAxis, NewAxis);
        assert_eq!(r.shape(), &[1, 3, 2, 4, 1, 1]);
    }

    #[test]
    fn test_oix() {
        // first index - one dimensional index tensor
        let t = CpuI32::linspace(1, 24, 24u8).reshape(&[4, 2, 3]);
        let i = CpuI32::new(&[2], &[2, 0]);
        let r = t.oix(IndexSpec::advanced().idx(i));
        assert_eq!(r.shape(), &[2, 2, 3]);
        assert_eq!(r.ravel(), &[13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6]);

        // second index - one dimensional index tensor
        let i = CpuI32::new(&[2], &[1, 0]);
        let r = t.oix(IndexSpec::advanced().idx(..).idx(i));
        assert_eq!(r.shape(), &[4, 2, 3]);
        assert_eq!(
            r.ravel(),
            &[
                4, 5, 6, 1, 2, 3, //
                10, 11, 12, 7, 8, 9, //
                16, 17, 18, 13, 14, 15, //
                22, 23, 24, 19, 20, 21
            ]
        );

        // third index - one dimensional index tensor
        let i = CpuI32::new(&[2], &[1, 0]);
        let r = t.oix(IndexSpec::advanced().idx(Ellipsis).idx(i));
        assert_eq!(r.shape(), &[4, 2, 2]);
        assert_eq!(
            r.ravel(),
            &[
                2, 1, 5, 4, //
                8, 7, 11, 10, //
                14, 13, 17, 16, //
                20, 19, 23, 22
            ]
        );

        // first index - two dimensional index tensor
        let i = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let r = t.oix(IndexSpec::advanced().idx(i));
        assert_eq!(r.shape(), &[2, 2, 2, 3]);
        assert_eq!(
            r.ravel(),
            &[
                13, 14, 15, 16, 17, 18, //
                1, 2, 3, 4, 5, 6, //
                7, 8, 9, 10, 11, 12, //
                19, 20, 21, 22, 23, 24
            ]
        );

        // all indexes - one-dimensional index tensors
        let i0 = CpuI32::new(&[4], &[2, 0, 1, 3]);
        let i1 = CpuI32::new(&[2], &[1, 0]);
        let i2 = CpuI32::new(&[2], &[2, 1]);
        let r = t.oix(IndexSpec::advanced().idx(i0).idx(i1).idx(i2));
        assert_eq!(r.shape(), &[4, 2, 2]);
        assert_eq!(
            r.ravel(),
            &[18, 17, 15, 14, 6, 5, 3, 2, 12, 11, 9, 8, 24, 23, 21, 20]
        );

        // all indexes - two-dimensional index tensor
        let i0 = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let i1 = CpuI32::new(&[2], &[1, 0]);
        let i2 = CpuI32::new(&[2], &[2, 1]);
        let r = t.oix(IndexSpec::advanced().idx(i0).idx(i1).idx(i2));
        assert_eq!(r.shape(), &[2, 2, 2, 2]);
        assert_eq!(
            r.ravel(),
            &[18, 17, 15, 14, 6, 5, 3, 2, 12, 11, 9, 8, 24, 23, 21, 20]
        );

        // all indexes - all two-dimensional index tensor
        let i0 = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let i1 = CpuI32::new(&[1, 2], &[1, 0]);
        let i2 = CpuI32::new(&[2, 1], &[2, 1]);
        let r = t.oix(IndexSpec::advanced().idx(i0).idx(i1).idx(i2));
        assert_eq!(r.shape(), &[2, 2, 1, 2, 2, 1]);
        assert_eq!(
            r.ravel(),
            &[18, 17, 15, 14, 6, 5, 3, 2, 12, 11, 9, 8, 24, 23, 21, 20]
        );
    }

    #[test]
    fn text_vix() {
        // first index - one dimensional index tensor
        let t = CpuI32::linspace(1, 24, 24u8).reshape(&[4, 2, 3]);
        let i = CpuI32::new(&[2], &[2, 0]);
        let r = t.vix(IndexSpec::advanced().idx(i));
        assert_eq!(r.shape(), &[2, 2, 3]);
        assert_eq!(r.ravel(), &[13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6]);

        // second index - one dimensional index tensor
        let i = CpuI32::new(&[2], &[1, 0]);
        let r = t.vix(IndexSpec::advanced().idx(..).idx(i));
        // compared to oix, the new dimensions are always added at the front.
        assert_eq!(r.shape(), &[2, 4, 3]);
        // permute to get the oix result.
        assert_eq!(
            r.permute(&[1, 0, 2]).ravel(),
            &[
                4, 5, 6, 1, 2, 3, //
                10, 11, 12, 7, 8, 9, //
                16, 17, 18, 13, 14, 15, //
                22, 23, 24, 19, 20, 21
            ]
        );

        // third index - one dimensional index tensor
        let i = CpuI32::new(&[2], &[1, 0]);
        let r = t.vix(IndexSpec::advanced().idx(Ellipsis).idx(i));
        assert_eq!(r.shape(), &[2, 4, 2]);
        assert_eq!(
            r.permute(&[1, 2, 0]).ravel(),
            &[
                2, 1, 5, 4, //
                8, 7, 11, 10, //
                14, 13, 17, 16, //
                20, 19, 23, 22
            ]
        );

        // first index - two dimensional index tensor
        let i = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let r = t.vix(IndexSpec::advanced().idx(i));
        assert_eq!(r.shape(), &[2, 2, 2, 3]);
        assert_eq!(
            r.ravel(),
            &[
                13, 14, 15, 16, 17, 18, //
                1, 2, 3, 4, 5, 6, //
                7, 8, 9, 10, 11, 12, //
                19, 20, 21, 22, 23, 24
            ]
        );

        // all indexes - one-dimensional index tensors
        let i0 = CpuI32::new(&[2], &[2, 0]);
        let i1 = CpuI32::new(&[2], &[1, 0]);
        let i2 = CpuI32::new(&[2], &[2, 1]);
        let r = t.vix(IndexSpec::advanced().idx(i0).idx(i1).idx(i2));
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.ravel(), &[18, 2]);

        // all indexes - all two-dimensional index tensor
        let i0 = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let i1 = CpuI32::new(&[1, 2], &[1, 0]);
        let i2 = CpuI32::new(&[2, 1], &[2, 1]);
        let r = t.vix(IndexSpec::advanced().idx(i0).idx(i1).idx(i2));
        assert_eq!(r.shape(), &[2, 2]);

        // 2 0      2,1,2   0,0,2
        // 1 3      1,1,1   3,0,1

        // 1 0

        // 2
        // 1
        assert_eq!(r.ravel(), &[18, 3, 11, 20]);

        // all indexes - two-dimensional index tensor
        let i0 = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let i1 = CpuI32::new(&[2], &[1, 0]);
        let i2 = CpuI32::new(&[2], &[2, 1]);
        let r = t.vix(IndexSpec::advanced().idx(i0).idx(i1).idx(i2));
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.ravel(), &[18, 2, 12, 20]);
    }
}
