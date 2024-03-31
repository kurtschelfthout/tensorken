use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use crate::{num::Bool, DiffableOps, Tensor};

/// A variation of `Index` and `IndexMut`, that returns the output
/// by value. Sadly, we can't use the standard Index trait, because
/// it requires that the output be a reference. But we want to be able
/// to return new tensors, which we can't give a lifetime long enough so
/// they can be returned from the index method.
/// This means we also can't use the actual [] syntax :(
/// I made the name as short as I could think of.
pub trait IndexValue<Idx> {
    type Output;

    fn at(&self, index: Idx) -> Self::Output;
}

/// A single index that specifies where to start counting from - the start of the axis, or the end.
/// In Python, the end is specified as a negative number, i.e. -1 means the last element.
/// -1 would be Tail(0) in this notation, which gives it a more pleasing symmetry.
#[derive(Clone, Copy, Debug)]
pub enum SingleIndex {
    Head(usize),
    Tail(usize),
}

pub enum Step {
    ToTail(usize),
    // ToHead(usize),
}

impl Step {
    fn get_step(&self) -> usize {
        match self {
            Step::ToTail(i) => *i,
        }
    }
}

pub enum IndexElement {
    // A single element in an axis.
    Single(SingleIndex),
    // A range of elements in an axis.
    Slice(SingleIndex, Step, SingleIndex),
    NewAxis,
    Ellipsis,
}

/// Specifies what to select along each axis.
#[derive(Default)]
pub struct IndexSpec {
    axes: Vec<IndexElement>,
}

/// Build an [`IndexSpec`] by chaining calls to [`IndexSpecBuilder::idx`].
pub trait IndexSpecBuilder<Idx> {
    #[must_use]
    fn idx(self, index: Idx) -> Self;
}

impl IndexSpecBuilder<Range<usize>> for IndexSpec {
    fn idx(self, index: Range<usize>) -> Self {
        self.idx(SingleIndex::Head(index.start)..SingleIndex::Head(index.end))
    }
}

impl IndexSpecBuilder<Range<SingleIndex>> for IndexSpec {
    fn idx(mut self, index: Range<SingleIndex>) -> Self {
        self.axes
            .push(IndexElement::Slice(index.start, Step::ToTail(1), index.end));
        self
    }
}

impl IndexSpecBuilder<RangeFrom<usize>> for IndexSpec {
    fn idx(self, index: RangeFrom<usize>) -> Self {
        self.idx(SingleIndex::Head(index.start)..SingleIndex::Tail(0))
    }
}

impl IndexSpecBuilder<RangeFrom<SingleIndex>> for IndexSpec {
    fn idx(mut self, index: RangeFrom<SingleIndex>) -> Self {
        self.axes.push(IndexElement::Slice(
            index.start,
            Step::ToTail(1),
            SingleIndex::Tail(0),
        ));
        self
    }
}

impl IndexSpecBuilder<RangeTo<usize>> for IndexSpec {
    fn idx(self, index: RangeTo<usize>) -> Self {
        self.idx(SingleIndex::Head(0)..SingleIndex::Head(index.end))
    }
}

impl IndexSpecBuilder<RangeTo<SingleIndex>> for IndexSpec {
    fn idx(mut self, index: RangeTo<SingleIndex>) -> Self {
        self.axes.push(IndexElement::Slice(
            SingleIndex::Head(0),
            Step::ToTail(1),
            index.end,
        ));
        self
    }
}

impl IndexSpecBuilder<RangeFull> for IndexSpec {
    fn idx(self, _: RangeFull) -> Self {
        self.idx(SingleIndex::Head(0)..SingleIndex::Tail(0))
    }
}

impl IndexSpecBuilder<usize> for IndexSpec {
    fn idx(mut self, index: usize) -> Self {
        self.axes
            .push(IndexElement::Single(SingleIndex::Head(index)));
        self
    }
}

#[must_use]
pub const fn hd(i: usize) -> IndexElement {
    IndexElement::Single(SingleIndex::Head(i))
}

#[must_use]
pub const fn tl(i: usize) -> IndexElement {
    IndexElement::Single(SingleIndex::Tail(i))
}

pub const ELLIPSIS: IndexElement = IndexElement::Ellipsis;

pub const NEW_AXIS: IndexElement = IndexElement::NewAxis;

impl IndexSpecBuilder<IndexElement> for IndexSpec {
    fn idx(mut self, element: IndexElement) -> Self {
        self.axes.push(element);
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

struct IndexResolution {
    limits: Vec<(usize, usize)>,
    shape: Vec<usize>,
}

impl IndexSpec {
    fn resolve(&self, shape: &[usize]) -> IndexResolution {
        let mut limits = Vec::with_capacity(shape.len());
        let mut new_shape = Vec::with_capacity(std::cmp::max(shape.len(), self.axes.len()));
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
                    limits.push((0, size));
                    new_shape.push(size);
                    idx_i += 1;
                    shape_i += 1;
                }
                Some(index_element) => match index_element {
                    IndexElement::Single(idx) => {
                        let s = idx.get_index(size - 1);
                        limits.push((s, s + 1));
                        // no change to new_shape - this dimension is squeezed out.
                        idx_i += 1;
                        shape_i += 1;
                    }
                    IndexElement::Slice(start, step, end) => {
                        // Gotcha here - we pass in size-1 because the last element is at index size-1.
                        let s = start.get_index(size - 1);
                        // Here we pass size, because the last element of a range is exclusive, so the max valid index is size.
                        let e = end.get_index(size);
                        limits.push((s, e));
                        new_shape.push((e - s) / step.get_step());
                        idx_i += 1;
                        shape_i += 1;
                    }
                    IndexElement::NewAxis => {
                        // no limits change because this is a new axis
                        new_shape.push(1);
                        idx_i += 1;
                    }
                    IndexElement::Ellipsis => {
                        // add a limit if there aren't enough remaining elements in the IndexSpec
                        let remaining_idx_elems = axes_len - idx_i - 1;
                        let remaining_shape_dims = shape.len() - shape_i;
                        if remaining_idx_elems < remaining_shape_dims {
                            limits.push((0, size));
                            new_shape.push(size);
                            shape_i += 1;
                        }
                        if remaining_idx_elems >= remaining_shape_dims {
                            idx_i += 1;
                        }
                    }
                },
            }
        }
        // we may have new axes to add at the end.
        while idx_i < self.axes.len() {
            match self.axes.get(idx_i) {
                Some(IndexElement::NewAxis) => new_shape.push(1),
                Some(IndexElement::Ellipsis) => (), // ignore
                _ => panic!("Invalid index spec."),
            }
            idx_i += 1;
        }
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        IndexResolution {
            limits,
            shape: new_shape,
        }
    }
}

impl<E: Bool, I: DiffableOps> IndexValue<IndexSpec> for Tensor<I::Repr<E>, E, I> {
    type Output = Self;

    /// Index a tensor. See [sl] to create [`IndexSpec`] types.
    fn at(&self, index: IndexSpec) -> Self::Output {
        let resolution = index.resolve(self.shape());
        self.crop(&resolution.limits).reshape(&resolution.shape)
    }
}

impl<E: Bool, I: DiffableOps> Tensor<I::Repr<E>, E, I> {
    /// Create a new index along the first axis.
    pub fn at1<T>(&self, index: T) -> Self
    where
        IndexSpec: IndexSpecBuilder<T>,
    {
        self.at(IndexSpec::default().idx(index))
    }

    /// Create a new index along the first two axes.
    pub fn at2<T1, T2>(&self, index1: T1, index2: T2) -> Self
    where
        IndexSpec: IndexSpecBuilder<T1> + IndexSpecBuilder<T2>,
    {
        self.at(IndexSpec::default().idx(index1).idx(index2))
    }

    /// Create a new index along the first three axes.
    pub fn at3<T1, T2, T3>(&self, index1: T1, index2: T2, index3: T3) -> Self
    where
        IndexSpec: IndexSpecBuilder<T1> + IndexSpecBuilder<T2> + IndexSpecBuilder<T3>,
    {
        self.at(IndexSpec::default().idx(index1).idx(index2).idx(index3))
    }

    /// Create a new index along the first four axes.
    pub fn at4<T1, T2, T3, T4>(&self, index1: T1, index2: T2, index3: T3, index4: T4) -> Self
    where
        IndexSpec: IndexSpecBuilder<T1>
            + IndexSpecBuilder<T2>
            + IndexSpecBuilder<T3>
            + IndexSpecBuilder<T4>,
    {
        self.at(IndexSpec::default()
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
        assert_eq!(t.at2(0, 0).to_scalar(), 1);
        assert_eq!(t.at2(0, 1).to_scalar(), 2);
        assert_eq!(t.at2(0, 2).to_scalar(), 3);
        assert_eq!(t.at2(1, 0).to_scalar(), 4);
        assert_eq!(t.at2(1, 1).to_scalar(), 5);
        assert_eq!(t.at2(1, 2).to_scalar(), 6);
    }

    #[test]
    fn test_at_single_index() {
        let t = CpuI32::new(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(t.at2(hd(0), hd(0)).to_scalar(), 1);
        assert_eq!(t.at2(hd(0), hd(1)).to_scalar(), 2);
        assert_eq!(t.at2(hd(0), hd(2)).to_scalar(), 3);
        assert_eq!(t.at2(hd(1), hd(0)).to_scalar(), 4);
        assert_eq!(t.at2(hd(1), hd(1)).to_scalar(), 5);
        assert_eq!(t.at2(hd(1), hd(2)).to_scalar(), 6);

        assert_eq!(t.at2(tl(0), tl(0)).to_scalar(), 6);
        assert_eq!(t.at2(tl(0), tl(1)).to_scalar(), 5);
        assert_eq!(t.at2(tl(0), tl(2)).to_scalar(), 4);
        assert_eq!(t.at2(tl(1), tl(0)).to_scalar(), 3);
        assert_eq!(t.at2(tl(1), tl(1)).to_scalar(), 2);
        assert_eq!(t.at2(tl(1), tl(2)).to_scalar(), 1);
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

        let r = t.at2(0, ..);
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.ravel(), &[1, 2, 3]);

        let r = t.at2(1, ..);
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.ravel(), &[4, 5, 6]);

        let r = t.at2(.., 0);
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.ravel(), &[1, 4]);

        let r = t.at2(0..1, ..);
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[1, 2, 3]);

        let r = t.at1(1..2);
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[4, 5, 6]);

        let r = t.at2(.., 1..2);
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.ravel(), &[2, 5]);
    }

    #[test]
    fn test_ellipsis() {
        let t = CpuI32::new(&[3, 2, 4], &(0..24).collect::<Vec<_>>());

        let r = t.at2(1, ELLIPSIS);
        assert_eq!(r.shape(), &[2, 4]);
        assert_eq!(r.ravel(), (8..16).collect::<Vec<_>>());

        let r = t.at2(ELLIPSIS, 1);
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.ravel(), vec![1, 5, 9, 13, 17, 21]);

        let r = t.at3(0, ELLIPSIS, 1);
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.ravel(), vec![1, 5]);

        let r = t.at4(0, ELLIPSIS, 1, 1);
        assert_eq!(r.shape(), &[1]);
        assert_eq!(r.ravel(), vec![5]);
    }

    #[test]
    fn test_new_axis() {
        let t = CpuI32::new(&[3, 2, 4], &(0..24).collect::<Vec<_>>());

        let r = t.at2(1, NEW_AXIS);
        assert_eq!(r.shape(), &[1, 2, 4]);

        let r = t.at4(1, NEW_AXIS, 1, ..4);
        assert_eq!(r.shape(), &[1, 4]);

        let r = t.at2(ELLIPSIS, NEW_AXIS);
        assert_eq!(r.shape(), &[3, 2, 4, 1]);
    }
}
