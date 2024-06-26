use std::{
    cmp::max,
    iter,
    marker::PhantomData,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
};

use crate::{
    num::{Bool, CastFrom, Num},
    Axes, DiffableOps, Shape, Tensor, ToCpu,
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

/// In addition to basic indexing (slicing), allow indexing with bool and int tensors.
/// Following a proposal for numpy: <https://numpy.org/neps/nep-0021-advanced-indexing.html/>,
/// there are two methods that differ in their handling of int tensor indexes.
pub trait AdvancedIndex<Idx> {
    /// Outer indexing. A straightforward generalization of slicing, with tensor indexes.
    /// Copying if tensor indexes are used.
    #[must_use]
    fn oix(&self, index: Idx) -> Self;

    /// Vectorized indexing. More powerful than oix, but also more complex. As opposed to
    /// `oix`, any int tensor indexes are broadcasted together.
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
    BoolTensor(Tensor<I::Repr<bool>, bool, I>),
}

pub enum IndexElement<I: DiffableOps> {
    // A single element in an axis.
    Single(SingleIndex),
    // A range of elements in an axis. The second element is inclusive if the bool is true.
    Slice(SingleIndex, SingleIndex, bool),
    // Create a new axis with size 1.
    NewAxis,
    // Keep the remaining dimensions as is.
    Ellipsis,
    // Fancy index - mask or int tensor.
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

/// Basic indexing is alwas non-copying. Start an index
/// specification with `IndexSpec::basic()` to ensure you don't copy.
impl<I: DiffableOps> IndexSpec<I, BasicIndexingWitness> {
    pub fn basic() -> Self {
        Self {
            axes: Vec::new(),
            witness: PhantomData,
        }
    }
}

/// Advanced indexing is always copying. Start an index
/// specification with `IndexSpec::advanced()` if you want to use advanced indexing.
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

impl<I: DiffableOps, W> IndexSpecBuilder<Range<usize>> for IndexSpec<I, W> {
    fn idx(self, index: Range<usize>) -> Self {
        self.idx(SingleIndex::Head(index.start)..SingleIndex::Head(index.end))
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<Range<SingleIndex>> for IndexSpec<I, W> {
    fn idx(mut self, index: Range<SingleIndex>) -> Self {
        self.axes
            .push(IndexElement::Slice(index.start, index.end, false));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeFrom<usize>> for IndexSpec<I, W> {
    fn idx(mut self, index: RangeFrom<usize>) -> Self {
        self.axes.push(IndexElement::Slice(
            SingleIndex::Head(index.start),
            SingleIndex::Tail(0),
            true,
        ));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeFrom<SingleIndex>> for IndexSpec<I, W> {
    fn idx(mut self, index: RangeFrom<SingleIndex>) -> Self {
        self.axes
            .push(IndexElement::Slice(index.start, SingleIndex::Tail(0), true));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeTo<usize>> for IndexSpec<I, W> {
    fn idx(mut self, index: RangeTo<usize>) -> Self {
        self.axes.push(IndexElement::Slice(
            SingleIndex::Head(0),
            SingleIndex::Head(index.end),
            false,
        ));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeTo<SingleIndex>> for IndexSpec<I, W> {
    fn idx(mut self, index: RangeTo<SingleIndex>) -> Self {
        self.axes
            .push(IndexElement::Slice(SingleIndex::Head(0), index.end, false));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<RangeFull> for IndexSpec<I, W> {
    fn idx(mut self, _: RangeFull) -> Self {
        self.axes.push(IndexElement::Slice(
            SingleIndex::Head(0),
            SingleIndex::Tail(0),
            true,
        ));
        self
    }
}

impl<I: DiffableOps, W> IndexSpecBuilder<usize> for IndexSpec<I, W> {
    fn idx(mut self, index: usize) -> Self {
        self.axes
            .push(IndexElement::Single(SingleIndex::Head(index)));
        self
    }
}

/// Select a single index along an axis, where `i`  is a zero-based index
/// starting from the head, i.e. the beginning of the dimension.
#[must_use]
pub const fn hd(i: usize) -> SingleIndex {
    SingleIndex::Head(i)
}

/// Select a single index along an axis, where `i` is a zero-based index
/// starting from the tail, i.e. the end of the dimension. `tl(0)` is the last element.
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

/// Indicate that a new axis with size 1 should be added at this point.
pub struct NewAxis;

impl<I: DiffableOps, W> IndexSpecBuilder<NewAxis> for IndexSpec<I, W> {
    fn idx(mut self, _: NewAxis) -> Self {
        self.axes.push(IndexElement::NewAxis);
        self
    }
}

/// Indicates that any remaining dimensions should be kept as is.
/// You can only use one `Ellipsis` per index spec.
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
    fn get_index(&self, max_valid_index: usize, clamp: usize) -> usize {
        match self {
            SingleIndex::Head(i) => *std::cmp::min(i, &clamp),
            SingleIndex::Tail(i) => max_valid_index.saturating_sub(*std::cmp::min(i, &clamp)),
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

    fn add_fancy_int_axis(&mut self, size: usize, fancy_element: &'a Fancy<I>) {
        self.limits.push((0, size));
        self.flips.push(false);
        self.shape.push(size);
        self.fancy.push(fancy_element);
    }

    fn add_fancy_bool_axis(&mut self, sizes: &[usize], fancy_element: &'a Fancy<I>, ndims: usize) {
        (0..ndims).for_each(|i| {
            self.limits.push((0, sizes[i]));
            self.flips.push(false);
            self.shape.push(sizes[i]);
        });
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
            // if the range is decreasing, we have e+1..s+1 because inclusive...exclusive is flipped.
            // E.g. 3..1 is 3, 2. or equivalent with (2..4).rev()
            // Add the limits in reverse order.
            self.limits.push((e + 1, s + 1));
            self.flips.push(true); // flip the axis
            self.shape.push(s - e);
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
                    // This is equivalent to adding implicit ellipsis at the end.
                    result.add_full_axis(size);
                    shape_i += 1;
                }
                Some(index_element) => match index_element {
                    IndexElement::Fancy(fancy_element) => {
                        match fancy_element {
                            Fancy::IntTensor(_) => {
                                result.add_fancy_int_axis(size, fancy_element);
                                shape_i += 1;
                            }
                            Fancy::BoolTensor(b) => {
                                result.add_fancy_bool_axis(
                                    &shape[shape_i..shape_i + b.shape().ndims()],
                                    fancy_element,
                                    b.shape().ndims(),
                                );
                                shape_i += b.shape().ndims();
                            }
                            Fancy::Full => {
                                result.add_full_axis(size);
                                shape_i += 1;
                            }
                        }
                        idx_i += 1;
                    }
                    IndexElement::Single(idx) => {
                        // translate the index to the actual index in the tensor
                        let s = idx.get_index(size - 1, size - 1);
                        result.add_single(s);
                        idx_i += 1;
                        shape_i += 1;
                    }
                    IndexElement::Slice(start, end, is_end_inclusive) => {
                        // Get the start index. Pass in size-1 because the last element has index == size-1.
                        let s = start.get_index(size - 1, size - 1);
                        // Get the end index. Pass size, because the last element of a range is exclusive, so the max valid index is size.
                        let e = if *is_end_inclusive {
                            end.get_index(size - 1, size - 1) + 1
                        } else {
                            end.get_index(size - 1, size)
                        };
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
    /// Index a tensor using basic indexing. See [`IndexSpec::basic()`] to build an [`IndexSpec`].
    fn ix(&self, index: IndexSpec<I, BasicIndexingWitness>) -> Self {
        // first do the basic indexing - no copy.
        let resolution = index.resolve_basic(self.shape());

        self.crop(&resolution.limits)
            .flip(&resolution.flips)
            .reshape(&resolution.shape)
    }
}

impl<I: DiffableOps + Clone> IndexSpecBuilder<&Tensor<I::Repr<i32>, i32, I>>
    for IndexSpec<I, AdvancedIndexingWitness>
{
    fn idx(mut self, t: &Tensor<I::Repr<i32>, i32, I>) -> Self {
        self.axes
            .push(IndexElement::Fancy(Fancy::IntTensor(t.clone())));
        self
    }
}

impl<I: DiffableOps + Clone> IndexSpecBuilder<&Tensor<I::Repr<bool>, bool, I>>
    for IndexSpec<I, AdvancedIndexingWitness>
{
    fn idx(mut self, t: &Tensor<I::Repr<bool>, bool, I>) -> Self {
        self.axes
            .push(IndexElement::Fancy(Fancy::BoolTensor(t.clone())));
        self
    }
}

impl<
        T,
        E: Num + CastFrom<bool>,
        I: DiffableOps<Repr<E> = T>
            + ToCpu<Repr<E> = T>
            + ToCpu<Repr<bool> = <I as DiffableOps>::Repr<bool>>,
    > AdvancedIndex<IndexSpec<I, AdvancedIndexingWitness>> for Tensor<T, E, I>
{
    fn oix(&self, index: IndexSpec<I, AdvancedIndexingWitness>) -> Self {
        // first do the basic indexing - no copy.
        let resolution = index.resolve_basic(self.shape());
        let basic = self
            .crop(&resolution.limits)
            .flip(&resolution.flips)
            .reshape(&resolution.shape);

        // then any advanced indexing - always copy.
        let mut result = basic;

        // the dimension we're at in the resulting shape after indexing
        let mut dim_result = 0;
        for (fancy, size) in resolution.fancy.iter().zip(resolution.shape) {
            match fancy {
                Fancy::Full => {
                    dim_result += 1;
                }
                Fancy::IntTensor(i) => {
                    oix_int(size, i, &mut result, dim_result);
                    dim_result += i.shape().ndims();
                }
                Fancy::BoolTensor(b) => {
                    ix_bool(b, &mut result, dim_result);
                    dim_result += 1;
                }
            };
        }
        result
    }

    fn vix(&self, index: IndexSpec<I, AdvancedIndexingWitness>) -> Self {
        // first do the basic indexing - no copy.
        let resolution = index.resolve_basic(self.shape());
        let basic = self
            .crop(&resolution.limits)
            .flip(&resolution.flips)
            .reshape(&resolution.shape);

        // then any advanced indexing - always copy.
        let mut result = basic;

        // The dimension we're at in the result, without any int tensor index dimensions added at the front.
        let mut dim_result_post = 0;
        // The number of dimensions added at the front of the result, by int tensor indexes.
        // This is the max number of dimensions seen in any int index tensor so far.
        let mut dim_result_pre = 0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for (fancy, size) in resolution.fancy.iter().zip(resolution.shape) {
            match fancy {
                Fancy::Full => dim_result_post += 1,
                Fancy::IntTensor(i) => {
                    // i_range.shape is [size]
                    let i_range =
                        Tensor::new(&[size], (0..size as i32).collect::<Vec<_>>().as_slice());
                    let mut i_shape = i.shape().to_vec();
                    i_shape.push(1);
                    // i_shape is [i.shape, 1]
                    let i_one_hot = i.reshape(&i_shape).eq(&i_range).cast::<E>();
                    // i_one_hot is thus [i.shape, size] via broadcasting on eq.
                    // Now, reshape i_one_hot to add any needed 1s before and after the size:
                    // [i.shape, 1, 1, ..., size, 1, 1, ...]
                    let mut i_one_hot_shape = i.shape().to_vec();
                    i_one_hot_shape.extend(iter::repeat(1).take(dim_result_post));
                    i_one_hot_shape.push(size);
                    i_one_hot_shape.extend(
                        iter::repeat(1).take(
                            result
                                .shape()
                                .ndims()
                                .saturating_sub(dim_result_pre + dim_result_post + 1),
                        ),
                    );
                    let i_one_hot = i_one_hot.reshape(&i_one_hot_shape);

                    result = &result * &i_one_hot;

                    dim_result_pre = max(dim_result_pre, i.shape().ndims());
                    let dim_result = dim_result_pre + dim_result_post;
                    result = result.sum(&[dim_result]).squeeze(&Axes::Axis(dim_result));

                    // "Moved" one dimension from the original to the prefix where int tensor indexes are.
                    // So no change for dim_result_post.
                    // dim_result_post += 0;
                }
                Fancy::BoolTensor(b) => {
                    let dim_result = dim_result_pre + dim_result_post;
                    ix_bool(b, &mut result, dim_result);
                    dim_result_post += 1;
                }
            };
        }
        result
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn ix_bool<
    T,
    E: Num + CastFrom<bool>,
    I: DiffableOps<Repr<E> = T>
        + ToCpu<Repr<E> = T>
        + ToCpu<Repr<bool> = <I as DiffableOps>::Repr<bool>>,
>(
    b: &Tensor<<I as DiffableOps>::Repr<bool>, bool, I>,
    result: &mut Tensor<T, E, I>,
    dim_result: usize,
) {
    // Create the equivalent int index tensor, then use oix_int.
    let vec = b.ravel();
    let i_vec: Vec<_> = vec
        .iter()
        .enumerate()
        .filter_map(|t| if *t.1 { Some(t.0 as i32) } else { None })
        .collect();
    let i_tensor: Tensor<_, i32, I> = Tensor::new(&[i_vec.len()], i_vec.as_slice());
    // flatten the dimensions of the result tensor that are being indexed by the bool tensor to a 1D vector.
    let mut result_shape = result.shape().to_vec();
    result_shape.splice(
        dim_result..(dim_result + b.shape().ndims()),
        [b.shape().size()],
    );
    *result = result.reshape(&result_shape);
    oix_int(b.shape().size(), &i_tensor, result, dim_result);
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn oix_int<T, E: Num + CastFrom<bool>, I: DiffableOps<Repr<E> = T> + ToCpu<Repr<E> = T>>(
    size: usize,
    i: &Tensor<<I as DiffableOps>::Repr<i32>, i32, I>,
    result: &mut Tensor<T, E, I>,
    dim_result: usize,
) {
    let mut i_range_shape = vec![size];
    let ones = vec![1; i.shape().ndims()];
    i_range_shape.extend(&ones);
    // i_range.shape is [size, 1, 1, ...]
    let i_range = Tensor::new(
        &i_range_shape,
        (0..i_range_shape.size() as i32)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    // i_one_hot.shape is [size, i.shape] by broadcasting on eq.
    let i_one_hot = i.eq(&i_range).cast::<E>();
    // make room in the result for the new dimensions by adding 1s after the current dimension, dim.
    // We need as many as there are dimentsions in i.
    let mut result_shape = result.shape().to_vec();
    result_shape.splice((dim_result + 1)..=dim_result, ones);

    // also reshape the i_one_hot to add any needed 1s after the current dimension. (Any needed 1s before are added by broadcasting)
    let mut i_one_hot_shape = i_one_hot.shape().to_vec();
    i_one_hot_shape.extend(vec![1; result.shape().ndims() - (dim_result + 1)]);

    *result = result.reshape(&result_shape);
    let i_one_hot = i_one_hot.reshape(&i_one_hot_shape);
    *result = &*result * &i_one_hot;
    *result = result.sum(&[dim_result]).squeeze(&Axes::Axis(dim_result));
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

impl<
        T,
        E: Num + CastFrom<bool>,
        I: DiffableOps<Repr<E> = T>
            + ToCpu<Repr<E> = T>
            + ToCpu<Repr<bool> = <I as DiffableOps>::Repr<bool>>,
    > Tensor<T, E, I>
{
    /// Shorthand for outer indexing along the first axis.
    pub fn oix1<T1>(&self, index: T1) -> Self
    where
        IndexSpec<I, AdvancedIndexingWitness>: IndexSpecBuilder<T1>,
    {
        self.oix(IndexSpec::advanced().idx(index))
    }

    /// Shorthand for outer indexing along the first two axes.
    pub fn oix2<T1, T2>(&self, index1: T1, index2: T2) -> Self
    where
        IndexSpec<I, AdvancedIndexingWitness>: IndexSpecBuilder<T1> + IndexSpecBuilder<T2>,
    {
        self.oix(IndexSpec::advanced().idx(index1).idx(index2))
    }

    /// Shorthand for outer indexing along the first three axes.
    pub fn oix3<T1, T2, T3>(&self, index1: T1, index2: T2, index3: T3) -> Self
    where
        IndexSpec<I, AdvancedIndexingWitness>:
            IndexSpecBuilder<T1> + IndexSpecBuilder<T2> + IndexSpecBuilder<T3>,
    {
        self.oix(IndexSpec::advanced().idx(index1).idx(index2).idx(index3))
    }

    /// Shorthand for outer indexing along the first four axes.
    pub fn oix4<T1, T2, T3, T4>(&self, index1: T1, index2: T2, index3: T3, index4: T4) -> Self
    where
        IndexSpec<I, AdvancedIndexingWitness>: IndexSpecBuilder<T1>
            + IndexSpecBuilder<T2>
            + IndexSpecBuilder<T3>
            + IndexSpecBuilder<T4>,
    {
        self.oix(
            IndexSpec::advanced()
                .idx(index1)
                .idx(index2)
                .idx(index3)
                .idx(index4),
        )
    }

    /// Shorthand for outer indexing along the first axis.
    pub fn vix1<T1>(&self, index: T1) -> Self
    where
        IndexSpec<I, AdvancedIndexingWitness>: IndexSpecBuilder<T1>,
    {
        self.vix(IndexSpec::advanced().idx(index))
    }

    /// Shorthand for outer indexing along the first two axes.
    pub fn vix2<T1, T2>(&self, index1: T1, index2: T2) -> Self
    where
        IndexSpec<I, AdvancedIndexingWitness>: IndexSpecBuilder<T1> + IndexSpecBuilder<T2>,
    {
        self.vix(IndexSpec::advanced().idx(index1).idx(index2))
    }

    /// Shorthand for outer indexing along the first three axes.
    pub fn vix3<T1, T2, T3>(&self, index1: T1, index2: T2, index3: T3) -> Self
    where
        IndexSpec<I, AdvancedIndexingWitness>:
            IndexSpecBuilder<T1> + IndexSpecBuilder<T2> + IndexSpecBuilder<T3>,
    {
        self.vix(IndexSpec::advanced().idx(index1).idx(index2).idx(index3))
    }

    /// Shorthand for outer indexing along the first four axes.
    pub fn vix4<T1, T2, T3, T4>(&self, index1: T1, index2: T2, index3: T3, index4: T4) -> Self
    where
        IndexSpec<I, AdvancedIndexingWitness>: IndexSpecBuilder<T1>
            + IndexSpecBuilder<T2>
            + IndexSpecBuilder<T3>
            + IndexSpecBuilder<T4>,
    {
        self.vix(
            IndexSpec::advanced()
                .idx(index1)
                .idx(index2)
                .idx(index3)
                .idx(index4),
        )
    }
}

#[cfg(test)]
mod tests {

    use crate::{Cpu32, CpuBool, CpuI32};

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

        let r = t.ix2(..tl(0), ..tl(0));
        assert_eq!(r.shape(), &[1, 2]);
        assert_eq!(r.ravel(), &[1, 2]);
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
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.ravel(), &[4, 5, 6]);

        let r = t.ix2(.., tl(1)..hd(0));
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.ravel(), &[2, 5]);

        let r = t.ix2(tl(0)..hd(0), tl(0)..hd(0));
        assert_eq!(r.shape(), &[1, 2]);
        assert_eq!(r.ravel(), &[6, 5]);

        let r = t.ix2(hd(1)..hd(0), hd(2)..hd(0));
        assert_eq!(r.shape(), &[1, 2]);
        assert_eq!(r.ravel(), &[6, 5]);
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
    fn test_oix_int() {
        // first index - one dimensional index tensor
        let t = CpuI32::linspace(1, 24, 24u8).reshape(&[4, 2, 3]);
        let i = CpuI32::new(&[2], &[2, 0]);
        let r = t.oix1(&i);
        assert_eq!(r.shape(), &[2, 2, 3]);
        assert_eq!(r.ravel(), &[13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6]);

        // second index - one dimensional index tensor
        let i = CpuI32::new(&[2], &[1, 0]);
        let r = t.oix2(.., &i);
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
        let r = t.oix2(Ellipsis, &i);
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
        let r = t.oix1(&i);
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
        let r = t.oix3(&i0, &i1, &i2);
        assert_eq!(r.shape(), &[4, 2, 2]);
        assert_eq!(
            r.ravel(),
            &[18, 17, 15, 14, 6, 5, 3, 2, 12, 11, 9, 8, 24, 23, 21, 20]
        );

        // all indexes - two-dimensional index tensor
        let i0 = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let i1 = CpuI32::new(&[2], &[1, 0]);
        let i2 = CpuI32::new(&[2], &[2, 1]);
        let r = t.oix3(&i0, &i1, &i2);
        assert_eq!(r.shape(), &[2, 2, 2, 2]);
        assert_eq!(
            r.ravel(),
            &[18, 17, 15, 14, 6, 5, 3, 2, 12, 11, 9, 8, 24, 23, 21, 20]
        );

        // all indexes - all two-dimensional index tensor
        let i0 = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let i1 = CpuI32::new(&[1, 2], &[1, 0]);
        let i2 = CpuI32::new(&[2, 1], &[2, 1]);
        let r = t.oix3(&i0, &i1, &i2);
        assert_eq!(r.shape(), &[2, 2, 1, 2, 2, 1]);
        assert_eq!(
            r.ravel(),
            &[18, 17, 15, 14, 6, 5, 3, 2, 12, 11, 9, 8, 24, 23, 21, 20]
        );
    }

    #[test]
    fn test_oix_bool() {
        // first index - one dimensional index tensor
        let t = CpuI32::linspace(1, 24, 24u8).reshape(&[4, 2, 3]);
        let i = CpuBool::new(&[4], &[false, false, true, false]);
        let r = t.oix1(&i);
        assert_eq!(r.shape(), &[1, 2, 3]);
        assert_eq!(r.ravel(), &[13, 14, 15, 16, 17, 18]);

        // second index - one dimensional index tensor
        let i = CpuBool::new(&[2], &[true, false]);
        let r = t.oix2(.., &i);
        assert_eq!(r.shape(), &[4, 1, 3]);
        assert_eq!(r.ravel(), &[1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]);

        // third index - one dimensional index tensor
        let i = CpuBool::new(&[3], &[true, false, false]);
        let r = t.oix2(Ellipsis, &i);
        assert_eq!(r.shape(), &[4, 2, 1]);
        assert_eq!(r.ravel(), &[1, 4, 7, 10, 13, 16, 19, 22]);

        // first index - two dimensional index tensor
        let i = CpuBool::new(
            &[4, 2],
            &[false, false, true, false, false, false, true, false],
        );
        let r = t.oix1(&i);
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.ravel(), &[7, 8, 9, 19, 20, 21]);

        // second index - two dimensional index tensor
        let i = CpuBool::new(&[2, 3], &[true, false, true, false, false, false]);
        let r = t.oix2(.., &i);
        assert_eq!(r.shape(), &[4, 2]);
        assert_eq!(r.ravel(), &[1, 3, 7, 9, 13, 15, 19, 21]);

        // all indexes - two-dimensional index tensors
        let t = t.reshape(&[2, 2, 2, 3]);
        let i1 = CpuBool::new(&[2, 2], &[true, false, true, false]);
        let i2 = CpuBool::new(&[2, 3], &[false, false, true, false, true, false]);
        let r = t.oix2(&i1, &i2);
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.ravel(), &[3, 5, 15, 17]);
    }

    #[test]
    fn text_vix_int() {
        // first index - one dimensional index tensor
        let t = CpuI32::linspace(1, 24, 24u8).reshape(&[4, 2, 3]);
        let i = CpuI32::new(&[2], &[2, 0]);
        let r = t.vix1(&i);
        assert_eq!(r.shape(), &[2, 2, 3]);
        assert_eq!(r.ravel(), &[13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6]);

        // second index - one dimensional index tensor
        let i = CpuI32::new(&[2], &[1, 0]);
        let r = t.vix2(.., &i);
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
        let r = t.vix2(Ellipsis, &i);
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
        let r = t.vix1(&i);
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
        let r = t.vix3(&i0, &i1, &i2);
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.ravel(), &[18, 2]);

        // all indexes - all two-dimensional index tensor
        let i0 = CpuI32::new(&[2, 2], &[2, 0, 1, 3]);
        let i1 = CpuI32::new(&[1, 2], &[1, 0]);
        let i2 = CpuI32::new(&[2, 1], &[2, 1]);
        let r = t.vix3(&i0, &i1, &i2);
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
        let r = t.vix3(&i0, &i1, &i2);
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.ravel(), &[18, 2, 12, 20]);
    }

    #[test]
    fn test_vix_bool() {
        // first index - one dimensional index tensor
        let t = CpuI32::linspace(1, 24, 24u8).reshape(&[4, 2, 3]);
        let i = CpuBool::new(&[4], &[false, false, true, false]);
        let r = t.vix1(&i);
        assert_eq!(r.shape(), &[1, 2, 3]);
        assert_eq!(r.ravel(), &[13, 14, 15, 16, 17, 18]);

        // second index - one dimensional index tensor
        let i = CpuBool::new(&[2], &[true, false]);
        let r = t.vix2(.., &i);
        assert_eq!(r.shape(), &[4, 1, 3]);
        assert_eq!(r.ravel(), &[1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]);

        // third index - one dimensional index tensor
        let i = CpuBool::new(&[3], &[true, false, false]);
        let r = t.vix2(Ellipsis, &i);
        assert_eq!(r.shape(), &[4, 2, 1]);
        assert_eq!(r.ravel(), &[1, 4, 7, 10, 13, 16, 19, 22]);

        // first index - two dimensional index tensor
        let i = CpuBool::new(
            &[4, 2],
            &[false, false, true, false, false, false, true, false],
        );
        let r = t.vix1(&i);
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.ravel(), &[7, 8, 9, 19, 20, 21]);

        // second index - two dimensional index tensor
        let i = CpuBool::new(&[2, 3], &[true, false, true, false, false, false]);
        let r = t.vix2(.., &i);
        assert_eq!(r.shape(), &[4, 2]);
        assert_eq!(r.ravel(), &[1, 3, 7, 9, 13, 15, 19, 21]);

        // all indexes - two-dimensional index tensors
        let t = t.reshape(&[2, 2, 2, 3]);
        let i1 = CpuBool::new(&[2, 2], &[true, false, true, false]);
        let i2 = CpuBool::new(&[2, 3], &[false, false, true, false, true, false]);
        let r = t.vix2(&i1, &i2);
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.ravel(), &[3, 5, 15, 17]);
    }

    #[test]
    fn test_pytorch_bug() {
        // test case as reported in https://dev-discuss.pytorch.org/t/how-does-advanced-indexing-work-when-i-combine-a-tensor-and-a-single-index/558
        let x = Cpu32::new(&[2, 2], &[1., 2., 3., 4.]);
        let m = CpuBool::new(&[2], &[true, true]);

        // ezyang example
        let r1 = x.oix1(&m).oix1(0);
        let r2 = x.oix2(&m, 0);

        assert_eq!(r1.shape(), &[2]);
        assert_eq!(r1.ravel(), &[1., 2.]);

        assert_eq!(r2.shape(), &[2]);
        assert_eq!(r2.ravel(), &[1., 3.]);

        // example from bug report
        let x = CpuI32::new(&[5, 5, 2], &(0..50).collect::<Vec<_>>());
        let m = CpuBool::new(&[5, 5], &(0..25).map(|x| x % 2 == 0).collect::<Vec<_>>());

        let r = x.oix1(&m);
        assert_eq!(r.shape(), &[13, 2]);

        let r = x.oix2(&m, 0);
        assert_eq!(r.shape(), &[13]);
        assert_eq!(
            r.ravel(),
            &[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        );
        // println!("{x}");
        // println!("{m}");
        // println!("{}", x.oix1(&m));
        // println!("{}", x.oix3(.., .., 0));
        // println!("{}", x.oix3(.., .., 0).oix1(&m));
        // println!("{r}");
    }

    #[test]
    fn test_nep_21() {
        // Various examples from https://numpy.org/neps/nep-0021-advanced-indexing.html
        let arr = CpuI32::ones(&[5, 6, 7, 8]);

        // Outer indexing https://numpy.org/neps/nep-0021-advanced-indexing.html#id2
        // Multiple indices are “orthogonal” and their result axes are inserted at the same place (they are not broadcasted):
        let i1 = &CpuI32::new(&[1], &[0]);
        let i2 = &CpuI32::new(&[2], &[0, 1]);
        let r = arr.oix4(.., i1, i2, ..);
        assert_eq!(r.shape(), &[5, 1, 2, 8]);
        let r = arr.oix4(.., i1, .., i2);
        assert_eq!(r.shape(), &[5, 1, 7, 2]);
        let r = arr.oix4(.., i1, 0, ..);
        assert_eq!(r.shape(), &[5, 1, 8]);
        let r = arr.oix4(.., i1, .., 0);
        assert_eq!(r.shape(), &[5, 1, 7]);

        // Boolean indices results are always inserted where the index is:
        let mut bindx_vec = vec![false; 56];
        bindx_vec[0] = true;
        let bindx = &CpuBool::new(&[7, 8], &bindx_vec);
        let r = arr.oix3(.., 0, bindx);
        assert_eq!(r.shape(), &[5, 1]);
        let r = arr.oix3(0, .., bindx);
        assert_eq!(r.shape(), &[6, 1]);

        // Nothing changed in the presence of other advanced indices since:
        let r = arr.oix3(i1, .., bindx);
        assert_eq!(r.shape(), &[1, 6, 1]);
        let r = arr.oix3(.., i2, bindx);
        assert_eq!(r.shape(), &[5, 2, 1]);

        // Vectorized/inner indexing https://numpy.org/neps/nep-0021-advanced-indexing.html#vectorized-inner-indexing
        // Multiple indices are broadcasted and iterated as one like fancy indexing, but the new axes are always inserted at the front:
        let r = arr.vix4(.., i1, i2, ..);
        assert_eq!(r.shape(), &[2, 5, 8]);
        let r = arr.vix4(.., i1, .., i2);
        assert_eq!(r.shape(), &[2, 5, 7]);
        let r = arr.vix4(.., i1, 0, ..);
        assert_eq!(r.shape(), &[1, 5, 8]);
        let r = arr.vix4(.., i1, .., 0);
        assert_eq!(r.shape(), &[1, 5, 7]);

        // Boolean indices results are always inserted where the index is, exactly as in oindex given how specific they are to the axes they operate on:
        let r = arr.vix3(.., 0, bindx);
        assert_eq!(r.shape(), &[5, 1]);
        let r = arr.vix3(0, .., bindx);
        assert_eq!(r.shape(), &[6, 1]);

        // But other advanced indices are again transposed to the front:
        let r = arr.vix3(i1, .., bindx);
        assert_eq!(r.shape(), &[1, 6, 1]);
        let r = arr.vix3(.., i2, bindx);
        assert_eq!(r.shape(), &[2, 5, 1]);
    }
}
