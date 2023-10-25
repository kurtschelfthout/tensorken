use std::{
    fmt::{Display, Formatter},
    iter,
    ops::{Add, Div, Mul, Neg, Range, RangeFrom, RangeFull, RangeTo, Sub},
};

use prettytable::{format, Cell, Table};

use crate::{
    num::Num,
    raw_tensor::RawTensor,
    raw_tensor_cpu::CpuRawTensor,
    raw_tensor_fuse::Fuse,
    raw_tensor_shape_tracker::ShapeTracker,
    raw_tensor_wgpu::WgpuRawTensor,
    tensor_mut::TensorMut,
    {Diffable, DiffableExt},
};

// Blanket implementation to translate from diffable tensor ops (Diffable) to low-level tensor ops (RawTensor).
impl<T: Num, TTensor: RawTensor<Elem = T>> Diffable for TTensor {
    type Elem = T;

    fn log(&self) -> Self {
        self.log()
    }
    fn exp(&self) -> Self {
        self.exp()
    }

    fn elementwise_add(&self, other: &Self) -> Self {
        self.add(other)
    }

    fn elementwise_sub(&self, other: &Self) -> Self {
        self.sub(other)
    }

    fn elementwise_mul(&self, other: &Self) -> Self {
        self.mul(other)
    }

    fn elementwise_div(&self, other: &Self) -> Self {
        self.div(other)
    }

    fn elementwise_pow(&self, other: &Self) -> Self {
        self.pow(other)
    }

    fn elementwise_eq(&self, other: &Self) -> Self {
        self.eq(other)
    }

    fn sum(&self, axes: &[usize]) -> Self {
        self.sum(axes)
    }

    fn max(&self, axes: &[usize]) -> Self {
        self.max(axes)
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        self.reshape(shape)
    }

    fn permute(&self, dims: &[usize]) -> Self {
        self.permute(dims)
    }

    fn expand(&self, shape: &[usize]) -> Self {
        self.expand(shape)
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        self.pad(padding)
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        self.crop(limits)
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn new(shape: &[usize], data: &[Self::Elem]) -> Self {
        TTensor::new(shape, data)
    }

    fn ravel(&self) -> Vec<Self::Elem> {
        self.to_cpu().ravel()
    }
}

/// The "high-level" tensor type - the face of the library.
/// Tensors support arithmetic traits like Add, Sub, Neg to overload mathematical operators.
/// Unlike on `RawTensor`, all operations are broadcasted for convenience.
/// Also, we add higher-level operators to it like `matmul`.
/// All operations are ultimately implemented in terms of the `Diffable` trait, which due
/// to the blanket implementation above, get translated ultimately to `RawTensor` operations.
/// This is nice, because to implement a new type of accelerator, you only need to implement `RawTensor`.
#[derive(Debug, Clone)]
#[must_use]
pub struct Tensor<T>(T);

impl<T: Num, TRawTensor: RawTensor<Elem = T>> Tensor<TRawTensor> {
    pub fn to_tensor_mut(&self) -> TensorMut<T> {
        TensorMut::new(self)
    }

    pub fn to_cpu(&self) -> Tensor<CpuRawTensor<T>> {
        Tensor(self.0.to_cpu())
    }
}

impl<T: Diffable> Diffable for Tensor<T> {
    type Elem = T::Elem;

    /// Apply the natural logarithm to each element.
    fn log(&self) -> Self {
        Tensor(self.0.log())
    }

    /// Apply exp to each element.
    fn exp(&self) -> Self {
        Tensor(self.0.exp())
    }

    fn elementwise_add(&self, other: &Self) -> Self {
        Tensor(self.0.elementwise_add(&other.0))
    }

    fn elementwise_sub(&self, other: &Self) -> Self {
        Tensor(self.0.elementwise_sub(&other.0))
    }

    fn elementwise_mul(&self, other: &Self) -> Self {
        Tensor(self.0.elementwise_mul(&other.0))
    }

    fn elementwise_div(&self, other: &Self) -> Self {
        Tensor(self.0.elementwise_div(&other.0))
    }

    /// Raise self to the power of other, element-wise.
    fn elementwise_pow(&self, exp: &Self) -> Self {
        Tensor(self.0.elementwise_pow(&exp.0))
    }

    // Return a new tensor with ones where elements are equal, and zero otherwise.
    fn elementwise_eq(&self, other: &Self) -> Self {
        Tensor(self.0.elementwise_eq(&other.0))
    }

    /// Reduce to sum along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    fn sum(&self, axes: &[usize]) -> Self {
        Tensor(self.0.sum(axes))
    }

    /// Reduce to max element along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    fn max(&self, axes: &[usize]) -> Self {
        Tensor(self.0.max(axes))
    }

    /// Reshape the tensor to the given shape.
    /// The number of elements must remain the same.
    /// Compare to numpy's `reshape`.
    fn reshape(&self, shape: &[usize]) -> Self {
        Tensor(self.0.reshape(shape))
    }

    /// Changes axes around according to the given permutation.
    /// Compare to numpy's `transpose`.
    fn permute(&self, dims: &[usize]) -> Self {
        Tensor(self.0.permute(dims))
    }

    /// Expand the tensor to the given shape. Only dimensions of length 1 can be expanded.
    /// Like numpy's `broadcast_to` but simpler - does not add dimensions of size 1.
    fn expand(&self, shape: &[usize]) -> Self {
        Tensor(self.0.expand(shape))
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        Tensor(self.0.pad(padding))
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        Tensor(self.0.crop(limits))
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn new(shape: &[usize], data: &[Self::Elem]) -> Self {
        Tensor(T::new(shape, data))
    }

    fn ravel(&self) -> Vec<Self::Elem> {
        self.0.ravel()
    }
}

crate::math_macros::impl_bin_op!(Add, add, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Sub, sub, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Mul, mul, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Div, div, Tensor<T: Diffable>);

crate::math_macros::impl_un_op!(Neg, neg, Tensor<T: Diffable>);

pub type Cpu32 = Tensor<ShapeTracker<Fuse<CpuRawTensor<f32>>>>;
pub type Wgpu32<'d> = Tensor<ShapeTracker<Fuse<WgpuRawTensor<'d, f32>>>>;

fn create_table<T: Num + Display>(tensor: &Tensor<CpuRawTensor<T>>, table: &mut Table) {
    let shape = tensor.shape();
    if shape.len() == 2 {
        let format = format::FormatBuilder::new()
            .column_separator(' ')
            .left_border('[')
            .right_border(']')
            .separators(
                &[format::LinePosition::Top, format::LinePosition::Bottom],
                format::LineSeparator::new(' ', ' ', ' ', ' '),
            )
            .padding(1, 0)
            .build();
        table.set_format(format);

        for r in 0..shape[0] {
            let row = table.add_empty_row();
            for c in 0..shape[1] {
                row.add_cell(Cell::new(&format!("{}", tensor.at(&[r, c]))));
            }
        }
    } else {
        table.set_format(*format::consts::FORMAT_BORDERS_ONLY);
        for r in 0..shape[0] {
            let row = table.add_empty_row();
            for c in 0..shape[1] {
                let mut table = Table::new();
                let tensor = tensor.at(r).at(c);
                create_table(&tensor, &mut table);
                row.add_cell(Cell::new(&format!("{table}")));
            }
        }
    }
}

impl<RT: RawTensor> Display for Tensor<RT>
where
    RT::Elem: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let cpu = if self.shape().len() % 2 == 0 {
            self.to_cpu()
        } else {
            self.reshape(&[&[1], self.shape()].concat()).to_cpu()
        };

        let mut table = Table::new();
        create_table(&cpu, &mut table);
        write!(f, "{table}")
    }
}

/// A variation of `Index` and `IndexMut`, that returns the output
/// by value. Sadly, we can't use the standard Index trait, because
/// it requires that the output be a reference. But we want to be able
/// to return new tensors, which we can't give a lifetime long enough so
/// they can be returned from the index method.
/// This means we also can't use the actual [] syntax :( I made the name
/// as short as I could think of.
/// TODO: use a macro to generate the implementations for ranges etc, + variadic versions for more dimensions.
pub trait IndexValue<Idx> {
    type Output;

    fn at(&self, index: Idx) -> Self::Output;
}

impl<T: Diffable> IndexValue<usize> for T {
    type Output = Self;

    /// Slices the tensor at the given index, in the first dimension.
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

impl<T: Diffable, const N: usize> IndexValue<&[usize; N]> for Tensor<T> {
    type Output = Self;

    /// Returns a tensor containing a single value at the given index. There must be at most as many indices as dimensions.
    fn at(&self, index: &[usize; N]) -> Self::Output {
        let mut limits = index.iter().map(|&i| (i, i + 1)).collect::<Vec<_>>();
        limits.extend(iter::repeat((0, 0)).take(self.shape().len() - limits.len()));
        self.crop(&limits)
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

/// Specifies what to slice along each axis. Any axes at the end that are
/// omitted, are not sliced.
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

#[must_use]
pub fn sl() -> Slice {
    Slice::default()
}

pub fn sl1<T>(index: T) -> Slice
where
    Slice: SliceIdx<T>,
{
    sl().idx(index)
}

pub fn sl2<T1, T2>(index1: T1, index2: T2) -> Slice
where
    Slice: SliceIdx<T1> + SliceIdx<T2>,
{
    sl1(index1).idx(index2)
}

pub fn sl3<T1, T2, T3>(index1: T1, index2: T2, index3: T3) -> Slice
where
    Slice: SliceIdx<T1> + SliceIdx<T2> + SliceIdx<T3>,
{
    sl2(index1, index2).idx(index3)
}

pub fn sl4<T1, T2, T3, T4>(index1: T1, index2: T2, index3: T3, index4: T4) -> Slice
where
    Slice: SliceIdx<T1> + SliceIdx<T2> + SliceIdx<T3> + SliceIdx<T4>,
{
    sl3(index1, index2, index3).idx(index4)
}

impl<T: Diffable> IndexValue<Slice> for T {
    type Output = Self;

    /// Slice the tensor.
    fn at(&self, index: Slice) -> Self::Output {
        let mut limits = index.crop_limits(self.shape());
        limits.extend(iter::repeat((0, 0)).take(self.shape().len() - limits.len()));
        self.crop(&limits)
    }
}

/// One of two traits to make it easy to write generic functions over tensors.
// TODO: find a better name?
#[allow(clippy::module_name_repetitions)]
pub trait TensorLike<'a>:
    'a
    + Clone
    + DiffableExt
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Add<&'a Self, Output = Self>
    + Sub<&'a Self, Output = Self>
    + Div<&'a Self, Output = Self>
    + Mul<&'a Self, Output = Self>
{
}

impl<'a, T> TensorLike<'a> for T where
    Self: 'a
        + Clone
        + DiffableExt
        + Neg<Output = Self>
        + Add<Output = Self>
        + Sub<Output = Self>
        + Div<Output = Self>
        + Mul<Output = Self>
        + Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Div<&'a Self, Output = Self>
        + Mul<&'a Self, Output = Self>
{
}

/// One of two traits to make it easy to write generic functions over tensors,
/// that can be differentiated.
#[allow(clippy::module_name_repetitions)]
pub trait TensorLikeRef<T>:
    Sized
    + Neg<Output = T>
    + Add<Output = T>
    + Sub<Output = T>
    + Div<Output = T>
    + Mul<Output = T>
    + Add<T, Output = T>
    + Sub<T, Output = T>
    + Div<T, Output = T>
    + Mul<T, Output = T>
{
}

impl<'a, T> TensorLikeRef<T> for &'a T where
    Self: Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + Mul<T, Output = T>
{
}

#[cfg(test)]
mod tests {

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
}
