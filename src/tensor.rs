use std::{
    fmt::{Display, Formatter},
    iter,
    ops::{Add, Div, Mul, Neg, Range, RangeFrom, RangeFull, RangeTo, Sub},
    sync::Once,
};

use prettytable::{format, Cell, Table};

use crate::{
    num::Num,
    raw_tensor::{RawTensor, RealizedRawTensor},
    raw_tensor_cpu::CpuRawTensor,
    raw_tensor_fuse::Fuse,
    raw_tensor_shape_tracker::ShapeTracker,
    raw_tensor_wgpu::WgpuRawTensor,
    tensor_mut::TensorMut,
    Shape, {Diffable, DiffableExt},
};

// Blanket implementation to translate from diffable tensor ops (Diffable) to low-level tensor ops (RawTensor).
impl<T: Num, TTensor: RawTensor<E = T>> Diffable for TTensor {
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
        Self::new(shape, data)
    }
}

/// The "high-level" tensor type - the face of the library.
/// Tensors support arithmetic traits like Add, Sub, Neg to overload mathematical operators.
/// Unlike on [`RawTensor`], all operations are broadcasted for convenience.
/// Also, we add higher-level operators to it like `matmul`.
/// All operations are ultimately implemented in terms of the [`Diffable`] trait, which due
/// to the blanket implementation above, get translated ultimately to [`RawTensor`] operations.
/// This is nice, because to implement a new type of accelerator, you only need to implement [`RawTensor`].
#[derive(Debug, Clone)]
#[must_use]
pub struct Tensor<T>(T);

impl<T: Num, TRawTensor: RealizedRawTensor<E = T>> Tensor<TRawTensor> {
    /// Create a new mutable tensor with self's shape and elements.
    pub fn to_tensor_mut(&self) -> TensorMut<T> {
        TensorMut::new(self)
    }

    /// Create a new Tensor that has any lazy operations realized.
    pub fn realize(&self) -> Self {
        Self(self.0.realize())
    }

    /// Create a new [`CpuRawTensor`] with self's shape and elements.
    pub fn to_cpu(&self) -> Tensor<CpuRawTensor<T>> {
        Tensor(self.0.to_cpu())
    }

    pub fn ravel(&self) -> Vec<T> {
        self.0.to_cpu().ravel()
    }

    /// If the tensor has only one element, return it.
    /// # Panics
    /// If the tensor does not have exactly one element.
    pub fn to_scalar(&self) -> T {
        assert!(self.shape().size() == 1);
        self.ravel()[0]
    }

    /// Returns a new tensor with a new axis inserted at the front of size `num_classes`.
    /// All elements are assumed to be integers in the range [0, `num_classes`).
    /// The new axis is used as a one-hot encoding of the elements.
    pub fn one_hot<N: Into<usize>>(&self, num_classes: N) -> Self {
        let nc: usize = num_classes.into();
        let mut data = vec![T::ZERO; self.shape().size() * nc];
        for (i, &x) in self.ravel().iter().enumerate() {
            data[i * nc + x.to_usize()] = T::ONE;
        }
        let mut new_shape = vec![];
        new_shape.extend(self.shape());
        new_shape.push(nc);
        Self::new(&new_shape, &data)
    }
}

impl From<&Wgpu32<'static>> for Cpu32 {
    fn from(wgpu: &Wgpu32<'static>) -> Self {
        Tensor::new(wgpu.shape(), &wgpu.ravel())
    }
}

impl From<&Cpu32> for Wgpu32<'static> {
    fn from(wgpu: &Cpu32) -> Self {
        Tensor::new(wgpu.shape(), &wgpu.ravel())
    }
}

impl<T: Diffable> Diffable for Tensor<T> {
    type Elem = T::Elem;

    /// Apply the natural logarithm to each element.
    fn log(&self) -> Self {
        Self(self.0.log())
    }

    /// Apply exp to each element.
    fn exp(&self) -> Self {
        Self(self.0.exp())
    }

    fn elementwise_add(&self, other: &Self) -> Self {
        Self(self.0.elementwise_add(&other.0))
    }

    fn elementwise_sub(&self, other: &Self) -> Self {
        Self(self.0.elementwise_sub(&other.0))
    }

    fn elementwise_mul(&self, other: &Self) -> Self {
        Self(self.0.elementwise_mul(&other.0))
    }

    fn elementwise_div(&self, other: &Self) -> Self {
        Self(self.0.elementwise_div(&other.0))
    }

    /// Raise self to the power of other, element-wise.
    fn elementwise_pow(&self, exp: &Self) -> Self {
        Self(self.0.elementwise_pow(&exp.0))
    }

    // Return a new tensor with ones where elements are equal, and zero otherwise.
    fn elementwise_eq(&self, other: &Self) -> Self {
        Self(self.0.elementwise_eq(&other.0))
    }

    /// Reduce to sum along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    fn sum(&self, axes: &[usize]) -> Self {
        Self(self.0.sum(axes))
    }

    /// Reduce to max element along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    fn max(&self, axes: &[usize]) -> Self {
        Self(self.0.max(axes))
    }

    /// Reshape the tensor to the given shape.
    /// The number of elements must remain the same.
    /// Compare to numpy's `reshape`.
    fn reshape(&self, shape: &[usize]) -> Self {
        Self(self.0.reshape(shape))
    }

    /// Changes axes around according to the given permutation.
    /// Compare to numpy's `transpose`.
    fn permute(&self, dims: &[usize]) -> Self {
        Self(self.0.permute(dims))
    }

    /// Expand the tensor to the given shape. Only dimensions of length 1 can be expanded.
    /// Like numpy's `broadcast_to` but simpler - does not add dimensions of size 1.
    fn expand(&self, shape: &[usize]) -> Self {
        Self(self.0.expand(shape))
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        Self(self.0.pad(padding))
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        Self(self.0.crop(limits))
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn new(shape: &[usize], data: &[Self::Elem]) -> Self {
        Self(T::new(shape, data))
    }
}

crate::math_macros::impl_bin_op!(Add, add, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Sub, sub, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Mul, mul, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Div, div, Tensor<T: Diffable>);

crate::math_macros::impl_un_op!(Neg, neg, Tensor<T: Diffable>);

pub type Cpu32 = Tensor<ShapeTracker<Fuse<CpuRawTensor<f32>>>>;
pub type Wgpu32<'d> = Tensor<ShapeTracker<Fuse<WgpuRawTensor<'d, f32>>>>;

static mut FORMAT_TENSOR: Option<format::TableFormat> = None;
static INIT_FORMAT_TENSOR: Once = Once::new();

/// Returns a reference to the global wgpu context, creating it if necessary.
fn get_pretty_format() -> &'static format::TableFormat {
    unsafe {
        INIT_FORMAT_TENSOR.call_once(|| {
            FORMAT_TENSOR = Some(
                format::FormatBuilder::new()
                    .column_separator(' ')
                    .borders('│')
                    .separators(
                        &[format::LinePosition::Top],
                        format::LineSeparator::new(' ', ' ', '┌', '┐'),
                    )
                    .separators(
                        &[format::LinePosition::Bottom],
                        format::LineSeparator::new(' ', ' ', '└', '┘'),
                    )
                    .padding(1, 1)
                    .build(),
            );
        });
        return FORMAT_TENSOR.as_ref().unwrap();
    }
}

static mut FORMAT_TENSOR_SINGLE_LINE: Option<format::TableFormat> = None;
static INIT_FORMAT_TENSOR_SINGLE_LINE: Once = Once::new();

fn get_single_line_format() -> &'static format::TableFormat {
    unsafe {
        INIT_FORMAT_TENSOR_SINGLE_LINE.call_once(|| {
            FORMAT_TENSOR_SINGLE_LINE = Some(
                format::FormatBuilder::new()
                    .column_separator(' ')
                    .left_border('[')
                    .right_border(']')
                    .padding(1, 0)
                    .build(),
            );
        });
        return FORMAT_TENSOR_SINGLE_LINE.as_ref().unwrap();
    }
}

// static mut FORMAT_TENSOR_NUMPY: Option<format::TableFormat> = None;
// static INIT_FORMAT_TENSOR_NUMPY: Once = Once::new();

// fn get_numpy_format() -> &'static format::TableFormat {
//     unsafe {
//         INIT_FORMAT_TENSOR_NUMPY.call_once(|| {
//             FORMAT_TENSOR_NUMPY = Some(
//                 format::FormatBuilder::new()
//                     .column_separator(' ')
//                     .left_border('[')
//                     .right_border(']')
//                     .separators(
//                         &[format::LinePosition::Top, format::LinePosition::Bottom],
//                         format::LineSeparator::new(' ', ' ', ' ', ' '),
//                     )
//                     .padding(1, 0)
//                     .build(),
//             );
//         });
//         return FORMAT_TENSOR_NUMPY.as_ref().unwrap();
//     }
// }

fn create_table<T: Num + Display>(
    tensor: &Tensor<CpuRawTensor<T>>,
    table: &mut Table,
    precision: Option<usize>,
) {
    let shape = tensor.shape();

    if shape.len() == 2 {
        table.set_format(if shape[0] == 1 {
            *get_single_line_format()
        } else {
            *get_pretty_format()
        });
        for r in 0..shape[0] {
            let row = table.add_empty_row();
            for c in 0..shape[1] {
                if precision.is_some() {
                    row.add_cell(Cell::new(&format!(
                        "{:.precision$}",
                        tensor.at(&[r, c]).to_scalar(),
                        precision = precision.unwrap()
                    )));
                } else {
                    row.add_cell(Cell::new(&format!("{}", tensor.at(&[r, c]).to_scalar(),)));
                }
            }
        }
    } else {
        table.set_format(*get_pretty_format());
        for r in 0..shape[0] {
            let row = table.add_empty_row();
            for c in 0..shape[1] {
                let mut table = Table::new();
                let tensor = tensor.at(&[r, c]);
                create_table(&tensor, &mut table, precision);
                row.add_cell(Cell::new(&format!("{table}")));
            }
        }
    }
}

impl<RT: RealizedRawTensor> Display for Tensor<RT>
where
    RT::E: Num + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let cpu = if self.shape().len() % 2 == 0 {
            self.to_cpu()
        } else {
            self.reshape(&[&[1], self.shape()].concat()).to_cpu()
        };

        let mut table = Table::new();
        create_table(&cpu, &mut table, f.precision());
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
pub trait IndexValue<Idx> {
    type Output;

    fn at(&self, index: Idx) -> Self::Output;
}

impl<T: Diffable> IndexValue<usize> for T {
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

impl<T: Diffable, const N: usize> IndexValue<&[usize; N]> for Tensor<T> {
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

impl<T: Diffable> IndexValue<Slice> for T {
    type Output = Self;

    /// Slice the tensor.
    fn at(&self, index: Slice) -> Self::Output {
        let mut limits = index.crop_limits(self.shape());
        limits.extend(iter::repeat((0, 0)).take(self.shape().len() - limits.len()));
        self.crop(&limits)
    }
}

/// One of two traits to make it easy to write differentiable functions.
/// The other one is [`TensorLikeRef`].
#[allow(clippy::module_name_repetitions)]
pub trait TensorLike<'a>:
    'a
    + Clone
    + Diffable
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
        + Diffable
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

/// One of two traits to make it easy to write differentiable functions.
/// The other one is [`TensorLike`].
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
