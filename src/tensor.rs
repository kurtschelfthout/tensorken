use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Add, Div, Mul, Neg, Sub},
};

use prettytable::{format, Cell, Table};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::{
    diffable::{Diffable, DiffableExt},
    num::Num,
    raw_tensor::RawTensor,
    raw_tensor_cpu::CpuRawTensor,
    raw_tensor_wgpu::WgpuRawTensor,
    shape::Shape,
    tensor_mut::TensorMut,
};

// Blanket implementation to translate from diffable tensor ops (Diffable) to low-level tensor ops (RawTensor).
impl<T: Num, TTensor: RawTensor<Elem = T>> Diffable for TTensor {
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

    fn zeros_like(&self) -> Self {
        TTensor::new(&vec![1; self.shape().ndims()], &[T::ZERO]).expand(self.shape())
    }

    fn ones_like(&self) -> Self {
        TTensor::new(&vec![1; self.shape().ndims()], &[T::ONE]).expand(self.shape())
    }

    fn shape(&self) -> &[usize] {
        self.shape()
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
pub struct Tensor<TRawTensor>(TRawTensor);

impl<T: Num, TRawTensor: RawTensor<Elem = T>> Tensor<TRawTensor> {
    /// Create a new tensor with the given shape and elements.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    pub fn new(shape: &[usize], data: &[T]) -> Self {
        Tensor(TRawTensor::new(shape, data))
    }

    pub fn scalar(value: T) -> Self {
        Tensor(TRawTensor::new(&[1], &[value]))
    }

    /// Create a new tensor with the given shape, and fill it with the given value.
    pub fn full(shape: &[usize], value: T) -> Self {
        Tensor(TRawTensor::new(&vec![1; shape.ndims()], &[value])).expand(shape)
    }

    /// Create a new tensor with the given shape, and fill it with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        Tensor::full(shape, T::ZERO)
    }

    /// Create a new 2-dimensional tensor with the ones the diagonal and zeros elsewhere.
    pub fn eye(dim: usize) -> Self {
        // kind of an pad/crop/expand/reshape stress test
        Self::scalar(T::ONE)
            .pad(&[(0, dim)])
            .reshape(&[1, dim + 1])
            .expand(&[dim, dim + 1])
            .reshape(&[dim * (dim + 1)])
            .crop(&[(0, dim * dim)])
            .reshape(&[dim, dim])
    }

    pub fn linspace(start: T, end: T, num: usize) -> Self {
        let mut data = Vec::with_capacity(num);
        let step = if num > 1 {
            let nf: T = T::from_usize(num);
            (end - start) / (nf - T::ONE)
        } else {
            T::ZERO
        };
        for i in 0..num {
            data.push(start + step * T::from_usize(i));
        }
        Self::new(&[num], &data)
    }

    /// Create a new tensor with the given shape, and fill it with random values from a standard normal distribution
    pub fn randn<R>(shape: &[usize], rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        rand_distr::StandardNormal: Distribution<T>,
    {
        // let normal = StandardNormal //.unwrap();
        let mut data: Vec<T> = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            data.push(rng.sample(StandardNormal));
        }

        Self::new(shape, &data)
    }

    /// Create a new tensor with the same shape as self, but all elements equal to given value.
    pub fn constant_like(&self, value: T) -> Self {
        Tensor::full(self.0.shape(), value)
    }

    /// Return the elements of the tensor as a Vec, i.e. on the CPU.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    pub fn ravel(&self) -> Vec<T> {
        self.0.ravel()
    }

    pub fn to_tensor_mut(&self) -> TensorMut<T> {
        TensorMut::new(self)
    }

    /// If the tensor has only one element, return it.
    /// # Panics
    /// If the tensor does not have exactly one element.
    pub fn to_scalar(&self) -> T {
        assert!(self.0.shape().size() == 1);
        self.0.ravel()[0]
    }

    pub fn to_cpu(&self) -> Tensor<CpuRawTensor<T>> {
        Tensor(self.0.to_cpu())
    }

    // fn fused_multiply_add(&self, other: &Self, axes: &[usize]) -> Self {
    //     self.broadcasted_apply(
    //         other,
    //         |a, b| Tensor(a.0.fused_multiply_add(&b.0, axes)),
    //         false,
    //     )
    // }
}

impl<T: Diffable> Diffable for Tensor<T> {
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

    fn zeros_like(&self) -> Self {
        Tensor(self.0.zeros_like())
    }

    fn ones_like(&self) -> Self {
        Tensor(self.0.ones_like())
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
}

crate::math_macros::impl_bin_op!(Add, add, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Sub, sub, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Mul, mul, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Div, div, Tensor<T: Diffable>);

crate::math_macros::impl_un_op!(Neg, neg, Tensor<T: Diffable>);

pub type Cpu32 = Tensor<CpuRawTensor<f32>>;
pub type Wgpu32<'d> = Tensor<WgpuRawTensor<'d, f32>>;

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
/// TODO: use a macro to generate the implementations for
/// ranges etc, + variadic versions for more dimensions.
pub trait IndexValue<Idx> {
    type Output;

    fn at(&self, index: Idx) -> Self::Output;
}

// TODO: this at should go to DiffableExt, as it can and should work for any Diffable.
impl<RT: RawTensor> IndexValue<usize> for Tensor<RT> {
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

// This at can only work for RawTensors, because it returns a scalar using to_scalar.
impl<RT: RawTensor, const N: usize> IndexValue<&[usize; N]> for Tensor<RT> {
    type Output = RT::Elem;

    /// Returns the value at the given index. There must be as many indices as there are dimensions.
    fn at(&self, index: &[usize; N]) -> Self::Output {
        let limits = index.iter().map(|&i| (i, i + 1)).collect::<Vec<_>>();
        self.crop(&limits).to_scalar()
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
