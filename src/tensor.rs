use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{
    num::{Float, Num, ZeroOne},
    raw_tensor::{CastInto, RawTensor, RealizedRawTensor},
    raw_tensor_cpu::CpuRawTensor,
    raw_tensor_fuse::Fuse,
    raw_tensor_shape_tracker::ShapeTracker,
    raw_tensor_wgpu::WgpuRawTensor,
    tensor_mut::TensorMut,
    Diffable, DiffableExt, Shape,
};

// Blanket implementation to translate from diffable tensor ops (Diffable) to low-level tensor ops (RawTensor).
impl<E, T: RawTensor<E = E>> Diffable for T {
    type Elem = E;

    fn log(&self) -> Self
    where
        Self::Elem: Float,
    {
        self.log()
    }

    fn exp(&self) -> Self
    where
        Self::Elem: Float,
    {
        self.exp()
    }

    fn elementwise_add(&self, other: &Self) -> Self
    where
        Self::Elem: Num,
    {
        self.add(other)
    }

    fn elementwise_sub(&self, other: &Self) -> Self
    where
        Self::Elem: Num,
    {
        self.sub(other)
    }

    fn elementwise_mul(&self, other: &Self) -> Self
    where
        Self::Elem: Num,
    {
        self.mul(other)
    }

    fn elementwise_div(&self, other: &Self) -> Self
    where
        Self::Elem: Num,
    {
        self.div(other)
    }

    fn elementwise_pow(&self, other: &Self) -> Self
    where
        Self::Elem: Float,
    {
        self.pow(other)
    }

    fn elementwise_eq(&self, other: &Self) -> Self
    where
        Self::Elem: ZeroOne,
    {
        self.eq(other)
    }

    fn sum(&self, axes: &[usize]) -> Self
    where
        Self::Elem: Num,
    {
        self.sum(axes)
    }

    fn max(&self, axes: &[usize]) -> Self
    where
        Self::Elem: ZeroOne,
    {
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

    fn pad(&self, padding: &[(usize, usize)]) -> Self
    where
        Self::Elem: ZeroOne,
    {
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

impl<T: Copy, TRawTensor: RealizedRawTensor<E = T>> Tensor<TRawTensor> {
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
    pub fn one_hot<N: Into<usize>>(&self, num_classes: N) -> Self
    where
        T: Num,
    {
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

impl<T: Diffable> Diffable for Tensor<T> {
    type Elem = T::Elem;

    /// Apply the natural logarithm to each element.
    fn log(&self) -> Self
    where
        Self::Elem: Float,
    {
        Self(self.0.log())
    }

    /// Apply exp to each element.
    fn exp(&self) -> Self
    where
        Self::Elem: Float,
    {
        Self(self.0.exp())
    }

    fn elementwise_add(&self, other: &Self) -> Self
    where
        Self::Elem: Num,
    {
        Self(self.0.elementwise_add(&other.0))
    }

    fn elementwise_sub(&self, other: &Self) -> Self
    where
        Self::Elem: Num,
    {
        Self(self.0.elementwise_sub(&other.0))
    }

    fn elementwise_mul(&self, other: &Self) -> Self
    where
        Self::Elem: Num,
    {
        Self(self.0.elementwise_mul(&other.0))
    }

    fn elementwise_div(&self, other: &Self) -> Self
    where
        Self::Elem: Num,
    {
        Self(self.0.elementwise_div(&other.0))
    }

    /// Raise self to the power of other, element-wise.
    fn elementwise_pow(&self, exp: &Self) -> Self
    where
        Self::Elem: Float,
    {
        Self(self.0.elementwise_pow(&exp.0))
    }

    // Return a new tensor with ones where elements are equal, and zero otherwise.
    fn elementwise_eq(&self, other: &Self) -> Self
    where
        Self::Elem: ZeroOne,
    {
        Self(self.0.elementwise_eq(&other.0))
    }

    /// Reduce to sum along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    fn sum(&self, axes: &[usize]) -> Self
    where
        Self::Elem: Num,
    {
        Self(self.0.sum(axes))
    }

    /// Reduce to max element along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    fn max(&self, axes: &[usize]) -> Self
    where
        Self::Elem: ZeroOne,
    {
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

    fn pad(&self, padding: &[(usize, usize)]) -> Self
    where
        Self::Elem: ZeroOne,
    {
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

impl<TTo, TFro> CastInto<Tensor<TTo>> for Tensor<TFro>
where
    TFro: CastInto<TTo>,
{
    fn cast(&self) -> Tensor<TTo> {
        Tensor(self.0.cast())
    }
}

crate::math_macros::impl_bin_op!(Add, add, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Sub, sub, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Mul, mul, Tensor<T: Diffable>);
crate::math_macros::impl_bin_op!(Div, div, Tensor<T: Diffable>);

crate::math_macros::impl_un_op!(Neg, neg, Tensor<T: Diffable>);

pub type Cpu32 = Tensor<ShapeTracker<Fuse<CpuRawTensor<f32>>>>;
pub type CpuI32 = Tensor<ShapeTracker<Fuse<CpuRawTensor<i32>>>>;
pub type CpuBool = Tensor<CpuRawTensor<bool>>;
pub type Wgpu32<'d> = Tensor<ShapeTracker<Fuse<WgpuRawTensor<'d, f32>>>>;
// pub type WgpuI32<'d> = Tensor<ShapeTracker<Fuse<WgpuRawTensor<'d, i32>>>>;
// pub type WgpuBool<'d> = Tensor<WgpuRawTensor<'d, bool>>;

impl From<&Wgpu32<'static>> for Cpu32 {
    fn from(wgpu: &Wgpu32<'static>) -> Self {
        Tensor::new(wgpu.shape(), &wgpu.ravel())
    }
}

impl From<&Cpu32> for Wgpu32<'static> {
    fn from(cpu: &Cpu32) -> Self {
        Tensor::new(cpu.shape(), &cpu.ravel())
    }
}

impl CastInto<CpuI32> for Tensor<CpuRawTensor<bool>> {
    fn cast(&self) -> CpuI32 {
        let c: Tensor<CpuRawTensor<i32>> = self.cast();
        Tensor::new(c.shape(), &c.ravel())
    }
}

impl CastInto<Cpu32> for Tensor<CpuRawTensor<bool>> {
    fn cast(&self) -> Cpu32 {
        let c: Tensor<CpuRawTensor<f32>> = self.cast();
        Tensor::new(c.shape(), &c.ravel())
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
