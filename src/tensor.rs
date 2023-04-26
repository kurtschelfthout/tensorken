use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

use bytemuck::Pod;

use crate::{
    num::Num, raw_tensor::RawTensor, raw_tensor_cpu::CpuRawTensor, raw_tensor_wgpu::WgpuRawTensor,
    shape_strider::Shape, tensor_mut::TensorMut,
};

/// The "high-level" tensor type - the face of the library.
/// Tensors support arithmetic traits like Add, Sub, Neg to overload mathematical operators.
/// Unlike on `RawTensor`, all operations are broadcasted for convenience.
/// Also, we add higher-level operators to it like `matmul`.
/// All operations are ultimately implemented in terms of the `RawTensor` trait - this is nice,
/// because to implement a new type of accelerator, you only need to implement `RawTensor`.
#[derive(Debug, Clone)]
#[must_use]
pub struct Tensor<TRawTensor>(TRawTensor);

impl<T: Copy + Num, TRawTensor: RawTensor<Elem = T>> Tensor<TRawTensor> {
    /// Create a new tensor with the given shape and elements.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    pub fn new(shape: &[usize], data: &[T]) -> Self {
        Tensor(TRawTensor::new(shape, data))
    }

    /// Create a new tensor with the given shape, and fill it with the given value.
    pub fn full(shape: &[usize], value: T) -> Self {
        Tensor(TRawTensor::new(&vec![1; shape.ndims()], &[value])).expand(shape)
    }

    /// Create a new tensor with the given shape, and fill it with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        Tensor::full(shape, T::ZERO)
    }

    /// Create a new tensor with the same shape as self, but all elements equal to given value.
    pub fn constant_like(&self, value: T) -> Self {
        Tensor::full(self.0.shape(), value)
    }

    /// Create a new tensor with the same shape as self, but all elements equal to zero.
    pub fn zeros_like(&self) -> Self {
        self.constant_like(T::ZERO)
    }

    /// Return the elements of the tensor as a Vec, i.e. on the CPU.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    pub fn ravel(&self) -> Vec<T> {
        self.0.ravel()
    }

    fn add(&self, other: &Self) -> Self {
        self.broadcasted_apply(other, |a, b| Tensor(a.0.add(&b.0)), false)
    }

    fn mul(&self, other: &Self) -> Self {
        self.broadcasted_apply(other, |a, b| Tensor(a.0.mul(&b.0)), false)
    }

    fn sub(&self, other: &Self) -> Self {
        self.broadcasted_apply(other, |a, b| Tensor(a.0.sub(&b.0)), false)
    }

    fn div(&self, other: &Self) -> Self {
        self.broadcasted_apply(other, |a, b| Tensor(a.0.div(&b.0)), false)
    }

    /// Raise self to the power of other, element-wise.
    pub fn pow(&self, exp: &Self) -> Self {
        self.broadcasted_apply(exp, |a, b| Tensor(a.0.pow(&b.0)), false)
    }

    // Return a new tensor with ones where elements are equal, and zero otherwise.
    pub fn eq(&self, other: &Self) -> Self {
        self.broadcasted_apply(other, |a, b| Tensor(a.0.eq(&b.0)), false)
    }

    /// Apply the natural logarithm to each element.
    pub fn log(&self) -> Self {
        Tensor(self.0.log())
    }

    /// Apply exp to each element.
    pub fn exp(&self) -> Self {
        Tensor(self.0.exp())
    }

    /// Reduce to sum along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    pub fn sum(&self, axes: &[usize]) -> Self {
        Tensor(self.0.sum(axes))
    }

    /// Reduce to max element along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    pub fn max(&self, axes: &[usize]) -> Self {
        Tensor(self.0.max(axes))
    }

    /// Reshape the tensor to the given shape.
    /// The number of elements must remain the same.
    /// Compare to numpy's `reshape`.
    pub fn reshape(&self, shape: &[usize]) -> Self {
        Tensor(self.0.reshape(shape))
    }

    /// Changes axes around according to the given permutation.
    /// Compare to numpy's `transpose`.
    pub fn permute(&self, dims: &[usize]) -> Self {
        Tensor(self.0.permute(dims))
    }

    /// Expand the tensor to the given shape. Only dimensions of length 1 can be expanded.
    /// Like numpy's `broadcast_to` but simpler - does not add dimensions of size 1.
    pub fn expand(&self, shape: &[usize]) -> Self {
        Tensor(self.0.expand(shape))
    }

    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    pub fn to_tensor_mut(&self) -> TensorMut<T> {
        TensorMut::new(self)
    }
}

macro_rules! impl_difftensor_tensor {
    ($op_trait:ident, $op_fn:ident) => {
        impl<T: RawTensor> $op_trait<&Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                Tensor::<_>::$op_fn(self, rhs)
            }
        }

        impl<T: RawTensor> $op_trait<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                Self::$op_fn(&self, rhs)
            }
        }

        impl<T: RawTensor> $op_trait<Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                Self::$op_fn(&self, &rhs)
            }
        }

        impl<T: RawTensor> $op_trait<Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                Self::$op_fn(self, &rhs)
            }
        }
    };
}

impl_difftensor_tensor!(Add, add);
impl_difftensor_tensor!(Sub, sub);
impl_difftensor_tensor!(Mul, mul);
impl_difftensor_tensor!(Div, div);

impl<T: RawTensor> Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
        self.zeros_like().sub(self)
    }
}

impl<T: RawTensor> Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
        self.zeros_like().sub(self)
    }
}

impl<TRawTensor: RawTensor> Tensor<TRawTensor> {
    fn broadcasted_apply(
        &self,
        other: &Self,
        f: impl Fn(&Self, &Self) -> Self,
        reverse: bool,
    ) -> Self {
        if self.shape().ndims() > other.shape().ndims() {
            // Rust tidbit: I originally did not have a reverse parameter,
            // but just called |a,b| f(b,a) in the recursive call. This doesn't work,
            // because it hits the recursion limit: https://stackoverflow.com/questions/54613966/error-reached-the-recursion-limit-while-instantiating-funcclosure
            return other.broadcasted_apply(self, f, !reverse);
        }

        if self.shape().ndims() == other.shape().ndims() {
            let res_shape = self
                .shape()
                .iter()
                .zip(other.shape().iter())
                .map(|(a, b)| *a.max(b))
                .collect::<Vec<_>>();
            let s_expanded = self.expand(&res_shape);
            let o_expanded = other.expand(&res_shape);
            if reverse {
                return f(&o_expanded, &s_expanded);
            }
            return f(&s_expanded, &o_expanded);
        }

        let num_ones_to_add = other.shape().len().saturating_sub(self.shape().len());
        let mut new_shape = vec![1; num_ones_to_add];
        new_shape.extend(self.shape());

        self.reshape(&new_shape)
            .broadcasted_apply(other, f, reverse)
    }

    /// Switch the two axes around.
    pub fn transpose(&self, axis0: usize, axis1: usize) -> Self {
        let mut axes = (0..self.shape().ndims()).collect::<Vec<_>>();
        axes.swap(axis0, axis1);
        self.permute(&axes)
    }

    /// Matrix multiplication, generalized to tensors.
    /// i.e. multiply [..., m, n] with [..., n, o] to [..., m, o]
    pub fn matmul(&self, other: &Self) -> Self {
        // self's shape from [..., m, n] to [..., m, 1, n]
        // using just reshape.
        let shape = self.shape();
        let self_shape = [
            &shape[..shape.ndims() - 1],
            &[1],
            &[shape[shape.ndims() - 1]],
        ]
        .concat();
        let l = self.reshape(&self_shape);

        // other's shape from [..., n, o] to [..., 1, o, n]
        // using reshape + transpose.
        let shape = other.shape();
        let other_shape = [
            &shape[..shape.ndims() - 2],
            &[1],
            &shape[shape.ndims() - 2..],
        ]
        .concat();
        let r = other
            .reshape(&other_shape)
            .transpose(other_shape.ndims() - 1, other_shape.ndims() - 2);

        // after multiply: [..., m, o, n]
        // after sum:      [..., m, o, 1]
        let summed = (l * r).sum(&[other_shape.ndims() - 1]);
        // after reshape:  [..., m, o]
        let s = summed.shape();
        summed.reshape(&s[..s.ndims() - 1])
    }
}

impl<T: Copy + Num> Tensor<CpuRawTensor<T>> {
    pub fn new_cpu(shape: &[usize], data: &[T]) -> Self {
        Tensor(CpuRawTensor::new(shape, data))
    }
}

impl<'d, T: Copy + Num + Pod> Tensor<WgpuRawTensor<'d, T>> {
    pub fn new_wgpu(shape: &[usize], data: &[T]) -> Self {
        Tensor(<WgpuRawTensor<T> as RawTensor>::new(shape, data))
    }
}
