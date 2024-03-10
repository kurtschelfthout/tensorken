use crate::num::{Float, Num, ZeroOne};

/// Counterpart for tinygrad's "low-level" operations in ops.py.
/// Represents the operations that a tensor implementation, be it on CPU or GPU, must implement.
/// All tensor operations in mlops are eventually translated to these operations.
/// As such it can be used to implement a new type of accelerator, but can also support
/// optimizations like fusing.
/// Think of `RawTensor` as the DSL for accelerators, in final style.
pub trait RawTensor {
    // Note: Elem is an associated type, not a generic parameter, for rather subtle reasons.
    // We often want to implement traits for e.g. Tensor<impl RawTensor> without having to mention
    // the element type, as the element type is not restricted by the implementation. See e.g. Add, Neg on Tensor:
    // impl<RT: RawTensor> Add for Tensor<RT> { ... }
    // If RawTensor would have a generic type, we'd have to mention it in the impl:
    // impl<T, RT: RawTensor<T>> Add for Tensor<RT> { ... }
    // but then rust complains that T is not restricted by the implementation.
    type E;

    // unary ops
    // ---------

    /// Apply exp to each element.
    #[must_use]
    fn exp(&self) -> Self
    where
        Self::E: Float;

    /// Apply the natural logarithm to each element.
    #[must_use]
    fn log(&self) -> Self
    where
        Self::E: Float;

    // binary ops
    // ----------

    /// Add self to other, element-wise.
    #[must_use]
    fn add(&self, other: &Self) -> Self
    where
        Self::E: Num;

    /// Subtract other from self, element-wise.
    #[must_use]
    fn sub(&self, other: &Self) -> Self
    where
        Self::E: Num;

    /// Multiply self by other, element-wise.
    #[must_use]
    fn mul(&self, other: &Self) -> Self
    where
        Self::E: Num;

    /// Divide self by other,  element-wise.
    #[must_use]
    fn div(&self, other: &Self) -> Self
    where
        Self::E: Num;

    /// Raise self to the power of other, element-wise.
    #[must_use]
    fn pow(&self, other: &Self) -> Self
    where
        Self::E: Float;

    /// Return a new tensor with ones where elements are equal, and zero otherwise.
    #[must_use]
    fn eq(&self, other: &Self) -> Self
    where
        Self::E: ZeroOne;

    // reduce ops
    // ----------

    #[must_use]
    fn sum(&self, axes: &[usize]) -> Self
    where
        Self::E: Num;

    /// Find the maximum of elements along the given axes.
    #[must_use]
    fn max(&self, axes: &[usize]) -> Self
    where
        Self::E: ZeroOne;

    // movement ops
    // ------------

    /// Reshape the tensor to the given shape.
    /// The number of elements must remain the same.
    /// Compare to numpy's `reshape`.
    #[must_use]
    fn reshape(&self, shape: &[usize]) -> Self;

    /// Changes axes around according to the given permutation.
    /// Compare to numpy's `transpose`.
    #[must_use]
    fn permute(&self, permutation: &[usize]) -> Self;

    /// Expand the tensor to the given shape. Only dimensions of length 1 can be expanded.
    /// Like numpy's `broadcast_to` but simpler - does not add dimensions of size 1.
    #[must_use]
    fn expand(&self, shape: &[usize]) -> Self;

    // Not yet implemented:
    // fn stride(&self, strides: &[usize]) -> Self;

    /// Pad the tensor with zeros according to the given padding.
    /// Like numpy's `pad`, but simpler - needs as many elements in `padding` as there
    /// are dimensions in the tensor.
    #[must_use]
    fn pad(&self, padding: &[(usize, usize)]) -> Self
    where
        Self::E: ZeroOne;

    /// Crop the tensor according to the given limits - taking a contiguous slice in each axis.
    /// Needs as many limits as there are dimensions in the tensor.
    #[must_use]
    fn crop(&self, limits: &[(usize, usize)]) -> Self;

    // creation
    // --------

    /// Create a new tensor with the given shape and elements.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    fn new(shape: &[usize], data: &[Self::E]) -> Self;

    // elimination
    // -----------

    /// Return the shape of the tensor.
    fn shape(&self) -> &[usize];

    // fused operations
    // ----------------

    /// Multiply self with other element-wise, and sum-reduce the given dimensions, in one fused
    /// operation.
    /// This operation is in `RawTensor` for performance reasons, as clearly functionally it is equivalent to `mul` + `sum`.
    /// However, usually hardware have specialized instructions for vectorized fused multiply-accumulate,
    /// or fused multiple-add, e.g. fma instruction in WebGPU. Also, for matrix multiplication, this avoids
    /// allocation of a typically large intermediate result tensor which holds the results of the un-accumulated
    /// element-wise multiplication, and so is essential.
    #[must_use]
    fn fused_multiply_add(&self, other: &Self, axes: &[usize]) -> Self
    where
        Self::E: Num;
}

#[allow(clippy::module_name_repetitions)]
pub trait RealizedRawTensor: RawTensor {
    /// Return the tensor on the CPU.
    fn to_cpu(&self) -> crate::CpuRawTensor<Self::E>;

    /// For tensors that have lazy operations, run them.
    #[must_use]
    fn realize(&self) -> Self;
}

// We need to be able to cast tensors from one element type to another.
// From trait doesn't work because convenient definitions like:
//
//impl<EFro, ETo: From<EFro>> From<CpuRawTensor<EFro>> for CpuRawTensor<ETo> {}
//
//overlap with the core crate's implementation of From<T> for T, i.e. everything is convertible to itself.
//Specifically, it overlaps for the case where EFro = ETo.

// So, our own trait it is.
pub trait CastInto<T> {
    fn cast(&self) -> T;
}
