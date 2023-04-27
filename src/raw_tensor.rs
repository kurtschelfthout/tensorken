use crate::num::Num;

/// Counterpart for tinygrad's "low-level" operations in ops.py.
/// Represents the operations that a tensor implementation, be it on CPU or GPU, must implement.
/// All tensor operations in mlops are eventually translated to these operations.
/// As such it can be used to implement a new type of accelerator, but can also support
/// optimizations like fusing.
/// Think of `RawTensor` as the DSL for accelerators, in final style.
pub trait RawTensor
where
    Self: Sized,
{
    type Elem: Num;

    // unary ops
    // ---------

    /// Apply exp to each element.
    #[must_use]
    fn exp(&self) -> Self;

    /// Apply the natural logarithm to each element.
    #[must_use]
    fn log(&self) -> Self;

    // binary ops
    // ----------

    /// Add self to other, element-wise.
    #[must_use]
    fn add(&self, other: &Self) -> Self;

    /// Subtract other from self, element-wise.
    #[must_use]
    fn sub(&self, other: &Self) -> Self;

    /// Multiply self by other, element-wise.
    #[must_use]
    fn mul(&self, other: &Self) -> Self;

    /// Divide self by other,  element-wise.
    #[must_use]
    fn div(&self, other: &Self) -> Self;

    /// Raise self to the power of other, element-wise.
    #[must_use]
    fn pow(&self, other: &Self) -> Self;

    /// Return a new tensor with ones where elements are equal, and zero otherwise.
    #[must_use]
    fn eq(&self, other: &Self) -> Self;

    // reduce ops
    // ----------

    #[must_use]
    fn sum(&self, axes: &[usize]) -> Self;

    /// Find the maximum of elements along the given axes.
    #[must_use]
    fn max(&self, axes: &[usize]) -> Self;

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
    fn pad(&self, padding: &[(usize, usize)]) -> Self;

    /// Crop the tensor according to the given limits - taking a contiguous slice in each axis.
    /// Needs as many limits as there are dimensions in the tensor.
    #[must_use]
    fn crop(&self, limits: &[(usize, usize)]) -> Self;

    // creation
    // --------

    /// Create a new tensor with the given shape and elements.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    fn new(shape: &[usize], data: &[Self::Elem]) -> Self;

    // elimination
    // -----------

    /// Return the shape of the tensor.
    fn shape(&self) -> &[usize];

    /// Return the elements of the tensor as a Vec, i.e. on the CPU.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    fn ravel(&self) -> Vec<Self::Elem>;
}
