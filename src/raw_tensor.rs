use crate::num::{Bool, CastFrom, Elem, Float, Num};

/// Counterpart for tinygrad's "low-level" operations (ops.py).
/// Represents the operations that a tensor implementation, be it on CPU or GPU, must implement.
/// All "medium-level" tensor operations (mlops.py) are eventually translated to these operations.
/// As such it can be used to implement a new type of accelerator, but can also support
/// optimizations like fusing.
/// Think of `RawTensorOps` as the DSL for accelerators, in final style.
#[allow(clippy::module_name_repetitions)]
pub trait RawTensorOps {
    /// The type of the tensor representation. It is parametrized by the element
    /// type E.
    type Repr<E: Clone>: Clone;

    // unary ops
    // ---------

    /// Apply exp to each element.
    fn exp<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E>;

    /// Apply the natural logarithm to each element.
    fn log<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E>;

    /// Cast the elements to another type.
    fn cast<EFro: Elem, ETo: CastFrom<EFro> + Elem>(t: &Self::Repr<EFro>) -> Self::Repr<ETo>;

    /// Run any delayed operations.
    fn realize<E: Clone>(t: &Self::Repr<E>) -> Self::Repr<E>;

    // binary ops
    // ----------

    /// Add self to other, element-wise.
    fn add<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    /// Subtract other from self, element-wise.
    fn sub<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    /// Multiply self by other, element-wise.
    fn mul<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    /// Divide self by other,  element-wise.
    fn div<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    /// Raise self to the power of other, element-wise.
    fn pow<E: Float>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    /// Return a new tensor with ones where elements are equal, and zero otherwise.
    fn eq<E: PartialEq + Elem>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<bool>;

    // reduce ops
    // ----------

    fn sum<E: Num>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E>;

    /// Find the maximum of elements along the given axes.
    fn max<E: Bool>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E>;

    // movement ops
    // ------------

    /// Reshape the tensor to the given shape.
    /// The number of elements must remain the same.
    /// Compare to numpy's `reshape`.
    fn reshape<E: Elem>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E>;

    /// Changes axes around according to the given permutation.
    /// Compare to numpy's `transpose`.
    fn permute<E: Clone>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E>;

    /// Expand the tensor to the given shape. Only dimensions of length 1 can be expanded.
    /// Like numpy's `broadcast_to` but simpler - does not add dimensions of size 1.
    fn expand<E: Clone>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E>;

    // Not yet implemented:
    // fn stride(t: &Self::Repr<E>, strides: &[usize]) -> Self;

    /// Pad the tensor with zeros according to the given padding.
    /// Like numpy's `pad`, but simpler - needs as many elements in `padding` as there
    /// are dimensions in the tensor.
    fn pad<E: Bool>(t: &Self::Repr<E>, padding: &[(usize, usize)]) -> Self::Repr<E>;

    /// Crop the tensor according to the given limits - taking a contiguous slice in each axis.
    /// Needs as many limits as there are dimensions in the tensor.
    fn crop<E: Clone>(t: &Self::Repr<E>, limits: &[(usize, usize)]) -> Self::Repr<E>;

    // creation
    // --------

    /// Create a new tensor with the given shape and elements.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    fn new<E: Elem>(shape: &[usize], data: &[E]) -> Self::Repr<E>;

    // elimination
    // -----------

    /// Return the shape of the tensor.
    fn shape<E: Clone>(t: &Self::Repr<E>) -> &[usize];

    // fused operations
    // ----------------

    /// Multiply self with other element-wise, and sum-reduce the given dimensions, in one fused
    /// operation.
    /// This operation is in `RawTensor` for performance reasons, as clearly functionally it is equivalent to `mul` + `sum`.
    /// However, usually hardware have specialized instructions for vectorized fused multiply-accumulate,
    /// or fused multiple-add, e.g. fma instruction in WebGPU. Also, for matrix multiplication, this avoids
    /// allocation of a typically large intermediate result tensor which holds the results of the un-accumulated
    /// element-wise multiplication, and so is essential.
    fn fused_multiply_add<E: Num>(
        lhs: &Self::Repr<E>,
        rhs: &Self::Repr<E>,
        axes: &[usize],
    ) -> Self::Repr<E>;
}

pub trait ToCpu: RawTensorOps {
    /// Return the tensor on the CPU.
    fn to_cpu<E: Elem>(t: &Self::Repr<E>) -> crate::CpuRawTensor<E>;
}
