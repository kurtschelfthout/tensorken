use crate::num::{Num, ZeroOne};

/// Contains "mid level" operations (this is tinygrad terminology) that are differentiable.
/// These are not dependent on Add,Mul etc, traits because we want to be able to have a blanket implementation
/// for anyting that implements the low-level ops. (which isn't possible with foreign traits)
/// For user convenience, we have Reverse and Tensor structs that take a Diffable as generic argument, and which
/// do implement Add, Mul etc.traits.
/// Although the chosen operations are broadly similar to tinygrad, unlike tinygrad but inspired by JAX and `DiffSharp`,
/// this allows for calculation of higher-order derivatives, and eventually mixed forward and reverse modes (forward is
/// not yet implemented)
pub trait Diffable {
    type Elem: ZeroOne + Num;

    #[must_use]
    fn log(&self) -> Self;
    #[must_use]
    fn exp(&self) -> Self;

    // These ops are all elementwise, meaning in particular they have identical shapes.
    // No broadcasting. It's important to implemented broadcasted ops in terms of these ops,
    // so that everything flows through correctly. For example, for reverse AD, these will
    // add an elementary diffable operation to the tape, and it would be quite complicated
    // and unnecessary to implement broadcasting in the add operation, especially since it is
    // defined in terms of the other movement ops on this level.
    // As a result, the add,sub etc we actually use are on DiffableExt.

    #[must_use]
    fn elementwise_add(&self, other: &Self) -> Self;
    #[must_use]
    fn elementwise_sub(&self, other: &Self) -> Self;
    #[must_use]
    fn elementwise_mul(&self, other: &Self) -> Self;
    #[must_use]
    fn elementwise_div(&self, other: &Self) -> Self;
    #[must_use]
    fn elementwise_pow(&self, exp: &Self) -> Self;
    #[must_use]
    fn elementwise_eq(&self, other: &Self) -> Self;

    #[must_use]
    fn sum(&self, axes: &[usize]) -> Self;
    #[must_use]
    fn max(&self, axes: &[usize]) -> Self;

    #[must_use]
    fn reshape(&self, shape: &[usize]) -> Self;
    #[must_use]
    fn permute(&self, dims: &[usize]) -> Self;
    #[must_use]
    fn expand(&self, shape: &[usize]) -> Self;
    #[must_use]
    fn pad(&self, padding: &[(usize, usize)]) -> Self;
    #[must_use]
    fn crop(&self, limits: &[(usize, usize)]) -> Self;

    /// Create a new tensor with the given shape and elements.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    fn new(shape: &[usize], data: &[Self::Elem]) -> Self;
    fn shape(&self) -> &[usize];
}
