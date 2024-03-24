use crate::{
    num::{Elem, Float, Num, ZeroOne},
    Shape,
};

/// Contains "mid level" operations (this is tinygrad terminology) that are differentiable.
/// These are not dependent on Add,Mul etc, traits because we want to be able to have a blanket implementation
/// for anyting that implements the low-level ops. (which isn't possible with foreign traits)
/// For user convenience, we have Reverse and Tensor structs that take a Diffable as generic argument, and which
/// do implement Add, Mul etc.traits.
/// Although the chosen operations are broadly similar to tinygrad, unlike tinygrad but inspired by JAX and `DiffSharp`,
/// this allows for calculation of higher-order derivatives, and eventually mixed forward and reverse modes (forward is
/// not yet implemented)
pub trait Diffable {
    type Repr<E: Clone>: Clone;

    fn log<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E>;

    fn exp<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E>;

    // These ops are all elementwise, meaning in particular they have identical shapes.
    // No broadcasting. It's important to implemented broadcasted ops in terms of these ops,
    // so that everything flows through correctly. For example, for reverse AD, these will
    // add an elementary diffable operation to the tape, and it would be quite complicated
    // and unnecessary to implement broadcasting in the add operation, especially since it is
    // defined in terms of the other movement ops on this level.
    // As a result, the add,sub etc we actually use are on Tensor.

    fn elementwise_add<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    fn elementwise_sub<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    fn elementwise_mul<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    fn elementwise_div<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    fn elementwise_pow<E: Float>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>;

    fn eq<E: PartialEq + Elem>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<bool>;

    fn sum<E: Num>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E>;

    fn max<E: Num + From<bool>>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E>;

    fn reshape<E: Elem>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E>;

    fn permute<E: Elem>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E>;

    fn expand<E: Num>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E>;

    fn pad<E: ZeroOne>(t: &Self::Repr<E>, padding: &[(usize, usize)]) -> Self::Repr<E>;

    fn crop<E: ZeroOne>(t: &Self::Repr<E>, limits: &[(usize, usize)]) -> Self::Repr<E>;

    /// Create a new tensor with the given shape and elements.
    /// The order of the elements is in increasing order of the last axis, then the second last, etc.
    fn new<E: Elem>(shape: &[usize], data: &[E]) -> Self::Repr<E>;

    fn shape<E: Clone>(t: &Self::Repr<E>) -> &[usize];

    fn cast<EFro: Elem, ETo: From<EFro> + Elem>(t: &Self::Repr<EFro>) -> Self::Repr<ETo>;

    // All the following are useful for the ad_ops. In principle all the methods
    // on Tensor could be implemented here, but due to the lack of dot notation and self,
    // it's quite cumbersome to write. So we only have the ones that are useful.
    fn full<E: Num>(shape: &[usize], value: E) -> Self::Repr<E> {
        let r = Self::new(&vec![1; shape.ndims()], &[value]);
        Self::expand::<E>(&r, shape)
    }

    fn zeros_like<E: Num>(t: &Self::Repr<E>) -> Self::Repr<E> {
        Self::full(Self::shape(t), E::ZERO)
    }

    fn ones_like<E: Num>(t: &Self::Repr<E>) -> Self::Repr<E> {
        Self::full(Self::shape(t), E::ONE)
    }

    fn neg<E: Num>(t: &Self::Repr<E>) -> Self::Repr<E> {
        let zeros = Self::zeros_like(t);
        Self::elementwise_sub(&zeros, t)
    }
}
