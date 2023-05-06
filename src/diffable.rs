use crate::Shape;

/// Contains "mid level" operations (this is tinygrad terminology) that are differentiable.
/// These are not dependent on Add,Mul etc, traits because we want to be able to have a blanket implementation
/// for anyting that implements the low-level ops. (which isn't possible with foreign traits)
/// For user convenience, we have Reverse and Tensor structs that take a Diffable as generic argument, and which
/// do implement Add, Mul etc.traits.
/// Although the chosen operations are broadly similar to tinygrad, unlike tinygrad but inspired by JAX and `DiffSharp`,
/// this allows for calculation of higher-order derivatives, and eventually mixed forward and reverse modes (forward is
/// not yet implemented)
pub trait Diffable {
    #[must_use]
    fn log(&self) -> Self;
    #[must_use]
    fn exp(&self) -> Self;

    #[must_use]
    fn add(&self, other: &Self) -> Self;
    #[must_use]
    fn sub(&self, other: &Self) -> Self;
    #[must_use]
    fn mul(&self, other: &Self) -> Self;
    #[must_use]
    fn div(&self, other: &Self) -> Self;
    #[must_use]
    fn pow(&self, exp: &Self) -> Self;
    #[must_use]
    fn eq(&self, other: &Self) -> Self;

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

    #[must_use]
    fn zeros_like(&self) -> Self;
    #[must_use]
    fn ones_like(&self) -> Self;
    fn shape(&self) -> &[usize];
}

/// These are operations that are based on the core Diffable operations.
/// Could have added those to Diffable itself, but this seems a bit tidier.
/// They can optionally be further split out by category.
#[allow(clippy::module_name_repetitions)]
pub trait DiffableExt: Diffable
where
    Self: Sized,
{
    // movement

    /// Switch the two axes around.
    #[must_use]
    fn transpose(&self, axis0: usize, axis1: usize) -> Self {
        let mut axes = (0..self.shape().ndims()).collect::<Vec<_>>();
        axes.swap(axis0, axis1);
        self.permute(&axes)
    }

    // math
    #[must_use]
    fn neg(&self) -> Self {
        self.zeros_like().sub(self)
    }

    #[must_use]
    fn reciprocal(&self) -> Self {
        self.ones_like().div(self)
    }

    /// Matrix multiplication, generalized to tensors.
    /// i.e. multiply [..., m, n] with [..., n, o] to [..., m, o]
    #[must_use]
    fn matmul(&self, other: &Self) -> Self {
        // self's shape from [..., m, n] to [..., m, 1, n]
        // using just reshape.
        let s = self.shape();
        let self_shape = [&s[..s.ndims() - 1], &[1, s[s.ndims() - 1]]].concat();
        let l = self.reshape(&self_shape);

        // other's shape from [..., n, o] to [..., 1, o, n]
        // using reshape + transpose.
        let s = other.shape();
        let other_shape = [&s[..s.ndims() - 2], &[1], &s[s.ndims() - 2..]].concat();
        let r = other
            .reshape(&other_shape)
            .transpose(other_shape.ndims() - 1, other_shape.ndims() - 2);

        // // after multiply: [..., m, o, n]
        // let prod = &l.mul(&r);
        // // after sum:      [..., m, o, 1]
        // let sum = prod.sum(&[prod.shape().ndims() - 1]);

        // fused multiply + sum
        let last_dim = max(l.shape().ndims(), r.shape().ndims()) - 1;
        let sum = l.fused_multiply_add(&r, &[last_dim]);

        // after reshape:  [..., m, o]
        let s = sum.shape();
        sum.reshape(&s[..s.ndims() - 1])
    }

    // activation functions
    #[must_use]
    fn sigmoid(&self) -> Self {
        self.ones_like().add(&self.neg().exp()).reciprocal()
    }
    #[must_use]
    fn tanh(&self) -> Self {
        let two = &(self.ones_like().add(&self.ones_like()));
        two.mul(&two.mul(self).sigmoid()).sub(&self.ones_like())
    }
}

impl<T: Diffable> DiffableExt for T {}
