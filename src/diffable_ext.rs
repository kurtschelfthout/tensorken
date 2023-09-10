use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::num::Num;
use crate::{Diffable, Shape};

pub(crate) fn broadcasted_apply<T: Diffable>(
    lhs: &T,
    rhs: &T,
    f: impl Fn(&T, &T) -> T,
    reverse: bool,
) -> T {
    if lhs.shape().ndims() > rhs.shape().ndims() {
        // Rust tidbit: I originally did not have a reverse parameter,
        // but just called |a,b| f(b,a) in the recursive call. This doesn't work,
        // because it hits the recursion limit: https://stackoverflow.com/questions/54613966/error-reached-the-recursion-limit-while-instantiating-funcclosure
        return broadcasted_apply(rhs, lhs, f, !reverse);
    }

    if lhs.shape().ndims() == rhs.shape().ndims() {
        let res_shape = lhs
            .shape()
            .iter()
            .zip(rhs.shape().iter())
            .map(|(a, b)| *a.max(b))
            .collect::<Vec<_>>();
        let lhs_expanded = lhs.expand(&res_shape);
        let rhs_expanded = rhs.expand(&res_shape);
        return if reverse {
            f(&rhs_expanded, &lhs_expanded)
        } else {
            f(&lhs_expanded, &rhs_expanded)
        };
    }

    let num_ones_to_add = rhs.shape().len().saturating_sub(lhs.shape().len());
    let mut new_shape = vec![1; num_ones_to_add];
    new_shape.extend(lhs.shape());

    broadcasted_apply(&lhs.reshape(&new_shape), rhs, f, reverse)
}

/// These are operations that are based on the core Diffable operations.
/// Could have added those to Diffable itself, but this seems a bit tidier.
/// They can optionally be further split out by category.
#[allow(clippy::module_name_repetitions)]
pub trait DiffableExt: Diffable
where
    Self: Sized,
{
    fn scalar(value: Self::Elem) -> Self {
        Self::new(&[1], &[value])
    }

    /// Create a new tensor with the given shape, and fill it with the given value.
    fn full(shape: &[usize], value: Self::Elem) -> Self {
        Self::new(&vec![1; shape.ndims()], &[value]).expand(shape)
    }

    /// Create a new tensor with the given shape, and fill it with zeros.
    fn zeros(shape: &[usize]) -> Self {
        Self::full(shape, Self::Elem::ZERO)
    }

    /// Create a new 2-dimensional tensor with the ones the diagonal and zeros elsewhere.
    fn eye(dim: usize) -> Self {
        // kind of an pad/crop/expand/reshape stress test
        Self::scalar(Self::Elem::ONE)
            .pad(&[(0, dim)])
            .reshape(&[1, dim + 1])
            .expand(&[dim, dim + 1])
            .reshape(&[dim * (dim + 1)])
            .crop(&[(0, dim * dim)])
            .reshape(&[dim, dim])
    }

    fn linspace(start: Self::Elem, end: Self::Elem, num: usize) -> Self {
        let mut data = Vec::with_capacity(num);
        let step = if num > 1 {
            let nf: Self::Elem = Self::Elem::from_usize(num);
            (end - start) / (nf - Self::Elem::ONE)
        } else {
            Self::Elem::ZERO
        };
        for i in 0..num {
            data.push(start + step * Self::Elem::from_usize(i));
        }
        Self::new(&[num], &data)
    }

    /// Create a new tensor with the given shape, and fill it with random values from a standard normal distribution
    fn randn<R>(shape: &[usize], rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        rand_distr::StandardNormal: Distribution<Self::Elem>,
    {
        // let normal = StandardNormal //.unwrap();
        let mut data: Vec<Self::Elem> = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            data.push(rng.sample(StandardNormal));
        }

        Self::new(shape, &data)
    }

    /// Create a new tensor with the same shape as self, but all elements equal to given value.
    #[must_use]
    fn constant_like(&self, value: Self::Elem) -> Self {
        Self::full(self.shape(), value)
    }

    #[must_use]
    fn zeros_like(&self) -> Self {
        self.constant_like(Self::Elem::ZERO)
    }

    #[must_use]
    fn ones_like(&self) -> Self {
        self.constant_like(Self::Elem::ONE)
    }

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
    fn add(&self, other: &Self) -> Self {
        broadcasted_apply(self, other, |a, b| a.elementwise_add(b), false)
    }

    #[must_use]
    fn sub(&self, other: &Self) -> Self {
        broadcasted_apply(self, other, |a, b| a.elementwise_sub(b), false)
    }

    #[must_use]
    fn mul(&self, other: &Self) -> Self {
        broadcasted_apply(self, other, |a, b| a.elementwise_mul(b), false)
    }

    #[must_use]
    fn div(&self, other: &Self) -> Self {
        broadcasted_apply(self, other, |a, b| a.elementwise_div(b), false)
    }

    #[must_use]
    fn pow(&self, other: &Self) -> Self {
        broadcasted_apply(self, other, |a, b| a.elementwise_pow(b), false)
    }

    #[must_use]
    fn eq(&self, other: &Self) -> Self {
        broadcasted_apply(self, other, |a, b| a.elementwise_eq(b), false)
    }

    #[must_use]
    fn neg(&self) -> Self {
        self.zeros_like().sub(self)
    }

    #[must_use]
    fn reciprocal(&self) -> Self {
        self.ones_like().div(self)
    }

    /// Matrix multiplication, generalized to tensors.
    /// i.e. multiply [..., m, n] with [..., n, o] to [..., m, o].
    ///
    /// Like in numpy's matmul:
    /// - If both arguments are 2-D they are multiplied like conventional matrices.
    /// - If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
    /// - If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
    /// - If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
    ///
    /// # Panics
    /// If one of the dimensions is 0 or if the inner dimensions don't match.
    #[must_use]
    fn matmul(&self, other: &Self) -> Self {
        let (l_nd, r_nd) = (self.shape().ndims(), other.shape().ndims());

        assert!(l_nd != 0, "matmul: lhs has no dimensions");
        assert!(r_nd != 0, "matmul: rhs has no dimensions");

        // the indices of the dimensions we're multiplying.
        let (l_n, r_n) = (l_nd - 1, r_nd.saturating_sub(2));
        {
            let l = self.shape()[l_n];
            let r = other.shape()[r_n];
            assert!(l == r, "matmul: inner dimensions don't match: {l} != {r}");
        }

        let prod = if l_nd == 1 {
            // if either of the dimensions is 1, we can just use elementwise mul.
            let s = self.shape();
            let l_shape = [s, &[1]].concat();
            let l = self.reshape(&l_shape);
            // .transpose(l_shape.ndims() - 1, l_shape.ndims() - 2);
            l.mul(other)
                .transpose(l_shape.ndims() - 1, l_shape.ndims() - 2)
        } else if r_nd == 1 {
            self.mul(other)
        } else {
            // self's shape from [..., m, n] to [..., m, 1, n]
            // using just reshape.
            let s = self.shape();
            let l_shape = [&s[..l_n], &[1, s[l_n]]].concat();
            let l = self.reshape(&l_shape);

            // other's shape from [..., n, o] to [..., 1, o, n]
            // using reshape + transpose.
            let s = other.shape();
            let r_shape = [&s[..r_n], &[1], &s[r_n..]].concat();
            let r = other
                .reshape(&r_shape)
                .transpose(r_shape.ndims() - 1, r_shape.ndims() - 2);

            // after multiply: [..., m, o, n]
            l.mul(&r)
        };

        // after sum:      [..., m, o, 1]
        let sum = prod.sum(&[prod.shape().ndims() - 1]);

        // note: counting on raw_tensor_fuse to make the mul and sum into a single operation.
        // otherwise, we'll blow up memory and time.

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
