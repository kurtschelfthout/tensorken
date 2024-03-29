use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use rand::Rng;
use rand_distr::Distribution;

use crate::{
    num::{Bool, CastFrom, CastTo, Elem, Float, Num},
    raw_tensor::{RawTensor, ToCpu},
    raw_tensor_cpu::{CpuRawTensor, CpuRawTensorImpl},
    tensor_mut::TensorMut,
    Diffable, Forward, ForwardImpl, Reverse, ReverseImpl, Shape,
};

// Blanket implementation to translate from diffable tensor ops (Diffable) to low-level tensor ops (RawTensor).
impl<I: RawTensor> Diffable for I {
    type Repr<E: Clone> = I::Repr<E>;

    fn log<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        I::log(t)
    }

    fn exp<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        I::exp(t)
    }

    fn elementwise_add<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        I::add(lhs, rhs)
    }

    fn elementwise_sub<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        I::sub(lhs, rhs)
    }

    fn elementwise_mul<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        I::mul(lhs, rhs)
    }

    fn elementwise_div<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        I::div(lhs, rhs)
    }

    fn elementwise_pow<E: Float>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        I::pow(lhs, rhs)
    }

    fn eq<E: PartialEq + Elem>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<bool> {
        I::eq(lhs, rhs)
    }

    fn sum<E: Num>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        I::sum(t, axes)
    }

    fn max<E: Num + CastFrom<bool>>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        I::max(t, axes)
    }

    fn reshape<E: Elem>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        I::reshape(t, shape)
    }

    fn permute<E: Clone>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E> {
        I::permute(t, permutation)
    }

    fn expand<E: Clone>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        I::expand(t, shape)
    }

    fn pad<E: Bool>(t: &Self::Repr<E>, padding: &[(usize, usize)]) -> Self::Repr<E> {
        I::pad(t, padding)
    }

    fn crop<E: Clone>(t: &Self::Repr<E>, limits: &[(usize, usize)]) -> Self::Repr<E> {
        I::crop(t, limits)
    }

    fn new<E: Elem>(shape: &[usize], data: &[E]) -> Self::Repr<E> {
        I::new(shape, data)
    }

    fn shape<E: Clone>(t: &Self::Repr<E>) -> &[usize] {
        I::shape(t)
    }

    fn cast<EFro: Elem, ETo: CastFrom<EFro> + Elem>(t: &Self::Repr<EFro>) -> Self::Repr<ETo> {
        I::cast(t)
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
pub struct Tensor<T, E: Clone, I: Diffable<Repr<E> = T>>(
    pub(crate) T,
    pub(crate) PhantomData<(E, I)>,
);

impl<T, E: Float, I: Diffable<Repr<E> = T>> Tensor<T, E, I> {
    /// Apply the natural logarithm to each element.
    #[must_use]
    pub fn log(&self) -> Self {
        Self(I::log::<E>(&self.0), PhantomData)
    }

    /// Apply exp to each element.
    #[must_use]
    pub fn exp(&self) -> Self {
        Self(I::exp::<E>(&self.0), PhantomData)
    }

    /// Raise self to the power of other, element-wise.
    #[must_use]
    pub fn elementwise_pow(&self, exp: &Self) -> Self {
        Self(I::elementwise_pow::<E>(&self.0, &exp.0), PhantomData)
    }
}

impl<T, E: Num, I: Diffable<Repr<E> = T>> Tensor<T, E, I> {
    #[must_use]
    pub fn elementwise_add(&self, other: &Self) -> Self {
        Self(I::elementwise_add::<E>(&self.0, &other.0), PhantomData)
    }

    #[must_use]
    pub fn elementwise_sub(&self, other: &Self) -> Self {
        Self(I::elementwise_sub::<E>(&self.0, &other.0), PhantomData)
    }

    #[must_use]
    pub fn elementwise_mul(&self, other: &Self) -> Self {
        Self(I::elementwise_mul::<E>(&self.0, &other.0), PhantomData)
    }

    #[must_use]
    pub fn elementwise_div(&self, other: &Self) -> Self {
        Self(I::elementwise_div::<E>(&self.0, &other.0), PhantomData)
    }

    /// Reduce to sum along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    #[must_use]
    pub fn sum(&self, axes: &[usize]) -> Self {
        Self(I::sum::<E>(&self.0, axes), PhantomData)
    }

    /// Expand the tensor to the given shape. Only dimensions of length 1 can be expanded.
    /// Like numpy's `broadcast_to` but simpler - does not add dimensions of size 1.
    #[must_use]
    pub fn expand(&self, shape: &[usize]) -> Self {
        Self(I::expand::<E>(&self.0, shape), PhantomData)
    }
}

impl<T, E: Bool, I: Diffable<Repr<E> = T>> Tensor<T, E, I> {
    #[must_use]
    pub fn pad(&self, padding: &[(usize, usize)]) -> Self {
        Self(I::pad::<E>(&self.0, padding), PhantomData)
    }

    #[must_use]
    pub fn crop(&self, limits: &[(usize, usize)]) -> Self {
        Self(I::crop::<E>(&self.0, limits), PhantomData)
    }
}

impl<T, E: Elem, I: Diffable<Repr<E> = T>> Tensor<T, E, I> {
    /// Reshape the tensor to the given shape.
    /// The number of elements must remain the same.
    /// Compare to numpy's `reshape`.
    #[must_use]
    pub fn reshape(&self, shape: &[usize]) -> Self {
        Self(I::reshape::<E>(&self.0, shape), PhantomData)
    }

    /// Changes axes around according to the given permutation.
    /// Compare to numpy's `transpose`.
    #[must_use]
    pub fn permute(&self, dims: &[usize]) -> Self {
        Self(I::permute::<E>(&self.0, dims), PhantomData)
    }

    pub fn new(shape: &[usize], data: &[E]) -> Self {
        Self(I::new::<E>(shape, data), PhantomData)
    }

    pub fn cast<ETo: CastFrom<E> + Elem>(&self) -> Tensor<<I as Diffable>::Repr<ETo>, ETo, I> {
        Tensor(I::cast::<E, ETo>(&self.0), PhantomData)
    }
}

impl<T, E: Clone, I: Diffable<Repr<E> = T>> Tensor<T, E, I> {
    /// Return a new tensor with true where elements are equal, and false elsewhere.
    pub fn elementwise_eq(&self, other: &Self) -> Tensor<<I as Diffable>::Repr<bool>, bool, I>
    where
        E: PartialEq + Elem,
    {
        Tensor(I::eq::<E>(&self.0, &other.0), PhantomData)
    }

    /// Reduce to max element along the given axes.
    /// Keeps the reduced dimensions, but with size 1.
    #[must_use]
    pub fn max(&self, axes: &[usize]) -> Self
    where
        E: Num + CastFrom<bool>,
    {
        Self(I::max::<E>(&self.0, axes), PhantomData)
    }

    pub fn shape(&self) -> &[usize] {
        I::shape::<E>(&self.0)
    }
}

/// Applies the non-broadcasted function f to lhs and rhs, adding broadcast.
fn broadcasted_apply_rec<T, TR, E: Num, I: Diffable<Repr<E> = T>>(
    lhs: &T,
    rhs: &T,
    f: impl Fn(&T, &T) -> TR,
    reverse: bool,
) -> TR {
    let lhss = I::shape::<E>(lhs);
    let rhss = I::shape::<E>(rhs);
    if lhss.ndims() > rhss.ndims() {
        // Rust tidbit: I originally did not have a reverse parameter,
        // but just called |a,b| f(b,a) in the recursive call. This doesn't work,
        // because it hits the recursion limit: https://stackoverflow.com/questions/54613966/error-reached-the-recursion-limit-while-instantiating-funcclosure
        return broadcasted_apply_rec::<T, TR, E, I>(rhs, lhs, f, !reverse);
    }

    if lhss.ndims() == rhss.ndims() {
        let res_shape: Vec<_> = lhss
            .iter()
            .zip(rhss.iter())
            .map(|(a, b)| *a.max(b))
            .collect();
        let lhs_expanded = I::expand::<E>(lhs, &res_shape);
        let rhs_expanded = I::expand::<E>(rhs, &res_shape);
        return if reverse {
            f(&rhs_expanded, &lhs_expanded)
        } else {
            f(&lhs_expanded, &rhs_expanded)
        };
    }

    let num_ones_to_add = rhss.len().saturating_sub(lhss.len());
    let mut new_shape = vec![1; num_ones_to_add];
    new_shape.extend(lhss);

    broadcasted_apply_rec::<T, TR, E, I>(&I::reshape::<E>(lhs, &new_shape), rhs, f, reverse)
}

fn broadcasted_apply<T, TR, E, ER, I, IR>(
    lhs: &Tensor<T, E, I>,
    rhs: &Tensor<T, E, I>,
    f: impl Fn(&T, &T) -> TR,
    reverse: bool,
) -> Tensor<TR, ER, IR>
where
    E: Num,
    ER: Elem,
    I: Diffable<Repr<E> = T>,
    IR: Diffable<Repr<ER> = TR>,
{
    Tensor(
        broadcasted_apply_rec::<T, TR, E, I>(&lhs.0, &rhs.0, f, reverse),
        PhantomData,
    )
}

pub enum Axes {
    All,
    Axis(usize),
}

impl<T, E: Elem, I: Diffable<Repr<E> = T>> Tensor<T, E, I> {
    pub fn lift_rev(&self) -> Tensor<<ReverseImpl<I> as Diffable>::Repr<E>, E, ReverseImpl<I>>
    where
        I: 'static,
        T: Clone,
    {
        Tensor(Reverse::Lift(self.0.clone()), PhantomData)
    }

    pub fn lift_fwd(&self) -> Tensor<<ForwardImpl<I> as Diffable>::Repr<E>, E, ForwardImpl<I>>
    where
        T: Clone,
    {
        Tensor(Forward::Lift(self.0.clone()), PhantomData)
    }

    /// Create a new tensor with a single element.
    pub fn scalar(value: E) -> Self {
        Self(I::new(&[1], &[value]), PhantomData)
    }

    /// Create a new tensor with the given shape, filled with the given value.
    pub fn full(shape: &[usize], value: E) -> Self
    where
        E: Num,
    {
        Self(I::full(shape, value), PhantomData)
    }

    /// Create a new tensor with the given shape, filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self
    where
        E: Num,
    {
        Self::full(shape, E::ZERO)
    }

    /// Create a new tensor with the given shape, filled with ones.
    pub fn ones(shape: &[usize]) -> Self
    where
        E: Num,
    {
        Self::full(shape, E::ONE)
    }

    /// Create a new 2-dimensional tensor with ones on the diagonal and zeros elsewhere.
    pub fn eye(dim: usize) -> Self
    where
        E: Num,
    {
        Self::scalar(E::ONE)
            .pad(&[(0, dim)])
            .reshape(&[1, dim + 1])
            .expand(&[dim, dim + 1])
            .reshape(&[dim * (dim + 1)])
            .crop(&[(0, dim * dim)])
            .reshape(&[dim, dim])
    }

    /// Create a new 1D tensor with `num` values linearly spaced between `start` and `end`.
    pub fn linspace<N: Into<usize> + Into<E> + Copy>(start: E, end: E, num: N) -> Self
    where
        E: Num,
    {
        let num_usize = num.into();
        let mut data = Vec::with_capacity(num_usize);
        let step = if num_usize > 1 {
            let nf: E = num.into();
            (end - start.clone()) / (nf - E::ONE)
        } else {
            E::ZERO
        };
        let mut point = start;
        for _i in 0..num_usize {
            data.push(point.clone());
            point = point + step.clone();
        }
        Self::new(&[num_usize], &data)
    }

    /// Create a new tensor with the given shape, and fill it with random values from a standard normal distribution
    pub fn randn<R>(shape: &[usize], rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        rand_distr::StandardNormal: Distribution<E>,
    {
        let mut data: Vec<E> = Vec::with_capacity(shape.iter().product());
        for _ in 0..shape.iter().product() {
            data.push(rng.sample(rand_distr::StandardNormal));
        }
        Self::new(shape, &data)
    }

    /// Create a new tensor with the same shape as self, but all elements equal to given value.
    #[must_use]
    pub fn constant_like(&self, value: E) -> Self
    where
        E: Num,
    {
        Self::full(self.shape(), value)
    }

    #[must_use]
    pub fn zeros_like(&self) -> Self
    where
        E: Num,
    {
        self.constant_like(E::ZERO)
    }

    #[must_use]
    pub fn ones_like(&self) -> Self
    where
        E: Num,
    {
        self.constant_like(E::ONE)
    }
    /// Switch the two axes around.
    #[must_use]
    pub fn transpose(&self, axis0: usize, axis1: usize) -> Self {
        let mut axes: Vec<_> = (0..self.shape().ndims()).collect();
        axes.swap(axis0, axis1);
        self.permute(&axes)
    }

    /// Remove all axes with size 1, or a specific axis if given.
    /// # Panics
    /// If the given axis is not of size 1.
    #[must_use]
    pub fn squeeze(&self, dim: &Axes) -> Self {
        let mut shape = self.shape().to_vec();
        if let Axes::Axis(dim) = dim {
            assert_eq!(shape[*dim], 1);
            shape.remove(*dim);
        } else {
            shape.retain(|&x| x != 1);
        }
        self.reshape(&shape)
    }

    /// Insert a new axis of length 1.
    #[must_use]
    pub fn expand_dims(&self, dim: usize) -> Self {
        let mut shape = self.shape().to_vec();
        shape.insert(dim, 1);
        self.reshape(&shape)
    }

    /// Join a sequence of tensors along an existing axis.
    /// # Panics
    /// If no tensors are given, i.e. `tensors` is empty.
    /// If the shapes of the tensors don't match along all axes except the given axis.
    pub fn concatenate(tensors: &[&Self], axis: usize) -> Self
    where
        E: Num,
    {
        assert!(!tensors.is_empty(), "concatenate: no tensors given");
        let mut shape = tensors[0].shape().to_vec();

        for tensor in &tensors[1..] {
            (0..tensors[0].shape().ndims()).for_each(|a| {
                if a != axis {
                    assert_eq!(
                        shape[a],
                        tensor.shape()[a],
                        "concatenate: shapes don't match"
                    );
                }
            });
            shape[axis] += tensor.shape()[axis];
        }

        let mut result = Self::zeros(&shape);
        let mut padding = vec![(0, 0); shape.len()];
        let mut cumsum = 0;
        for tensor in tensors {
            padding[axis] = (cumsum, shape[axis] - tensor.shape()[axis] - cumsum);
            cumsum += tensor.shape()[axis];
            result = result.add(&tensor.pad(&padding));
        }
        result
    }

    /// Stack a sequence of tensors along a new axis.
    /// # Panics
    /// If no tensors are given, i.e. `tensors` is empty.
    pub fn stack(tensors: &[&Self], axis: usize) -> Self
    where
        E: Num,
    {
        assert!(!tensors.is_empty(), "stack: no tensors given");
        let ts: Vec<_> = tensors.iter().map(|t| t.expand_dims(axis)).collect();
        Self::concatenate(&ts.iter().collect::<Vec<_>>(), axis)
    }

    // math

    #[must_use]
    pub fn add(&self, other: &Self) -> Self
    where
        E: Num,
    {
        broadcasted_apply::<T, T, E, E, I, I>(self, other, I::elementwise_add::<E>, false)
    }

    #[must_use]
    pub fn sub(&self, other: &Self) -> Self
    where
        E: Num,
    {
        broadcasted_apply::<T, T, E, E, I, I>(self, other, I::elementwise_sub::<E>, false)
    }

    #[must_use]
    pub fn mul(&self, other: &Self) -> Self
    where
        E: Num,
    {
        broadcasted_apply::<T, T, E, E, I, I>(self, other, I::elementwise_mul::<E>, false)
    }

    #[must_use]
    pub fn div(&self, other: &Self) -> Self
    where
        E: Num,
    {
        broadcasted_apply::<T, T, E, E, I, I>(self, other, I::elementwise_div::<E>, false)
    }

    #[must_use]
    pub fn pow(&self, other: &Self) -> Self
    where
        E: Float,
    {
        broadcasted_apply::<T, T, E, E, I, I>(self, other, I::elementwise_pow::<E>, false)
    }

    #[must_use]
    pub fn eq(&self, other: &Self) -> Tensor<<I as Diffable>::Repr<bool>, bool, I>
    where
        E: Num,
    {
        broadcasted_apply::<T, I::Repr<bool>, E, bool, I, I>(self, other, I::eq::<E>, false)
    }

    #[must_use]
    pub fn neg(&self) -> Self
    where
        E: Num,
    {
        self.zeros_like().sub(self)
    }

    #[must_use]
    pub fn reciprocal(&self) -> Self
    where
        E: Num,
    {
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
    pub fn matmul(&self, other: &Self) -> Self
    where
        E: Num,
    {
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

    /// Dot product of two tensors.
    ///
    /// From numpy's documentation:
    /// - If both a and b are 1-D arrays, it is inner product of vectors.
    /// - If both a and b are 2-D, it is matrix multiplication, but using matmul is preferred.
    /// - If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    /// - If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b.
    ///
    /// # Panics
    /// If one of the dimensions is 0 or if the inner dimensions don't match.
    #[must_use]
    pub fn dot(&self, other: &Self) -> Self
    where
        E: Num,
    {
        let (l_nd, r_nd) = (self.shape().ndims(), other.shape().ndims());

        assert!(l_nd != 0, "dot: lhs has no dimensions");
        assert!(r_nd != 0, "dot: rhs has no dimensions");

        if l_nd == 1 && r_nd == 1 {
            return self.mul(other).sum(&[0]);
        } else if l_nd == 2 && r_nd == 2 {
            return self.matmul(other);
        }

        // sum product over l_n and r_n
        let (l_n, r_n) = (l_nd - 1, r_nd.saturating_sub(2));
        {
            let l = self.shape()[l_n];
            let r = other.shape()[r_n];
            assert!(l == r, "dot: inner dimensions don't match: {l} != {r}");
        }

        // self reshape from [..., m, n] to [..., m, 1, ..., 1, n]
        // number of 1s = r_nd - 1
        let s = self.shape();
        let ones = vec![1; r_nd - 1];
        let l_shape = [&s[..l_n], &ones, &[s[l_n]]].concat();
        let l = self.reshape(&l_shape);

        // other transpose + reshape from [..., n, o] to [1, ..., 1, ..., o, n]
        let s = other.shape();
        // transpose first - this does nothing if other is 1d
        let other = other.transpose(s.ndims() - 1, s.ndims().saturating_sub(2));
        // shape s has likely changed
        let s = other.shape();
        let ones = vec![1; l_nd - 1];
        let r_shape = [&ones, s].concat();
        let r = other
            //.transpose(r_shape.ndims() - 1, r_shape.ndims() - 2)
            .reshape(&r_shape);

        let prod = l.mul(&r);

        // after sum:      [..., m, ..., o, 1]
        let sum = prod.sum(&[prod.shape().ndims() - 1]);

        // note: counting on raw_tensor_fuse to make the mul and sum into a single operation.
        // otherwise, we'll blow up memory and time.

        // after reshape:  [..., m, ..., o]
        let s = sum.shape();
        sum.reshape(&s[..s.ndims() - 1])
    }

    // activation functions
    #[must_use]
    pub fn sigmoid(&self) -> Self
    where
        E: Float,
    {
        self.ones_like().add(&self.neg().exp()).reciprocal()
    }
    #[must_use]
    pub fn tanh(&self) -> Self
    where
        E: Float,
    {
        let two = &(self.ones_like().add(&self.ones_like()));
        two.mul(&two.mul(self).sigmoid()).sub(&self.ones_like())
    }
}

impl<T, E: Clone, I: ToCpu<Repr<E> = T>> Tensor<T, E, I> {
    /// Create a new mutable tensor with self's shape and elements.
    pub fn to_tensor_mut(&self) -> TensorMut<E>
    where
        E: Elem,
    {
        TensorMut::new(self)
    }

    /// Create a new Tensor that has any lazy operations realized.
    #[must_use]
    pub fn realize(&self) -> Self {
        Self(I::realize::<E>(&self.0), PhantomData)
    }

    /// Create a new [`CpuRawTensor`] with self's shape and elements.
    pub fn to_cpu(&self) -> Tensor<CpuRawTensor<E>, E, CpuRawTensorImpl>
    where
        E: Elem,
    {
        Tensor(I::to_cpu::<E>(&self.0), PhantomData)
    }

    pub fn ravel(&self) -> Vec<E>
    where
        E: Elem,
    {
        self.to_cpu().0.ravel()
    }

    /// If the tensor has only one element, return it.
    /// # Panics
    /// If the tensor does not have exactly one element.
    pub fn to_scalar(&self) -> E
    where
        E: Elem,
    {
        assert!(self.shape().size() == 1);
        self.ravel()[0].clone()
    }

    /// Returns a new tensor with a new axis inserted at the back, of size `num_classes`.
    /// All elements are assumed to be integers in the range [0, `num_classes`).
    /// The new axis is used as a one-hot encoding of the elements.
    ///
    /// # Panics
    /// If any element can't be converted to a usize.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn one_hot<N>(&self, num_classes: N) -> Self
    where
        N: CastTo<usize>,
        E: Num + CastTo<usize>,
    {
        let nc: usize = num_classes.cast_to();
        let mut data = vec![E::ZERO; self.shape().size() * nc];
        for (i, x) in self.ravel().iter().enumerate() {
            data[i * nc + x.cast_to()] = E::ONE;
        }
        let mut new_shape = Vec::with_capacity(self.shape().ndims() + 1);
        new_shape.extend(self.shape());
        new_shape.push(nc);
        Self::new(&new_shape, &data)
    }
}

impl<T: Clone, E: Clone, I: 'static + Diffable<Repr<E> = T>> Tensor<Reverse<T>, E, ReverseImpl<I>> {
    pub fn primal(&self) -> Tensor<T, E, I> {
        Tensor(self.0.primal().clone(), PhantomData)
    }
}

impl<T: Clone, E: Clone, I: 'static + Diffable<Repr<E> = T>> Tensor<Forward<T>, E, ForwardImpl<I>> {
    pub fn primal(&self) -> Tensor<T, E, I> {
        Tensor(self.0.primal().clone(), PhantomData)
    }
}

crate::math_macros::impl_bin_op!(Add, add, Tensor<T, E, I>);
crate::math_macros::impl_bin_op!(Sub, sub, Tensor<T, E, I>);
crate::math_macros::impl_bin_op!(Mul, mul, Tensor<T, E, I>);
crate::math_macros::impl_bin_op!(Div, div, Tensor<T, E, I>);

crate::math_macros::impl_un_op!(Neg, neg, Tensor<T, E, I>);
