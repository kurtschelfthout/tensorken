use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

use bytemuck::Pod;

use crate::{
    tiny_mlops::Diffable, tiny_num::Num, tiny_raw_tensor::RawTensor,
    tiny_raw_tensor_cpu::CpuRawTensor, tiny_raw_tensor_wgpu::raw_tensor::WgpuRawTensor,
    tiny_shape_strider::Shape,
};

/// The "high-level" tensor type. The Tensor struct exists mostly for
/// ergonomic reasons: it allows us to implement various arithmetic traits
/// like Add, Sub, Neg to overload mathematical operators.
/// Also, we add convenience operators on it

// Blanket implementation to translate from mid-level tensor ops (Diffable) to low-level tensor ops (RawTensor).
impl<T: Num, TTensor: RawTensor<Elem = T>> Diffable for TTensor {
    fn zeros_like(&self) -> Self {
        TTensor::new(&vec![1; self.shape().ndims()], &[T::ZERO]).expand(self.shape())
    }
    fn ones_like(&self) -> Self {
        TTensor::new(&vec![1; self.shape().ndims()], &[T::ONE]).expand(self.shape())
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn add(&self, other: &Self) -> Self {
        self.add(other)
    }

    fn sub(&self, other: &Self) -> Self {
        self.sub(other)
    }

    fn mul(&self, other: &Self) -> Self {
        self.mul(other)
    }

    fn div(&self, other: &Self) -> Self {
        self.div(other)
    }

    fn pow(&self, other: &Self) -> Self {
        self.pow(other)
    }

    fn eq(&self, other: &Self) -> Self {
        self.eq(other)
    }

    fn log(&self) -> Self {
        self.log()
    }

    fn exp(&self) -> Self {
        self.exp()
    }

    fn sum(&self, axes: &[usize]) -> Self {
        self.sum(axes)
    }

    fn max(&self, axes: &[usize]) -> Self {
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
}

#[derive(Debug, Clone)]
pub struct Tensor<TOps>(TOps);

impl<T: Copy + Num, TRawTensor: RawTensor<Elem = T>> Tensor<TRawTensor> {
    pub fn new(shape: &[usize], data: &[T]) -> Self {
        Tensor(TRawTensor::new(shape, data))
    }

    pub fn constant_like(&self, value: T) -> Self {
        Tensor(TRawTensor::new(&vec![1; self.0.shape().ndims()], &[value])).expand(self.0.shape())
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.0.ravel()
    }
}

impl<TOps: Diffable> Diffable for Tensor<TOps> {
    fn zeros_like(&self) -> Self {
        Tensor(self.0.zeros_like())
    }

    fn ones_like(&self) -> Self {
        Tensor(self.0.ones_like())
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

    fn pow(&self, exp: &Self) -> Self {
        self.broadcasted_apply(exp, |a, b| Tensor(a.0.pow(&b.0)), false)
    }

    fn eq(&self, other: &Self) -> Self {
        self.broadcasted_apply(other, |a, b| Tensor(a.0.eq(&b.0)), false)
    }

    fn log(&self) -> Self {
        Tensor(self.0.log())
    }

    fn exp(&self) -> Self {
        Tensor(self.0.exp())
    }

    fn sum(&self, axes: &[usize]) -> Self {
        Tensor(self.0.sum(axes))
    }

    fn max(&self, axes: &[usize]) -> Self {
        Tensor(self.0.max(axes))
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        Tensor(self.0.reshape(shape))
    }

    fn permute(&self, dims: &[usize]) -> Self {
        Tensor(self.0.permute(dims))
    }

    fn expand(&self, shape: &[usize]) -> Self {
        Tensor(self.0.expand(shape))
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
}

macro_rules! impl_difftensor_tensor {
    ($op_trait:ident, $op_fn:ident) => {
        impl<T: Diffable> $op_trait<&Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                Diffable::$op_fn(self, rhs)
            }
        }

        impl<T: Diffable> $op_trait<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                Diffable::$op_fn(&self, rhs)
            }
        }

        impl<T: Diffable> $op_trait<Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                Diffable::$op_fn(&self, &rhs)
            }
        }

        impl<T: Diffable> $op_trait<Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                Diffable::$op_fn(self, &rhs)
            }
        }
    };
}

impl_difftensor_tensor!(Add, add);
impl_difftensor_tensor!(Sub, sub);
impl_difftensor_tensor!(Mul, mul);
impl_difftensor_tensor!(Div, div);

impl<T: Diffable> Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
        self.zeros_like().sub(self)
    }
}

impl<T: Diffable> Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Tensor<T> {
        self.zeros_like().sub(self)
    }
}

/// One of two traits to make it easy to write generic functions over tensors,
/// that can be differentiated.
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

/// One of two traits to make it easy to write generic functions over tensors,
/// that can be differentiated.
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

impl<TDiffable: Diffable> Tensor<TDiffable> {
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
            } else {
                return f(&s_expanded, &o_expanded);
            }
        }

        let num_ones_to_add = other.shape().len().saturating_sub(self.shape().len());
        let mut new_shape = vec![1; num_ones_to_add];
        new_shape.extend(self.shape());

        self.reshape(&new_shape)
            .broadcasted_apply(other, f, reverse)
    }

    pub fn transpose(&self, axis0: usize, axis1: usize) -> Self {
        let mut axes = (0..self.shape().ndims()).collect::<Vec<_>>();
        axes.swap(axis0, axis1);
        self.permute(&axes)
    }

    /// Matrix multiplication, generalized to tensors.
    /// i.e. multiply [..., m, n] with [..., n, o] to [..., m, o]
    pub fn dot(&self, other: &Self) -> Self {
        // self's shape from [..., m, n] to [..., m, 1, n]
        // using just reshape.
        let lshape = self.shape();
        let lshape = [
            &lshape[..lshape.ndims() - 1],
            &[1],
            &[lshape[lshape.ndims() - 1]],
        ]
        .concat();
        let l = self.reshape(&lshape);

        // other's shape from [..., n, o] to [..., 1, o, n]
        // using reshape + transpose.
        let rshape = other.shape();
        let rshape = [
            &rshape[..rshape.ndims() - 2],
            &[1],
            &rshape[rshape.ndims() - 2..],
        ]
        .concat();
        let r = other
            .reshape(&rshape)
            .transpose(rshape.ndims() - 1, rshape.ndims() - 2);

        // after multiply: [..., m, o, n]
        // after sum:      [..., m, o, 1]
        let summed = (l * r).sum(&[rshape.ndims() - 1]);
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

#[cfg(test)]
mod tests {

    use super::*;

    fn assert_tensor_eq<T1: RawTensor<Elem = f32>, T2: RawTensor<Elem = f32>>(
        a: &Tensor<T1>,
        b: &Tensor<T2>,
    ) {
        let (a, b) = (a.to_vec(), b.to_vec());
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(a, b)| (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-2),
            "\r\nleft : {:?}\r\nright: {:?}",
            a,
            b
        )
    }

    fn assert_vec_eq(a: &[f32], b: &[f32]) {
        assert!(
            a.iter()
                .zip(b.iter())
                // big tolerance here - pow on the GPU is esp. sensitive to rounding errors
                .all(|(a, b)| (a.is_nan() && b.is_nan()) || (a - b).abs() < 1.5),
            "\r\nleft : {:?}\r\nright: {:?}",
            a,
            b
        )
    }

    // a few functions that are "compile time" tests - to check that the
    // TernsorLike traits are having the right effect.
    fn fun<'t, T>(t1: &'t T, t2: &'t T) -> T
    where
        T: TensorLike<'t>,
        &'t T: TensorLikeRef<T>,
    {
        let r1 = t1.exp(); // DiffTensor ops
        let r2 = t2.log();
        let r3 = t1 + r1; // &T + T
        let r4 = r2 - t2; // T - &T
        let r5 = t1 / t2; // T / T
        let r6 = r3 / r4.exp(); // T / T
        let r7 = t1 * t2; // &T * &T
        r6 + r5 + r7
    }

    #[test]
    fn test_tensorlike() {
        let (shape, data) = (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t_wgpu = &Tensor::new_wgpu(shape, data);
        let t_cpu = &Tensor::new_cpu(shape, data);

        let r_cpu = fun(t_cpu, t_cpu);
        let r_gpu = fun(t_wgpu, t_wgpu);
        assert_tensor_eq(&r_cpu, &r_gpu);
    }

    #[test]
    fn test_elementwise_ops() {
        let shape = &[2, 3];
        let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t2d = &[6.0, 7.0, 8.0, 9.0, 10.0, 12.0];
        let t1 = &Tensor::new_cpu(shape, t1d);
        let t2 = &Tensor::new_cpu(shape, t2d);
        elementwise_ops(t1, t2);

        let t1 = &Tensor::new_wgpu(shape, t1d);
        let t2 = &Tensor::new_wgpu(shape, t2d);
        elementwise_ops(t1, t2);
    }

    fn elementwise_ops<T: RawTensor<Elem = f32>>(t1: &Tensor<T>, t2: &Tensor<T>) {
        let r1 = t1.exp();
        assert_vec_eq(
            &r1.to_vec(),
            &[
                2.7182817, 7.389056, 20.085537, 54.59815, 148.41316, 403.4288,
            ],
        );
        let r2 = t1.log();
        assert_vec_eq(
            &r2.to_vec(),
            &[0.0, 0.69314724, 1.0986124, 1.3862945, 1.6094381, 1.7917596],
        );
        let r3 = t1 + t2;
        assert_eq!(r3.to_vec(), vec![7.0, 9.0, 11.0, 13.0, 15.0, 18.0]);
        let r4 = t1 - t2;
        assert_eq!(r4.to_vec(), vec![-5.0, -5.0, -5.0, -5.0, -5.0, -6.0]);
        let r5 = t1 / t2;
        assert_eq!(
            r5.to_vec(),
            vec![0.16666667, 0.2857143, 0.375, 0.44444445, 0.5, 0.5]
        );
        let r6 = t1 * t2;
        assert_eq!(r6.to_vec(), vec![6.0, 14.0, 24.0, 36.0, 50.0, 72.0]);

        let r7 = t1.eq(t2);
        assert_eq!(r7.to_vec(), vec![0.0; 6]);
    }

    #[test]
    fn test_broadcasted_ops() {
        let t1s = &[1, 1, 2, 3];
        let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t2s = &t1s[3..];
        let t2d = &[6.0, 7.0, 8.0];
        let t3d = &[1.0, 2.0, 3.0];

        let t1 = &Tensor::new_cpu(t1s, t1d);
        let t2 = &Tensor::new_cpu(t2s, t2d);
        let t3 = &Tensor::new_cpu(t2s, t3d);
        broadcasted_ops(t1, t2, t3);

        let t1 = &Tensor::new_wgpu(t1s, t1d);
        let t2 = &Tensor::new_wgpu(t2s, t2d);
        let t3 = &Tensor::new_wgpu(t2s, t3d);
        broadcasted_ops(t1, t2, t3);
    }

    fn broadcasted_ops<T: RawTensor<Elem = f32>>(t1: &Tensor<T>, t2: &Tensor<T>, t3: &Tensor<T>) {
        let (r3, r3r) = (t1 + t2, t2 + t1);
        assert_eq!(r3.to_vec(), vec![7.0, 9.0, 11.0, 10.0, 12.0, 14.0]);
        assert_eq!(r3r.to_vec(), vec![7.0, 9.0, 11.0, 10.0, 12.0, 14.0]);
        let (r4, r4r) = (t1 - t2, t2 - t1);
        assert_eq!(r4.to_vec(), vec![-5.0, -5.0, -5.0, -2.0, -2.0, -2.0]);
        assert_eq!(r4r.to_vec(), vec![5.0, 5.0, 5.0, 2.0, 2.0, 2.0]);
        let (r5, r5r) = (t1 / t2, t2 / t1);
        assert_eq!(
            r5.to_vec(),
            vec![0.16666667, 0.2857143, 0.375, 0.6666667, 0.71428573, 0.75]
        );
        assert_eq!(r5r.to_vec(), vec![6.0, 3.5, 2.6666667, 1.5, 1.4, 1.3333334]);
        let (r6, r6r) = (t1 * t2, t2 * t1);
        assert_eq!(r6.to_vec(), vec![6.0, 14.0, 24.0, 24.0, 35.0, 48.0]);
        assert_eq!(r6r.to_vec(), vec![6.0, 14.0, 24.0, 24.0, 35.0, 48.0]);

        let (r7, r7r) = (t1.pow(t2), t2.pow(t1));
        assert_vec_eq(
            &r7.to_vec(),
            &[1.0, 128.0, 6561.0, 4096.0, 78125.0, 1679616.0],
        );
        assert_vec_eq(
            &r7r.to_vec(),
            &[6.0, 49.0, 512.0, 1296.0, 16807.0, 262144.0],
        );

        let (r8, r8r) = (t1.eq(t3), t3.eq(t1));
        assert_eq!(r8.to_vec(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(r8r.to_vec(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_reduce_ops() {
        let shape = &[2, 3];
        let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = &Tensor::new_cpu(shape, t1d);
        reduce_ops(t1);

        let t1 = &Tensor::new_wgpu(shape, t1d);
        reduce_ops(t1);
    }

    fn reduce_ops<T: RawTensor<Elem = f32>>(t1: &Tensor<T>) {
        let r1 = t1.sum(&[0]);
        assert_eq!(r1.to_vec(), vec![5.0, 7.0, 9.0]);
        let r2 = t1.sum(&[1]);
        assert_eq!(r2.to_vec(), vec![6.0, 15.0]);
        let r3 = t1.max(&[0]);
        assert_eq!(r3.to_vec(), vec![4.0, 5.0, 6.0]);
        let r4 = t1.max(&[1]);
        assert_eq!(r4.to_vec(), vec![3.0, 6.0]);
    }

    #[test]
    fn test_movement_ops() {
        let shape = &[2, 3];
        let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = &Tensor::new_cpu(shape, t1d);
        movement_ops(t1);

        let t1 = &Tensor::new_wgpu(shape, t1d);
        movement_ops(t1);
    }

    fn movement_ops<T: RawTensor<Elem = f32>>(t1: &Tensor<T>) {
        let r1 = t1.reshape(&[3, 2]);
        assert_eq!(&[3, 2], r1.shape());
        assert_eq!(r1.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let r2 = t1.permute(&[1, 0]);
        assert_eq!(&[3, 2], r2.shape());
        assert_eq!(r2.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let r3 = t1.expand(&[2, 2, 3]);
        assert_eq!(&[2, 2, 3], r3.shape());
    }

    #[test]
    fn test_2x3_dot_3x2() {
        let t1d = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t2d = &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let t1 = &Tensor::new_cpu(&[2, 3], t1d);
        let t2 = &Tensor::new_cpu(&[3, 2], t2d);

        do_2x3_dot_3x2(t1, t2);

        let t1 = &Tensor::new_wgpu(&[2, 3], t1d);
        let t2 = &Tensor::new_wgpu(&[3, 2], t2d);
        do_2x3_dot_3x2(t1, t2)
    }

    fn do_2x3_dot_3x2<T: RawTensor<Elem = f32>>(t1: &Tensor<T>, t2: &Tensor<T>) {
        let r1 = t1.dot(t2);
        assert_eq!(r1.shape(), &[2, 2]);
        assert_eq!(r1.to_vec(), vec![20.0, 14.0, 56.0, 41.0]);
    }

    #[test]
    fn test_2x3x5_dot_2x5x2() {
        let t1d = (0..30).map(|x| x as f32).collect::<Vec<_>>();
        let t2d = (0..20).map(|x| x as f32).collect::<Vec<_>>();
        let t1 = &Tensor::new_cpu(&[2, 3, 5], &t1d);
        let t2 = &Tensor::new_cpu(&[2, 5, 2], &t2d);
        do_2x3x5_dot_2x5x2(t1, t2);

        let t1 = &Tensor::new_wgpu(&[2, 3, 5], &t1d);
        let t2 = &Tensor::new_wgpu(&[2, 5, 2], &t2d);
        do_2x3x5_dot_2x5x2(t1, t2);
    }

    fn do_2x3x5_dot_2x5x2<T: RawTensor<Elem = f32>>(t1: &Tensor<T>, t2: &Tensor<T>) {
        let r1 = t1.dot(t2);
        assert_eq!(r1.shape(), &[2, 3, 2]);
        assert_eq!(
            r1.to_vec(),
            vec![
                60.0, 70.0, 160.0, 195.0, 260.0, 320.0, 1210.0, 1295.0, 1560.0, 1670.0, 1910.0,
                2045.0
            ]
        );
    }
}
