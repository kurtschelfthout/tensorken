use std::fmt::Debug;
use std::ops::Add;
use std::sync::Arc;

use crate::num::Num;
use crate::raw_tensor::RawTensor;
use crate::shape::Shape;
use crate::shape_strider::{ShapeStrider, TensorIndexIterator};

/// Implementation of `RawTensor` for CPU.
/// The "numpy" part of the tensor library.

// in numpy, this is a byte buffer and dtypes are used to interpret the bytes.
// Ignoring that here.
#[derive(Debug)]
struct Buffer<T> {
    data: Vec<T>,
}

/// Operations avoid copying the buffer if possible, but buffers are read-only,
/// and can be shared between multiple tensors (e.g. with different shapes).
/// As a result, buffers are reference counted. Cloning a `CpuRawTensor` is cheap.
#[derive(Clone)]
pub struct CpuRawTensor<T> {
    buffer: Arc<Buffer<T>>,
    strider: ShapeStrider,
}

impl<T: Debug> Debug for CpuRawTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CpuRawTensor({:?}, {:?})", self.buffer, self.strider)
    }
}

impl<T: Copy> CpuRawTensor<T> {
    /// Return a new tensor with the given shape and data.
    /// Panics if the shape and data are not compatible.
    /// Assumes the data is laid out contiguously, in row-major order.
    pub(crate) fn new_into(shape: &[usize], data: Vec<T>) -> Self {
        assert!(shape.size() == data.len(), "Shape size {} and data len {} must match - either too few or too many elements in data.", shape.size(), data.len());

        let strider = ShapeStrider::contiguous(shape);

        let buffer = Arc::new(Buffer { data });

        Self { buffer, strider }
    }

    /// Return a new tensor with the same buffer as this one, but
    /// with a different shape. Assumes the new shape is compatible.
    fn with_strider(&self, strider: ShapeStrider) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            strider,
        }
    }

    /// Return a new tensor with the same shape as this one, but
    /// using the given Vec as data.
    /// Assumes the data is laid out contiguously, in row-major order.
    fn with_contiguous_data(&self, data: Vec<T>) -> Self {
        Self::new_into(self.strider.shape(), data)
    }

    /// Return a new tensor with the same shape as this one, but
    /// copied into a new buffer contiguously.
    fn contiguous(&self) -> Self {
        let data = self.ravel();
        self.with_contiguous_data(data)
    }

    /// Return a new tensor with the same shape as self, after applying f to each element.
    /// Allocates a new buffer, resulting tensor is contiguous.
    fn map(&self, f: impl Fn(T) -> T) -> Self {
        let mut result = Vec::with_capacity(self.strider.size());
        for &x in &self.buffer.data {
            result.push(f(x));
        }

        self.with_contiguous_data(result)
    }

    /// Return a new tensor with the same shape as self and other, after applying f to each pair of elements.
    /// Panics if the shapes are not identical.
    /// Allocates a new buffer, resulting tensor is contiguous.
    fn zip(&self, other: &Self, f: impl Fn(T, T) -> T) -> Self {
        self.strider.validate_can_zip(&other.strider).unwrap();

        let mut result: Vec<_> = self.into_iter().collect();
        for (x, y) in result.iter_mut().zip(other.into_iter()) {
            *x = f(*x, y);
        }

        self.with_contiguous_data(result)
    }

    /// Return a new tensor with the given axes reduced to length 1, via at given function.
    /// The reduced dimensions are not removed.
    /// Allocates a new buffer, resulting tensor is contiguous.
    fn reduce(&self, default: T, f: impl Fn(T, T) -> T, axes: &[usize]) -> Self {
        self.strider.validate_can_reduce(axes).unwrap();

        let (strider, reducer) = self.strider.reduce(axes);
        let mut result_buffer = vec![default; strider.size()];

        for self_index in self.strider.iter_tensor_index() {
            let self_buffer_index = self.strider.buffer_index(&self_index);
            let result_buffer_index = reducer.buffer_index(&self_index);
            result_buffer[result_buffer_index] = f(
                result_buffer[result_buffer_index],
                self.buffer.data[self_buffer_index],
            );
        }

        Self {
            strider,
            buffer: Arc::new(Buffer {
                data: result_buffer,
            }),
        }
    }

    fn fused_zip_reduce(
        &self,
        other: &Self,
        axes: &[usize],
        default: T,
        fzr: impl Fn(T, T, T) -> T,
    ) -> Self {
        self.strider.validate_can_zip(&other.strider).unwrap();
        self.strider.validate_can_reduce(axes).unwrap();

        let (strider_0, reducer_0) = self.strider.reduce(axes);

        let mut result_buffer = vec![default; strider_0.size()];

        for (self_index, other_index) in self
            .strider
            .iter_tensor_index()
            .zip(other.strider.iter_tensor_index())
        {
            let self_buffer_index = self.strider.buffer_index(&self_index);
            let other_buffer_index = other.strider.buffer_index(&other_index);
            let result_buffer_index = reducer_0.buffer_index(&self_index);
            result_buffer[result_buffer_index] = fzr(
                result_buffer[result_buffer_index],
                self.buffer.data[self_buffer_index],
                other.buffer.data[other_buffer_index],
            );
        }

        Self {
            strider: strider_0,
            buffer: Arc::new(Buffer {
                data: result_buffer,
            }),
        }
    }

    #[allow(dead_code)]
    fn strides(&self) -> &[usize] {
        self.strider.strides()
    }

    #[must_use]
    pub fn ravel(&self) -> Vec<T> {
        self.into_iter().collect()
    }
}

pub struct CpuRawTensorIterator<'a, T> {
    tensor: &'a CpuRawTensor<T>,
    index_iter: TensorIndexIterator<'a>,
}

impl<'a, T: Copy> Iterator for CpuRawTensorIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter
            .next()
            .map(|index| self.tensor.buffer.data[self.tensor.strider.buffer_index(&index)])
    }
}

impl<'a, T: Copy> IntoIterator for &'a CpuRawTensor<T> {
    type Item = T;
    type IntoIter = CpuRawTensorIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        CpuRawTensorIterator {
            tensor: self,
            index_iter: self.strider.iter_tensor_index(),
        }
    }
}

impl<T: Num> RawTensor for CpuRawTensor<T> {
    type Elem = T;
    fn exp(&self) -> Self {
        self.map(Num::exp)
    }

    fn log(&self) -> Self {
        self.map(Num::log)
    }

    fn add(&self, other: &Self) -> Self {
        self.zip(other, |x, y| x + y)
    }

    fn sub(&self, other: &Self) -> Self {
        self.zip(other, |x, y| x - y)
    }

    fn mul(&self, other: &Self) -> Self {
        self.zip(other, |x, y| x * y)
    }

    fn div(&self, other: &Self) -> Self {
        self.zip(other, |x, y| x / y)
    }

    fn pow(&self, other: &Self) -> Self {
        self.zip(other, Num::powf)
    }

    fn eq(&self, other: &Self) -> Self {
        self.zip(other, |x, y| if x == y { T::ONE } else { T::ZERO })
    }

    fn sum(&self, axes: &[usize]) -> Self {
        self.reduce(T::ZERO, Add::add, axes)
    }

    fn max(&self, axes: &[usize]) -> Self {
        self.reduce(T::MIN, |x, y| if x > y { x } else { y }, axes)
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        // good explanation of striding and reshaping - and when copy is needed:
        // https://ajcr.net/stride-guide-part-1/
        self.strider.validate_can_reshape(shape).unwrap();

        if let Ok(strider) = self.strider.reshape(shape) {
            return self.with_strider(strider);
        }
        self.contiguous().reshape(shape)
    }

    fn permute(&self, permutation: &[usize]) -> Self {
        self.strider.validate_can_permute(permutation).unwrap();

        let strider = self.strider.permute(permutation);
        self.with_strider(strider)
    }

    fn expand(&self, shape: &[usize]) -> Self {
        self.strider.validate_can_expand(shape).unwrap();

        let strider = self.strider.expand(shape).unwrap();
        self.with_strider(strider)
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        self.strider.validate_can_pad(padding).unwrap();

        let strider = self.strider.pad(padding);
        let mut buffer = vec![T::ZERO; strider.size()];
        for mut index in self.strider.iter_tensor_index() {
            // first get the value in the unpadded tensor
            let value = self.buffer.data[self.strider.buffer_index(&index)];
            // change the index to take padding into account
            for (i, (l, _)) in index.iter_mut().zip(padding) {
                *i += l;
            }
            // write the value to the padded tensor's buffer
            let buffer_index = strider.buffer_index(&index);
            buffer[buffer_index] = value;
        }
        Self {
            strider,
            buffer: Arc::new(Buffer { data: buffer }),
        }
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        self.strider.validate_can_crop(limits).unwrap();

        let strider = self.strider.crop(limits);
        self.with_strider(strider)
    }

    fn new(shape: &[usize], data: &[Self::Elem]) -> Self {
        Self::new_into(shape, data.to_vec())
    }

    fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    fn to_cpu(&self) -> CpuRawTensor<Self::Elem> {
        self.clone()
    }

    fn fused_multiply_add(&self, other: &Self, axes: &[usize]) -> Self {
        self.fused_zip_reduce(other, axes, T::ZERO, |acc, x, y| acc + x * y)
    }
}

#[cfg(test)]
mod tests {

    use std::{iter::repeat, vec};

    use super::*;

    fn make_vec(len: u16) -> Vec<f32> {
        (0..len).map(f32::from).collect()
    }
    #[test]
    fn test_ravel() {
        let t = CpuRawTensor::new_into(&[2, 3, 4], make_vec(24));
        assert_eq!(t.ravel(), make_vec(24));
    }

    fn test_reshape_24(orig_shape: &[usize], new_shape: &[usize], expected_strides: &[usize]) {
        let t = CpuRawTensor::new_into(orig_shape, make_vec(24));
        let t = t.reshape(new_shape);
        assert_eq!(t.shape(), new_shape);
        assert_eq!(t.strides(), expected_strides);
        assert_eq!(t.ravel(), make_vec(24));
    }

    #[test]
    fn test_reshape() {
        test_reshape_24(&[24], &[3, 2, 4], &[8, 4, 1]);
        test_reshape_24(&[2, 1, 3, 1, 4], &[2, 3, 4], &[12, 4, 1]);
        test_reshape_24(&[2, 1, 3, 1, 4], &[2, 3, 4, 1], &[12, 4, 1, 1]);
    }

    fn test_permute_24(orig_shape: &[usize], permutation: &[usize], expected_shape: &[usize]) {
        let t = CpuRawTensor::new_into(orig_shape, make_vec(24));
        let tp = t.permute(permutation);
        assert_eq!(tp.shape(), expected_shape);
        assert_ne!(tp.strides(), t.strides());
        let ravel_data = &tp.ravel();
        assert_ne!(ravel_data, &tp.buffer.data);

        let rev_perm = (0..permutation.len())
            .map(|i| permutation.iter().position(|&x| x == i).unwrap())
            .collect::<Vec<_>>();
        let tpp = tp.permute(&rev_perm);
        assert_eq!(tpp.shape(), orig_shape);
        assert_eq!(tpp.strides(), t.strides());
        assert_eq!(tpp.ravel(), t.ravel());
    }

    #[test]
    fn test_permute() {
        test_permute_24(&[6, 4], &[1, 0], &[4, 6]);
        test_permute_24(&[2, 3, 4], &[2, 0, 1], &[4, 2, 3]);
    }

    #[test]
    fn test_reshape_permute_reshape() {
        // test from tinygrad abstractions.py
        let t = CpuRawTensor::new_into(&[10, 10], make_vec(100));

        let tp = t.permute(&[1, 0]);
        assert_eq!(tp.shape(), &[10, 10]);
        assert_eq!(tp.strides(), &[1, 10]);

        let tpr = tp.reshape(&[5, 2, 5, 2]);
        assert_eq!(tpr.shape(), &[5, 2, 5, 2]);
        assert_eq!(tpr.strides(), &[2, 1, 20, 10]);

        let tpcopy = tpr.reshape(&[100]);
        assert_eq!(tpcopy.shape(), &[100]);
        assert_eq!(tpcopy.strides(), &[1]);

        let tprr = tpr.reshape(&[10, 10]);
        assert_eq!(tprr.shape(), &[10, 10]);
        assert_eq!(tprr.strides(), &[1, 10]);

        let tprrt = tprr.permute(&[1, 0]);
        assert_eq!(tprrt.shape(), &[10, 10]);
        assert_eq!(tprrt.strides(), &[10, 1]);
    }

    #[test]
    fn test_expand_scalar() {
        let t = CpuRawTensor::new_into(&[1], vec![42.0]);
        let t = t.expand(&[4]);

        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.strides(), &[0]);
        assert_eq!(t.ravel(), repeat(42.0).take(4).collect::<Vec<_>>());
    }

    #[test]
    fn test_expand_3x1() {
        let t = CpuRawTensor::new_into(&[3, 1], make_vec(3));
        let t = t.expand(&[3, 5]);

        assert_eq!(t.shape(), &[3, 5]);
        assert_eq!(t.strides(), &[1, 0]);
    }

    #[test]
    fn test_expand_1x2x3x4() {
        let t = CpuRawTensor::new_into(&[1, 2, 3, 4], make_vec(24));
        let t = t.expand(&[5, 2, 3, 4]);

        assert_eq!(t.shape(), &[5, 2, 3, 4]);
        assert_eq!(t.strides(), &[0, 12, 4, 1]);
        assert_eq!(
            t.ravel(),
            make_vec(24)
                .into_iter()
                .cycle()
                .take(5 * 24)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_unary_ops() {
        let t = CpuRawTensor::new_into(&[2, 3], make_vec(6));
        let t = t.exp();
        assert_eq!(t.shape(), &[2, 3]);
        let t = t.log();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(
            t.ravel().iter().map(|x| x.floor()).collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_binary_ops() {
        let t1 = CpuRawTensor::new_into(&[2, 3], make_vec(6));
        let t2 = CpuRawTensor::new_into(&[2, 3], make_vec(6));
        let t = t1.add(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ravel(), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let t = t1.sub(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ravel(), vec![0.0; 6]);
        let t = t1.mul(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ravel(), vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
        let t = t1.div(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.buffer.data[t.strider.buffer_index(&[0, 0])].is_nan());
        assert_eq!(t.ravel()[1..6], vec![1.0; 5]);
        let t = t1.eq(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ravel(), vec![1.0; 6]);
        let t = t1.eq(&t2.sub(&t1));
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.buffer.data[t.strider.buffer_index(&[0, 0])], 1.0);
        assert_eq!(t.ravel()[1..6], vec![0.0; 5]);
    }

    #[test]
    fn test_binary_ops_different_strides() {
        let t1 = CpuRawTensor::new_into(&[1, 1], vec![20.0]).expand(&[2, 3]);
        let t2 = CpuRawTensor::new_into(&[2, 3], make_vec(6));
        let t = t1.add(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ravel(), vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0]);
        let t = t1.sub(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ravel(), vec![20.0, 19.0, 18.0, 17.0, 16.0, 15.0]);
        let t = t1.mul(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ravel(), vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0]);
        let t = t1.div(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(
            t.ravel()[1..6], // first one is nan
            vec![20.0, 10.0, 6.666_666_6, 5.0, 4.0]
        );
    }

    #[test]
    fn test_reduce_ops_empty() {
        let t: CpuRawTensor<f32> = CpuRawTensor::new_into(&[], vec![]);
        let s = t.sum(&[]);
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.buffer.data, vec![]);

        let t: CpuRawTensor<f32> = CpuRawTensor::new_into(&[], vec![]);
        let s = t.max(&[]);
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.buffer.data, vec![]);
    }

    #[test]
    fn test_reduce_ops() {
        let t = CpuRawTensor::new_into(&[2, 3], make_vec(6));
        let s = t.sum(&[0]);
        assert_eq!(s.shape(), &[1, 3]);
        assert_eq!(s.buffer.data, vec![3.0, 5.0, 7.0]);
        let s = t.sum(&[1]);
        assert_eq!(s.shape(), &[2, 1]);
        assert_eq!(s.buffer.data, vec![3.0, 12.0]);
        let s = t.sum(&[0, 1]);
        assert_eq!(s.shape(), &[1, 1]);
        assert_eq!(s.buffer.data, vec![15.0]);

        let s = t.max(&[0]);
        assert_eq!(s.shape(), &[1, 3]);
        assert_eq!(s.buffer.data, vec![3.0, 4.0, 5.0]);
        let s = t.max(&[1]);
        assert_eq!(s.shape(), &[2, 1]);
        assert_eq!(s.buffer.data, vec![2.0, 5.0]);
        let s = t.max(&[0, 1]);
        assert_eq!(s.shape(), &[1, 1]);
        assert_eq!(s.buffer.data, vec![5.0]);
    }

    #[test]
    fn test_reduce_ops_non_contiguous() {
        let t = CpuRawTensor::new_into(&[2, 3], make_vec(6)).permute(&[1, 0]);
        let s = t.sum(&[0]);
        assert_eq!(s.shape(), &[1, 2]);
        assert_eq!(s.buffer.data, vec![3.0, 12.0]);
        let s = t.sum(&[1]);
        assert_eq!(s.shape(), &[3, 1]);
        assert_eq!(s.buffer.data, vec![3.0, 5.0, 7.0]);
        let s = t.sum(&[0, 1]);
        assert_eq!(s.shape(), &[1, 1]);
        assert_eq!(s.buffer.data, vec![15.0]);
    }

    #[test]
    fn test_crop() {
        let orig_shape = &[2, 3, 4];
        let t = CpuRawTensor::new_into(orig_shape, make_vec(24));

        // crop single dimension
        let s = t.crop(&[(0, 1), (0, 3), (0, 4)]);
        assert_eq!(s.ravel(), make_vec(12));

        let s = t.crop(&[(1, 2), (0, 3), (0, 4)]);
        assert_eq!(
            s.ravel(),
            make_vec(12).iter().map(|x| x + 12.0).collect::<Vec<_>>()
        );

        // crop nothing
        let s = t.crop(&[(0, 2), (0, 3), (0, 4)]);
        assert_eq!(s.shape(), orig_shape);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.ravel(), t.ravel());

        let s = t.crop(&[(0, 2), (0, 3), (1, 3)]);
        assert_eq!(s.shape(), &[2, 3, 2]);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(
            s.ravel(),
            vec![1.0, 2.0, 5., 6., 9., 10., 13., 14., 17., 18., 21., 22.]
        );

        // keep cropping
        let s = s.crop(&[(0, 1), (1, 2), (0, 2)]);
        assert_eq!(s.shape(), &[1, 1, 2]);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.ravel(), vec![5.0, 6.0]);

        // crop to single element
        let s = s.crop(&[(0, 1), (0, 1), (1, 2)]);
        assert_eq!(s.shape(), &[1, 1, 1]);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.ravel(), vec![6.0]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_pad() {
        let orig_shape = &[2, 3, 4];
        let t = CpuRawTensor::new_into(orig_shape, make_vec(24));

        // pad nothing
        let s = t.pad(&[(0, 0), (0, 0), (0, 0)]);
        assert_eq!(s.shape(), orig_shape);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.ravel(), t.ravel());

        // pad everything
        let padding = &[(1, 2), (3, 4), (5, 6)];
        let s = t.pad(padding);
        assert_eq!(s.shape(), &[5, 10, 15]);
        assert_eq!(s.strides(), &[150, 15, 1]);
        let s_raveled = s.ravel();
        assert_eq!(s_raveled.iter().filter(|&&x| x != 0.0).count(), 23);
        assert_eq!(s_raveled[s.strider.buffer_index(&[1, 3, 6])], 1.0);
        assert_eq!(s_raveled[s.strider.buffer_index(&[1, 3, 7])], 2.0);
    }

    #[test]
    fn test_fused_multiply_add() {
        // contiguous
        let t1 = CpuRawTensor::new_into(&[2, 3], make_vec(6));
        let t2 = t1.add(&CpuRawTensor::new_into(&[2, 3], make_vec(6)));

        let r = t1.fused_multiply_add(&t2, &[0]);
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.buffer.data, vec![18.0, 34.0, 58.0]);

        let r = t1.fused_multiply_add(&t2, &[1]);
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.buffer.data, vec![10.0, 100.0]);

        let r = t1.fused_multiply_add(&t2, &[0, 1]);
        assert_eq!(r.shape(), &[1, 1]);
        assert_eq!(r.buffer.data, vec![110.0]);

        // different strides
        let t1 = CpuRawTensor::new_into(&[1, 1], vec![8.0]).expand(&[2, 3]);
        let t2 = CpuRawTensor::new_into(&[2, 3], make_vec(6));

        let r = t1.fused_multiply_add(&t2, &[0]);
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.buffer.data, vec![24.0, 40.0, 56.0]);

        let r = t1.fused_multiply_add(&t2, &[1]);
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.buffer.data, vec![24.0, 96.0]);

        let r = t1.fused_multiply_add(&t2, &[0, 1]);
        assert_eq!(r.shape(), &[1, 1]);
        assert_eq!(r.buffer.data, vec![120.0]);

        // non_contiguous
        let t1 = CpuRawTensor::new_into(&[2, 3], make_vec(6)).permute(&[1, 0]);
        let t2 = t1.add(&CpuRawTensor::new_into(&[2, 3], make_vec(6)).permute(&[1, 0]));
        let r = t1.fused_multiply_add(&t2, &[0]);
        assert_eq!(r.shape(), &[1, 2]);
        assert_eq!(r.buffer.data, vec![10.0, 100.0]);

        let r = t1.fused_multiply_add(&t2, &[1]);
        assert_eq!(r.shape(), &[3, 1]);
        assert_eq!(r.buffer.data, vec![18.0, 34.0, 58.0]);

        let r = t1.fused_multiply_add(&t2, &[0, 1]);
        assert_eq!(r.shape(), &[1, 1]);
        assert_eq!(r.buffer.data, vec![110.0]);
    }
}
