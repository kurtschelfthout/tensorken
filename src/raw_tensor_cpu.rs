use std::fmt::Debug;
use std::ops::Add;
use std::rc::Rc;

use crate::num::Num;
use crate::raw_tensor::RawTensor;
use crate::shape_strider::{Shape, ShapeStrider, TensorIndexIterator};

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
    buffer: Rc<Buffer<T>>,
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
    fn new(shape: &[usize], data: Vec<T>) -> Self {
        assert!(shape.size() == data.len(), "Shape size {} and data len {} must match - either too few or too many elements in data.", shape.size(), data.len());

        let strider = ShapeStrider::contiguous(shape);

        let buffer = Rc::new(Buffer { data });

        Self { buffer, strider }
    }

    /// Return a new tensor with the same buffer as this one, but
    /// with a different shape. Assumes the new shape is compatible.
    fn with_strider(&self, strider: ShapeStrider) -> Self {
        Self {
            buffer: Rc::clone(&self.buffer),
            strider,
        }
    }

    /// Return a new tensor with the same shape as this one, but
    /// using the given Vec as data.
    /// Assumes the data is laid out contiguously, in row-major order.
    fn with_contiguous_data(&self, data: Vec<T>) -> Self {
        Self::new(self.strider.shape(), data)
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
            buffer: Rc::new(Buffer {
                data: result_buffer,
            }),
        }
    }

    #[allow(dead_code)]
    fn get(&self, index: &[usize]) -> T {
        self.buffer.data[self.strider.buffer_index(index)]
    }

    #[allow(dead_code)]
    fn strides(&self) -> &[usize] {
        self.strider.strides()
    }

    fn ravel(&self) -> Vec<T> {
        self.into_iter().collect()
    }
}

pub struct CpuRawTensorIterator<'a, T> {
    tensor: &'a CpuRawTensor<T>,
    index_iter: TensorIndexIterator<'a>, // index: Vec<usize>,
                                         // exhausted: bool,
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

    fn new(shape: &[usize], data: &[Self::Elem]) -> Self {
        Self::new(shape, data.to_vec())
    }

    fn shape(&self) -> &[usize] {
        self.strider.shape()
    }

    fn ravel(&self) -> Vec<Self::Elem> {
        self.ravel()
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
        let t = CpuRawTensor::new(&[2, 3, 4], make_vec(24));
        assert_eq!(t.ravel(), make_vec(24));
    }

    fn test_reshape_24(orig_shape: &[usize], new_shape: &[usize], expected_strides: &[usize]) {
        let t = CpuRawTensor::new(orig_shape, make_vec(24));
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
        let t = CpuRawTensor::new(orig_shape, make_vec(24));
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
        let t = CpuRawTensor::new(&[10, 10], make_vec(100));

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
        let t = CpuRawTensor::new(&[1], vec![42.0]);
        let t = t.expand(&[5, 4]);

        assert_eq!(t.shape(), &[5, 4]);
        assert_eq!(t.strides(), &[0, 0]);
        assert_eq!(t.ravel(), repeat(42.0).take(20).collect::<Vec<_>>());
    }

    #[test]
    fn test_expand_3x1() {
        let t = CpuRawTensor::new(&[3, 1], make_vec(3));
        let t = t.expand(&[15, 3, 5]);

        assert_eq!(t.shape(), &[15, 3, 5]);
        assert_eq!(t.strides(), &[0, 1, 0]);
    }

    #[test]
    fn test_expand_2x3x4() {
        let t = CpuRawTensor::new(&[2, 3, 4], make_vec(24));
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
        let t = CpuRawTensor::new(&[2, 3], make_vec(6));
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
        let t1 = CpuRawTensor::new(&[2, 3], make_vec(6));
        let t2 = CpuRawTensor::new(&[2, 3], make_vec(6));
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
        assert!(t.get(&[0, 0]).is_nan());
        assert_eq!(t.ravel()[1..6], vec![1.0; 5]);
        let t = t1.eq(&t2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ravel(), vec![1.0; 6]);
        let t = t1.eq(&t2.sub(&t1));
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.ravel()[1..6], vec![0.0; 5]);
    }

    #[test]
    fn test_binary_ops_different_strides() {
        let t1 = CpuRawTensor::new(&[1], vec![20.0]).expand(&[2, 3]);
        let t2 = CpuRawTensor::new(&[2, 3], make_vec(6));
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
        let t: CpuRawTensor<f32> = CpuRawTensor::new(&[], vec![]);
        let s = t.sum(&[]);
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.buffer.data, vec![]);

        let t: CpuRawTensor<f32> = CpuRawTensor::new(&[], vec![]);
        let s = t.max(&[]);
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.buffer.data, vec![]);
    }

    #[test]
    fn test_reduce_ops() {
        let t = CpuRawTensor::new(&[2, 3], make_vec(6));
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
}
