use std::ops::{Index, IndexMut};

use crate::{
    diffable::Diffable,
    raw_tensor::{RawTensor, RealizedRawTensor},
    shape::Shape,
    shape_strider::ShapeStrider,
    tensor::Tensor,
};

// Implementation note:
// In as much as possible, TensorMut is conceptually a client of Tensor.
// Tensor only calls TensorMut::new from its to_tensor_mut.

// TODO: rename to TensorDirect or something - this is useful for fast
// (maybe mutable) access to an underlying buffer, which does not use lazy, fused or derivable operations.
// e.g. for display or debug purposes.

/// A mutable tensor, which owns its buffer.
/// This is useful for implementing algorithms that mutate tensors in-place,
/// and initializing tensors from outside data.
pub struct TensorMut<T> {
    buffer: Vec<T>,
    strider: ShapeStrider,
}

impl<T: Copy> TensorMut<T> {
    /// Create a new tensor with the same shape and elements as the given tensor.
    /// Copies all the `Tensor`'s data.
    pub fn new<RT: RealizedRawTensor<E = T>>(from: &Tensor<RT>) -> Self {
        let buffer = from.ravel();
        let strider = ShapeStrider::contiguous(from.shape());
        Self { buffer, strider }
    }

    /// Create a new tensor with the given shape and elements.
    /// Copies all the `TensorMut`'s data.
    pub fn to_tensor<RT: RawTensor<E = T>>(&self) -> Tensor<RT> {
        // note: to avoid the copy here, could add an `into_new` method to RawTensor
        // which consumes the buffer & shape.
        Tensor::<RT>::new(self.strider.shape(), &self.buffer)
    }

    fn validate_index(&self, index: &[usize]) {
        assert_eq!(
            index.len(),
            self.strider.shape().ndims(),
            "Index has wrong number of dimensions"
        );
        for (i, &dim) in index.iter().enumerate() {
            assert!(
                dim < self.strider.shape()[i],
                "Index out of bounds: {:?} >= {:?}",
                dim,
                self.strider.shape()[i]
            );
        }
    }

    fn index(&self, index: &[usize]) -> &T {
        self.validate_index(index);
        &self.buffer[self.strider.buffer_index(index)]
    }

    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        self.validate_index(index);
        &mut self.buffer[self.strider.buffer_index(index)]
    }
}

impl<T: Copy> Index<&[usize]> for TensorMut<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        self.index(index)
    }
}

impl<T: Copy> IndexMut<&[usize]> for TensorMut<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        self.index_mut(index)
    }
}
