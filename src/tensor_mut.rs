use std::ops::{Index, IndexMut};

use crate::{
    num::Elem, raw_tensor::ToCpu, shape::Shape, shape_strider::ShapeStrider, tensor::Tensor,
};

// Implementation note:
// In as much as possible, TensorMut is conceptually a client of Tensor.
// Tensor only calls TensorMut::new from its to_tensor_mut.

// Uuseful for fast access to an underlying buffer, which does not use lazy, fused or derivable operations.
// e.g. for display or debug purposes.

/// A mutable tensor, which owns its buffer.
/// This is useful for implementing algorithms that mutate tensors in-place,
/// and initializing tensors from outside data.
/// Since it owns its buffer, regular `Index` and `IndexMut` are implemented.
pub struct TensorMut<E> {
    buffer: Vec<E>,
    strider: ShapeStrider,
}

impl<E: Elem> TensorMut<E> {
    /// Create a new tensor with the same shape and elements as the given tensor.
    /// Copies all the `Tensor`'s data.
    pub fn new<T, I: ToCpu<Repr<E> = T>>(from: &Tensor<T, E, I>) -> Self {
        let buffer = from.ravel();
        let strider = ShapeStrider::contiguous(from.shape());
        Self { buffer, strider }
    }

    /// Create a new tensor with the given shape and elements.
    /// Copies all the `TensorMut`'s data.
    #[must_use]
    pub fn to_tensor<T, I: ToCpu<Repr<E> = T>>(&self) -> Tensor<T, E, I> {
        // note: to avoid the copy here, could add an `into_new` method to RawTensor
        // which consumes the buffer & shape.
        Tensor::<T, E, I>::new(self.strider.shape(), &self.buffer)
    }
}

impl<E: Copy> TensorMut<E> {
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

    fn index(&self, index: &[usize]) -> &E {
        self.validate_index(index);
        &self.buffer[self.strider.buffer_index(index)]
    }

    fn index_mut(&mut self, index: &[usize]) -> &mut E {
        self.validate_index(index);
        &mut self.buffer[self.strider.buffer_index(index)]
    }
}

impl<E: Copy> Index<&[usize]> for TensorMut<E> {
    type Output = E;

    fn index(&self, index: &[usize]) -> &Self::Output {
        self.index(index)
    }
}

impl<T: Copy> IndexMut<&[usize]> for TensorMut<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        self.index_mut(index)
    }
}
