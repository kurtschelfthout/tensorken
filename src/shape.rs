/// A trait for types that can be used as shapes for tensors,
/// with some convenience methods for working with shapes.
pub trait Shape {
    /// Returns the shape as a slice.
    fn shape(&self) -> &[usize];

    /// Returns the number of dimensions.
    fn ndims(&self) -> usize {
        self.shape().len()
    }

    /// Returns the total number of elements.
    fn size(&self) -> usize {
        if self.ndims() == 0 {
            0
        } else {
            self.shape().iter().product()
        }
    }
}

impl Shape for &[usize] {
    fn shape(&self) -> &[usize] {
        self
    }
}

impl Shape for Vec<usize> {
    fn shape(&self) -> &[usize] {
        self
    }
}
