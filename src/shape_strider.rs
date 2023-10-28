use crate::shape::Shape;

impl Shape for ShapeStrider {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// A struct that encapsulates how tensor indexes map to buffer offsets.
/// To figure out the mapping, it uses a shape and a set of strides.
/// The strides indicate how many elements to skip in the buffer to reach the next element along each dimension.
#[derive(Debug, Clone)]
pub(crate) struct ShapeStrider {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl ShapeStrider {
    /// Create a new strider with no dimensions.
    pub(crate) fn empty() -> Self {
        Self {
            shape: vec![],
            strides: vec![],
            offset: 0,
        }
    }

    /// Create a new strider with the given shape,
    /// striding through a contiguous buffer in row-major order.
    pub(crate) fn contiguous(shape: &[usize]) -> Self {
        if shape.is_empty() {
            return Self::empty();
        }
        let shape = shape.to_vec();
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn buffer_index(&self, index: &[usize]) -> usize {
        self.offset
            + index
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>()
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        self.strides == Self::contiguous(self.shape()).strides()
    }

    fn is_valid_index(&self, index: &[usize]) -> bool {
        !self.shape.is_empty() // can't index into empty shape at all
            && self.shape.ndims() == index.len()
            && index.iter().zip(self.shape.iter()).all(|(i, s)| i < s)
    }

    /// Iterate over tensor indexes, in increasing order.
    pub(crate) fn iter_tensor_index(&self) -> TensorIndexIterator {
        TensorIndexIterator::new(self)
    }

    pub(crate) fn validate_can_zip(&self, other: &Self) -> Result<(), String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shapes must match: {:?} != {:?}",
                self.shape, other.shape
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_can_reduce(&self, axes: &[usize]) -> Result<(), String> {
        if axes.iter().any(|a| *a >= self.shape.len()) {
            return Err(format!(
                "One or more axes are out of bounds. ndims: {:?} axes: {:?}",
                self.shape.ndims(),
                axes
            ));
        }
        Ok(())
    }

    /// Remove all dimensions of length 1.
    pub(crate) fn squeeze(&self) -> Self {
        let mut shape = Vec::with_capacity(self.shape.ndims());
        let mut strides = Vec::with_capacity(self.shape.ndims());
        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if dim != 1 {
                shape.push(dim);
                strides.push(stride);
            }
        }
        Self {
            shape,
            strides,
            offset: self.offset,
        }
    }

    /// Return a new, contiguous strider with the given axes reduced to length 1,
    /// as well as a strider with the same shape as result, but with the given axes'
    /// strides set to 0. This second strider can be used to iterate over the
    /// to-be-reduced axes while accumulating.
    pub(crate) fn reduce(&self, axes: &[usize]) -> (Self, Self) {
        let mut shape = self.shape.clone();
        for &axis in axes {
            shape[axis] = 1;
        }
        let result = Self::contiguous(&shape);

        let mut strides = result.strides.clone();
        for &axis in axes {
            strides[axis] = 0;
        }
        let reducer = Self {
            shape,
            strides,
            offset: 0,
        };

        (result, reducer)
    }

    pub(crate) fn validate_can_reshape(&self, shape: &[usize]) -> Result<(), String> {
        if self.size() != shape.size() {
            return Err(format!(
                "cannot reshape tensor of size {} to shape {:?} - size must be the same.",
                self.size(),
                shape
            ));
        }
        Ok(())
    }

    /// Attempt to reshape to the given new shape. Returns Ok if the reshape is possible without copying the buffer, Err otherwise.
    pub(crate) fn reshape(&self, new_shape: &[usize]) -> Result<Self, String> {
        assert!(
            self.size() == new_shape.size(),
            "cannot reshape tensor of shape {:?} to shape {:?} - size must be the same. {} != {}",
            self.shape(),
            new_shape,
            self.shape().size(),
            new_shape.size()
        );
        let newnd = new_shape.ndims();
        let mut new_strides = vec![0; newnd];

        let squeezed = self.squeeze();
        let old_shape = &squeezed.shape;
        let oldnd = squeezed.shape.ndims();
        let old_strides = &squeezed.strides;

        let (mut oi, mut oj) = (0, 1);
        let (mut ni, mut nj) = (0, 1);
        while (ni < newnd) && (oi < oldnd) {
            // First find the dimensions in both old and new we can combine -
            // by checking that the number of elements in old and new is the same.
            let mut np = new_shape[ni];
            let mut op = old_shape[oi];
            // This loop always ends, because we're checking that the size of old and
            // new shapes are the same.
            while np != op {
                if np < op {
                    np *= new_shape[nj];
                    nj += 1;
                } else {
                    op *= old_shape[oj];
                    oj += 1;
                }
            }

            // Check if the strides in the old dimensions to combine are contiguous enough.
            // We have to be able to use a single stride for the combined dimension.
            for ok in oi..oj - 1 {
                if old_strides[ok] != old_strides[ok + 1] * old_shape[ok + 1] {
                    return Err(format!(
                        "cannot reshape tensor of shape {old_shape:?} to shape {new_shape:?} without copying."
                    ));
                }
            }

            // now calculate new strides - going back to front as usual.
            new_strides[nj - 1] = old_strides[oj - 1];
            for nk in (ni + 1..nj).rev() {
                new_strides[nk - 1] = new_strides[nk] * new_shape[nk];
            }

            ni = nj;
            nj += 1;
            oi = oj;
            oj += 1;
        }

        let last_stride = if ni >= 1 { new_strides[ni - 1] } else { 1 };
        for new_stride in new_strides.iter_mut().take(newnd).skip(ni) {
            *new_stride = last_stride;
        }

        Ok(Self {
            shape: new_shape.to_vec(),
            strides: new_strides,
            offset: self.offset,
        })
    }

    pub(crate) fn validate_can_permute(&self, permutation: &[usize]) -> Result<(), String> {
        if permutation.iter().any(|x| *x >= self.shape.ndims()) {
            return Err("Invalid permutation: at least one target axis is greater than number of dimensions.".to_string());
        }
        if permutation.len() * (permutation.len() - 1) / 2 != permutation.iter().sum() {
            return Err(
                "Invalid permutation: all axes must be specified exactly once.".to_string(),
            );
        }
        Ok(())
    }

    /// Apply the given shape permutation.
    pub(crate) fn permute(&self, permutation: &[usize]) -> Self {
        let mut shape = Vec::with_capacity(self.shape.ndims());
        let mut strides = Vec::with_capacity(self.shape.ndims());
        for &i in permutation {
            shape.push(self.shape[i]);
            strides.push(self.strides[i]);
        }
        Self {
            shape,
            strides,
            offset: self.offset,
        }
    }

    pub(crate) fn validate_can_expand(&self, shape: &[usize]) -> Result<(), String> {
        if self.shape.ndims() > shape.ndims() {
            return Err(format!(
                "Cannot expand tensor to shape {:?} from shape {:?} - new shape has fewer dimensions",
                shape, self.shape
            ));
        }
        Ok(())
    }

    /// Expand the tensor to the given shape.
    /// Returns an error if the tensor cannot be expanded to the given shape.
    pub(crate) fn expand(&self, shape: &[usize]) -> Result<Self, String> {
        let ndims = shape.ndims();
        let mut new_shape = Vec::with_capacity(ndims);
        let mut new_strides = Vec::with_capacity(ndims);
        for (fro_dim, to_dim) in (0..self.shape.ndims()).rev().zip((0..shape.ndims()).rev()) {
            if self.shape[fro_dim] == shape[to_dim] {
                new_shape.push(self.shape[fro_dim]);
                new_strides.push(self.strides[fro_dim]);
            } else if self.shape[fro_dim] == 1 {
                new_shape.push(shape[to_dim]);
                new_strides.push(0);
            } else {
                return Err(format!(
                    "Cannot expand tensor to shape {:?} from shape {:?}",
                    shape, self.shape
                ));
            }
        }
        new_shape.reverse();
        new_strides.reverse();
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    pub(crate) fn pad(&self, padding: &[(usize, usize)]) -> Self {
        let mut new_shape = self.shape.clone();
        for (i, &(before, after)) in padding.iter().enumerate() {
            new_shape[i] = self.shape[i] + before + after;
        }
        Self::contiguous(&new_shape)
    }

    pub(crate) fn validate_can_pad(&self, padding: &[(usize, usize)]) -> Result<(), String> {
        if padding.len() != self.shape.ndims() {
            return Err(format!(
                "Cannot pad tensor of shape {:?} with padding {:?} - padding must have same number of dimensions as tensor.",
                self.shape, padding
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_can_crop(&self, limits: &[(usize, usize)]) -> Result<(), String> {
        if limits.len() != self.shape.ndims() {
            return Err(format!(
                "Cannot crop tensor of shape {:?} with limits {:?} - limits must have same number of dimensions as tensor.",
                self.shape, limits
            ));
        }
        for (i, &(start, end)) in limits.iter().enumerate() {
            if start >= end {
                return Err(format!(
                    "Cannot crop tensor of shape {:?} with limits {:?} - start must be less than end for each dimension.",
                    self.shape, limits
                ));
            }
            if end > self.shape[i] {
                return Err(format!(
                    "Cannot crop tensor of shape {:?} with limits {:?} - end must be less than or equal to shape of tensor for each dimension.",
                    self.shape, limits
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn crop(&self, limits: &[(usize, usize)]) -> ShapeStrider {
        let offset = self.buffer_index(&limits.iter().map(|&(start, _)| start).collect::<Vec<_>>());
        let shape = limits
            .iter()
            .map(|&(start, end)| end - start)
            .collect::<Vec<_>>();
        Self {
            shape,
            strides: self.strides.clone(),
            offset,
        }
    }
}

pub(crate) struct TensorIndexIterator<'a> {
    strider: &'a ShapeStrider,
    index: Vec<usize>,
    exhausted: bool,
}

impl<'a> TensorIndexIterator<'a> {
    pub(crate) fn new(strider: &'a ShapeStrider) -> Self {
        let index = vec![0; strider.shape().ndims()];
        let exhausted = !strider.is_valid_index(&index);
        Self {
            strider,
            index,
            exhausted,
        }
    }
}

impl<'a> Iterator for TensorIndexIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let result = self.index.clone();
        for i in (0..self.strider.shape.ndims()).rev() {
            self.index[i] += 1;
            if self.index[i] < self.strider.shape[i] {
                break;
            }
            self.index[i] = 0;
        }
        self.exhausted = self.index.iter().all(|e| *e == 0);
        Some(result)
    }
}
