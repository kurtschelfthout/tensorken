use crate::RawTensor;

/// An implementation of
impl RawTensor for String {
    type E = f32;

    fn exp(&self) -> Self {
        format!("{self}.exp()")
    }

    fn log(&self) -> Self {
        format!("{self}.log()")
    }

    fn add(&self, other: &Self) -> Self {
        format!("({self} + {other})")
    }

    fn sub(&self, other: &Self) -> Self {
        format!("({self} - {other})")
    }

    fn mul(&self, other: &Self) -> Self {
        format!("({self} * {other})")
    }

    fn div(&self, other: &Self) -> Self {
        format!("({self} / {other})")
    }

    fn pow(&self, other: &Self) -> Self {
        format!("{self}.pow({other})")
    }

    fn eq(&self, other: &Self) -> Self {
        format!("({self} == {other})")
    }

    fn sum(&self, axes: &[usize]) -> Self {
        format!("{self}.sum({axes:?})")
    }

    fn max(&self, axes: &[usize]) -> Self {
        format!("{self}.max({axes:?})")
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        format!("{self}.reshape({shape:?})")
    }

    fn permute(&self, permutation: &[usize]) -> Self {
        format!("{self}.permute({permutation:?})")
    }

    fn expand(&self, shape: &[usize]) -> Self {
        format!("{self}.expand({shape:?})")
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        format!("{self}.pad({padding:?})")
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        format!("{self}.crop({limits:?})")
    }

    fn new(shape: &[usize], data: &[Self::E]) -> Self {
        format!("new({shape:?}, {data:?})")
    }

    fn shape(&self) -> &[usize] {
        panic!("shape() not implemented for String. Try ShapeTracker<String> instead.")
    }

    fn fused_multiply_add(&self, other: &Self, axes: &[usize]) -> Self {
        format!("{self}.fused_multiply_add({other}, {axes:?})")
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_shape_tracks() {
        let t1: String = RawTensor::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: String = RawTensor::new(&[2, 2], &[5., 6., 7., 8.]);
        let r = t1.exp().add(&t2.log());
        assert_eq!(
            r,
            "(new([2, 2], [1.0, 2.0, 3.0, 4.0]).exp() + new([2, 2], [5.0, 6.0, 7.0, 8.0]).log())"
        );
    }

    #[test]
    fn test_symbolic() {
        let t1: String = "A".to_string();
        let t2: String = "B".to_string();
        let r = t1.exp().add(&t2.log());
        assert_eq!(r, "(A.exp() + B.log())");
    }
}
