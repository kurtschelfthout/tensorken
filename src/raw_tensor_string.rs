use crate::RawTensor;

/// An implementation of
impl RawTensor for String {
    type Elem = f32;

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

    fn new(shape: &[usize], data: &[Self::Elem]) -> Self {
        format!("new({shape:?}, {data:?})")
    }

    fn shape(&self) -> &[usize] {
        panic!("shape() not implemented for String. Try ShapeTracker<String> instead.")
    }

    fn to_cpu(&self) -> crate::CpuRawTensor<Self::Elem> {
        panic!("to_cpu() not implemented for String.")
    }

    fn fused_multiply_add(&self, other: &Self, axes: &[usize]) -> Self {
        format!("{self}.fused_multiply_add({other}, {axes:?})")
    }
}
