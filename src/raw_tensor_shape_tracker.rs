use crate::{
    num::{Float, Num, ZeroOne},
    raw_tensor::RealizedRawTensor,
    shape_strider::ShapeStrider,
    RawTensor, Shape,
};

#[derive(Clone)]
pub struct ShapeTracker<T>(ShapeStrider, T);

impl<TRaw: RawTensor> ShapeTracker<TRaw> {
    fn new(shape: &[usize], data: &[TRaw::E]) -> Self {
        Self(ShapeStrider::contiguous(shape), TRaw::new(shape, data))
    }
}

/// This implementation passes every operation through
/// to self.1, except for shape.
impl<TRaw: RawTensor> RawTensor for ShapeTracker<TRaw> {
    type E = TRaw::E;

    fn exp(&self) -> Self
    where
        Self::E: Float,
    {
        Self(self.0.clone(), self.1.exp())
    }

    fn log(&self) -> Self
    where
        Self::E: Float,
    {
        Self(self.0.clone(), self.1.log())
    }

    fn add(&self, other: &Self) -> Self
    where
        Self::E: Num,
    {
        Self(self.0.clone(), self.1.add(&other.1))
    }

    fn sub(&self, other: &Self) -> Self
    where
        Self::E: Num,
    {
        Self(self.0.clone(), self.1.sub(&other.1))
    }

    fn mul(&self, other: &Self) -> Self
    where
        Self::E: Num,
    {
        Self(self.0.clone(), self.1.mul(&other.1))
    }

    fn div(&self, other: &Self) -> Self
    where
        Self::E: Num,
    {
        Self(self.0.clone(), self.1.div(&other.1))
    }

    fn pow(&self, other: &Self) -> Self
    where
        Self::E: Float,
    {
        Self(self.0.clone(), self.1.pow(&other.1))
    }

    fn eq(&self, other: &Self) -> Self
    where
        Self::E: ZeroOne,
    {
        Self(self.0.clone(), self.1.eq(&other.1))
    }

    fn sum(&self, axes: &[usize]) -> Self
    where
        Self::E: Num,
    {
        Self(self.0.reduce(axes).0, self.1.sum(axes))
    }

    fn max(&self, axes: &[usize]) -> Self
    where
        Self::E: ZeroOne,
    {
        Self(self.0.reduce(axes).0, self.1.max(axes))
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        Self(
            self.0
                .reshape(shape)
                .unwrap_or(ShapeStrider::contiguous(shape)),
            self.1.reshape(shape),
        )
    }

    fn permute(&self, permutation: &[usize]) -> Self {
        Self(self.0.permute(permutation), self.1.permute(permutation))
    }

    fn expand(&self, shape: &[usize]) -> Self {
        Self(self.0.expand(shape).unwrap(), self.1.expand(shape))
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self
    where
        Self::E: ZeroOne,
    {
        Self(self.0.pad(padding), self.1.pad(padding))
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        Self(self.0.crop(limits), self.1.crop(limits))
    }

    fn new(shape: &[usize], data: &[Self::E]) -> Self {
        Self::new(shape, data)
    }

    fn shape(&self) -> &[usize] {
        // the only operation that is not passed on to the underlying rawtensor.
        self.0.shape()
    }

    fn fused_multiply_add(&self, other: &Self, axes: &[usize]) -> Self
    where
        Self::E: Num,
    {
        Self(
            self.0.reduce(axes).0,
            self.1.fused_multiply_add(&other.1, axes),
        )
    }
}

impl<TRaw: RealizedRawTensor> RealizedRawTensor for ShapeTracker<TRaw> {
    fn to_cpu(&self) -> crate::CpuRawTensor<Self::E> {
        self.1.to_cpu()
    }

    fn realize(&self) -> Self {
        Self(self.0.clone(), self.1.realize())
    }
}

#[cfg(test)]
mod tests {

    use crate::CpuRawTensor;

    use super::*;

    #[test]
    fn test_shape_tracks() {
        let t1: ShapeTracker<CpuRawTensor<f32>> = RawTensor::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: ShapeTracker<CpuRawTensor<f32>> = RawTensor::new(&[2, 2], &[5., 6., 7., 8.]);
        let result = [
            t1.exp(),
            t1.log(),
            t1.add(&t2),
            t1.sub(&t2),
            t1.mul(&t2),
            t1.div(&t2),
            t1.pow(&t2),
            t1.eq(&t2),
            t1.reshape(&[1, 4]),
            t1.crop(&[(0, 1), (0, 1)]),
            t1.pad(&[(1, 2), (3, 5)]),
            t1.reshape(&[1, 4]).expand(&[4, 4]),
            t1.sum(&[0]),
            t1.max(&[0]),
            t1.permute(&[1, 0]),
            t1.fused_multiply_add(&t2, &[0]),
        ];
        for result in &result {
            assert_eq!(result.1.shape(), result.shape());
        }
    }

    #[test]
    fn test_shape_is_not_passed_through() {
        // check that calling shape does not panic - it would if passed through to String as RawTensor
        let t1: ShapeTracker<String> = RawTensor::new(&[2, 2], &[1., 2., 3., 4.]);
        t1.shape();
    }
}
