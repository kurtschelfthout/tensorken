use core::panic;

use crate::{
    num::{Float, Num},
    raw_tensor::{CastInto, RealizedRawTensor},
    RawTensor,
};

#[derive(Clone, Debug)]
pub enum Fuse<T> {
    Mul(T, T),
    NotMul(T),
}

impl<T> Fuse<T> {
    fn new(t: T) -> Self {
        Self::NotMul(t)
    }
}

impl<TRaw: RawTensor + Clone + 'static> Fuse<TRaw>
where
    TRaw::E: Num,
{
    fn run(&self) -> TRaw {
        match self {
            Self::Mul(lhs, rhs) => lhs.mul(rhs),
            Self::NotMul(t) => t.clone(),
        }
    }

    pub fn realize(&self) -> Self {
        match self {
            Self::Mul(lhs, rhs) => Self::new(lhs.mul(rhs)),
            Self::NotMul(_) => self.clone(),
        }
    }
}

fn unary_no_fuse<TRaw: RawTensor + 'static>(
    s: &Fuse<TRaw>,
    f: impl Fn(&TRaw) -> TRaw + 'static,
) -> Fuse<TRaw>
where
    TRaw::E: Num,
{
    match s {
        Fuse::Mul(lhs, rhs) => Fuse::NotMul(f(&lhs.mul(rhs))),
        Fuse::NotMul(t) => Fuse::NotMul(f(t)),
    }
}

fn binary_no_fuse<TRaw: RawTensor + 'static>(
    lhs: &Fuse<TRaw>,
    rhs: &Fuse<TRaw>,
    f: impl Fn(&TRaw, &TRaw) -> TRaw + 'static,
) -> Fuse<TRaw>
where
    TRaw::E: Num,
{
    match (lhs, rhs) {
        (Fuse::Mul(lhs1, rhs1), Fuse::Mul(lhs2, rhs2)) => {
            Fuse::new(f(&lhs1.mul(rhs1), &lhs2.mul(rhs2)))
        }
        (Fuse::Mul(lhs1, rhs1), Fuse::NotMul(t)) => Fuse::NotMul(f(&lhs1.mul(rhs1), t)),
        (Fuse::NotMul(t), Fuse::Mul(lhs2, rhs2)) => Fuse::NotMul(f(t, &lhs2.mul(rhs2))),
        (Fuse::NotMul(t1), Fuse::NotMul(t2)) => Fuse::NotMul(f(t1, t2)),
    }
}

// fn combine_axes(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
//     let mut axes = lhs.to_vec();
//     axes.extend_from_slice(rhs);
//     axes.sort_unstable();
//     axes.dedup();
//     axes
// }

impl<TRaw: RawTensor + Clone + 'static> RawTensor for Fuse<TRaw>
where
    // Since fusing mul and sum only makes sense for Num types,
    // we can constrain that for all methods here. This means
    // we don't need to add individual where clauses to each method,
    // except where Float is required.
    TRaw::E: Num,
{
    type E = TRaw::E;

    fn exp(&self) -> Self
    where
        Self::E: Float,
    {
        unary_no_fuse(self, TRaw::exp)
    }

    fn log(&self) -> Self
    where
        Self::E: Float,
    {
        unary_no_fuse(self, TRaw::log)
    }

    fn add(&self, other: &Self) -> Self {
        binary_no_fuse(self, other, TRaw::add)
    }

    fn sub(&self, other: &Self) -> Self {
        binary_no_fuse(self, other, TRaw::sub)
    }

    fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Fuse::Mul(lhs1, rhs1), Fuse::Mul(lhs2, rhs2)) => {
                Fuse::Mul(lhs1.mul(rhs1), lhs2.mul(rhs2))
            }
            (Fuse::Mul(lhs1, rhs1), Fuse::NotMul(t)) => Fuse::Mul(lhs1.mul(rhs1), t.clone()),
            (Fuse::NotMul(t), Fuse::Mul(lhs2, rhs2)) => Fuse::Mul(t.clone(), lhs2.mul(rhs2)),
            (Fuse::NotMul(t1), Fuse::NotMul(t2)) => Fuse::Mul(t1.clone(), t2.clone()),
        }
    }

    fn div(&self, other: &Self) -> Self {
        binary_no_fuse(self, other, TRaw::div)
    }

    fn pow(&self, other: &Self) -> Self
    where
        TRaw::E: Float,
    {
        binary_no_fuse(self, other, TRaw::pow)
    }

    fn eq(&self, other: &Self) -> Self {
        binary_no_fuse(self, other, TRaw::eq)
    }

    fn sum(&self, axes: &[usize]) -> Self {
        match self {
            Fuse::Mul(lhs, rhs) => Fuse::NotMul(lhs.fused_multiply_add(rhs, axes)),
            Fuse::NotMul(t) => Fuse::NotMul(t.sum(axes)),
        }
    }

    fn max(&self, axes: &[usize]) -> Self {
        let vec = axes.to_vec();
        unary_no_fuse(self, move |x| x.max(&vec))
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        let vec = shape.to_vec();
        unary_no_fuse(self, move |x| x.reshape(&vec))
    }

    fn permute(&self, permutation: &[usize]) -> Self {
        let vec = permutation.to_vec();
        unary_no_fuse(self, move |x| x.permute(&vec))
    }

    fn expand(&self, shape: &[usize]) -> Self {
        let vec = shape.to_vec();
        unary_no_fuse(self, move |x| x.expand(&vec))
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        let vec = padding.to_vec();
        unary_no_fuse(self, move |x| x.pad(&vec))
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        let vec = limits.to_vec();
        unary_no_fuse(self, move |x| x.crop(&vec))
    }

    fn new(shape: &[usize], data: &[Self::E]) -> Self {
        let traw = TRaw::new(shape, data);
        Self::new(traw)
    }

    fn shape(&self) -> &[usize] {
        panic!("shape() can't be implemented for Fuse. Wrap it in ShapeTracker.")
    }

    fn fused_multiply_add(&self, other: &Self, axes: &[usize]) -> Self {
        let vec = axes.to_vec();
        binary_no_fuse(self, other, move |lhs, rhs| {
            lhs.fused_multiply_add(rhs, &vec)
        })
    }
}

impl<TRaw: RealizedRawTensor + Clone + 'static> RealizedRawTensor for Fuse<TRaw>
where
    TRaw::E: Num,
{
    fn to_cpu(&self) -> crate::CpuRawTensor<Self::E> {
        self.run().to_cpu()
    }

    fn realize(&self) -> Self {
        Self::new(self.run())
    }
}

// TTo is Num here so the `sum` after `cast` is allowed. That should be fine,
// means we'll be able to cast from bool -> i32 and bool -> f32 which is most
// of what's needed.
impl<TFro: 'static + RawTensor + Clone, TTo: RawTensor> CastInto<Fuse<TTo>> for Fuse<TFro>
where
    TFro: CastInto<TTo>,
    TTo::E: Num,
    TFro::E: Num,
{
    fn cast(&self) -> Fuse<TTo> {
        Fuse::new(self.run().cast())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_mul_sum_fuses() {
        let t1: Fuse<String> = RawTensor::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: Fuse<String> = RawTensor::new(&[2, 2], &[5., 6., 7., 8.]);
        let actual = t1.mul(&t2).sum(&[0]).run();
        let expected = t1.fused_multiply_add(&t2, &[0]).run();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_complicated() {
        let t1: Fuse<String> = RawTensor::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: Fuse<String> = RawTensor::new(&[2, 2], &[5., 6., 7., 8.]);
        let actual = t1.sum(&[0]).mul(&t2.sum(&[0])).sum(&[1]).run();
        let expected = t1.sum(&[0]).fused_multiply_add(&t2.sum(&[0]), &[1]).run();
        assert_eq!(expected, actual);
    }
}
