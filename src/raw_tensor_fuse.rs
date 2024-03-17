use core::panic;
use std::rc::Rc;

use crate::{
    num::{Float, Num},
    raw_tensor::{CastInto, RealizedRawTensor},
    RawTensor,
};

enum FuseCtx {
    Sum(Vec<usize>),
    NotSum,
}

#[derive(Clone)]
pub struct Fuse<T>(Rc<dyn Fn(&FuseCtx) -> T>);

impl<T> std::fmt::Debug for Fuse<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fuse")
    }
}

impl<T> Fuse<T> {
    fn new(f: impl Fn(&FuseCtx) -> T + 'static) -> Self {
        Self(Rc::new(f))
    }

    fn run(&self) -> T {
        (self.0)(&FuseCtx::NotSum)
    }
}

impl<TRaw: RawTensor + Clone + 'static> Fuse<TRaw>
where
    TRaw::E: Num,
{
    fn from_raw_tensor(raw_tensor: TRaw) -> Self {
        Self::new(move |ctx| match ctx {
            FuseCtx::Sum(axes2) => raw_tensor.sum(axes2),
            FuseCtx::NotSum => raw_tensor.clone(),
        })
    }

    pub fn realize(&self) -> Self {
        Self::from_raw_tensor(self.run())
    }
}

fn unary_no_fuse<TRaw: RawTensor + 'static>(
    s: &Fuse<TRaw>,
    f: impl Fn(&TRaw) -> TRaw + 'static,
) -> Fuse<TRaw>
where
    TRaw::E: Num,
{
    let k = Rc::clone(&s.0);
    let nextctx = FuseCtx::NotSum;
    Fuse::new(move |ctx| match ctx {
        FuseCtx::Sum(axes) => f(&k(&nextctx)).sum(axes),
        FuseCtx::NotSum => f(&k(&nextctx)),
    })
}

fn binary_no_fuse<TRaw: RawTensor + 'static>(
    lhs: &Fuse<TRaw>,
    rhs: &Fuse<TRaw>,
    f: fn(&TRaw, &TRaw) -> TRaw,
) -> Fuse<TRaw>
where
    TRaw::E: Num,
{
    let f_lhs = Rc::clone(&lhs.0);
    let f_rhs = Rc::clone(&rhs.0);
    let nextctx = FuseCtx::NotSum;
    Fuse::new(move |ctx| match ctx {
        FuseCtx::Sum(axes) => f(&f_lhs(&nextctx), &f_rhs(&nextctx)).sum(axes),
        FuseCtx::NotSum => f(&f_lhs(&nextctx), &f_rhs(&nextctx)),
    })
}

fn combine_axes(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    let mut axes = lhs.to_vec();
    axes.extend_from_slice(rhs);
    axes.sort_unstable();
    axes.dedup();
    axes
}

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
        let f_lhs = Rc::clone(&self.0);
        let f_rhs = Rc::clone(&other.0);
        let nextctx = FuseCtx::NotSum;
        Self::new(move |ctx| match ctx {
            FuseCtx::Sum(axes) => f_lhs(&nextctx).fused_multiply_add(&f_rhs(&nextctx), axes),
            FuseCtx::NotSum => f_lhs(&nextctx).mul(&f_rhs(&nextctx)),
        })
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
        let f = Rc::clone(&self.0);
        let my_axes = axes.to_vec();
        Self::new(move |ctx| match ctx {
            FuseCtx::Sum(sum_axes) => f(&FuseCtx::Sum(combine_axes(&my_axes, sum_axes))),
            FuseCtx::NotSum => f(&FuseCtx::Sum(my_axes.clone())),
        })
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
        Self::from_raw_tensor(traw)
    }

    fn shape(&self) -> &[usize] {
        panic!("shape() can't be implemented for Fuse. Wrap it in ShapeTracker.")
    }

    fn fused_multiply_add(&self, other: &Self, axes: &[usize]) -> Self {
        let f_lhs = Rc::clone(&self.0);
        let f_rhs = Rc::clone(&other.0);
        let nextctx = FuseCtx::NotSum;
        let my_axes = axes.to_vec();
        Self::new(move |ctx| match ctx {
            FuseCtx::Sum(axes) => {
                f_lhs(&nextctx).fused_multiply_add(&f_rhs(&nextctx), &combine_axes(&my_axes, axes))
            }
            FuseCtx::NotSum => f_lhs(&nextctx).fused_multiply_add(&f_rhs(&nextctx), &my_axes),
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
        Self::from_raw_tensor(self.run())
    }
}

// TTo is Num here so the `sum` after `cast` is allowed. That should be fine,
// means we'll be able to cast from bool -> i32 and bool -> f32 which is most
// of what's needed.
impl<TFro: 'static + RawTensor, TTo: RawTensor> CastInto<Fuse<TTo>> for Fuse<TFro>
where
    TFro: CastInto<TTo>,
    TTo::E: Num,
{
    fn cast(&self) -> Fuse<TTo> {
        let k = Rc::clone(&self.0);
        let nextctx = FuseCtx::NotSum;
        Fuse::new(move |ctx| match ctx {
            FuseCtx::Sum(axes) => k(&nextctx).cast().sum(axes),
            FuseCtx::NotSum => k(&nextctx).cast(),
        })
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
    fn test_fma_sum_fuses() {
        let t1: Fuse<String> = RawTensor::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: Fuse<String> = RawTensor::new(&[2, 2], &[5., 6., 7., 8.]);
        let actual = t1.fused_multiply_add(&t2, &[1]).sum(&[0]).run();
        let expected = t1.fused_multiply_add(&t2, &[0, 1]).run();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_mul_sum_sum_fuses() {
        let t1: Fuse<String> = RawTensor::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: Fuse<String> = RawTensor::new(&[2, 2], &[5., 6., 7., 8.]);
        let actual = t1.mul(&t2).sum(&[0]).sum(&[1]).run();
        let expected = t1.fused_multiply_add(&t2, &[0, 1]).run();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_sum_sum_fuses() {
        let t1: Fuse<String> = RawTensor::new(&[2, 2], &[1., 2., 3., 4.]);
        let actual = t1.sum(&[0]).sum(&[1]).run();
        let expected = t1.sum(&[0, 1]).run();
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
