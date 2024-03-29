use std::marker::PhantomData;

use crate::{
    num::{Bool, CastFrom, Elem, Float, Num},
    raw_tensor::RealizedRawTensor,
    RawTensor,
};

/// Fuse is now easy to understand.
/// It delays the application of `mul` until the following operation is called.
/// This is the `Mul` variant. The function allows the operation following `mul`
/// to calculate it, if it is not a `sum`, i.e. if `mul(...).sum(...)` can't be merged.
/// We need to embed this as a thunk, because operations that follow `mul` do not necessarily
/// have the right constraint on `E` to have `mul` be visible. E.g. in `mul(...).eq(&t2)` the `eq`
/// cannot call `I::mul`.
/// If the current operation is not a `mul`, we just calculate as usual using the underlying implementation.
#[derive(Clone, Debug)]
pub enum Fuse<T> {
    Mul(T, T, fn(&T, &T) -> T),
    Val(T),
}

impl<T> Fuse<T> {
    fn new(t: T) -> Self {
        Self::Val(t)
    }
}

impl<T: Clone> Fuse<T> {
    fn run(&self) -> T {
        match self {
            Self::Mul(lhs, rhs, f) => f(lhs, rhs),
            Self::Val(t) => t.clone(),
        }
    }

    fn realize(&self) -> Self {
        match self {
            Self::Mul(lhs, rhs, f) => Self::new(f(lhs, rhs)),
            Self::Val(_) => self.clone(),
        }
    }
}

fn unary_no_fuse<T, TR>(s: &Fuse<T>, op: impl Fn(&T) -> TR) -> Fuse<TR> {
    match s {
        Fuse::Mul(lhs, rhs, f) => Fuse::Val(op(&f(lhs, rhs))),
        Fuse::Val(v) => Fuse::Val(op(v)),
    }
}

fn binary_no_fuse<T, TR>(lhs: &Fuse<T>, rhs: &Fuse<T>, op: impl Fn(&T, &T) -> TR) -> Fuse<TR> {
    match (lhs, rhs) {
        (Fuse::Mul(lhs1, rhs1, f1), Fuse::Mul(lhs2, rhs2, f2)) => {
            Fuse::new(op(&f1(lhs1, rhs1), &f2(lhs2, rhs2)))
        }
        (Fuse::Mul(lhs1, rhs1, f), Fuse::Val(t)) => Fuse::Val(op(&f(lhs1, rhs1), t)),
        (Fuse::Val(t), Fuse::Mul(lhs2, rhs2, f)) => Fuse::Val(op(t, &f(lhs2, rhs2))),
        (Fuse::Val(t1), Fuse::Val(t2)) => Fuse::Val(op(t1, t2)),
    }
}

// fn combine_axes(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
//     let mut axes = lhs.to_vec();
//     axes.extend_from_slice(rhs);
//     axes.sort_unstable();
//     axes.dedup();
//     axes
// }

#[derive(Debug, Clone)]
pub struct FuseImpl<I>(PhantomData<I>);

impl<I: RawTensor> RawTensor for FuseImpl<I> {
    type Repr<E> = Fuse<I::Repr<E>> where E: Clone, Self::Repr<E>: Clone;

    fn exp<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        unary_no_fuse(t, I::exp)
    }

    fn log<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        unary_no_fuse(t, I::log)
    }

    fn cast<EFro: Elem, ETo: CastFrom<EFro> + Elem>(t: &Self::Repr<EFro>) -> Self::Repr<ETo> {
        unary_no_fuse(t, I::cast)
    }

    fn add<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        binary_no_fuse(lhs, rhs, I::add)
    }

    fn sub<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        binary_no_fuse(lhs, rhs, I::sub)
    }

    fn mul<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        match (lhs, rhs) {
            (Fuse::Mul(lhs1, rhs1, f1), Fuse::Mul(lhs2, rhs2, f2)) => {
                Fuse::Mul(f1(lhs1, rhs1), f2(lhs2, rhs2), I::mul)
            }
            (Fuse::Mul(lhs1, rhs1, f), Fuse::Val(t)) => Fuse::Mul(f(lhs1, rhs1), t.clone(), I::mul),
            (Fuse::Val(t), Fuse::Mul(lhs2, rhs2, f)) => Fuse::Mul(t.clone(), f(lhs2, rhs2), I::mul),
            (Fuse::Val(t1), Fuse::Val(t2)) => Fuse::Mul(t1.clone(), t2.clone(), I::mul),
        }
    }

    fn div<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        binary_no_fuse(lhs, rhs, I::div)
    }

    fn pow<E: Float>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        binary_no_fuse(lhs, rhs, I::pow)
    }

    fn eq<E: PartialEq + Elem>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<bool> {
        binary_no_fuse(lhs, rhs, I::eq)
    }

    fn sum<E: Num>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        match t {
            Fuse::Mul(lhs, rhs, _) => Fuse::Val(I::fused_multiply_add(lhs, rhs, axes)),
            Fuse::Val(t) => Fuse::Val(I::sum(t, axes)),
        }
    }

    fn max<E: Bool>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        let vec = axes.to_vec();
        unary_no_fuse(t, move |x| I::max(x, &vec))
    }

    fn reshape<E: Elem>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        let vec = shape.to_vec();
        unary_no_fuse(t, move |x| I::reshape(x, &vec))
    }

    fn permute<E: Clone>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E> {
        let vec = permutation.to_vec();
        unary_no_fuse(t, move |x| I::permute(x, &vec))
    }

    fn expand<E: Clone>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        let vec = shape.to_vec();
        unary_no_fuse(t, move |x| I::expand(x, &vec))
    }

    fn pad<E: Bool>(t: &Self::Repr<E>, padding: &[(usize, usize)]) -> Self::Repr<E> {
        let vec = padding.to_vec();
        unary_no_fuse(t, move |x| I::pad(x, &vec))
    }

    fn crop<E: Clone>(t: &Self::Repr<E>, limits: &[(usize, usize)]) -> Self::Repr<E> {
        let vec = limits.to_vec();
        unary_no_fuse(t, move |x| I::crop(x, &vec))
    }

    fn new<E: Elem>(shape: &[usize], data: &[E]) -> Self::Repr<E> {
        let traw = I::new(shape, data);
        Fuse::new(traw)
    }

    fn shape<E: Clone>(t: &Self::Repr<E>) -> &[usize] {
        match t {
            Fuse::Mul(lhs, _, _) => I::shape(lhs),
            Fuse::Val(t) => I::shape(t),
        }
    }

    fn fused_multiply_add<E: Num>(
        lhs: &Self::Repr<E>,
        rhs: &Self::Repr<E>,
        axes: &[usize],
    ) -> Self::Repr<E> {
        let vec = axes.to_vec();
        binary_no_fuse(lhs, rhs, move |lhs, rhs| {
            I::fused_multiply_add(lhs, rhs, &vec)
        })
    }
}

impl<I: RealizedRawTensor> RealizedRawTensor for FuseImpl<I> {
    fn to_cpu<E: Elem>(t: &Self::Repr<E>) -> crate::CpuRawTensor<E> {
        I::to_cpu(&t.run())
    }

    fn realize<E: Clone>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.realize()
    }
}

#[cfg(test)]
mod tests {

    use crate::shape_strider::ShapeStrider;

    use super::*;

    type I = FuseImpl<crate::raw_tensor_string::StringImpl>;

    #[test]
    fn test_mul_sum_fuses() {
        let t1: Fuse<(ShapeStrider, String)> = I::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: Fuse<(ShapeStrider, String)> = I::new(&[2, 2], &[5., 6., 7., 8.]);
        let actual = I::sum::<f32>(&I::mul::<f32>(&t1, &t2), &[0]).run();
        let expected = I::fused_multiply_add::<f32>(&t1, &t2, &[0]).run();
        assert_eq!(expected.1, actual.1);
    }

    #[test]
    fn test_complicated() {
        let t1: Fuse<(ShapeStrider, String)> = I::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: Fuse<(ShapeStrider, String)> = I::new(&[2, 2], &[5., 6., 7., 8.]);
        let actual = I::sum::<f32>(
            &I::mul::<f32>(&I::sum::<f32>(&t1, &[0]), &I::sum::<f32>(&t2, &[0])),
            &[1],
        )
        .run();
        let expected = I::fused_multiply_add::<f32>(
            &I::sum::<f32>(&t1, &[0]),
            &I::sum::<f32>(&t2, &[0]),
            &[1],
        )
        .run();
        assert_eq!(expected.1, actual.1);
    }
}
