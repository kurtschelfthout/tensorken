use crate::RawTensor;

pub(crate) struct StringImpl;

/// Rawtensor for String - a poor man's symbolic execution.
impl RawTensor for StringImpl {
    type Repr<E: Clone> = String;

    fn exp<E: crate::num::Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        format!("{t}.exp()")
    }

    fn log<E: crate::num::Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        format!("{t}.log()")
    }

    fn cast<EFro: Clone, ETo: From<EFro> + Clone>(t: &Self::Repr<EFro>) -> Self::Repr<ETo> {
        format!("{t}.cast()")
    }

    fn add<E: crate::num::Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        format!("({lhs} + {rhs})")
    }

    fn sub<E: crate::num::Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        format!("({lhs} - {rhs})")
    }

    fn mul<E: crate::num::Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        format!("({lhs} * {rhs})")
    }

    fn div<E: crate::num::Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        format!("({lhs} / {rhs})")
    }

    fn pow<E: crate::num::Float>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        format!("{lhs}.pow({rhs})")
    }

    fn eq<E: PartialEq + Clone>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<bool> {
        format!("{lhs}.eq({rhs})")
    }

    fn sum<E: crate::num::Num>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        format!("{t}.sum({axes:?})")
    }

    fn max<E: crate::num::ZeroOne>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        format!("{t}.max({axes:?})")
    }

    fn reshape<E: Clone>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        format!("{t}.reshape({shape:?})")
    }

    fn permute<E: Clone>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E> {
        format!("{t}.permute({permutation:?})")
    }

    fn expand<E: Clone>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        format!("{t}.expand({shape:?})")
    }

    fn pad<E: crate::num::ZeroOne>(t: &Self::Repr<E>, padding: &[(usize, usize)]) -> Self::Repr<E> {
        format!("{t}.pad({padding:?})")
    }

    fn crop<E: Clone>(t: &Self::Repr<E>, limits: &[(usize, usize)]) -> Self::Repr<E> {
        format!("{t}.crop({limits:?})")
    }

    fn new<E: Clone>(shape: &[usize], data: &[E]) -> Self::Repr<E> {
        // TODO: consider adding E: Debug so we can print the data here.
        format!("new({shape:?}, {:?})", data.len())
    }

    fn shape<E: Clone>(_: &Self::Repr<E>) -> &[usize] {
        panic!("shape() not implemented for String.")
    }

    fn fused_multiply_add<E: crate::num::Num>(
        lhs: &Self::Repr<E>,
        rhs: &Self::Repr<E>,
        axes: &[usize],
    ) -> Self::Repr<E> {
        format!("{lhs}.fused_multiply_add({rhs}, {axes:?})")
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    type I = StringImpl;

    #[test]
    fn test_shape_tracks() {
        let t1: &String = &I::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: &String = &I::new(&[2, 2], &[5., 6., 7., 8.]);
        let r = I::log::<f32>(&I::add::<f32>(&I::exp::<f32>(t1), t2));
        assert_eq!(r, "(new([2, 2], 4).exp() + new([2, 2], 4)).log()");
    }

    #[test]
    fn test_symbolic() {
        let t1: &String = &"A".to_string();
        let t2: &String = &"B".to_string();
        let r = I::add::<f32>(&I::exp::<f32>(t1), &I::log::<f32>(t2));
        assert_eq!(r, "(A.exp() + B.log())");
    }
}
