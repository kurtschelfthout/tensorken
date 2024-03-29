use crate::{
    num::{Bool, CastFrom, Float, Num},
    shape_strider::ShapeStrider,
    RawTensorOps, Shape,
};

#[derive(Clone, Debug)]
pub enum StringImpl {}

/// Rawtensor for String with shape tracking - a poor man's symbolic execution.
impl RawTensorOps for StringImpl {
    type Repr<E: Clone> = (ShapeStrider, String);

    fn exp<E: Float>((s, t): &Self::Repr<E>) -> Self::Repr<E> {
        (s.clone(), format!("{t}.exp()"))
    }

    fn log<E: Float>((s, t): &Self::Repr<E>) -> Self::Repr<E> {
        (s.clone(), format!("{t}.log()"))
    }

    fn cast<EFro: Clone, ETo: CastFrom<EFro> + Clone>(
        (s, t): &Self::Repr<EFro>,
    ) -> Self::Repr<ETo> {
        (s.clone(), format!("{t}.cast()"))
    }

    fn realize<E: Clone>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.clone()
    }

    fn add<E: Num>((ls, lt): &Self::Repr<E>, (rs, rt): &Self::Repr<E>) -> Self::Repr<E> {
        ls.validate_can_zip(rs).unwrap();
        (ls.clone(), format!("({lt} + {rt})"))
    }

    fn sub<E: Num>((ls, lt): &Self::Repr<E>, (rs, rt): &Self::Repr<E>) -> Self::Repr<E> {
        ls.validate_can_zip(rs).unwrap();
        (ls.clone(), format!("({lt} - {rt})"))
    }

    fn mul<E: Num>((ls, lt): &Self::Repr<E>, (rs, rt): &Self::Repr<E>) -> Self::Repr<E> {
        ls.validate_can_zip(rs).unwrap();
        (ls.clone(), format!("({lt} * {rt})"))
    }

    fn div<E: Num>((ls, lt): &Self::Repr<E>, (rs, rt): &Self::Repr<E>) -> Self::Repr<E> {
        ls.validate_can_zip(rs).unwrap();
        (ls.clone(), format!("({lt} / {rt})"))
    }

    fn pow<E: Float>((ls, lt): &Self::Repr<E>, (rs, rt): &Self::Repr<E>) -> Self::Repr<E> {
        ls.validate_can_zip(rs).unwrap();
        (ls.clone(), format!("{lt}.pow({rt})"))
    }

    fn eq<E: PartialEq + Clone>(
        (ls, lt): &Self::Repr<E>,
        (rs, rt): &Self::Repr<E>,
    ) -> Self::Repr<bool> {
        ls.validate_can_zip(rs).unwrap();
        (ls.clone(), format!("{lt}.eq({rt})"))
    }

    fn sum<E: Num>((s, t): &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        s.validate_can_reduce(axes).unwrap();
        (s.reduce(axes).0, format!("{t}.sum({axes:?})"))
    }

    fn max<E: Bool>((s, t): &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        s.validate_can_reduce(axes).unwrap();
        (s.reduce(axes).0, format!("{t}.max({axes:?})"))
    }

    fn reshape<E: Clone>((s, t): &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        if let Ok(strider) = s.reshape(shape) {
            return (strider, format!("{t}.reshape({shape:?})"));
        }
        (
            ShapeStrider::contiguous(shape),
            format!("{t}.reshape({shape:?})"),
        )
    }

    fn permute<E: Clone>((s, t): &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E> {
        (
            s.permute(permutation),
            format!("{t}.permute({permutation:?})"),
        )
    }

    fn expand<E: Clone>((s, t): &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        (s.expand(shape).unwrap(), format!("{t}.expand({shape:?})"))
    }

    fn pad<E: Bool>((s, t): &Self::Repr<E>, padding: &[(usize, usize)]) -> Self::Repr<E> {
        (s.pad(padding), format!("{t}.pad({padding:?})"))
    }

    fn crop<E: Clone>((s, t): &Self::Repr<E>, limits: &[(usize, usize)]) -> Self::Repr<E> {
        (s.crop(limits), format!("{t}.crop({limits:?})"))
    }

    fn new<E: Clone>(shape: &[usize], data: &[E]) -> Self::Repr<E> {
        (
            ShapeStrider::contiguous(shape),
            format!("new({shape:?}, {:?})", data.len()),
        )
    }

    fn shape<E: Clone>(t: &Self::Repr<E>) -> &[usize] {
        t.0.shape()
    }

    fn fused_multiply_add<E: Num>(
        (ls, lt): &Self::Repr<E>,
        (rs, rt): &Self::Repr<E>,
        axes: &[usize],
    ) -> Self::Repr<E> {
        ls.validate_can_zip(rs).unwrap();
        ls.validate_can_reduce(axes).unwrap();
        (
            ls.reduce(axes).0,
            format!("{lt}.fused_multiply_add({rt}, {axes:?})"),
        )
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    type I = StringImpl;

    #[test]
    fn test_shape_tracks() {
        let t1: &(ShapeStrider, String) = &I::new(&[2, 2], &[1., 2., 3., 4.]);
        let t2: &(ShapeStrider, String) = &I::new(&[2, 2], &[5., 6., 7., 8.]);
        let r = I::log::<f32>(&I::add::<f32>(&I::exp::<f32>(t1), t2));
        assert_eq!(r.1, "(new([2, 2], 4).exp() + new([2, 2], 4)).log()");
    }

    #[test]
    fn test_symbolic() {
        let t1: &(ShapeStrider, String) = &(ShapeStrider::contiguous(&[1]), "A".to_string());
        let t2: &(ShapeStrider, String) = &(ShapeStrider::contiguous(&[1]), "B".to_string());
        let r = I::add::<f32>(&I::exp::<f32>(t1), &I::log::<f32>(t2));
        assert_eq!(r.1, "(A.exp() + B.log())");
    }
}
