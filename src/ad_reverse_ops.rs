use crate::Diffable;

/// A trait that represents the operation on the primal value, and
/// returns a `UnaryRevOp`, which is the operation on the adjoints in the
/// reverse pass.
/// This design allows the reverse pass to reuse calculations in the forward
/// pass.
pub trait UnaryOp<TTensor>: UnaryRevOp<TTensor>
where
    //this can be avoided by making Self the last element of the tuple.
    Self: Sized,
{
    type Args: ?Sized;
    fn f(a: &TTensor, args: &Self::Args) -> (Self, TTensor);
}

/// Same as `UnaryOp`, but for binary operations.
pub trait BinaryOp<TTensor>: BinaryRevOp<TTensor>
where
    Self: Sized,
{
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor);
}

/// A trait that represents a unary operation on the adjoints in the reverse pass.
pub trait UnaryRevOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor;
}

/// Same as `UnaryRevOp`, but for binary operations.
pub trait BinaryRevOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor;
    fn df_dfdb(&self, df: &TTensor) -> TTensor;
}

pub(crate) struct AddOp;

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for AddOp {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        (AddOp, a.add(b))
    }
}

impl<TTensor: Clone + Diffable> BinaryRevOp<TTensor> for AddOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.clone()
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        df.clone()
    }
}

pub(crate) struct MulOp<TTensor>(TTensor, TTensor);

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for MulOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        (MulOp(a.clone(), b.clone()), a.mul(b))
    }
}

impl<TTensor: Diffable> BinaryRevOp<TTensor> for MulOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.mul(&self.1)
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        df.mul(&self.0)
    }
}

pub(crate) struct SubOp;

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for SubOp {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        (SubOp, a.sub(b))
    }
}

impl<TTensor: Clone + Diffable> BinaryRevOp<TTensor> for SubOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.clone()
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        df.zeros_like().sub(df)
    }
}

pub(crate) struct DivOp<TTensor>(TTensor, TTensor);

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for DivOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        (DivOp(a.clone(), b.clone()), a.div(b))
    }
}

impl<TTensor: Diffable> BinaryRevOp<TTensor> for DivOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.div(&self.1)
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        let b2 = self.1.mul(&self.1);
        df.zeros_like().sub(df).mul(&self.0).div(&b2)
    }
}

pub(crate) struct PowOp<TTensor>(TTensor, TTensor, TTensor);

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for PowOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        let r = a.pow(b);
        (PowOp(a.clone(), b.clone(), r.clone()), r)
    }
}

impl<TTensor: Clone + Diffable> BinaryRevOp<TTensor> for PowOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.mul(&self.1.mul(&self.2.div(&self.0)))
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        df.mul(&self.0.log().mul(&self.2))
    }
}

pub(crate) struct EqOp;

impl<TTensor: Diffable> BinaryOp<TTensor> for EqOp {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        (EqOp, a.eq(b))
    }
}

impl<TTensor: Diffable> BinaryRevOp<TTensor> for EqOp {
    fn df_dfda(&self, _: &TTensor) -> TTensor {
        panic!("Equality is not differentiable");
    }

    fn df_dfdb(&self, _: &TTensor) -> TTensor {
        panic!("Equality is not differentiable");
    }
}

pub(crate) struct LogOp<TTensor>(TTensor);

impl<TTensor: Clone + Diffable> UnaryOp<TTensor> for LogOp<TTensor> {
    type Args = ();
    fn f(a: &TTensor, _: &Self::Args) -> (Self, TTensor) {
        (LogOp(a.clone()), a.log())
    }
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for LogOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.div(&self.0)
    }
}

pub(crate) struct ExpOp<TTensor>(TTensor);

impl<TTensor: Clone + Diffable> UnaryOp<TTensor> for ExpOp<TTensor> {
    type Args = ();
    fn f(a: &TTensor, _: &Self::Args) -> (Self, TTensor) {
        let r = a.exp();
        (ExpOp(r.clone()), r)
    }
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for ExpOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.mul(&self.0)
    }
}

pub(crate) struct SumOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for SumOp {
    type Args = [usize];
    fn f(a: &TTensor, axes: &Self::Args) -> (Self, TTensor) {
        let r = a.sum(axes);
        (SumOp(a.shape().to_vec()), r)
    }
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for SumOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.expand(&self.0)
    }
}

pub(crate) struct MaxOp<TTensor>(TTensor, TTensor);

impl<TTensor: Clone + Diffable> UnaryOp<TTensor> for MaxOp<TTensor> {
    type Args = [usize];
    fn f(a: &TTensor, axes: &Self::Args) -> (Self, TTensor) {
        let r = a.max(axes);
        (MaxOp(a.clone(), r.clone()), r)
    }
}

fn shape_to_axes(old_shape: &[usize], new_shape: &[usize]) -> Vec<usize> {
    assert!(
        old_shape.len() == new_shape.len(),
        "shape_to_axes: old_shape.len() != new_shape.len()"
    );
    old_shape
        .iter()
        .zip(new_shape.iter())
        .enumerate()
        .filter_map(|(i, (a, b))| if a == b { None } else { Some(i) })
        .collect()
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for MaxOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        let max_is_1s = self.0.eq(&self.1.expand(self.0.shape()));
        let div = max_is_1s
            .sum(&shape_to_axes(max_is_1s.shape(), df.shape()))
            .expand(self.0.shape());
        let max_is_amount = max_is_1s.div(&div);
        let df_expanded = df.expand(self.0.shape());

        max_is_amount.mul(&df_expanded)
    }
}

pub(crate) struct ExpandOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for ExpandOp {
    type Args = [usize];
    fn f(a: &TTensor, new_shape: &Self::Args) -> (Self, TTensor) {
        let r = a.expand(new_shape);
        (ExpandOp(a.shape().to_vec()), r)
    }
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for ExpandOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.sum(&shape_to_axes(df.shape(), &self.0))
    }
}

pub(crate) struct ReshapeOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for ReshapeOp {
    type Args = [usize];
    fn f(a: &TTensor, new_shape: &Self::Args) -> (Self, TTensor) {
        let r = a.reshape(new_shape);
        (ReshapeOp(a.shape().to_vec()), r)
    }
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for ReshapeOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.reshape(&self.0)
    }
}

pub(crate) struct PermuteOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for PermuteOp {
    type Args = [usize];
    fn f(a: &TTensor, order: &Self::Args) -> (Self, TTensor) {
        (PermuteOp(order.to_vec()), a.permute(order))
    }
}

// like numpy argsort: returns the indices that would sort an array.
// Here only used to invert the permutation in the backward pass.
fn argsort(v: &[usize]) -> Vec<usize> {
    let mut v: Vec<_> = v.iter().enumerate().collect();
    v.sort_by_key(|&(_, k)| *k);
    v.into_iter().map(|(i, _)| i).collect()
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for PermuteOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.permute(&argsort(&self.0))
    }
}

pub(crate) struct PadOp(Vec<(usize, usize)>);

impl<TTensor: Diffable> UnaryOp<TTensor> for PadOp {
    type Args = [(usize, usize)];
    fn f(a: &TTensor, padding: &Self::Args) -> (Self, TTensor) {
        let r = a.pad(padding);
        let limits = padding
            .iter()
            .zip(a.shape())
            .map(|((pl, _), s)| (*pl, pl + s))
            .collect::<Vec<_>>();
        (PadOp(limits), r)
    }
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for PadOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.crop(&self.0)
    }
}

pub(crate) struct CropOp(Vec<(usize, usize)>);

impl<TTensor: Diffable> UnaryOp<TTensor> for CropOp {
    type Args = [(usize, usize)];
    fn f(a: &TTensor, limits: &Self::Args) -> (Self, TTensor) {
        let r = a.crop(limits);
        let padding = limits
            .iter()
            .zip(a.shape())
            .map(|((l0, l1), s)| (*l0, s - l1))
            .collect::<Vec<_>>();
        (CropOp(padding), r)
    }
}

impl<TTensor: Diffable> UnaryRevOp<TTensor> for CropOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.pad(&self.0)
    }
}
