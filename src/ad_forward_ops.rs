use crate::{
    ad_ops::{UnaryDiffOp, UnaryOp},
    Diffable,
};

// ------end of element-wise ops------
// now the implementations start to diverge.

pub(crate) struct SumOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for SumOp {
    type Args = [usize];
    fn f(a: &TTensor, axes: &Self::Args) -> (Self, TTensor) {
        let r = a.sum(axes);
        (SumOp(axes.to_vec()), r)
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for SumOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.sum(&self.0)
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

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for MaxOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        let max_is_1s = self.0.elementwise_eq(&self.1.expand(self.0.shape()));
        max_is_1s.elementwise_mul(df)
    }
}

pub(crate) struct ExpandOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for ExpandOp {
    type Args = [usize];
    fn f(a: &TTensor, new_shape: &Self::Args) -> (Self, TTensor) {
        let r = a.expand(new_shape);
        (ExpandOp(new_shape.to_vec()), r)
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for ExpandOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.expand(&self.0)
    }
}

pub(crate) struct ReshapeOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for ReshapeOp {
    type Args = [usize];
    fn f(a: &TTensor, new_shape: &Self::Args) -> (Self, TTensor) {
        let r = a.reshape(new_shape);
        (ReshapeOp(new_shape.to_vec()), r)
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for ReshapeOp {
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

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for PermuteOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.permute(&self.0)
    }
}

pub(crate) struct PadOp(Vec<(usize, usize)>);

impl<TTensor: Diffable> UnaryOp<TTensor> for PadOp {
    type Args = [(usize, usize)];
    fn f(a: &TTensor, padding: &Self::Args) -> (Self, TTensor) {
        let r = a.pad(padding);
        (PadOp(padding.to_vec()), r)
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for PadOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.pad(&self.0)
    }
}

pub(crate) struct CropOp(Vec<(usize, usize)>);

impl<TTensor: Diffable> UnaryOp<TTensor> for CropOp {
    type Args = [(usize, usize)];
    fn f(a: &TTensor, limits: &Self::Args) -> (Self, TTensor) {
        let r = a.crop(limits);
        (CropOp(limits.to_vec()), r)
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for CropOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.crop(&self.0)
    }
}
