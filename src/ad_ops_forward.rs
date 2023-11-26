use crate::{
    ad_ops::{UnaryDiffOp, UnaryOp},
    Diffable,
};

pub(crate) struct SumOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for SumOp {
    type Args = [usize];
    fn f(a: &TTensor, axes: &Self::Args) -> (TTensor, Self) {
        let r = a.sum(axes);
        (r, SumOp(axes.to_vec()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for SumOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.sum(&self.0)
    }
}

pub(crate) struct MaxOp<TTensor>(TTensor, TTensor);

impl<TTensor: Clone + Diffable> UnaryOp<TTensor> for MaxOp<TTensor> {
    type Args = [usize];
    fn f(a: &TTensor, axes: &Self::Args) -> (TTensor, Self) {
        let r = a.max(axes);
        (r.clone(), MaxOp(a.clone(), r))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for MaxOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor {
        let max_is_1s = self.0.elementwise_eq(&self.1.expand(self.0.shape()));
        max_is_1s.elementwise_mul(d)
    }
}

pub(crate) struct ExpandOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for ExpandOp {
    type Args = [usize];
    fn f(a: &TTensor, new_shape: &Self::Args) -> (TTensor, Self) {
        let r = a.expand(new_shape);
        (r, ExpandOp(new_shape.to_vec()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for ExpandOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.expand(&self.0)
    }
}

pub(crate) struct ReshapeOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for ReshapeOp {
    type Args = [usize];
    fn f(a: &TTensor, new_shape: &Self::Args) -> (TTensor, Self) {
        let r = a.reshape(new_shape);
        (r, ReshapeOp(new_shape.to_vec()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for ReshapeOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.reshape(&self.0)
    }
}

pub(crate) struct PermuteOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for PermuteOp {
    type Args = [usize];
    fn f(a: &TTensor, order: &Self::Args) -> (TTensor, Self) {
        (a.permute(order), PermuteOp(order.to_vec()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for PermuteOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.permute(&self.0)
    }
}

pub(crate) struct PadOp(Vec<(usize, usize)>);

impl<TTensor: Diffable> UnaryOp<TTensor> for PadOp {
    type Args = [(usize, usize)];
    fn f(a: &TTensor, padding: &Self::Args) -> (TTensor, Self) {
        let r = a.pad(padding);
        (r, PadOp(padding.to_vec()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for PadOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.pad(&self.0)
    }
}

pub(crate) struct CropOp(Vec<(usize, usize)>);

impl<TTensor: Diffable> UnaryOp<TTensor> for CropOp {
    type Args = [(usize, usize)];
    fn f(a: &TTensor, limits: &Self::Args) -> (TTensor, Self) {
        let r = a.crop(limits);
        (r, CropOp(limits.to_vec()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for CropOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.crop(&self.0)
    }
}
