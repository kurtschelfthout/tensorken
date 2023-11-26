use crate::{
    ad_ops::{UnaryDiffOp, UnaryOp},
    Diffable,
};

pub(crate) struct SumOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for SumOp {
    type Args = [usize];
    fn f(a: &TTensor, axes: &Self::Args) -> (TTensor, Self) {
        let r = a.sum(axes);
        (r, SumOp(a.shape().to_vec()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for SumOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.expand(&self.0)
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

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for MaxOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor {
        let max_is_1s = self.0.elementwise_eq(&self.1.expand(self.0.shape()));
        let div = max_is_1s
            .sum(&shape_to_axes(max_is_1s.shape(), d.shape()))
            .expand(self.0.shape());
        let max_is_amount = max_is_1s.elementwise_div(&div);
        let df_expanded = d.expand(self.0.shape());

        max_is_amount.elementwise_mul(&df_expanded)
    }
}

pub(crate) struct ExpandOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for ExpandOp {
    type Args = [usize];
    fn f(a: &TTensor, new_shape: &Self::Args) -> (TTensor, Self) {
        let r = a.expand(new_shape);
        (r, ExpandOp(a.shape().to_vec()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for ExpandOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.sum(&shape_to_axes(d.shape(), &self.0))
    }
}

pub(crate) struct ReshapeOp(Vec<usize>);

impl<TTensor: Diffable> UnaryOp<TTensor> for ReshapeOp {
    type Args = [usize];
    fn f(a: &TTensor, new_shape: &Self::Args) -> (TTensor, Self) {
        let r = a.reshape(new_shape);
        (r, ReshapeOp(a.shape().to_vec()))
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

// like numpy argsort: returns the indices that would sort an array.
// Here only used to invert the permutation in the backward pass.
fn argsort(v: &[usize]) -> Vec<usize> {
    let mut v: Vec<_> = v.iter().enumerate().collect();
    v.sort_by_key(|&(_, k)| *k);
    v.into_iter().map(|(i, _)| i).collect()
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for PermuteOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.permute(&argsort(&self.0))
    }
}

pub(crate) struct PadOp(Vec<(usize, usize)>);

impl<TTensor: Diffable> UnaryOp<TTensor> for PadOp {
    type Args = [(usize, usize)];
    fn f(a: &TTensor, padding: &Self::Args) -> (TTensor, Self) {
        let r = a.pad(padding);
        let limits = padding
            .iter()
            .zip(a.shape())
            .map(|((pl, _), s)| (*pl, pl + s))
            .collect::<Vec<_>>();
        (r, PadOp(limits))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for PadOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.crop(&self.0)
    }
}

pub(crate) struct CropOp(Vec<(usize, usize)>);

impl<TTensor: Diffable> UnaryOp<TTensor> for CropOp {
    type Args = [(usize, usize)];
    fn f(a: &TTensor, limits: &Self::Args) -> (TTensor, Self) {
        let r = a.crop(limits);
        let padding = limits
            .iter()
            .zip(a.shape())
            .map(|((l0, l1), s)| (*l0, s - l1))
            .collect::<Vec<_>>();
        (r, CropOp(padding))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for CropOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.pad(&self.0)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_argsort() {
        assert_eq!(argsort(&[0, 1]), [0, 1]);
        assert_eq!(argsort(&[1, 0]), [1, 0]);
        assert_eq!(argsort(&[2, 0, 1]), [1, 2, 0]);
        assert_eq!(argsort(&[0, 1, 2]), [0, 1, 2]);
    }
}
