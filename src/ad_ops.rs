use crate::{Diffable, DiffableExt};

/// A trait that represents the operation on the primal value, and
/// returns a `UnaryDiffOp`, which is the operation on the adjoints in the
/// reverse pass.
/// This design allows the derivative calculation to reuse result from the primal calculation.
pub trait UnaryOp<TTensor> {
    type Args: ?Sized;
    fn f(a: &TTensor, args: &Self::Args) -> (TTensor, Self);
}

/// Same as `UnaryOp`, but for binary operations.
pub trait BinaryOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (TTensor, Self);
}

/// Propagate the derivative of a unary operation.
pub trait UnaryDiffOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor;
}

/// Propagate the derivative of a binary operation.
pub trait BinaryDiffOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor;
    fn dfdb(&self, d: &TTensor) -> TTensor;
}

// The rest of this file are implementations of the above traits for element-wise operations.
// They are the same for forward and reverse mode, and so we can share them.
// Forward-mode specific ops are in ad_ops_forward, and reverse-mode specific ops are in ad_ops_reverse.

pub(crate) struct AddOp;

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for AddOp {
    fn f(a: &TTensor, b: &TTensor) -> (TTensor, Self) {
        (a.elementwise_add(b), Self)
    }
}

impl<TTensor: Clone + Diffable> BinaryDiffOp<TTensor> for AddOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.clone()
    }

    fn dfdb(&self, d: &TTensor) -> TTensor {
        d.clone()
    }
}

pub(crate) struct MulOp<TTensor>(TTensor, TTensor);

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for MulOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (TTensor, Self) {
        (a.elementwise_mul(b), Self(a.clone(), b.clone()))
    }
}

impl<TTensor: Diffable> BinaryDiffOp<TTensor> for MulOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.elementwise_mul(&self.1)
    }

    fn dfdb(&self, d: &TTensor) -> TTensor {
        d.elementwise_mul(&self.0)
    }
}

pub(crate) struct SubOp;

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for SubOp {
    fn f(a: &TTensor, b: &TTensor) -> (TTensor, Self) {
        (a.elementwise_sub(b), Self)
    }
}

impl<TTensor: Clone + DiffableExt> BinaryDiffOp<TTensor> for SubOp {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.clone()
    }

    fn dfdb(&self, d: &TTensor) -> TTensor {
        d.zeros_like().elementwise_sub(d)
    }
}

pub(crate) struct DivOp<TTensor>(TTensor, TTensor);

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for DivOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (TTensor, Self) {
        (a.elementwise_div(b), Self(a.clone(), b.clone()))
    }
}

impl<TTensor: Diffable> BinaryDiffOp<TTensor> for DivOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.elementwise_div(&self.1)
    }

    fn dfdb(&self, d: &TTensor) -> TTensor {
        let b2 = self.1.elementwise_mul(&self.1);
        d.zeros_like()
            .elementwise_sub(d)
            .elementwise_mul(&self.0)
            .elementwise_div(&b2)
    }
}

pub(crate) struct PowOp<TTensor>(TTensor, TTensor, TTensor);

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for PowOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (TTensor, Self) {
        let r = a.elementwise_pow(b);
        (r.clone(), Self(a.clone(), b.clone(), r))
    }
}

impl<TTensor: Clone + Diffable> BinaryDiffOp<TTensor> for PowOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.elementwise_mul(&self.1.elementwise_mul(&self.2.elementwise_div(&self.0)))
    }

    fn dfdb(&self, d: &TTensor) -> TTensor {
        // if d is zero, then this causes unnecessary computation which can also lead to NaNs.
        d.elementwise_mul(&self.0.log().elementwise_mul(&self.2))
    }
}

pub(crate) struct LogOp<TTensor>(TTensor);

impl<TTensor: Clone + Diffable> UnaryOp<TTensor> for LogOp<TTensor> {
    type Args = ();
    fn f(a: &TTensor, (): &Self::Args) -> (TTensor, Self) {
        (a.log(), LogOp(a.clone()))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for LogOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.elementwise_div(&self.0)
    }
}

pub(crate) struct ExpOp<TTensor>(TTensor);

impl<TTensor: Clone + Diffable> UnaryOp<TTensor> for ExpOp<TTensor> {
    type Args = ();
    fn f(a: &TTensor, (): &Self::Args) -> (TTensor, Self) {
        let r = a.exp();
        (r.clone(), ExpOp(r))
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for ExpOp<TTensor> {
    fn dfda(&self, d: &TTensor) -> TTensor {
        d.elementwise_mul(&self.0)
    }
}
