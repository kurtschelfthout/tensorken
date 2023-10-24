use crate::{Diffable, DiffableExt};

/// A trait that represents the operation on the primal value, and
/// returns a `UnaryRevOp`, which is the operation on the adjoints in the
/// reverse pass.
/// This design allows the reverse pass to reuse calculations in the forward
/// pass.
pub trait UnaryOp<TTensor>
where
    //this can be avoided by making Self the last element of the tuple.
    Self: Sized,
{
    type Args: ?Sized;
    fn f(a: &TTensor, args: &Self::Args) -> (Self, TTensor);
}

/// Same as `UnaryOp`, but for binary operations.
pub trait BinaryOp<TTensor>
where
    Self: Sized,
{
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor);
}

/// A trait that represents a unary operation on the adjoints in the reverse pass.
pub trait UnaryDiffOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor;
}

/// Same as `UnaryRevOp`, but for binary operations.
pub trait BinaryDiffOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor;
    fn df_dfdb(&self, df: &TTensor) -> TTensor;
}

pub(crate) struct AddOp;

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for AddOp {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        (AddOp, a.elementwise_add(b))
    }
}

impl<TTensor: Clone + Diffable> BinaryDiffOp<TTensor> for AddOp {
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
        (MulOp(a.clone(), b.clone()), a.elementwise_mul(b))
    }
}

impl<TTensor: Diffable> BinaryDiffOp<TTensor> for MulOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.elementwise_mul(&self.1)
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        df.elementwise_mul(&self.0)
    }
}

pub(crate) struct SubOp;

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for SubOp {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        (SubOp, a.elementwise_sub(b))
    }
}

impl<TTensor: Clone + DiffableExt> BinaryDiffOp<TTensor> for SubOp {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.clone()
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        df.zeros_like().elementwise_sub(df)
    }
}

pub(crate) struct DivOp<TTensor>(TTensor, TTensor);

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for DivOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        (DivOp(a.clone(), b.clone()), a.elementwise_div(b))
    }
}

impl<TTensor: Diffable> BinaryDiffOp<TTensor> for DivOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.elementwise_div(&self.1)
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        let b2 = self.1.elementwise_mul(&self.1);
        df.zeros_like()
            .elementwise_sub(df)
            .elementwise_mul(&self.0)
            .elementwise_div(&b2)
    }
}

pub(crate) struct PowOp<TTensor>(TTensor, TTensor, TTensor);

impl<TTensor: Clone + Diffable> BinaryOp<TTensor> for PowOp<TTensor> {
    fn f(a: &TTensor, b: &TTensor) -> (Self, TTensor) {
        let r = a.elementwise_pow(b);
        (PowOp(a.clone(), b.clone(), r.clone()), r)
    }
}

impl<TTensor: Clone + Diffable> BinaryDiffOp<TTensor> for PowOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.elementwise_mul(&self.1.elementwise_mul(&self.2.elementwise_div(&self.0)))
    }

    fn df_dfdb(&self, df: &TTensor) -> TTensor {
        // if df is zero, then this causes unnecessary computation which can also lead to NaNs.
        df.elementwise_mul(&self.0.log().elementwise_mul(&self.2))
    }
}

pub(crate) struct LogOp<TTensor>(TTensor);

impl<TTensor: Clone + Diffable> UnaryOp<TTensor> for LogOp<TTensor> {
    type Args = ();
    fn f(a: &TTensor, _: &Self::Args) -> (Self, TTensor) {
        (LogOp(a.clone()), a.log())
    }
}

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for LogOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.elementwise_div(&self.0)
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

impl<TTensor: Diffable> UnaryDiffOp<TTensor> for ExpOp<TTensor> {
    fn df_dfda(&self, df: &TTensor) -> TTensor {
        df.elementwise_mul(&self.0)
    }
}
