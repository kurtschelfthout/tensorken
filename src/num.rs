use std::ops::{Add, Div, Mul, Sub};

/// A number that has both zero and one values.
pub trait ZeroOne {
    /// The zero value.
    const ZERO: Self;
    /// The one value.
    const ONE: Self;
}

/// A trait with basic requirements of numbers stored in tensors.
/// Currently only f32 is supported.
pub trait Num:
    ZeroOne
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + PartialEq
    + PartialOrd
    + Copy
{
    /// The minimum value.
    const MIN: Self;

    /// Convert to usize.
    /// This is really only so Tensor can implement `one_hot`.
    /// TODO: remove once we have int tensors.
    fn to_usize(self) -> usize;
}

pub trait Float: Num {
    /// Apply exponential function.
    #[must_use]
    fn exp(self) -> Self;
    /// Apply the natural logarithm.
    #[must_use]
    fn log(self) -> Self;
    /// Raise self to the power of given exponent.
    #[must_use]
    fn powf(self, exponent: Self) -> Self;
}

impl ZeroOne for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Num for f32 {
    const MIN: Self = f32::MIN;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn to_usize(self) -> usize {
        self as _
    }
}

impl Float for f32 {
    fn exp(self) -> Self {
        self.exp()
    }

    fn log(self) -> Self {
        self.log(std::f32::consts::E)
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
}
