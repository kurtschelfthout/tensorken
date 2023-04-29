use std::ops::{Add, Div, Mul, Sub};

/// A trait with basic requirements of numbers stored in tensors.
/// Currently only f32 is supported.
pub trait Num:
    Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + PartialEq
    + PartialOrd
    + Copy
{
    /// The zero value.
    const ZERO: Self;
    /// The one value.
    const ONE: Self;
    /// The minimum value.
    const MIN: Self;

    /// Convert from usize.
    /// This is really only so Tensor can implement linspace.
    fn from_usize(n: usize) -> Self;
    /// Apply exponential function.
    fn exp(self) -> Self;
    /// Apply the natural logarithm.
    fn log(self) -> Self;
    /// Raise self to the power of given exponent.
    fn powf(self, exponent: Self) -> Self;
}

impl Num for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN: Self = f32::MIN;

    fn exp(self) -> Self {
        self.exp()
    }

    fn log(self) -> Self {
        self.log(std::f32::consts::E)
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_usize(n: usize) -> Self {
        n as f32
    }
}
