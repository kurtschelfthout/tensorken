use std::{
    f32::consts::E,
    ops::{Add, Div, Mul, Sub},
};

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
        self.log(E)
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
}
