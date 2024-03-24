use std::{
    borrow::Cow,
    ops::{Add, Div, Mul, Sub},
};

pub trait Elem: 'static + Clone
where
    Self: Sized,
{
    const WGPU_ELEMENT_NAME: &'static str;
    const WGPU_ELEMENT_SIZE: usize = std::mem::size_of::<Self>();
    fn from_buffer(array: &[u8]) -> Vec<Self>;
    fn to_buffer(array: &[Self]) -> Cow<'_, [u8]>;
}

impl Elem for f32 {
    const WGPU_ELEMENT_NAME: &'static str = "f32";

    fn from_buffer(array: &[u8]) -> Vec<Self> {
        bytemuck::cast_slice(array).to_vec()
    }

    fn to_buffer(array: &[Self]) -> Cow<'_, [u8]> {
        Cow::Borrowed(bytemuck::cast_slice(array))
    }
}

impl Elem for i32 {
    const WGPU_ELEMENT_NAME: &'static str = "i32";

    fn from_buffer(array: &[u8]) -> Vec<Self> {
        bytemuck::cast_slice(array).to_vec()
    }

    fn to_buffer(array: &[Self]) -> Cow<'_, [u8]> {
        Cow::Borrowed(bytemuck::cast_slice(array))
    }
}

// for bool, I could not get copying to a array<bool> buffer to work.
// So we need to pick a different type on the GPU side. Since most usage
// right now is eq followed cast to f32 in the diffable operation for max,
// I'm picking f32 so that usage remains a no-op.
impl Elem for bool {
    const WGPU_ELEMENT_NAME: &'static str = "f32";
    const WGPU_ELEMENT_SIZE: usize = 4;
    fn from_buffer(array: &[u8]) -> Vec<Self> {
        let f32s: &[f32] = bytemuck::cast_slice(array);
        f32s.iter().map(|x| *x != 0.0).collect()
    }

    fn to_buffer(array: &[Self]) -> Cow<'_, [u8]> {
        let f32s: Vec<f32> = array.iter().map(|i| f32::from(*i)).collect();
        Cow::Owned(bytemuck::cast_slice(&f32s).to_vec())
    }
}

/// A number that has both zero and one values.
pub trait ZeroOne: 'static + Copy + PartialEq + PartialOrd + Elem {
    /// The zero value.
    const ZERO: Self;
    /// The one value.
    const ONE: Self;
    /// The minimum value.
    const MIN: Self;
}

/// A trait with basic requirements of numbers stored in tensors.
/// Currently only f32 is supported.
pub trait Num:
    ZeroOne
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
{
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
    const MIN: Self = f32::MIN;
}

impl Num for f32 {
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

impl ZeroOne for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = i32::MIN;
}

impl Num for i32 {
    #[allow(clippy::cast_sign_loss)]
    fn to_usize(self) -> usize {
        self as _
    }
}

impl ZeroOne for bool {
    const ZERO: Self = false;
    const ONE: Self = true;
    const MIN: Self = false;
}
