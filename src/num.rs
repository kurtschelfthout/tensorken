use std::{
    borrow::Cow,
    ops::{Add, Div, Mul, Sub},
};
/// The basic requirements for anything stored in a tensor.
/// This is all WGPU-specific for the moment. It would be better
/// to put these requirements on `WgpuRawTensor`, but that's not easily
/// possible with the GAT-based `RawTensor` trait.
pub trait Elem: 'static + Clone
where
    Self: Sized,
{
    /// The name of the corresponding WGSL type.
    const WGPU_ELEMENT_NAME: &'static str;
    /// The size in bytes of the corresponding WGSL type.
    const WGPU_ELEMENT_SIZE: usize = std::mem::size_of::<Self>();
    /// Convert a byte buffer to a vector of elements. For loading from the GPU.
    fn from_buffer(array: &[u8]) -> Vec<Self>;
    /// Convert a slice of elements to a byte buffer. For storage to the GPU.
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
// is `eq`` followed by `cast`` to f32 in the AD operation for max,
// I'm picking f32 so that particular usage remains a no-op.
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

/// A number that has zero and one values, a minimum value, can be compared,
/// and has addition and multiplication.
pub trait Bool: 'static + Elem + PartialEq + PartialOrd {
    /// The zero value.
    const ZERO: Self;
    /// The one value.
    const ONE: Self;
    /// The minimum value.
    const MIN: Self;
}

/// In addition to `ZeroOne`, support arithmetic operations.
pub trait Num:
    Bool
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
{
}

/// In addition to `Num`, support floating point operations.
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

impl Bool for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN: Self = f32::MIN;
}

impl Num for f32 {}

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

impl Bool for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = i32::MIN;
}

impl Num for i32 {}

impl Bool for bool {
    const ZERO: Self = false;
    const ONE: Self = true;
    const MIN: Self = false;
}

/// Unfortunately there is no built-in Rust trait
/// for primitive cast operations. So we define our own.
pub trait CastTo<E> {
    fn cast_to(&self) -> E;
}

impl<T> CastTo<T> for T
where
    T: Copy,
{
    fn cast_to(&self) -> T {
        *self
    }
}

pub trait CastFrom<E> {
    fn cast_from(e: E) -> Self;
}

impl<EFro, ETo> CastFrom<EFro> for ETo
where
    EFro: CastTo<ETo>,
{
    fn cast_from(e: EFro) -> Self {
        e.cast_to()
    }
}

impl CastTo<f32> for i32 {
    #[allow(clippy::cast_precision_loss)]
    fn cast_to(&self) -> f32 {
        *self as f32
    }
}

impl CastTo<i32> for f32 {
    #[allow(clippy::cast_possible_truncation)]
    fn cast_to(&self) -> i32 {
        *self as i32
    }
}

impl CastTo<usize> for i32 {
    #[allow(clippy::cast_sign_loss)]
    fn cast_to(&self) -> usize {
        *self as usize
    }
}

impl CastTo<usize> for u32 {
    fn cast_to(&self) -> usize {
        *self as usize
    }
}

impl CastTo<i32> for bool {
    fn cast_to(&self) -> i32 {
        i32::from(*self)
    }
}

impl CastTo<f32> for bool {
    #[allow(clippy::cast_precision_loss)]
    fn cast_to(&self) -> f32 {
        i32::from(*self) as f32
    }
}
