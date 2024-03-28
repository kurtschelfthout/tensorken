use crate::{
    CpuRawTensor, CpuRawTensorImpl, Diffable, Forward, ForwardImpl, Fuse, FuseImpl, Reverse,
    ReverseImpl, ShapeStrider, StringImpl, Tensor, WgpuRawTensor, WgpuRawTensorImpl,
};

/// A type alias for a tensor on the CPU with element type `E`.
pub type Cpu<E> = Tensor<Fuse<CpuRawTensor<E>>, E, FuseImpl<CpuRawTensorImpl>>;

/// A type alias for a tensor on the CPU with element type `f32`.
pub type Cpu32 = Cpu<f32>;

/// A type alias for a tensor on the CPU with element type `i32`.
pub type CpuI32 = Cpu<i32>;

/// A type alias for a tensor on the CPU with element type `bool`.
pub type CpuBool = Cpu<bool>;

/// A type alias for a tensor on the GPU with element type `E`.
pub type Wgpu<E> = Tensor<Fuse<WgpuRawTensor<'static, E>>, E, FuseImpl<WgpuRawTensorImpl>>;
/// A type alias for a tensor on the GPU with element type `f32`.
pub type Wgpu32 = Wgpu<f32>;
/// A type alias for a tensor on the GPU with element type `i32`.
pub type WgpuI32 = Wgpu<i32>;
/// A type alias for a tensor on the GPU with element type `bool`.
pub type WgpuBool = Wgpu<bool>;

/// A type alias for a tensor that is abstractly executed. It tracks its shape, strides, and offset,
/// and the operations that are applied to it as a string.
pub type Sym = Tensor<Fuse<(ShapeStrider, String)>, i32, FuseImpl<StringImpl>>;

impl From<&Wgpu32> for Cpu32 {
    fn from(wgpu: &Wgpu32) -> Self {
        Tensor::new(wgpu.shape(), &wgpu.ravel())
    }
}

impl From<&Cpu32> for Wgpu32 {
    fn from(cpu: &Cpu32) -> Self {
        Tensor::new(cpu.shape(), &cpu.ravel())
    }
}

/// Experimental trait. It makes it easier to write a function that
/// can be called in normal, forward or reverse mode, in the sense that you only need
/// one type argument `D: Diff` instead of `T, E, I` with constraints.
/// However, it doesn't work well with type inference on the call site, so the D argument
/// needs to be specified explicitly.
pub trait Diff: Clone {
    type T: Clone;
    type E: Clone;
    type I: 'static + Diffable<Repr<Self::E> = Self::T> + Clone;
    type Fwd: Diff<T = Forward<Self::T>, E = Self::E, I = ForwardImpl<Self::I>>;
    type Rev: Diff<T = Reverse<Self::T>, E = Self::E, I = ReverseImpl<Self::I>>;
}

impl<T: Clone, E: Clone, I: 'static + Diffable<Repr<E> = T> + Clone> Diff for Tensor<T, E, I> {
    type T = T;
    type E = E;
    type I = I;
    type Fwd = Tensor<Forward<Self::T>, Self::E, ForwardImpl<Self::I>>;
    type Rev = Tensor<Reverse<Self::T>, Self::E, ReverseImpl<Self::I>>;
}

pub type Tensorken<Tsr> = Tensor<<Tsr as Diff>::T, <Tsr as Diff>::E, <Tsr as Diff>::I>;
pub type TensorFwd<Tsr> = Tensorken<<Tsr as Diff>::Fwd>;
pub type TensorRev<Tsr> = Tensorken<<Tsr as Diff>::Rev>;
