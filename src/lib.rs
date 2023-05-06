#![warn(clippy::pedantic)]
// Gives many wrong warnings e.g. for vjp calls.
// Doesn't realize that those need a closure for the HRTB.
#![allow(clippy::redundant_closure_for_method_calls)]

mod ad_reverse;
mod ad_reverse_ops;
mod diffable;
mod math_macros;
mod num;
mod raw_tensor;
mod raw_tensor_cpu;
mod raw_tensor_wgpu;
mod shape;
mod shape_strider;
pub mod tensor;
mod tensor;
pub mod tensor_mut;
mod tensor_mut;
mod wgpu_context;

pub use ad_reverse::{vjp1, vjpn, Reverse};
pub use diffable::{Diffable, DiffableExt};
pub use raw_tensor::RawTensor;
pub use raw_tensor_cpu::CpuRawTensor;
pub use raw_tensor_wgpu::WgpuRawTensor;
pub use shape::Shape;
pub use tensor::{Cpu32, IndexValue, Tensor, TensorLike, TensorLikeRef, Wgpu32};
