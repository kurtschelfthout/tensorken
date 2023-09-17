#![warn(clippy::pedantic)]
// Gives many wrong warnings e.g. for vjp calls.
// Doesn't realize that those need a closure for the HRTB.
#![allow(clippy::redundant_closure_for_method_calls)]

mod ad_reverse;
mod ad_reverse_ops;
mod diffable;
mod diffable_ext;
mod math_macros;
pub mod num;
mod raw_tensor;
mod raw_tensor_cpu;
mod raw_tensor_fuse;
mod raw_tensor_shape_tracker;
mod raw_tensor_string;
mod raw_tensor_wgpu;
mod shape;
mod shape_strider;
pub mod tensor;
pub mod tensor_mut;
mod wgpu_context;

pub use ad_reverse::{
    grad1, grad2, jacrev1, jacrev2, jacrevn, value_and_grad1, value_and_grad2, value_and_gradn,
    vjpn, PullBack, Reverse,
};
pub use diffable::Diffable;
pub use diffable_ext::DiffableExt;
pub use raw_tensor::RawTensor;
pub use raw_tensor_cpu::CpuRawTensor;
pub use raw_tensor_wgpu::WgpuRawTensor;
pub use shape::Shape;
pub use tensor::{Cpu32, IndexValue, Tensor, TensorLike, TensorLikeRef, Wgpu32};
