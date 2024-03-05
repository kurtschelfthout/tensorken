#![warn(clippy::pedantic)]

mod ad_forward;
mod ad_ops;
mod ad_ops_forward;
mod ad_ops_reverse;
mod ad_reverse;
mod ad_trace;
mod diffable;
mod diffable_ext;
mod indexing;
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
pub mod tensor_display;
pub mod tensor_mut;
mod wgpu_context;
pub use ad_forward::{
    diff1, diff2, jacfwd, jvpn, value_and_diff1, value_and_diff2, value_and_diffn, Forward,
};
pub use ad_reverse::{
    grad1, grad2, jacrev, value_and_grad1, value_and_grad2, value_and_gradn, vjpn, PullBack,
    Reverse,
};
pub use diffable::Diffable;
pub use diffable_ext::{Axes, DiffableExt};
pub use indexing::{sl, sl1, sl2, sl3, sl4, IndexValue, Slice, SliceIdx};
pub use raw_tensor::{RawTensor, RealizedRawTensor};
pub use raw_tensor_cpu::CpuRawTensor;
pub use raw_tensor_wgpu::WgpuRawTensor;
pub use shape::Shape;
pub use tensor::{Cpu32, CpuBool, CpuI32, Tensor, TensorLike, TensorLikeRef, Wgpu32};

// TODO of general interest
// - Make vjp and jvp and friends all be N-to-N arguments instead of N-to-1.
// - Treat zero and one as a special case for efficiency (also avoids NaNs, see jvp_test for pow). Maybe as a RawTensor implementation like Fuse.
// - Try a jax style vmap implementation of Diffable
