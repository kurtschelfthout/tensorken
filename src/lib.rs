#![warn(clippy::pedantic)]

mod ad_forward;
mod ad_ops;
mod ad_ops_forward;
mod ad_ops_reverse;
mod ad_reverse;
mod ad_trace;
mod diffable;
mod indexing;
mod math_macros;
pub mod num;
mod raw_tensor;
mod raw_tensor_cpu;
mod raw_tensor_fuse;
mod raw_tensor_string;
mod raw_tensor_wgpu;
mod shape;
mod shape_strider;
mod tensor;
mod tensor_display;
mod tensor_mut;
mod type_magic;
mod wgpu_context;
pub use ad_forward::{
    diff1, diff2, jacfwd, jvpn, value_and_diff1, value_and_diff2, value_and_diffn, Forward,
    ForwardImpl,
};
pub use ad_reverse::{
    grad1, grad2, jacrev, value_and_grad1, value_and_grad2, value_and_gradn, vjpn, PullBack,
    Reverse, ReverseImpl,
};
pub use diffable::DiffableOps;
pub use indexing::{sl, sl1, sl2, sl3, sl4, IndexValue, Slice, SliceIdx};
pub use raw_tensor::{RawTensorOps, ToCpu};
pub use raw_tensor_cpu::{CpuRawTensor, CpuRawTensorImpl};
pub use raw_tensor_fuse::{Fuse, FuseImpl};
pub use raw_tensor_string::StringImpl;
pub use raw_tensor_wgpu::{WgpuRawTensor, WgpuRawTensorImpl};
pub use shape::Shape;
pub use shape_strider::ShapeStrider;
pub use tensor::{Axes, Tensor};
pub use type_magic::{
    Cpu, Cpu32, CpuBool, CpuI32, Diff, Sym, TensorFwd, TensorRev, Tensorken, Wgpu, Wgpu32,
    WgpuBool, WgpuI32,
};

// TODO of general interest
// - Make vjp and jvp and friends all be N-to-N arguments instead of N-to-1.
// - Treat zero and one as a special case for efficiency (also avoids NaNs, see jvp_test for pow). Maybe as a RawTensor implementation like Fuse.
// - Try a jax style vmap implementation of Diffable
