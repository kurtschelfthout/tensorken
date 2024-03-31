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
pub use indexing::{hd, tl, IndexSpec, IndexSpecBuilder, IndexValue, ELLIPSIS, NEW_AXIS};
pub use raw_tensor::{RawTensorOps, ToCpu};
pub use raw_tensor_cpu::{CpuRawTensor, CpuRawTensorImpl};
pub use raw_tensor_fuse::{Fuse, FuseImpl};
pub use raw_tensor_string::StringImpl;
pub use raw_tensor_wgpu::{WgpuRawTensor, WgpuRawTensorImpl};
pub use shape::Shape;
pub use shape_strider::ShapeStrider;
pub use tensor::{Axes, Tensor};
pub use type_magic::{
    Cpu, Cpu32, CpuBool, CpuI32, Diff, Sym, TensorBase, TensorFwd, TensorRev, Wgpu, Wgpu32,
    WgpuBool, WgpuI32,
};

// TODO:
// - Make vjp and jvp and friends all be N-to-N arguments instead of N-to-1.
// - Treat zero and one as a special case for efficiency (also avoids NaNs, see jvp_test for pow). Maybe as a RawTensorOps implementation like Fuse.
// - Try a jax style vmap implementation of Diffable
// - The DiffableOps trait inflates the requirements on E, the element type somewhat artificially. A good example is that broadcasted `eq` does not
//   work on bools, because it requires Num. This is because broadcasting requires expand, which requires sum for reverse-mode AD.
//   One somewhat hacky solution is to implement add/mul/sum in Bool (as or/and/or)
//   Another perhaps more principled solution would be to have two `eq`, `add` etc functions on `Tensor`, one that uses `RawTensorOps` and so don't have
//   the inflated requirements, and one that uses `DiffableOps` for input into jvp/vjp (it doesn't matter there that there are more requirements, because
//   we need Num anyway to differentiate). This however seems to require implementing all the operations on Tensor twice.
// - JAX supports differentiating through container types, i.e. tensors in lists, tuples etc.
// - cast is not currently diffable, it's always lifted. It's a bit tricky to implement cast because of how it would mean having two parameters for `UnaryOp<T>`,
//   and I'm not currently sure how to best handle that.
// - consider implementing a numerical gradient checker for testing.
