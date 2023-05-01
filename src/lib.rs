#![warn(clippy::pedantic)]
// Gives many wrong warnings e.g. for vjp calls.
// Doesn't realize that those need a closure for the HRTB.
#![allow(clippy::redundant_closure_for_method_calls)]

mod ad_reverse;
pub mod diffable_ops;
mod num;
pub mod raw_tensor;
pub mod raw_tensor_cpu;
pub mod raw_tensor_wgpu;
pub mod shape;
mod shape_strider;
pub mod tensor;
pub mod tensor_mut;
mod wgpu_context;
