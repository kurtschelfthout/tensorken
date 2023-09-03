# Tensorken: A Fun, Hackable, GPU-Accelerated, Neural Network library in Rust, Written by an Idiot

(work in progress)

Understanding deep learning from the perspective of a programmer, by building a deep learning framework from the ground up, in the spirit of [tinygrad](https://github.com/geohot/tinygrad) and [micrograd](https://github.com/karpathy/micrograd).

- Fun and hackable: most importantly Tensorken doesn't take itself too seriously. It's meant to be small, hackable, easy to understand and change above all else. If you want something usable for real work, look elsewhere.
- GPU-Accelerated: For the moment Tensorken runs on the GPU via [wgpu](https://wgpu.rs/), Rust's implementation of WebGPU. Accelerated comes with a grain of salt: tensor operations are much faster than the bundled but very naïve CPU implementation.
- Neural network: Getting less aspirational every month. There are basic tensor operations that run on CPU and GPU and a prototype reverse-mode autodiff implementation with a JAX-style API.
- Rust: No particular reason other than that I'm learning Rust.
- Written by an idiot: Hi there! I know nothing about neural network or GPU programming. As a result, anything and everything in here may be slow, backward, wrong, or stupid, and that's not an exclusive or exhaustive list.

## "Tensorken"?

The suffix -ke means "small" in Flemish. So tensorke is a small tensor. Some Flemish dialects append an extra n, which sounds better in English.

## The shoulders of giants

Some cool stuff I looked at while figuring out how to build this.

- [micrograd](https://github.com/karpathy/micrograd)
- [tinygrad](https://github.com/geohot/tinygrad)
- [MiniTorch](https://minitorch.github.io/module0/module0/)
- [JAX](https://jax.readthedocs.io/en/latest/autodidax.html#jax-core-machinery)
- [DiffSharp](https://diffsharp.github.io/)

## Getting started

Just clone the repo and run `cargo run --example tour`. Then explore the code - it's not big!

Emerging posts with more explanation of the various stages of development of Tensorken. Intended to be read in order.

- [v0.1](https://github.com/kurtschelfthout/tensorken/releases/tag/v0.1): [Fun and Hackable Tensors in Rust, From Scratch](https://getcode.substack.com/p/fun-and-hackable-tensors-in-rust).

- [v0.2](https://github.com/kurtschelfthout/tensorken/releases/tag/v0.2): [Massively Parallel Fun with GPUs: Accelerating Tensors in Rust](https://getcode.substack.com/p/massively-parallel-fun-with-gpus).
