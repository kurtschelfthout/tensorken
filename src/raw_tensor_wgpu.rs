use std::{
    borrow::Cow,
    fmt::{Debug, Formatter},
    marker::PhantomData,
    sync::Arc,
};

use wgpu::util::DeviceExt;

use crate::{
    num::{Bool, CastFrom, Elem, Float, Num},
    raw_tensor::{RawTensorOps, ToCpu},
    shape::Shape,
    shape_strider::{ShapeStrider, Stride},
    wgpu_context::{get_wgpu_device, WgpuContext, WorkgroupSize},
    CpuRawTensor,
};

// Misc WGSL notes/tips:
// - WGSL aggressively prunes dead code. If you don't use a binding in WGSL, that binding gets removed.
//   If you have declared it in the rust wgpu code as a binding group, there will
//   be a failure at runtime, saying the number of bindings doesn't correspond.
// - I've been told it's a good idea to cache compute pipelines, as this avoids recompiling WGSL.
// - It's necessary to poll the device after submission of a command buffer, otherwise under high
//   load the GPU will run out of memory.
// - It's important to distinguish between the workgroup size, and the number of workgroups, which I've called `workgroup_count` in the code.
//   - workgroup size: fixed in the shader using `@workgroup_size`. The number of threads that execute in parallel within a workgroup.
//     These threads all execute the same code in sync (even with branches - if one of the threads doesn't need to take a branch, it'll still
//     fake-execute the code, i.e. do nothing useful) and can access typically faster shared workgroup memory. The idea is that these threads
//     all execute the same code, but on different data (SIMD) and are phyiscally close together on the GPU.
//   - workgroup count: set at dispatch time. The number of workgroups that are executed. Each workgroup has its own set of threads,
//     so the total number of invocations of the shader is workgroup size * workgroup count. Workgroups may or may not be executed in parallel,
//     this is decided by the GPU scheduler.
// - There are limits in WebGPU on the number of workgroups that can be dispatched, and the number of threads per workgroup.
// - Both workgroup size and workgroup count can be specified separately in x,y, and z dimensions.

/// Operations avoid copying the buffer if possible, but buffers are read-only,
/// and can be shared between multiple tensors (e.g. with different shapes).
/// As a result, buffers are reference counted. Cloning a `WgpuRawTensor` is cheap.
/// The buffer lives in GPU memory.
#[derive(Clone)]
pub struct WgpuRawTensor<'a, E> {
    buffer: Arc<wgpu::Buffer>,
    strider: ShapeStrider,
    context: &'a WgpuContext,
    element: std::marker::PhantomData<E>,
}

impl<E> Debug for WgpuRawTensor<'_, E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(format!("WgpuRawTensor<{}>", std::any::type_name::<E>()).as_str())
            .field("strider", &self.strider)
            .finish()
    }
}

impl<'a, E> WgpuRawTensor<'a, E> {
    /// Return a new tensor with the same buffer as this one, but
    /// with a different shape. Assumes the new shape is compatible.
    fn with_strider(&self, strider: ShapeStrider) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            strider,
            context: self.context,
            element: std::marker::PhantomData,
        }
    }

    fn device(&self) -> &wgpu::Device {
        &self.context.device
    }

    fn queue(&self) -> &wgpu::Queue {
        &self.context.queue
    }

    #[allow(dead_code)]
    fn strides(&self) -> &[Stride] {
        self.strider.strides()
    }
}

type ReduceInfo<'a> = (&'a [usize], &'a [usize], &'a [usize], usize);

impl<'a, E: Elem> WgpuRawTensor<'a, E> {
    fn byte_size(size: usize) -> u64 {
        u64::try_from(size * E::WGPU_ELEMENT_SIZE).unwrap()
    }

    /// Creates a new `WgpuRawTensor` from a shape and CPU data.
    /// The data is copied to the GPU.
    /// # Panics
    /// Panics if shape and data size do not match.
    fn new(shape: &[usize], cpu_data: &[E], device: &'a WgpuContext) -> Self {
        assert!(
            shape.size() == cpu_data.len(),
            "Shape and data size mismatch"
        );

        let strider = ShapeStrider::contiguous(shape);
        match E::to_buffer(cpu_data) {
            Cow::Borrowed(slice) => {
                let buffer = device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Tensor (new)"),
                        contents: slice,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });

                Self {
                    buffer: Arc::new(buffer),
                    strider,
                    context: device,
                    element: std::marker::PhantomData,
                }
            }
            Cow::Owned(vec) => {
                let buffer = device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Tensor (new)"),
                        contents: vec.as_slice(),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });

                Self {
                    buffer: Arc::new(buffer),
                    strider,
                    context: device,
                    element: std::marker::PhantomData,
                }
            }
        }
    }

    /// Get the buffer that encodes various strides and shapes of the inputs.
    /// Is used for all operations, so is a mess of a signature.
    fn get_strides_and_shapes_buffer(
        &self,
        chunk_size: usize,
        other: Option<&ShapeStrider>,
        output_strider: &ShapeStrider,
        reduce: Option<ReduceInfo>,
        padding: Option<&[(usize, usize)]>,
    ) -> wgpu::Buffer {
        let mut contents = Vec::with_capacity(8 * self.strider.shape().ndims() + 3);
        contents.push(self.strider.shape().ndims());
        contents.push(self.strider.offset());
        if let Some(other) = other {
            contents.push(other.offset());
        }
        contents.push(chunk_size);
        if let Some((_, _, _, reduce_size)) = reduce {
            contents.push(reduce_size);
        }

        contents.extend(self.strider.pos_strides());
        if let Some(other) = other {
            contents.extend(other.pos_strides());
        }

        contents.extend(output_strider.pos_strides());
        contents.extend(self.strider.shape());

        if let Some((reduced_strides, reduced_shape, output_shape, _)) = reduce {
            contents.extend(reduced_strides);
            contents.extend(reduced_shape);
            contents.extend(output_shape);
        }

        if let Some(padding) = padding {
            for (start, _) in padding {
                contents.push(*start);
            }
            for (_, end) in padding {
                contents.push(*end);
            }
        }

        let contents_u32: Vec<u32> = contents
            .iter()
            .map(|x| u32::try_from(*x).unwrap())
            .collect();

        let buffer = self
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                usage: wgpu::BufferUsages::STORAGE,
                label: Some("shapes and strides buffer"),
                contents: bytemuck::cast_slice(&contents_u32),
            });
        buffer
    }

    /// Make a new output buffer, to store the result of an operation,
    /// and so a new tensor.
    fn make_output_buffer(&self, size: usize, operation: &str) -> wgpu::Buffer {
        let size = Self::byte_size(size);
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Tensor ({operation})")),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Get a bind group for unary operations, i.e. that take one tensor as input.
    fn get_bind_group_unary(
        &self,
        pipeline: &wgpu::ComputePipeline,
        output_buffer: &wgpu::Buffer,
        strides_and_shapes: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: strides_and_shapes.as_entire_binding(),
                },
            ],
        })
    }

    /// Get a bind group for zip operations.
    fn get_bind_group_zip(
        &self,
        other: &Self,
        pipeline: &wgpu::ComputePipeline,
        output_buffer: &wgpu::Buffer,
        strides_and_shapes: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: other.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: strides_and_shapes.as_entire_binding(),
                },
            ],
        })
    }

    /// Submit the compute pipeline for execution to the GPU, and wait for it to complete.
    fn encode_and_submit(
        &self,
        compute_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroup_count: usize,
    ) {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            let workgroup_count = u32::try_from(workgroup_count).unwrap();
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Submit commands to the GPU, and wait synchronously for them to complete.
        // This naive approach makes CPU-GPU interaction entirely synchronous.
        // In principle, it's not necessary for correctness to wait for each command to finish.
        // The output buffer needs to be created and filled before it is used in further tensor
        // operations, but wgpu tracks such dependencies automatically, and inserts the necessary
        // barriers.
        // However, submitting a large number of compute passes without waiting for the result
        // can cause resource (sometimes memory) exhaustion, and the GPU device crashes. Without this,
        // benchmarks like matmul fail at higher sizes.
        // See https://github.com/gfx-rs/wgpu/issues/3806
        let index = self.queue().submit(Some(encoder.finish()));
        self.device()
            .poll(wgpu::Maintain::WaitForSubmissionIndex(index));
    }

    fn pipeline_for(
        &self,
        operation: &'static str,
        element_out: &'static str,
        workgroup_size: WorkgroupSize,
    ) -> Arc<wgpu::ComputePipeline> {
        self.context
            .pipeline_for(operation, E::WGPU_ELEMENT_NAME, element_out, workgroup_size)
    }
    /// Create a new tensor with given buffer and strider, passing self's context along.
    fn with_buffer_strider(&self, buffer: wgpu::Buffer, strider: ShapeStrider) -> Self {
        WgpuRawTensor {
            buffer: Arc::new(buffer),
            strider,
            context: self.context,
            element: self.element,
        }
    }

    /// Max number of threads in a workgroup, as defined by WebGPU.
    const MAX_WORKGROUP_SIZE: usize = 256;
    /// Max number of workgroups in a dispatch, as defined by WebGPU.
    const MAX_WORKGROUP_COUNT: usize = 65535;

    /// Return:
    /// - workgroup size (the number of threads in each workgroup, x and y. y is always 1.),
    /// - workgroup count (the number of workgroups to run)
    /// - chunk size (the number of elements each thread will process)
    /// This is based on the size of the output tensor, as for all operations except
    /// reduce, that's the maximum amount of parallelism we can have.
    /// The idea is to maximize workgroup size, then workgroup count, then chunk size -
    /// this should maximize parallelism.
    fn counts_n_sizes(output_tensor_size: usize) -> ((usize, usize), usize, usize) {
        if output_tensor_size <= Self::MAX_WORKGROUP_SIZE {
            return ((output_tensor_size, 1), 1, 1);
        }

        let workgroup_count =
            (output_tensor_size + Self::MAX_WORKGROUP_SIZE - 1) / Self::MAX_WORKGROUP_SIZE;
        if workgroup_count <= Self::MAX_WORKGROUP_COUNT {
            return ((Self::MAX_WORKGROUP_SIZE, 1), workgroup_count, 1);
        }

        let chunk_size =
            (output_tensor_size + Self::MAX_WORKGROUP_COUNT - 1) / Self::MAX_WORKGROUP_COUNT;
        (
            (Self::MAX_WORKGROUP_SIZE, 1),
            Self::MAX_WORKGROUP_COUNT,
            chunk_size,
        )
    }

    fn clamp(v: usize, min: usize, max: usize) -> usize {
        if v < min {
            min
        } else if v > max {
            max
        } else {
            v
        }
    }

    /// Return:
    /// - workgroup size (the number of threads in each workgroup, x and y.),
    /// - workgroup count (the number of workgroups to run)
    /// - chunk size (the number of elements each thread will process)
    /// Specifically for the reduce operation.
    /// Reduce is suboptimal if we'd only base it on the output tensor size: in the extreme,
    /// reducing to a single element achieves no parallelism at all.
    /// The idea here is to check if we have enough parallelism in the output tensor to
    /// parallelize "the normal way". If the output tensor size is smaller than the max workgroup size,
    /// we also parallelize the reduction itself.
    fn counts_n_sizes_reduce(
        input_tensor_size: usize,
        output_tensor_size: usize,
    ) -> (WorkgroupSize, usize, usize) {
        if output_tensor_size <= Self::MAX_WORKGROUP_SIZE {
            // Parallelize the reduction. e.g. reducing a tensor to a single scalar.

            // First parallelize on the output size as much as possible.
            let workgroup_size_x = output_tensor_size;

            // Then parallelize the reduction.
            // This is the number of elements that need to be reduced to a single output element.
            let reduce_size = input_tensor_size / output_tensor_size;
            // We can only have MAX_WORKGROUP_SIZE threads total.
            let max_workgroup_size_y = Self::MAX_WORKGROUP_SIZE / workgroup_size_x;
            // Arbitrarily choose to reduce at least 64 elements per thread.
            let workgroup_size_y = Self::clamp(reduce_size / 64, 1, max_workgroup_size_y);
            let workgroup_size = (workgroup_size_x, workgroup_size_y);
            let (workgroup_count, chunk_size) = (1, 1);
            // dbg!(
            //     output_tensor_size,
            //     workgroup_size,
            //     workgroup_count,
            //     chunk_size
            // );
            return (workgroup_size, workgroup_count, chunk_size);
        }

        // if the output tensor is larger than the max workgroup size, we don't parallelize
        // the reduce. I.e. each thread reduces one subtensor to a single element.
        // (this will check MAX_WORKGROUP_SIZE again, but ok)
        Self::counts_n_sizes(output_tensor_size)
    }

    /// Return a new tensor with the same shape as self, after applying f to each element.
    /// Allocates a new buffer. Resulting tensor is contiguous.
    fn map(&self, operation: &'static str) -> Self {
        let output_strider = ShapeStrider::contiguous(self.strider.shape());

        let (workgroup_size, workgroup_count, chunk_size) =
            Self::counts_n_sizes(output_strider.size());

        let output_buffer = self.make_output_buffer(output_strider.size(), operation);

        let compute_pipeline = self.pipeline_for(operation, E::WGPU_ELEMENT_NAME, workgroup_size);
        let strides_and_shapes =
            self.get_strides_and_shapes_buffer(chunk_size, None, &output_strider, None, None);
        let bind_group = self.get_bind_group_unary(
            compute_pipeline.as_ref(),
            &output_buffer,
            &strides_and_shapes,
        );
        self.encode_and_submit(compute_pipeline.as_ref(), &bind_group, workgroup_count);

        self.with_buffer_strider(output_buffer, output_strider)
    }

    /// Return a new tensor with the same shape as self, cast to the desired element type `TTo`.
    /// Allocates a new buffer. Resulting tensor is contiguous.
    fn cast<TTo: Elem>(&self) -> WgpuRawTensor<'a, TTo> {
        let output_strider = ShapeStrider::contiguous(self.strider.shape());

        let (workgroup_size, workgroup_count, chunk_size) =
            Self::counts_n_sizes(output_strider.size());

        let output_buffer = self.make_output_buffer(output_strider.size(), TTo::WGPU_ELEMENT_NAME);

        let compute_pipeline = self.pipeline_for(
            TTo::WGPU_ELEMENT_NAME,
            TTo::WGPU_ELEMENT_NAME,
            workgroup_size,
        );
        let strides_and_shapes =
            self.get_strides_and_shapes_buffer(chunk_size, None, &output_strider, None, None);
        let bind_group = self.get_bind_group_unary(
            compute_pipeline.as_ref(),
            &output_buffer,
            &strides_and_shapes,
        );
        self.encode_and_submit(compute_pipeline.as_ref(), &bind_group, workgroup_count);

        WgpuRawTensor {
            buffer: Arc::new(output_buffer),
            strider: output_strider,
            context: self.context,
            element: std::marker::PhantomData,
        }
    }

    /// Return a new tensor with the same shape as self and other, after applying f to each pair of elements.
    /// Allocates a new buffer. Resulting tensor is contiguous.
    fn zip(&self, other: &Self, operation: &'static str) -> Self {
        self.strider.validate_can_zip(&other.strider).unwrap();

        let output_strider = ShapeStrider::contiguous(self.strider.shape());

        let (workgroup_size, workgroup_count, chunk_size) =
            Self::counts_n_sizes(output_strider.size());
        let output_buffer = self.make_output_buffer(output_strider.size(), operation);
        let compute_pipeline = self.pipeline_for(operation, E::WGPU_ELEMENT_NAME, workgroup_size);
        let strides_and_shapes = self.get_strides_and_shapes_buffer(
            chunk_size,
            Some(&other.strider),
            &output_strider,
            None,
            None,
        );
        let bind_group = self.get_bind_group_zip(
            other,
            compute_pipeline.as_ref(),
            &output_buffer,
            &strides_and_shapes,
        );
        self.encode_and_submit(compute_pipeline.as_ref(), &bind_group, workgroup_count);

        WgpuRawTensor {
            buffer: Arc::new(output_buffer),
            strider: output_strider,
            context: self.context,
            element: PhantomData,
        }
    }

    /// Return a new tensor with the given axes reduced.
    /// Allocates a new buffer. Resulting tensor is contiguous.
    fn reduce(&self, axes: &[usize], operation: &'static str) -> Self {
        self.strider.validate_can_reduce(axes).unwrap();

        let (output_strider, _) = self.strider.reduce(axes);

        let mut reduced_shape = vec![1usize; self.strider.shape().ndims()];
        for &axis in axes {
            reduced_shape[axis] = self.strider.shape()[axis];
        }
        let reduced_strider = ShapeStrider::contiguous(&reduced_shape);

        let (workgroup_size, workgroup_count, chunk_size) =
            Self::counts_n_sizes_reduce(self.strider.size(), output_strider.size());
        let output_buffer = self.make_output_buffer(output_strider.size(), operation);
        let compute_pipeline = self.pipeline_for(operation, E::WGPU_ELEMENT_NAME, workgroup_size);
        let strides_and_shapes = self.get_strides_and_shapes_buffer(
            chunk_size,
            None,
            &output_strider,
            Some((
                &reduced_strider.pos_strides(),
                reduced_strider.shape(),
                output_strider.shape(),
                reduced_strider.size(),
            )),
            None,
        );
        let bind_group = self.get_bind_group_unary(
            compute_pipeline.as_ref(),
            &output_buffer,
            &strides_and_shapes,
        );
        self.encode_and_submit(compute_pipeline.as_ref(), &bind_group, workgroup_count);

        self.with_buffer_strider(output_buffer, output_strider)
    }

    /// Pad the tensor with the given padding.
    /// Allocates a new buffer. Resulting tensor is contiguous.
    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        self.strider.validate_can_pad(padding).unwrap();

        let output_strider = self.strider.pad(padding);
        let output_buffer = self.make_output_buffer(output_strider.size(), "pad");

        let (workgroup_size, workgroup_count, chunk_size) =
            Self::counts_n_sizes(output_strider.size());
        let compute_pipeline = self.pipeline_for("pad", E::WGPU_ELEMENT_NAME, workgroup_size);
        let strides_and_shapes = self.get_strides_and_shapes_buffer(
            chunk_size,
            None,
            &output_strider,
            None,
            Some(padding),
        );
        let bind_group = self.get_bind_group_unary(
            compute_pipeline.as_ref(),
            &output_buffer,
            &strides_and_shapes,
        );
        self.encode_and_submit(compute_pipeline.as_ref(), &bind_group, workgroup_count);

        self.with_buffer_strider(output_buffer, output_strider)
    }

    /// Elementwise multiply of self with other, followed by summing along the given axes, in
    /// a single operation.
    /// Allocates a new buffer. Resulting tensor is contiguous.
    fn fused_multiply_add_impl(&self, other: &Self, axes: &[usize]) -> Self {
        self.strider.validate_can_zip(&other.strider).unwrap();
        self.strider.validate_can_reduce(axes).unwrap();

        let (output_strider, _) = self.strider.reduce(axes);

        let mut reduced_shape = vec![1usize; self.strider.shape().ndims()];
        for &axis in axes {
            reduced_shape[axis] = self.strider.shape()[axis];
        }
        let reduced_strider = ShapeStrider::contiguous(&reduced_shape);

        let (workgroup_size, workgroup_count, chunk_size) =
            Self::counts_n_sizes(output_strider.size());
        let output_buffer = self.make_output_buffer(output_strider.size(), "fma");
        let compute_pipeline =
            self.pipeline_for("fused_mul_add", E::WGPU_ELEMENT_NAME, workgroup_size);
        let strides_and_shapes = self.get_strides_and_shapes_buffer(
            chunk_size,
            Some(&other.strider),
            &output_strider,
            Some((
                &reduced_strider.pos_strides(),
                reduced_strider.shape(),
                output_strider.shape(),
                reduced_strider.size(),
            )),
            None,
        );
        let bind_group = self.get_bind_group_zip(
            other,
            compute_pipeline.as_ref(),
            &output_buffer,
            &strides_and_shapes,
        );
        self.encode_and_submit(compute_pipeline.as_ref(), &bind_group, workgroup_count);

        self.with_buffer_strider(output_buffer, output_strider)
    }

    /// Returns a contiguous copy of the tensor.
    fn contiguous(&self) -> Self {
        self.map("id")
    }

    /// Returns a contiguous vector of the tensor's elements, on the CPU.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn ravel(&self) -> Vec<E> {
        if !self.strider.is_contiguous() {
            let t = self.contiguous();
            return t.ravel();
        }

        let size = Self::byte_size(self.strider.shape().size());
        let offset = Self::byte_size(self.strider.offset());
        let staging_buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Add copy operation to command encoder - copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&self.buffer, offset, &staging_buffer, 0, size);

        // Submits command encoder for processing
        let index = self.queue().submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        self.device()
            .poll(wgpu::Maintain::WaitForSubmissionIndex(index));
        let poll_result = pollster::block_on(receiver.receive());

        if let Some(Ok(())) = poll_result {
            let data = buffer_slice.get_mapped_range();
            let result = E::from_buffer(&data);

            // Resource cleanup dance - apparently needs to happen in this order.
            drop(data);
            staging_buffer.unmap();

            result
        } else {
            panic!("Failed to read buffer from GPU")
        }
    }
}

#[derive(Debug, Clone)]
pub enum WgpuRawTensorImpl {}

impl RawTensorOps for WgpuRawTensorImpl {
    type Repr<E: Clone> = WgpuRawTensor<'static, E>;

    fn exp<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.map("exp")
    }

    fn log<E: Float>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.map("log")
    }

    fn cast<EFro: Elem, ETo: CastFrom<EFro> + Elem>(t: &Self::Repr<EFro>) -> Self::Repr<ETo> {
        t.cast()
    }

    fn realize<E: Clone>(t: &Self::Repr<E>) -> Self::Repr<E> {
        t.clone()
    }

    fn add<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.zip(rhs, "add")
    }

    fn sub<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.zip(rhs, "sub")
    }

    fn mul<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.zip(rhs, "mul")
    }

    fn div<E: Num>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.zip(rhs, "div")
    }

    fn pow<E: Float>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E> {
        lhs.zip(rhs, "pow")
    }

    fn eq<E: PartialEq + Elem>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<bool> {
        lhs.zip(rhs, "eq").cast()
    }

    fn sum<E: Num>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        t.reduce(axes, "sum")
    }

    fn max<E: Bool>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E> {
        t.reduce(axes, "max")
    }

    fn reshape<E: Elem>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        t.strider.validate_can_reshape(shape).unwrap();

        if let Ok(strider) = t.strider.reshape(shape) {
            return t.with_strider(strider);
        }
        WgpuRawTensorImpl::reshape(&t.contiguous(), shape)
    }

    fn permute<E: Clone>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E> {
        t.strider.validate_can_permute(permutation).unwrap();

        let strider = t.strider.permute(permutation);
        t.with_strider(strider)
    }

    fn expand<E: Clone>(t: &Self::Repr<E>, shape: &[usize]) -> Self::Repr<E> {
        t.strider.validate_can_expand(shape).unwrap();

        let strider = t.strider.expand(shape).unwrap();
        t.with_strider(strider)
    }

    fn flip<E: Clone>(t: &Self::Repr<E>, flip: &[bool]) -> Self::Repr<E> {
        t.strider.validate_can_flip(flip).unwrap();

        let strider = t.strider.flip(flip);
        t.with_strider(strider)
    }

    fn pad<E: Bool>(t: &Self::Repr<E>, padding: &[(usize, usize)]) -> Self::Repr<E> {
        t.strider.validate_can_pad(padding).unwrap();

        t.pad(padding)
    }

    fn crop<E: Clone>(t: &Self::Repr<E>, limits: &[(usize, usize)]) -> Self::Repr<E> {
        t.strider.validate_can_crop(limits).unwrap();

        let strider = t.strider.crop(limits);
        t.with_strider(strider)
    }

    fn new<E: Elem>(shape: &[usize], data: &[E]) -> Self::Repr<E> {
        WgpuRawTensor::new(shape, data, get_wgpu_device())
    }

    fn shape<E: Clone>(t: &Self::Repr<E>) -> &[usize] {
        t.strider.shape()
    }

    fn fused_multiply_add<E: Num>(
        lhs: &Self::Repr<E>,
        rhs: &Self::Repr<E>,
        axes: &[usize],
    ) -> Self::Repr<E> {
        lhs.fused_multiply_add_impl(rhs, axes)
    }
}

impl ToCpu for WgpuRawTensorImpl {
    fn to_cpu<E: Elem>(t: &Self::Repr<E>) -> crate::CpuRawTensor<E> {
        CpuRawTensor::new_into(Self::shape(t), t.ravel())
    }
}

#[cfg(test)]
mod tests {

    use std::{f32::consts::E, iter::repeat};

    use crate::wgpu_context::get_wgpu_device;

    use super::*;

    type I = WgpuRawTensorImpl;

    fn assert_pos_strides<E: Clone>(t: &WgpuRawTensor<E>) -> Vec<usize> {
        t.strides()
            .iter()
            .map(|s| match s {
                Stride::Pos(i) => *i,
                Stride::Neg(_) => panic!("Expected positive strides, found {s:?}"),
            })
            .collect::<Vec<_>>()
    }

    fn assert_vec_eq(a: &[f32], b: &[f32]) {
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(a, b)| (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-2),
            "\r\nleft : {a:?}\r\nright: {b:?}"
        );
    }

    fn assert_vec_eq_num<T: Eq + Debug>(a: &[T], b: &[T]) {
        assert!(
            a.iter().zip(b.iter()).all(|(a, b)| a == b),
            "\r\nleft : {a:?}\r\nright: {b:?}"
        );
    }

    #[test]
    fn test_single_unary_ops() {
        let input = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let tensor = WgpuRawTensor::new(&[3, 2], &input, get_wgpu_device());
        let result = tensor.map("exp").ravel();
        // NOTE: there may be small differences between the GPU and CPU floating point ops.
        assert_vec_eq(
            &result,
            &input.iter().map(|x| x.exp()).collect::<Vec<f32>>(),
        );

        let result = tensor.map("log").ravel();
        assert_vec_eq(
            &result,
            &input.iter().map(|x| x.log(E)).collect::<Vec<f32>>(),
        );
    }

    #[test]
    fn test_sequential_unary_ops() {
        let input = vec![1.0, 2.0, -1.0, -2.0, 3.0, -3.0, 4.0, 5.0, 6.0];
        let tensor = WgpuRawTensor::new(&[3, 3], &input, get_wgpu_device());
        let result = tensor.map("exp").map("log").ravel();
        // NOTE: there may be small differences between the GPU and CPU floating point ops.
        assert_vec_eq(
            &result,
            &input.iter().map(|x| x.exp().log(E)).collect::<Vec<f32>>(),
        );
    }

    #[allow(clippy::float_cmp)]
    fn apply_binary_op(op: &str, a: f32, b: f32) -> f32 {
        match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" => a / b,
            // Rust returns something for powers of negative bases,
            // which in general, for real exponents, is a complex number.
            // WebGPU does not do that - depending on the device, it seems to
            // either return NaN (NVidia) or some number (Intel).
            // We just make sure to call it only with positive numbers.
            "pow" => a.powf(b),
            "eq" => f32::from(a == b),
            _ => panic!("unknown op: {op}"),
        }
    }

    #[test]
    fn test_binary_ops() {
        let i1 = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let t1 = WgpuRawTensor::new(&[3, 2], &i1, get_wgpu_device());
        let i2 = vec![6.0, 7.0, 8.0, -6.0, -7.0, -8.0];
        let t2 = WgpuRawTensor::new(&[3, 2], &i2, get_wgpu_device());
        for op in WgpuContext::ZIP_OPS {
            if op == "pow" {
                continue; // see separate test
            }
            let result = t1.zip(&t2, op).ravel();
            assert_vec_eq(
                &result,
                &i1.iter()
                    .zip(i2.iter())
                    .map(|(a, b)| apply_binary_op(op, *a, *b))
                    .collect::<Vec<f32>>(),
            );
        }
    }

    fn apply_binary_op_i32(op: &str, a: i32, b: i32) -> i32 {
        match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" => a / b,
            "eq" => i32::from(a == b),
            _ => panic!("unknown op: {op}"),
        }
    }

    #[test]
    fn test_binary_ops_i32() {
        let i1 = vec![1, 2, 3, -1, -2, -3];
        let t1 = WgpuRawTensor::new(&[3, 2], &i1, get_wgpu_device());
        let i2 = vec![6, 7, 8, -6, -7, -8];
        let t2 = WgpuRawTensor::new(&[3, 2], &i2, get_wgpu_device());
        for op in WgpuContext::ZIP_OPS {
            if op == "pow" {
                continue; // not for i32
            }
            let result = t1.zip(&t2, op).ravel();
            assert_vec_eq_num(
                &result,
                &i1.iter()
                    .zip(i2.iter())
                    .map(|(a, b)| apply_binary_op_i32(op, *a, *b))
                    .collect::<Vec<i32>>(),
            );
        }
    }

    #[test]
    fn test_eq_bool() {
        let i1 = vec![true, true, false, false, true, false];
        let t1 = WgpuRawTensor::new(&[3, 2], &i1, get_wgpu_device());
        let i2 = vec![true, false, false, true, true, false];
        let t2 = WgpuRawTensor::new(&[3, 2], &i2, get_wgpu_device());

        let result = I::eq(&t1, &t2).ravel();
        assert_vec_eq_num(
            &result,
            &i1.iter()
                .zip(i2.iter())
                .map(|(a, b)| a == b)
                .collect::<Vec<bool>>(),
        );
    }

    #[test]
    fn test_cast() {
        let b = WgpuRawTensor::new(
            &[3, 2],
            &[true, true, false, false, true, false],
            get_wgpu_device(),
        );
        let i = b.cast::<i32>();
        let f = b.cast::<f32>();
        assert_eq!(i.ravel(), vec![1, 1, 0, 0, 1, 0]);
        assert_eq!(f.ravel(), vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_pow() {
        let i1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 8.0];
        let t1 = WgpuRawTensor::new(&[3, 2], &i1, get_wgpu_device());
        let i2 = vec![6.0, 7.0, 8.0, -2.0, -5.0, -10.0];
        let t2 = WgpuRawTensor::new(&[3, 2], &i2, get_wgpu_device());
        let op = "pow";
        let result = t1.zip(&t2, op).ravel();
        assert_vec_eq(
            &result,
            &i1.iter()
                .zip(i2.iter())
                .map(|(a, b)| apply_binary_op(op, *a, *b))
                .collect::<Vec<f32>>(),
        );
    }

    #[test]
    fn test_non_contiguous_binary_ops() {
        let t1 = WgpuRawTensor::new(
            &[3, 2],
            &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
            get_wgpu_device(),
        );
        let t1 = I::permute(&t1, &[1, 0]);
        let t2 = WgpuRawTensor::new(
            &[3, 2],
            &[6.0, 7.0, 8.0, -6.0, -7.0, -8.0],
            get_wgpu_device(),
        );
        let t2 = I::reshape(&t2, &[2, 3]);
        for op in WgpuContext::ZIP_OPS {
            if op == "pow" {
                continue; // see separate test
            }
            let actual = t1.zip(&t2, op).ravel();
            let expected = t1
                .ravel()
                .into_iter()
                .zip(t2.ravel().into_iter())
                .map(|(a, b)| apply_binary_op(op, a, b))
                .collect::<Vec<f32>>();
            assert_vec_eq(&actual, &expected);
        }
    }

    #[test]
    fn test_non_contiguous_pow() {
        let t1 = WgpuRawTensor::new(&[3, 2], &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0], get_wgpu_device());
        let t1 = I::permute(&t1, &[1, 0]);
        let t2 = WgpuRawTensor::new(
            &[3, 2],
            &[6.0, 7.0, 8.0, -6.0, -7.0, -8.0],
            get_wgpu_device(),
        );
        let t2 = I::reshape(&t2, &[2, 3]);

        let op = "pow";
        let actual = t1.zip(&t2, op).ravel();
        let expected = t1
            .ravel()
            .into_iter()
            .zip(t2.ravel())
            .map(|(a, b)| apply_binary_op(op, a, b))
            .collect::<Vec<f32>>();
        assert_vec_eq(&actual, &expected);
    }

    fn make_vec(len: u16) -> Vec<f32> {
        (0..len).map(f32::from).collect()
    }

    #[test]
    fn test_ravel() {
        let input = make_vec(24);
        let t = WgpuRawTensor::new(&[2, 3, 4], &input, get_wgpu_device());
        assert_eq!(t.ravel(), input);
    }

    fn test_reshape_24(orig_shape: &[usize], new_shape: &[usize], expected_strides: &[usize]) {
        let input = make_vec(24);
        let t = WgpuRawTensor::new(orig_shape, &input, get_wgpu_device());
        let t = I::reshape(&t, new_shape);
        assert_eq!(I::shape(&t), new_shape);
        assert_eq!(assert_pos_strides(&t), expected_strides);
        assert_eq!(t.ravel(), make_vec(24));
    }

    #[test]
    fn test_reshape() {
        test_reshape_24(&[24], &[3, 2, 4], &[8, 4, 1]);
        test_reshape_24(&[2, 1, 3, 1, 4], &[2, 3, 4], &[12, 4, 1]);
        test_reshape_24(&[2, 1, 3, 1, 4], &[2, 3, 4, 1], &[12, 4, 1, 1]);
    }

    fn test_permute_24(orig_shape: &[usize], permutation: &[usize], expected_shape: &[usize]) {
        let input = make_vec(24);
        let t = WgpuRawTensor::new(orig_shape, &input, get_wgpu_device());
        let tp = I::permute(&t, permutation);
        assert_eq!(I::shape(&tp), expected_shape);
        assert_ne!(tp.strides(), t.strides());

        let rev_perm = (0..permutation.len())
            .map(|i| permutation.iter().position(|&x| x == i).unwrap())
            .collect::<Vec<_>>();
        let tpp = I::permute(&tp, &rev_perm);
        assert_eq!(I::shape(&tpp), orig_shape);
        assert_eq!(tpp.strides(), t.strides());
        assert_eq!(&tpp.ravel(), &t.ravel());
    }

    #[test]
    fn test_permute() {
        test_permute_24(&[6, 4], &[1, 0], &[4, 6]);
        test_permute_24(&[2, 3, 4], &[2, 0, 1], &[4, 2, 3]);
    }

    #[test]
    fn test_expand() {
        let t = WgpuRawTensor::new(&[1, 1], &[42.0], get_wgpu_device());
        let t = I::expand(&t, &[5, 4]);

        assert_eq!(I::shape(&t), &[5, 4]);
        assert_eq!(&t.strides(), &[Stride::ZERO, Stride::ZERO]);
        assert_eq!(t.ravel(), repeat(42.0).take(20).collect::<Vec<_>>());
    }

    #[test]
    fn test_reduce_ops() {
        let t = WgpuRawTensor::new(&[2, 3], &make_vec(6), get_wgpu_device());
        let s = I::sum(&t, &[0]);
        assert_eq!(I::shape(&s), &[1, 3]);
        assert_eq!(s.ravel(), vec![3.0, 5.0, 7.0]);
        let s = I::sum(&t, &[1]);
        assert_eq!(I::shape(&s), &[2, 1]);
        assert_eq!(s.ravel(), vec![3.0, 12.0]);
        let s = I::sum(&t, &[0, 1]);
        assert_eq!(I::shape(&s), &[1, 1]);
        assert_eq!(s.ravel(), vec![15.0]);

        let s = I::max(&t, &[0]);
        assert_eq!(I::shape(&s), &[1, 3]);
        assert_eq!(s.ravel(), vec![3.0, 4.0, 5.0]);
        let s = I::max(&t, &[1]);
        assert_eq!(I::shape(&s), &[2, 1]);
        assert_eq!(s.ravel(), vec![2.0, 5.0]);
        let s = I::max(&t, &[0, 1]);
        assert_eq!(I::shape(&s), &[1, 1]);
        assert_eq!(s.ravel(), vec![5.0]);

        let t = WgpuRawTensor::new(&[1, 1, 1], &[1.0], get_wgpu_device());
        let t = I::expand(&t, &[4, 2, 4]);
        assert_eq!(t.ravel(), vec![1.0; 32]);
        let s = I::sum(&t, &[0]);
        assert_eq!(I::shape(&s), &[1, 2, 4]);
        assert_eq!(s.ravel(), vec![4.0; 8]);
    }

    #[test]
    fn test_reduce_ops_non_contiguous() {
        let t = WgpuRawTensor::new(&[2, 3], &make_vec(6), get_wgpu_device());
        let t = I::permute(&t, &[1, 0]);
        let s = I::sum(&t, &[0]);
        assert_eq!(I::shape(&s), &[1, 2]);
        assert_eq!(s.ravel(), vec![3.0, 12.0]);
        let s = I::sum(&t, &[1]);
        assert_eq!(I::shape(&s), &[3, 1]);
        assert_eq!(s.ravel(), vec![3.0, 5.0, 7.0]);
        let s = I::sum(&t, &[0, 1]);
        assert_eq!(I::shape(&s), &[1, 1]);
        assert_eq!(s.ravel(), vec![15.0]);
    }

    #[test]
    fn test_reduce_ops_parallel() {
        let t = WgpuRawTensor::new(&[2, 128], &make_vec(256), get_wgpu_device());
        let s = I::sum(&t, &[0]);
        assert_eq!(I::shape(&s), &[1, 128]);
        let expected: Vec<_> = (128i16..128 + 256).step_by(2).map(f32::from).collect();
        assert_eq!(s.ravel(), expected);

        let s = I::sum(&t, &[1]);
        assert_eq!(I::shape(&s), &[2, 1]);
        assert_eq!(s.ravel(), vec![8128.0, 24512.0]);

        let s = I::sum(&t, &[0, 1]);
        assert_eq!(I::shape(&s), &[1, 1]);
        assert_eq!(s.ravel(), vec![8128.0 + 24512.0]);

        // No powers of two
        let t = WgpuRawTensor::new(&[60, 60], &make_vec(3600), get_wgpu_device());
        let s = I::max(&t, &[0]);
        assert_eq!(I::shape(&s), &[1, 60]);
        let expected: Vec<_> = (3540i16..3600).map(f32::from).collect();
        assert_eq!(s.ravel(), expected);

        let s = I::max(&t, &[1]);
        assert_eq!(I::shape(&s), &[60, 1]);
        let expected: Vec<_> = (59i16..3600).step_by(60).map(f32::from).collect();
        assert_eq!(s.ravel(), expected);

        let s = I::sum(&t, &[0, 1]);
        assert_eq!(I::shape(&s), &[1, 1]);
        assert_eq!(s.ravel(), vec![6_478_200.0]);
    }

    #[test]
    fn test_reduce_ops_non_contiguous_parallel() {
        let t = WgpuRawTensor::new(&[2, 128], &make_vec(256), get_wgpu_device());
        let t = I::permute(&t, &[1, 0]);
        let s = I::sum(&t, &[0]);
        assert_eq!(I::shape(&s), &[1, 2]);
        assert_eq!(s.ravel(), vec![8128.0, 24512.0]);

        let s = I::sum(&t, &[1]);
        assert_eq!(I::shape(&s), &[128, 1]);

        let expected: Vec<_> = (128i16..128 + 256).step_by(2).map(f32::from).collect();
        assert_eq!(s.ravel(), expected);

        let s = I::sum(&t, &[0, 1]);
        assert_eq!(I::shape(&s), &[1, 1]);
        assert_eq!(s.ravel(), vec![8128.0 + 24512.0]);
    }

    #[test]
    fn test_crop() {
        let orig_shape = &[2, 3, 4];
        let t = WgpuRawTensor::new(orig_shape, &make_vec(24), get_wgpu_device());

        // crop single dimension
        let s = I::crop(&t, &[(0, 1), (0, 3), (0, 4)]);
        assert_eq!(s.ravel(), make_vec(12));

        let s = I::crop(&t, &[(1, 2), (0, 3), (0, 4)]);
        assert_eq!(
            s.ravel(),
            make_vec(12).iter().map(|x| x + 12.0).collect::<Vec<_>>()
        );

        // crop nothing
        let s = I::crop(&t, &[(0, 2), (0, 3), (0, 4)]);
        assert_eq!(I::shape(&s), orig_shape);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.ravel(), t.ravel());

        let s = I::crop(&t, &[(0, 2), (0, 3), (1, 3)]);
        assert_eq!(I::shape(&s), &[2, 3, 2]);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(
            s.ravel(),
            vec![1.0, 2.0, 5., 6., 9., 10., 13., 14., 17., 18., 21., 22.]
        );

        // keep cropping
        let s = I::crop(&s, &[(0, 1), (1, 2), (0, 2)]);
        assert_eq!(I::shape(&s), &[1, 1, 2]);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.ravel(), vec![5.0, 6.0]);

        // crop to single element
        let s = I::crop(&s, &[(0, 1), (0, 1), (1, 2)]);
        assert_eq!(I::shape(&s), &[1, 1, 1]);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.ravel(), vec![6.0]);
    }

    #[test]
    fn test_pad() {
        let orig_shape = &[2, 3, 4];
        let t = WgpuRawTensor::new(orig_shape, &(0..24).collect::<Vec<_>>(), get_wgpu_device());

        // pad nothing
        let s = t.pad(&[(0, 0), (0, 0), (0, 0)]);
        assert_eq!(I::shape(&s), orig_shape);
        assert_eq!(s.strides(), t.strides());
        assert_eq!(s.ravel(), t.ravel());

        // pad a little
        let padding = &[(1, 1), (1, 1), (1, 1)];
        let s = t.pad(padding);
        assert_eq!(I::shape(&s), &[4, 5, 6]);
        assert_eq!(assert_pos_strides(&s), &[30, 6, 1]);
        let s_raveled = s.ravel();
        assert_eq!(s_raveled.len(), I::shape(&s).size());
        assert_eq!(s_raveled.iter().filter(|&&x| x != 0).count(), 23);
        assert_eq!(s_raveled[s.strider.buffer_index(&[1, 1, 2])], 1);
        assert_eq!(s_raveled[s.strider.buffer_index(&[1, 1, 3])], 2);

        // pad a lot in one dimension
        let padding = &[(20, 0), (0, 0), (0, 0)];
        let s = t.pad(padding);
        assert_eq!(I::shape(&s), &[22, 3, 4]);
        assert_eq!(assert_pos_strides(&s), &[12, 4, 1]);
        let s_raveled = s.ravel();
        assert_eq!(s_raveled.iter().filter(|&&x| x != 0).count(), 23);
        assert_eq!(s_raveled[s.strider.buffer_index(&[20, 0, 1])], 1);
        assert_eq!(s_raveled[s.strider.buffer_index(&[20, 0, 2])], 2);

        // pad a lot
        let padding = &[(1, 2), (3, 4), (5, 6)];
        let s = t.pad(padding);
        assert_eq!(I::shape(&s), &[5, 10, 15]);
        assert_eq!(assert_pos_strides(&s), &[150, 15, 1]);
        let s_raveled = s.ravel();
        assert_eq!(s_raveled.iter().filter(|&&x| x != 0).count(), 23);
        assert_eq!(s_raveled[s.strider.buffer_index(&[1, 3, 6])], 1);
        assert_eq!(s_raveled[s.strider.buffer_index(&[1, 3, 7])], 2);
    }

    #[test]
    fn test_pad_reshape_expand_crop() {
        let dim = 2;
        let t = WgpuRawTensor::new(&[1], &[1.0], get_wgpu_device());
        let t = I::pad(&t, &[(0, dim)]);
        let t = I::reshape(&t, &[1, dim + 1]);
        let t = I::expand(&t, &[dim, dim + 1]);
        let t = I::reshape(&t, &[dim * (dim + 1)]);
        let t = I::crop(&t, &[(0, dim * dim)]);
        assert_eq!(I::shape(&t), &[4]);
        assert_eq!(t.ravel(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_fused_multiply_add() {
        // contiguous
        let t1 = WgpuRawTensor::new(&[2, 3], &make_vec(6), get_wgpu_device());
        let t2 = I::add(
            &t1,
            &WgpuRawTensor::new(&[2, 3], &make_vec(6), get_wgpu_device()),
        );

        let r = I::fused_multiply_add(&t1, &t2, &[0]);
        assert_eq!(I::shape(&r), &[1, 3]);
        assert_eq!(r.ravel(), vec![18.0, 34.0, 58.0]);

        let r = I::fused_multiply_add(&t1, &t2, &[1]);
        assert_eq!(I::shape(&r), &[2, 1]);
        assert_eq!(r.ravel(), vec![10.0, 100.0]);

        let r = I::fused_multiply_add(&t1, &t2, &[0, 1]);
        assert_eq!(I::shape(&r), &[1, 1]);
        assert_eq!(r.ravel(), vec![110.0]);

        // different strides
        let t1 = WgpuRawTensor::new(&[1, 1], &[8.0], get_wgpu_device());
        let t1 = I::expand(&t1, &[2, 3]);
        let t2 = WgpuRawTensor::new(&[2, 3], &make_vec(6), get_wgpu_device());

        let r = I::fused_multiply_add(&t1, &t2, &[0]);
        assert_eq!(I::shape(&r), &[1, 3]);
        assert_eq!(r.ravel(), vec![24.0, 40.0, 56.0]);

        let r = I::fused_multiply_add(&t1, &t2, &[1]);
        assert_eq!(I::shape(&r), &[2, 1]);
        assert_eq!(r.ravel(), vec![24.0, 96.0]);

        let r = I::fused_multiply_add(&t1, &t2, &[0, 1]);
        assert_eq!(I::shape(&r), &[1, 1]);
        assert_eq!(r.ravel(), vec![120.0]);

        // non_contiguous
        let t1 = WgpuRawTensor::new(&[2, 3], &make_vec(6), get_wgpu_device());
        let t1 = I::permute(&t1, &[1, 0]);
        let t2 = WgpuRawTensor::new(&[2, 3], &make_vec(6), get_wgpu_device());
        let t2 = I::permute(&t2, &[1, 0]);
        let t2 = I::add(&t1, &t2);

        let r = I::fused_multiply_add(&t1, &t2, &[0]);
        assert_eq!(I::shape(&r), &[1, 2]);
        assert_eq!(r.ravel(), vec![10.0, 100.0]);

        let r = I::fused_multiply_add(&t1, &t2, &[1]);
        assert_eq!(I::shape(&r), &[3, 1]);
        assert_eq!(r.ravel(), vec![18.0, 34.0, 58.0]);

        let r = I::fused_multiply_add(&t1, &t2, &[0, 1]);
        assert_eq!(I::shape(&r), &[1, 1]);
        assert_eq!(r.ravel(), vec![110.0]);
    }
}
