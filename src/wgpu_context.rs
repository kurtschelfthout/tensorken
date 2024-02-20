use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{Arc, Once, RwLock},
};

/// Size of a workgroup (number of threads) in X and Y dimensions. Z dimension is always 1 at the moment, it's not included.
pub(crate) type WorkgroupSize = (usize, usize);

/// A singleton GPU context for the process. Holds a `wgpu::Device` and a `wgpu::Queue`, and a cache of compute
/// pipelines. The latter to avoid recompiling WGSL shaders.
/// Does double duty as the pipeline/shader builder, using straightforward string replacements.
pub(crate) struct WgpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pipelines: RwLock<HashMap<(&'static str, WorkgroupSize), Arc<wgpu::ComputePipeline>>>,
}

impl WgpuContext {
    // shader files
    const MAP_SHADER: &'static str = include_str!("shaders/map.wgsl");
    const ZIP_SHADER: &'static str = include_str!("shaders/zip.wgsl");
    const REDUCE_SHADER: &'static str = include_str!("shaders/reduce.wgsl");
    const PAD_SHADER: &'static str = include_str!("shaders/pad.wgsl");
    const FUSED_MUL_ADD_SHADER: &'static str = include_str!("shaders/fused_mul_add.wgsl");

    // low tech templating: these strings are replaced in shaders
    const REPLACE_OP_NAME: &'static str = "replace_me_with_actual_operation";
    const REPLACE_DEFAULT_NAME: &'static str = "replace_me_with_actual_default()";
    const REPLACE_UNARY_OP_DEF: &'static str =
        r"fn replace_me_with_actual_operation(in: f32) -> f32 { discard; }";
    const REPLACE_BINARY_OP_DEF: &'static str =
        r"fn replace_me_with_actual_operation(in_1: f32, in_2: f32) -> f32 { discard; }";
    const REPLACE_REDUCE_DEFAULT_DEF: &'static str =
        r"fn replace_me_with_actual_default() -> f32 { discard; }";
    const REPLACE_REDUCE_THREAD_COUNT_CONST: &'static str = r"const REDUCE_THREADS: u32 = 64u;";
    const REPLACE_REDUCE_INTERMEDIATE_BUFFER_SIZE_CONST: &'static str =
        r"const INTERMEDIATE_SIZE: u32 = 64u;";
    const REPLACE_WORKGROUP_SIZE: &'static str = "@workgroup_size(64)";

    // supported operations. crate visible for testing.
    pub(crate) const MAP_OPS: [&'static str; 3] = ["exp", "log", "id"];
    pub(crate) const ZIP_OPS: [&'static str; 6] = ["add", "sub", "mul", "div", "pow", "eq"];
    pub(crate) const RED_OPS: [&'static str; 2] = ["sum", "max"];
    pub(crate) const PAD_OP: &'static str = "pad";
    pub(crate) const FUSED_MUL_ADD_OP: &'static str = "fused_mul_add";

    pub(crate) fn new() -> Self {
        let (device, queue) = Self::get_device().unwrap();
        // device.start_capture(); // Uncomment to capture a trace for debugging
        Self {
            device,
            queue,
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    fn memo_compute_pipeline(
        &self,
        operation: &'static str,
        module: &wgpu::ShaderModule,
        pipelines: &mut std::sync::RwLockWriteGuard<
            HashMap<(&str, WorkgroupSize), Arc<wgpu::ComputePipeline>>,
        >,
        workgroup_size: WorkgroupSize,
    ) {
        const ENTRY_POINT: &str = "call";
        let compute_pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(operation),
                layout: None,
                module,
                entry_point: ENTRY_POINT,
            },
        ));
        pipelines.insert((operation, workgroup_size), compute_pipeline);
    }

    /// Create a shader module, replacing the `workgroup_size` placeholder with the actual workgroup size.
    fn create_shader_module(
        &self,
        operation: &str,
        shader_source: &str,
        workgroup_size: WorkgroupSize,
    ) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(operation),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_source.replace(
                    Self::REPLACE_WORKGROUP_SIZE,
                    &format!(
                        "@workgroup_size({}, {})",
                        workgroup_size.0, workgroup_size.1
                    ),
                ))),
            })
    }

    /// Retrieve or build and memoize a compute pipeline for the given operation and workgroup size.
    pub(crate) fn pipeline_for(
        &self,
        operation: &'static str,
        workgroup_size: WorkgroupSize,
    ) -> Arc<wgpu::ComputePipeline> {
        {
            let pipelines = self.pipelines.read().unwrap();
            if let Some(r) = pipelines.get(&(operation, workgroup_size)) {
                return Arc::clone(r);
            }
        }

        let mut pipelines = self.pipelines.write().unwrap();

        let module = if operation == Self::PAD_OP {
            self.create_shader_module(operation, Self::PAD_SHADER, workgroup_size)
        } else if operation == Self::FUSED_MUL_ADD_OP {
            self.create_shader_module(operation, Self::FUSED_MUL_ADD_SHADER, workgroup_size)
        } else if Self::MAP_OPS.contains(&operation) {
            self.create_shader_module(
                operation,
                &Self::MAP_SHADER
                    .replace(Self::REPLACE_UNARY_OP_DEF, "")
                    .replace(Self::REPLACE_OP_NAME, operation),
                workgroup_size,
            )
        } else if Self::ZIP_OPS.contains(&operation) {
            self.create_shader_module(
                operation,
                &Self::ZIP_SHADER
                    .replace(Self::REPLACE_BINARY_OP_DEF, "")
                    .replace(Self::REPLACE_OP_NAME, operation),
                workgroup_size,
            )
        } else if Self::RED_OPS.contains(&operation) {
            self.create_shader_module(
                operation,
                &Self::REDUCE_SHADER
                    .replace(Self::REPLACE_BINARY_OP_DEF, "")
                    .replace(Self::REPLACE_OP_NAME, operation)
                    .replace(Self::REPLACE_REDUCE_DEFAULT_DEF, "")
                    .replace(Self::REPLACE_DEFAULT_NAME, &operation.to_uppercase())
                    .replace(
                        Self::REPLACE_REDUCE_THREAD_COUNT_CONST,
                        &format!("const REDUCE_THREADS: u32 = {}u;", workgroup_size.1),
                    )
                    .replace(
                        Self::REPLACE_REDUCE_INTERMEDIATE_BUFFER_SIZE_CONST,
                        &format!(
                            "const INTERMEDIATE_SIZE: u32 = {}u;",
                            workgroup_size.0 * workgroup_size.1
                        ),
                    ),
                workgroup_size,
            )
        } else {
            panic!("Unsupported operation: {operation}");
        };

        self.memo_compute_pipeline(operation, &module, &mut pipelines, workgroup_size);
        Arc::clone(pipelines.get(&(operation, workgroup_size)).unwrap())
    }

    /// Get the wgpu device and queue.
    async fn get_device_async() -> Option<(wgpu::Device, wgpu::Queue)> {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // let backends = wgpu::Backends::all;
        // println!("backends: {backends:?}");

        // let adapters = instance.enumerate_adapters(backends);
        // for adapter in adapters {
        //     let info = adapter.get_info();
        //     println!("adapter: {:?}", info);
        // }

        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .expect("No suitable GPU adapters found on the system!");
        let info = adapter.get_info();
        println!(
            "Using {:#?} {} with {:#?} backend (Env vars: WGPU_POWER_PREF (low|high), WGPU_ADAPTER_NAME, WGPU_BACKEND).",
            info.device_type, info.name, info.backend
        );

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let r = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None, // Some(Path::new("./trace")),
            )
            .await
            .unwrap();
        Some(r)
    }

    /// Get the wgpu device and queue, synchronously.
    fn get_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(Self::get_device_async())
    }
}

static mut WGPU_DEVICE: Option<WgpuContext> = None;
static INIT_WGPU_DEVICE: Once = Once::new();

/// Returns a reference to the global wgpu context, creating it if necessary.
pub(crate) fn get_wgpu_device() -> &'static WgpuContext {
    unsafe {
        INIT_WGPU_DEVICE.call_once(|| {
            WGPU_DEVICE = Some(WgpuContext::new());
        });
        return WGPU_DEVICE.as_ref().unwrap();
    }
}
