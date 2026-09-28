use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use half::f16;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgba};
use rayon::prelude::*;
use std::num::NonZero;

#[cfg(not(target_os = "android"))]
use tauri::Manager;
use wgpu::util::{DeviceExt, TextureDataOrder};

use crate::camera_profile::CameraProfile;
use crate::image_processing::{AllAdjustments, GpuContext, MAX_MASKS};
use crate::lut_processing::Lut;
use crate::{AppState, GpuImageCache};

#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderOutputPrecision {
    #[default]
    EightBit,
    SixteenBit,
}

pub struct RenderRequest<'a> {
    pub adjustments: AllAdjustments,
    pub mask_bitmaps: &'a [ImageBuffer<Luma<u8>, Vec<u8>>],
    pub lut: Option<Arc<Lut>>,
    pub roi: Option<Roi>,
    pub camera_profile: Option<Arc<CameraProfile>>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DisplayTransform {
    pub rect: [f32; 4],
    pub clip: [f32; 4],
    pub window: [f32; 2],
    pub pixelated: f32,
    pub _pad: f32,
    pub bg_primary: [f32; 4],
    pub bg_secondary: [f32; 4],
}

pub struct WgpuDisplay {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub transform_buffer: wgpu::Buffer,
    pub latest_transform: DisplayTransform,
    pub device_pixel_ratio: f32,
    pub current_bind_group: Option<wgpu::BindGroup>,
}

impl WgpuDisplay {
    #[cfg(target_os = "linux")]
    fn fit_to_window(&mut self, device: &wgpu::Device) {
        let scale = self.device_pixel_ratio.round().max(1.0) as u32;
        // Wayland needs the buffer size to be a multiple of the scale.
        let fit = |px: f32| ((px.round() as u32) / scale * scale).max(scale);
        let (width, height) = (
            fit(self.latest_transform.window[0]),
            fit(self.latest_transform.window[1]),
        );
        crate::wayland_subsurface::set_buffer_scale(scale as i32);
        crate::wayland_subsurface::sync_position();
        if (width, height) != (self.config.width, self.config.height) {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(device, &self.config);
        }
    }

    /// Shows `view`, the whole photo with its mips, from the next `render` on.
    pub fn show(&mut self, device: &wgpu::Device, view: &wgpu::TextureView) {
        self.current_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Display Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
    }

    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        #[cfg(target_os = "linux")]
        self.fit_to_window(device);
        queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::bytes_of(&self.latest_transform),
        );
        if let Some(bind_group) = &self.current_bind_group {
            let output = match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(tex)
                | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(device, &self.config);
                    match self.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(tex)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
                        _ => panic!("Failed to acquire surface texture"),
                    }
                }
                _ => return,
            };
            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: self.latest_transform.bg_primary[0] as f64,
                                g: self.latest_transform.bg_primary[1] as f64,
                                b: self.latest_transform.bg_primary[2] as f64,
                                a: self.latest_transform.bg_primary[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: NonZero::new(0),
                });
                let clip_x1 = self.latest_transform.clip[0].max(0.0);
                let clip_y1 = self.latest_transform.clip[1].max(0.0);
                let clip_x2 =
                    (self.latest_transform.clip[0] + self.latest_transform.clip[2]).max(0.0);
                let clip_y2 =
                    (self.latest_transform.clip[1] + self.latest_transform.clip[3]).max(0.0);

                let final_clip_x = clip_x1.floor() as u32;
                let final_clip_y = clip_y1.floor() as u32;
                let final_clip_w = (clip_x2.ceil() as u32).saturating_sub(final_clip_x);
                let final_clip_h = (clip_y2.ceil() as u32).saturating_sub(final_clip_y);

                let max_x = self.config.width;
                let max_y = self.config.height;

                if final_clip_x < max_x && final_clip_y < max_y {
                    let clamped_width = final_clip_w.min(max_x - final_clip_x);
                    let clamped_height = final_clip_h.min(max_y - final_clip_y);

                    if clamped_width > 0 && clamped_height > 0 {
                        rpass.set_scissor_rect(
                            final_clip_x,
                            final_clip_y,
                            clamped_width,
                            clamped_height,
                        );

                        rpass.set_pipeline(&self.pipeline);
                        rpass.set_bind_group(0, bind_group, &[]);
                        rpass.draw(0..4, 0..1);
                    }
                }
            }
            queue.submit(Some(encoder.finish()));
            output.present();
        }
    }
}

pub fn get_or_init_gpu_context(
    state: &tauri::State<AppState>,
    _app_handle: &tauri::AppHandle,
) -> Result<GpuContext, String> {
    #[cfg(not(target_os = "android"))]
    let app_handle = _app_handle;

    let mut context_lock = match state.gpu_context.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            log::warn!(
                "GPU context lock was poisoned (previous crash). Wiping state to self-heal."
            );
            let mut guard = poisoned.into_inner();
            *guard = None;
            guard
        }
    };
    if let Some(context) = &*context_lock {
        return Ok(context.clone());
    }

    #[allow(unused_mut)]
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();

    #[cfg(target_os = "windows")]
    if std::env::var("WGPU_BACKEND").is_err() {
        instance_desc.backends = wgpu::Backends::PRIMARY;
    }

    let flag_path = state.gpu_crash_flag_path.lock().unwrap().clone();
    if let Some(p) = &flag_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(p, "initializing_gpu");
    }

    let instance = wgpu::Instance::new(instance_desc);

    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    let surface_opt = {
        let settings = crate::app_settings::load_settings(app_handle.clone()).unwrap_or_default();
        let use_wgpu_renderer = settings.use_wgpu_renderer.unwrap_or(true);

        if use_wgpu_renderer {
            if let Some(window) = app_handle.get_webview_window("main") {
                match instance.create_surface(window) {
                    Ok(surface) => Some(surface),
                    Err(e) => {
                        log::warn!(
                            "Failed to create surface, falling back to compute-only: {}",
                            e
                        );
                        if let Some(p) = &flag_path {
                            let _ = std::fs::remove_file(p);
                        }
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    #[cfg(target_os = "linux")]
    let surface_opt = {
        let settings = crate::app_settings::load_settings(app_handle.clone()).unwrap_or_default();
        if settings.use_wgpu_renderer.unwrap_or(false) {
            app_handle
                .get_webview_window("main")
                .and_then(|window| crate::wayland_subsurface::create_surface(&window, &instance))
        } else {
            None
        }
    };

    #[cfg(target_os = "android")]
    let surface_opt: Option<wgpu::Surface> = None;

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: surface_opt.as_ref(),
        ..Default::default()
    }))
    .map_err(|e| {
        if let Some(p) = &flag_path {
            let _ = std::fs::remove_file(p);
        }
        format!("Failed to find a wgpu adapter: {}", e)
    })?;

    let mut required_features = wgpu::Features::empty();
    if adapter
        .features()
        .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
    {
        required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    }

    let limits = adapter.limits();

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Processing Device"),
        required_features,
        required_limits: limits.clone(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| {
        if let Some(p) = &flag_path {
            let _ = std::fs::remove_file(p);
        }
        e.to_string()
    })?;

    device.on_uncaptured_error(Arc::new(|err: wgpu::Error| {
        log::error!("[wgpu-error] {}", err);
    }));

    if let Some(p) = &flag_path {
        let _ = std::fs::remove_file(p);
    }

    #[cfg(not(target_os = "android"))]
    let display_opt = if let Some(surface) = surface_opt {
        let window = app_handle
            .get_webview_window("main")
            .ok_or("Failed to get main window")?;

        let swapchain_caps = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                // Vulkan on Wayland may list 16-bit formats first, which need a device feature.
                matches!(
                    f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .or_else(|| {
                swapchain_caps
                    .formats
                    .iter()
                    .copied()
                    .find(|f| !f.is_srgb())
            })
            .unwrap_or(swapchain_caps.formats[0]);
        log::info!("Display surface format: {:?}", swapchain_format);

        // Keep Linux off Opaque: Hyprland <= 0.56 then draws the page over the subsurface
        // unblended (hyprwm/Hyprland#15958).
        let alpha_mode = if cfg!(target_os = "windows")
            && swapchain_caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::Opaque)
        {
            wgpu::CompositeAlphaMode::Opaque
        } else if swapchain_caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else if swapchain_caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
        {
            wgpu::CompositeAlphaMode::PostMultiplied
        } else {
            swapchain_caps.alpha_modes[0]
        };

        let size = window
            .inner_size()
            .unwrap_or(tauri::PhysicalSize::new(1280, 720));
        let config = wgpu::SurfaceConfiguration {
            width: size.width.max(1),
            height: size.height.max(1),
            format: swapchain_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            // Mesa's Wayland FIFO blocks without frame callbacks (hidden window).
            present_mode: if cfg!(target_os = "linux")
                && swapchain_caps
                    .present_modes
                    .contains(&wgpu::PresentMode::Mailbox)
            {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            },
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Display BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Display Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Display Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: NonZero::new(0),
            cache: None,
        });

        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Buffer"),
            size: std::mem::size_of::<DisplayTransform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Display Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Some(WgpuDisplay {
            surface,
            config,
            pipeline,
            bind_group_layout,
            transform_buffer,
            latest_transform: DisplayTransform {
                rect: [0.0, 0.0, 100.0, 100.0],
                clip: [0.0, 0.0, 10000.0, 10000.0],
                window: [size.width as f32, size.height as f32],
                pixelated: 0.0,
                _pad: 0.0,
                bg_primary: [24.0 / 255.0, 24.0 / 255.0, 24.0 / 255.0, 1.0],
                bg_secondary: [35.0 / 255.0, 35.0 / 255.0, 35.0 / 255.0, 1.0],
            },
            sampler,
            device_pixel_ratio: window.scale_factor().unwrap_or(1.0) as f32,
            current_bind_group: None,
        })
    } else {
        None
    };

    #[cfg(target_os = "android")]
    let display_opt = None;

    let new_context = GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        limits,
        display: Arc::new(std::sync::Mutex::new(display_opt)),
    };
    *context_lock = Some(new_context.clone());
    Ok(new_context)
}

fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    size: wgpu::Extent3d,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, String> {
    let unpadded_bytes_per_row = bytes_per_pixel * size.width;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);
    let output_buffer_size = (padded_bytes_per_row * size.height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );

    queue.submit(Some(encoder.finish()));
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        })
        .map_err(|e| format!("Failed while polling mapped GPU buffer: {}", e))?;
    let map_result = rx
        .recv()
        .map_err(|e| format!("Failed receiving GPU map result: {}", e))?;
    map_result.map_err(|e| e.to_string())?;

    let padded_data = buffer_slice.get_mapped_range().to_vec();
    output_buffer.unmap();

    if padded_bytes_per_row == unpadded_bytes_per_row {
        Ok(padded_data)
    } else {
        let mut unpadded_data = Vec::with_capacity((unpadded_bytes_per_row * size.height) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        Ok(unpadded_data)
    }
}

/// Waits until the GPU has finished `submission` (all work if `None`).
fn wait_for_gpu(
    device: &wgpu::Device,
    submission: Option<wgpu::SubmissionIndex>,
) -> Result<(), String> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: submission,
            timeout: Some(Duration::from_secs(10)),
        })
        .map(|_| ())
        .map_err(|e| format!("Failed waiting for the GPU: {}", e))
}

/// RGBA f16 texels of `img`, converted on all cores.
fn to_rgba_f16(img: &DynamicImage) -> Vec<[f16; 4]> {
    fn convert<const CHANNELS: usize>(samples: &[f32]) -> Vec<[f16; 4]> {
        samples
            .par_chunks_exact(CHANNELS)
            .map(|pixel| {
                let mut texel = [f16::ONE; 4];
                for (channel, sample) in texel.iter_mut().zip(pixel) {
                    *channel = f16::from_f32(*sample);
                }
                texel
            })
            .collect()
    }
    match img {
        DynamicImage::ImageRgb32F(rgb) => convert::<3>(rgb.as_raw()),
        DynamicImage::ImageRgba32F(rgba) => convert::<4>(rgba.as_raw()),
        _ => convert::<4>(img.to_rgba32f().as_raw()),
    }
}

fn mip_view(texture: &wgpu::Texture, level: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        base_mip_level: level,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

/// Encodes one pass of an 8x8-workgroup compute shader over `size`, binding `resources`
/// in order to its group 0.
fn encode_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    resources: &[wgpu::BindingResource],
    size: wgpu::Extent3d,
) {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: resource.clone(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    radius: u32,
    _pad: [u32; 3],
}

/// Radii of the blurs the main pass samples (sharpness, tonal, clarity, structure), in
/// pixels of an image whose short side is 1080 px. They grow with the image.
const BLUR_RADII: [f32; 4] = [1.0, 3.5, 8.0, 40.0];

/// A photo on the GPU, with the blurs and guided-filter coefficients the main pass
/// samples. None of them depends on the sliders, so they are made once per image and
/// geometry and every render reuses them.
pub struct GpuImage {
    pub width: u32,
    pub height: u32,
    pub is_raw: u32,
    view: wgpu::TextureView,
    blurs: [wgpu::TextureView; 4],
    gf_coeffs: wgpu::TextureView,
    gf_dehaze: wgpu::TextureView,
}

const GF_SHORT_EDGE: u32 = 1080;
const GF_CLARITY_RADIUS: f32 = 8.0;
const GF_STRUCTURE_RADIUS: f32 = 40.0;
const GF_CLARITY_EPS: f32 = 0.25;
const GF_STRUCTURE_EPS: f32 = 0.5;
const GF_COARSE_DIV: u32 = 4;
const GF_TONAL_RADIUS: f32 = 10.0;
const GF_TONAL_EPS: f32 = 0.25;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GfParams {
    radius: u32,
    is_raw: u32,
    eps: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FlareParams {
    amount: f32,
    is_raw: u32,
    exposure: f32,
    brightness: f32,
    contrast: f32,
    whites: f32,
    aspect_ratio: f32,
    _pad: f32,
}

pub struct GpuProcessor {
    context: GpuContext,
    h_blur_pipeline: wgpu::ComputePipeline,
    v_blur_pipeline: wgpu::ComputePipeline,
    mipmap_pipeline: wgpu::ComputePipeline,

    flare_bgl_0: wgpu::BindGroupLayout,
    flare_bgl_1: wgpu::BindGroupLayout,
    flare_threshold_pipeline: wgpu::ComputePipeline,
    flare_ghosts_pipeline: wgpu::ComputePipeline,
    flare_params_buffer: wgpu::Buffer,
    flare_threshold_view: wgpu::TextureView,
    flare_ghosts_view: wgpu::TextureView,
    flare_final_view: wgpu::TextureView,
    linear_sampler: wgpu::Sampler,

    gf_bgl: wgpu::BindGroupLayout,
    gf_downsample_pipeline: wgpu::ComputePipeline,
    gf_box_down_pipeline: wgpu::ComputePipeline,
    gf_blur_h_pipeline: wgpu::ComputePipeline,
    gf_blur_v_pipeline: wgpu::ComputePipeline,
    gf_coeffs_pipeline: wgpu::ComputePipeline,
    gf_pack_pipeline: wgpu::ComputePipeline,

    main_bgl: wgpu::BindGroupLayout,
    main_pipeline: wgpu::ComputePipeline,
    high_precision_entries: Vec<wgpu::BindGroupLayoutEntry>,
    high_precision_pipeline: std::sync::OnceLock<HighPrecisionPipeline>,
    high_precision_tile: std::sync::OnceLock<Tile>,
    tile: Tile,
    adjustments_buffer: wgpu::Buffer,
    dummy_view: wgpu::TextureView,
    dummy_masks_view: wgpu::TextureView,
    dummy_lut_view: wgpu::TextureView,
    dummy_lut_sampler: wgpu::Sampler,
    dummy_camera_profile_buffer: wgpu::Buffer,
    /// Counts the shader makes for the Curves graph's histogram; each render starts it afresh.
    curve_histogram: wgpu::Buffer,
}

/// The main pass writing 16-bit floats, for exports. Built on first use.
struct HighPrecisionPipeline {
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// Output of one tile of a render that is read back to the CPU.
struct Tile {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Tile {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tile Output Texture"),
            size: wgpu::Extent3d {
                width: TILE_SIZE,
                height: TILE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }
}

enum RenderedPixels {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

pub(crate) fn main_shader_source() -> String {
    [
        include_str!("shaders/shader.wgsl"),
        include_str!("shaders/camera_profile.wgsl"),
    ]
    .join("\n")
}

pub(crate) fn high_precision_shader_source() -> String {
    main_shader_source().replace("rgba8unorm, write>", "rgba16float, write>")
}

const FLARE_MAP_SIZE: u32 = 512;
const MAX_MASK_BINDINGS: u32 = 1;
const CURVE_HISTOGRAM_BYTES: u64 = 4 * 256 * 4;
const TILE_SIZE: u32 = 2048;

impl GpuProcessor {
    pub fn new(context: GpuContext) -> Result<Self, String> {
        let device = &context.device;

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });
        let mipmap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mipmap Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mipmap.wgsl").into()),
        });
        let compute_pipeline = |module: &wgpu::ShaderModule, entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let h_blur_pipeline = compute_pipeline(&blur_shader, "horizontal_blur");
        let v_blur_pipeline = compute_pipeline(&blur_shader, "vertical_blur");
        let mipmap_pipeline = compute_pipeline(&mipmap_shader, "main");

        let flare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flare Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flare.wgsl").into()),
        });

        let flare_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let flare_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare BGL 1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let flare_threshold_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Flare Threshold Layout"),
                bind_group_layouts: &[Some(&flare_bgl_0)],
                immediate_size: 0,
            });

        let flare_ghosts_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flare Ghosts Layout"),
            bind_group_layouts: &[Some(&flare_bgl_0), Some(&flare_bgl_1)],
            immediate_size: 0,
        });

        let flare_threshold_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flare Threshold Pipeline"),
                layout: Some(&flare_threshold_layout),
                module: &flare_shader,
                entry_point: Some("threshold_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let flare_ghosts_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flare Ghosts Pipeline"),
                layout: Some(&flare_ghosts_layout),
                module: &flare_shader,
                entry_point: Some("ghosts_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let flare_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flare Params Buffer"),
            size: std::mem::size_of::<FlareParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flare_tex_desc = wgpu::TextureDescriptor {
            label: Some("Flare Tex"),
            size: wgpu::Extent3d {
                width: FLARE_MAP_SIZE,
                height: FLARE_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let flare_threshold_texture = device.create_texture(&flare_tex_desc);
        let flare_threshold_view = flare_threshold_texture.create_view(&Default::default());
        let flare_ghosts_texture = device.create_texture(&flare_tex_desc);
        let flare_ghosts_view = flare_ghosts_texture.create_view(&Default::default());
        let flare_final_texture = device.create_texture(&flare_tex_desc);
        let flare_final_view = flare_final_texture.create_view(&Default::default());

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let gf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Guided Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/guided.wgsl").into()),
        });

        let gf_texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let gf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Guided Filter BGL"),
            entries: &[
                gf_texture_entry(0),
                gf_texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let gf_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Guided Filter Pipeline Layout"),
            bind_group_layouts: &[Some(&gf_bgl)],
            immediate_size: 0,
        });

        let gf_pipeline = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&gf_layout),
                module: &gf_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let gf_downsample_pipeline = gf_pipeline("gf_downsample");
        let gf_box_down_pipeline = gf_pipeline("gf_box_down");
        let gf_blur_h_pipeline = gf_pipeline("gf_blur_h");
        let gf_blur_v_pipeline = gf_pipeline("gf_blur_v");
        let gf_coeffs_pipeline = gf_pipeline("gf_coeffs");
        let gf_pack_pipeline = gf_pipeline("gf_pack");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image Processing Shader"),
            source: wgpu::ShaderSource::Wgsl(main_shader_source().into()),
        });

        let mut bind_group_layout_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 10 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 13 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 14 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 11 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 12 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });

        let main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main BGL"),
            entries: &bind_group_layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&main_bgl)],
            immediate_size: 0,
        });

        let main_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        bind_group_layout_entries[1].ty = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let adjustments_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjustments Buffer"),
            size: std::mem::size_of::<AllAdjustments>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_texture_desc = wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let dummy_view = device
            .create_texture(&dummy_texture_desc)
            .create_view(&Default::default());

        let dummy_masks_view = device
            .create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 2,
                },
                format: wgpu::TextureFormat::R8Unorm,
                ..dummy_texture_desc
            })
            .create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        let dummy_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            dimension: wgpu::TextureDimension::D3,
            ..dummy_texture_desc
        });
        let dummy_lut_view = dummy_lut_texture.create_view(&Default::default());
        let dummy_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let dummy_camera_profile_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Disabled Camera Profile Buffer"),
                contents: &CameraProfile::disabled_gpu_bytes(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let curve_histogram = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Curve Histogram Buffer"),
            size: CURVE_HISTOGRAM_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            tile: Tile::new(device, wgpu::TextureFormat::Rgba8Unorm),
            context,
            h_blur_pipeline,
            v_blur_pipeline,
            mipmap_pipeline,
            flare_bgl_0,
            flare_bgl_1,
            flare_threshold_pipeline,
            flare_ghosts_pipeline,
            flare_params_buffer,
            flare_threshold_view,
            flare_ghosts_view,
            flare_final_view,
            linear_sampler,
            gf_bgl,
            gf_downsample_pipeline,
            gf_box_down_pipeline,
            gf_blur_h_pipeline,
            gf_blur_v_pipeline,
            gf_coeffs_pipeline,
            gf_pack_pipeline,
            main_bgl,
            main_pipeline,
            high_precision_entries: bind_group_layout_entries,
            high_precision_pipeline: std::sync::OnceLock::new(),
            high_precision_tile: std::sync::OnceLock::new(),
            adjustments_buffer,
            dummy_view,
            dummy_masks_view,
            dummy_lut_view,
            dummy_lut_sampler,
            dummy_camera_profile_buffer,
            curve_histogram,
        })
    }

    pub fn build_guided_coeffs(
        &self,
        input_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        is_raw: u32,
    ) -> (wgpu::TextureView, wgpu::TextureView) {
        let device = &self.context.device;
        let queue = &self.context.queue;

        let short = width.min(height).max(1);
        let k = (GF_SHORT_EDGE as f32 / short as f32).min(1.0);
        let lw = ((width as f32 * k).round() as u32).max(1);
        let lh = ((height as f32 * k).round() as u32).max(1);
        let low_short = lw.min(lh) as f32;
        let cw = lw.div_ceil(GF_COARSE_DIV).max(1);
        let ch = lh.div_ceil(GF_COARSE_DIV).max(1);

        let make_view = |label: &'static str, w: u32, h: u32| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };

        let make_params = |base_radius: f32, eps: f32, div: f32| {
            let radius = (base_radius * low_short / 1080.0 / div).ceil().max(1.0) as u32;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Guided Filter Params"),
                contents: bytemuck::bytes_of(&GfParams {
                    radius,
                    is_raw,
                    eps,
                    _pad: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };

        let moments = make_view("GF Moments", lw, lh);
        let tmp = make_view("GF Ping Pong", lw, lh);
        let blurred = make_view("GF Blurred", lw, lh);
        let ab = make_view("GF Coeffs", lw, lh);
        let ab_clarity = make_view("GF Coeffs Clarity", lw, lh);
        let c_moments = make_view("GF Coarse Moments", cw, ch);
        let c_tmp = make_view("GF Coarse Ping Pong", cw, ch);
        let c_blurred = make_view("GF Coarse Blurred", cw, ch);
        let c_ab = make_view("GF Coarse Coeffs", cw, ch);

        let params_clarity = make_params(GF_CLARITY_RADIUS, GF_CLARITY_EPS, 1.0);
        let params_tonal = make_params(GF_TONAL_RADIUS, GF_TONAL_EPS, 1.0);
        let params_coarse =
            make_params(GF_STRUCTURE_RADIUS, GF_STRUCTURE_EPS, GF_COARSE_DIV as f32);

        let mut encoder = device.create_command_encoder(&Default::default());

        let mut pass = |pipeline: &wgpu::ComputePipeline,
                        src_a: &wgpu::TextureView,
                        src_b: &wgpu::TextureView,
                        dst: &wgpu::TextureView,
                        params: &wgpu::Buffer,
                        (w, h): (u32, u32)| {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Guided Filter BG"),
                layout: &self.gf_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_a),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src_b),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        };
        let fine = (lw, lh);
        let coarse = (cw, ch);

        pass(
            &self.gf_downsample_pipeline,
            input_view,
            input_view,
            &moments,
            &params_clarity,
            fine,
        );

        pass(
            &self.gf_box_down_pipeline,
            &moments,
            &moments,
            &c_moments,
            &params_coarse,
            coarse,
        );
        pass(
            &self.gf_blur_h_pipeline,
            &c_moments,
            &c_moments,
            &c_tmp,
            &params_coarse,
            coarse,
        );
        pass(
            &self.gf_blur_v_pipeline,
            &c_tmp,
            &c_tmp,
            &c_blurred,
            &params_coarse,
            coarse,
        );
        pass(
            &self.gf_coeffs_pipeline,
            &c_blurred,
            &c_blurred,
            &c_ab,
            &params_coarse,
            coarse,
        );
        pass(
            &self.gf_blur_h_pipeline,
            &c_ab,
            &c_ab,
            &c_tmp,
            &params_coarse,
            coarse,
        );
        pass(
            &self.gf_blur_v_pipeline,
            &c_tmp,
            &c_tmp,
            &c_moments,
            &params_coarse,
            coarse,
        );

        for (params, out) in [(&params_clarity, &ab_clarity), (&params_tonal, &blurred)] {
            pass(
                &self.gf_blur_h_pipeline,
                &moments,
                &moments,
                &tmp,
                params,
                fine,
            );
            pass(&self.gf_blur_v_pipeline, &tmp, &tmp, &blurred, params, fine);
            pass(
                &self.gf_coeffs_pipeline,
                &blurred,
                &blurred,
                &ab,
                params,
                fine,
            );
            pass(&self.gf_blur_h_pipeline, &ab, &ab, &tmp, params, fine);
            pass(&self.gf_blur_v_pipeline, &tmp, &tmp, out, params, fine);
        }

        pass(
            &self.gf_pack_pipeline,
            &ab_clarity,
            &blurred,
            &ab,
            &params_clarity,
            fine,
        );

        queue.submit(Some(encoder.finish()));
        (ab, c_moments)
    }

    fn high_precision_pipeline(&self) -> &HighPrecisionPipeline {
        self.high_precision_pipeline.get_or_init(|| {
            let device = &self.context.device;
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("High Precision Main BGL"),
                entries: &self.high_precision_entries,
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("High Precision Pipeline Layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("High Precision Image Processing Shader"),
                source: wgpu::ShaderSource::Wgsl(high_precision_shader_source().into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("High Precision Compute Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("HIGH_PRECISION_OUTPUT", 1.0)],
                    ..Default::default()
                },
                cache: None,
            });
            HighPrecisionPipeline { bgl, pipeline }
        })
    }

    fn high_precision_tile(&self) -> &Tile {
        self.high_precision_tile
            .get_or_init(|| Tile::new(&self.context.device, wgpu::TextureFormat::Rgba16Float))
    }

    /// Uploads `image` and builds its blurs and guided-filter coefficients.
    pub fn upload(&self, image: &DynamicImage, is_raw: u32) -> GpuImage {
        let device = &self.context.device;
        let (width, height) = image.dimensions();
        let texture = device.create_texture_with_data(
            &self.context.queue,
            &wgpu::TextureDescriptor {
                label: Some("Input Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(&to_rgba_f16(image)),
        );
        let view = texture.create_view(&Default::default());

        let short_side = width.min(height) as f32;
        let mut encoder = device.create_command_encoder(&Default::default());
        let blurs = BLUR_RADII.map(|radius| {
            let radius = (radius * short_side / 1080.0).ceil().max(1.0) as u32;
            self.encode_blur(&mut encoder, &view, radius)
        });
        self.context.queue.submit(Some(encoder.finish()));
        let (gf_coeffs, gf_dehaze) = self.build_guided_coeffs(&view, width, height, is_raw);

        GpuImage {
            width,
            height,
            is_raw,
            view,
            blurs,
            gf_coeffs,
            gf_dehaze,
        }
    }

    fn blur_texture(&self, label: &str, size: wgpu::Extent3d) -> wgpu::TextureView {
        self.context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
    }

    /// Blurs `source` horizontally, then vertically, into a new texture of its size.
    fn encode_blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        radius: u32,
    ) -> wgpu::TextureView {
        let device = &self.context.device;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur Params Buffer"),
            contents: bytemuck::bytes_of(&BlurParams { radius, _pad: [0; 3] }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let size = source.texture().size();
        let across = self.blur_texture("Blur Ping Pong Texture", size);
        let blurred = self.blur_texture("Blur Texture", size);
        for (pipeline, input, output) in [
            (&self.h_blur_pipeline, source, &across),
            (&self.v_blur_pipeline, &across, &blurred),
        ] {
            encode_pass(
                device,
                encoder,
                pipeline,
                &[
                    wgpu::BindingResource::TextureView(input),
                    wgpu::BindingResource::TextureView(output),
                    params.as_entire_binding(),
                ],
                size,
            );
        }
        blurred
    }

    /// Uploads what one render reads besides the image (masks, LUT) and runs its flare
    /// passes. Returns the main pass's bind group, writing into `output`.
    fn bind_main_pass(
        &self,
        layout: &wgpu::BindGroupLayout,
        image: &GpuImage,
        request: &RenderRequest,
        output: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let device = &self.context.device;
        let queue = &self.context.queue;

        let mask_texture_view = if request.mask_bitmaps.is_empty() {
            self.dummy_masks_view.clone()
        } else {
            let mask_layer_count = request.mask_bitmaps.len().clamp(2, MAX_MASKS) as u32;
            let buffer_size =
                (image.width as usize) * (image.height as usize) * (mask_layer_count as usize);
            let mut mask_texture_data = Vec::with_capacity(buffer_size);
            for mask_bitmap in request.mask_bitmaps.iter().take(MAX_MASKS) {
                mask_texture_data.extend_from_slice(mask_bitmap.as_raw());
            }
            mask_texture_data.resize(buffer_size, 0);
            device
                .create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("Full Mask Texture Array"),
                        size: wgpu::Extent3d {
                            width: image.width,
                            height: image.height,
                            depth_or_array_layers: mask_layer_count,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    TextureDataOrder::MipMajor,
                    &mask_texture_data,
                )
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                })
        };

        let (lut_texture_view, lut_sampler) = if let Some(lut_arc) = &request.lut {
            let lut_data = &lut_arc.data;
            let size = lut_arc.size;
            let mut rgba_lut_data_f16 = Vec::with_capacity(lut_data.len() / 3 * 4);
            for chunk in lut_data.as_chunks::<3>().0 {
                rgba_lut_data_f16.push(f16::from_f32(chunk[0]));
                rgba_lut_data_f16.push(f16::from_f32(chunk[1]));
                rgba_lut_data_f16.push(f16::from_f32(chunk[2]));
                rgba_lut_data_f16.push(f16::ONE);
            }
            let lut_texture = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("LUT 3D Texture"),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                bytemuck::cast_slice(&rgba_lut_data_f16),
            );
            let view = lut_texture.create_view(&Default::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (view, sampler)
        } else {
            (self.dummy_lut_view.clone(), self.dummy_lut_sampler.clone())
        };

        let camera_profile_buffer = request.camera_profile.as_ref().map(|profile| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Profile Buffer"),
                contents: &profile.gpu_bytes(),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });

        let adjustments = &request.adjustments;
        if adjustments.global.flare_amount > 0.0 {
            let mut encoder = device.create_command_encoder(&Default::default());

            let aspect_ratio = if image.height > 0 {
                image.width as f32 / image.height as f32
            } else {
                1.0
            };
            let f_params = FlareParams {
                amount: adjustments.global.flare_amount,
                is_raw: adjustments.global.is_raw_image,
                exposure: adjustments.global.exposure,
                brightness: adjustments.global.brightness,
                contrast: adjustments.global.contrast,
                whites: adjustments.global.whites,
                aspect_ratio,
                _pad: 0.0,
            };
            queue.write_buffer(&self.flare_params_buffer, 0, bytemuck::bytes_of(&f_params));

            let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG0"),
                layout: &self.flare_bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&image.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.flare_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.flare_threshold_pipeline);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 16, FLARE_MAP_SIZE / 16, 1);
            }

            let bg0_ghosts = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG0 Ghosts"),
                layout: &self.flare_bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&image.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_final_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.flare_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                ],
            });

            let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG1"),
                layout: &self.flare_bgl_1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_ghosts_view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.flare_ghosts_pipeline);
                cpass.set_bind_group(0, &bg0_ghosts, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 16, FLARE_MAP_SIZE / 16, 1);
            }

            queue.submit(Some(encoder.finish()));
        }
        let mut bind_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&image.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(output),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.adjustments_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&mask_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 3 + MAX_MASK_BINDINGS,
                resource: wgpu::BindingResource::TextureView(&lut_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 4 + MAX_MASK_BINDINGS,
                resource: wgpu::BindingResource::Sampler(&lut_sampler),
            },
        ];
        for (binding, blur) in (5 + MAX_MASK_BINDINGS..).zip(&image.blurs) {
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(blur),
            });
        }
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: 9 + MAX_MASK_BINDINGS,
            resource: wgpu::BindingResource::TextureView(
                if adjustments.global.flare_amount > 0.0 {
                    &self.flare_ghosts_view
                } else {
                    &self.dummy_view
                },
            ),
        });
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: 10 + MAX_MASK_BINDINGS,
            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
        });
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: 11 + MAX_MASK_BINDINGS,
            resource: wgpu::BindingResource::TextureView(&image.gf_coeffs),
        });
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: 12 + MAX_MASK_BINDINGS,
            resource: wgpu::BindingResource::TextureView(&image.gf_dehaze),
        });
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: 13 + MAX_MASK_BINDINGS,
            resource: camera_profile_buffer
                .as_ref()
                .unwrap_or(&self.dummy_camera_profile_buffer)
                .as_entire_binding(),
        });
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: 14 + MAX_MASK_BINDINGS,
            resource: self.curve_histogram.as_entire_binding(),
        });

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main Bind Group"),
            layout,
            entries: &bind_group_entries,
        })
    }

    /// Records the main pass over `size` pixels of the image, starting at `offset`. The
    /// offset goes through the shared adjustments buffer, so submit before recording another.
    fn encode_main_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        adjustments: &AllAdjustments,
        offset: (u32, u32),
        size: wgpu::Extent3d,
    ) {
        let mut adjustments = *adjustments;
        (adjustments.tile_offset_x, adjustments.tile_offset_y) = offset;
        self.context.queue.write_buffer(
            &self.adjustments_buffer,
            0,
            bytemuck::bytes_of(&adjustments),
        );
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
    }

    /// Renders all of `image` into mip 0 of `output`, then rebuilds its other mips.
    fn render_to_texture(
        &self,
        image: &GpuImage,
        request: RenderRequest,
        output: &wgpu::Texture,
    ) -> wgpu::SubmissionIndex {
        let device = &self.context.device;
        let bind_group = self.bind_main_pass(&self.main_bgl, image, &request, &mip_view(output, 0));
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.curve_histogram, 0, None);
        self.encode_main_pass(
            &mut encoder,
            &self.main_pipeline,
            &bind_group,
            &request.adjustments,
            (0, 0),
            output.size(),
        );
        for level in 1..output.mip_level_count() {
            encode_pass(
                device,
                &mut encoder,
                &self.mipmap_pipeline,
                &[
                    wgpu::BindingResource::TextureView(&mip_view(output, level - 1)),
                    wgpu::BindingResource::TextureView(&mip_view(output, level)),
                ],
                output
                    .size()
                    .mip_level_size(level, wgpu::TextureDimension::D2),
            );
        }
        self.context.queue.submit(Some(encoder.finish()))
    }

    /// Renders `request.roi` of `image` (all of it without one) tile by tile, reading each
    /// tile back. Returns the pixels and their size.
    fn render_to_cpu(
        &self,
        image: &GpuImage,
        request: RenderRequest,
        output_precision: RenderOutputPrecision,
    ) -> Result<(RenderedPixels, u32, u32), String> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let bounds = request.roi.unwrap_or(Roi {
            x: 0,
            y: 0,
            width: image.width,
            height: image.height,
        });
        let (pipeline, layout, tile, bytes_per_pixel) = match output_precision {
            RenderOutputPrecision::EightBit => (&self.main_pipeline, &self.main_bgl, &self.tile, 4),
            RenderOutputPrecision::SixteenBit => {
                let hp = self.high_precision_pipeline();
                (&hp.pipeline, &hp.bgl, self.high_precision_tile(), 8)
            }
        };
        let bind_group = self.bind_main_pass(layout, image, &request, &tile.view);
        queue.write_buffer(&self.curve_histogram, 0, &[0; CURVE_HISTOGRAM_BYTES as usize]);

        let output_len = (bounds.width * bounds.height * 4) as usize;
        let mut final_pixels = match output_precision {
            RenderOutputPrecision::EightBit => RenderedPixels::U8(vec![0u8; output_len]),
            RenderOutputPrecision::SixteenBit => RenderedPixels::U16(vec![0u16; output_len]),
        };

        for y_start in (bounds.y..bounds.y + bounds.height).step_by(TILE_SIZE as usize) {
            for x_start in (bounds.x..bounds.x + bounds.width).step_by(TILE_SIZE as usize) {
                let tile_size = wgpu::Extent3d {
                    width: TILE_SIZE.min(bounds.x + bounds.width - x_start),
                    height: TILE_SIZE.min(bounds.y + bounds.height - y_start),
                    depth_or_array_layers: 1,
                };
                let mut encoder = device.create_command_encoder(&Default::default());
                self.encode_main_pass(
                    &mut encoder,
                    pipeline,
                    &bind_group,
                    &request.adjustments,
                    (x_start, y_start),
                    tile_size,
                );
                queue.submit(Some(encoder.finish()));
                let tile_data =
                    read_texture(device, queue, &tile.texture, 0, tile_size, bytes_per_pixel)?;

                let row_bytes = (tile_size.width * bytes_per_pixel) as usize;
                for (row, source) in tile_data.chunks_exact(row_bytes).enumerate() {
                    let final_y = (y_start - bounds.y) as usize + row;
                    let final_x = (x_start - bounds.x) as usize;
                    let offset = (final_y * bounds.width as usize + final_x) * 4;
                    let samples = tile_size.width as usize * 4;
                    match &mut final_pixels {
                        RenderedPixels::U8(final_pixels) => {
                            final_pixels[offset..offset + samples].copy_from_slice(source);
                        }
                        RenderedPixels::U16(final_pixels) => {
                            for (sample, bits) in final_pixels[offset..offset + samples]
                                .iter_mut()
                                .zip(source.as_chunks::<2>().0)
                            {
                                let value = f16::from_bits(u16::from_ne_bytes(*bits)).to_f32();
                                *sample = if value.is_finite() {
                                    (value.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16
                                } else {
                                    0
                                };
                            }
                        }
                    }
                }
            }
        }

        Ok((final_pixels, bounds.width, bounds.height))
    }

    /// The Curves graph's histogram counts from the last render: luma, red, green and
    /// blue, 256 bins each.
    fn read_curve_histogram(&self) -> Result<Vec<u32>, String> {
        let device = &self.context.device;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Curve Histogram Readback"),
            size: CURVE_HISTOGRAM_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.curve_histogram, 0, &staging, 0, CURVE_HISTOGRAM_BYTES);
        let submission = self.context.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        wait_for_gpu(device, Some(submission))?;
        rx.recv()
            .map_err(|e| format!("Failed receiving GPU map result: {}", e))?
            .map_err(|e| e.to_string())?;
        let counts = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(counts)
    }
}

/// Locks a cache slot; if a panic poisoned it, empties it instead of propagating.
fn lock_or_reset<'a, T>(
    slot: &'a std::sync::Mutex<Option<T>>,
    name: &str,
) -> std::sync::MutexGuard<'a, Option<T>> {
    slot.lock().unwrap_or_else(|poisoned| {
        log::warn!("{} lock was poisoned. Resetting to self-heal.", name);
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    })
}

fn lock_processor<'a>(
    context: &GpuContext,
    state: &'a AppState,
) -> Result<std::sync::MutexGuard<'a, Option<GpuProcessor>>, String> {
    let mut processor = lock_or_reset(&state.gpu_processor, "GPU processor");
    if processor.is_none() {
        let start = Instant::now();
        *processor = Some(GpuProcessor::new(context.clone())?);
        log::info!("Created the GPU processor in {:?}", start.elapsed());
    }
    Ok(processor)
}

/// The editor's photo for direct display: uploaded once per image and geometry, then
/// rendered in full on every edit into `output`, whose mips serve every zoom level.
pub struct DisplayImage {
    source: Weak<DynamicImage>,
    image: GpuImage,
    output: wgpu::Texture,
}

/// Renders `image`, the editor's photo at full resolution, onto the direct display and
/// sends a small copy of it to the histogram and waveform. Zooming and panning then only
/// redraw the display. `image` is uploaded, and its blurs built, only when it is new.
pub fn render_to_display(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    image: &Arc<DynamicImage>,
    request: RenderRequest,
    analytics_config: Option<crate::AnalyticsConfig>,
) -> Result<(), String> {
    let device = &context.device;
    let processor = lock_processor(context, state)?;
    let processor = processor.as_ref().unwrap();

    let mut display_image = lock_or_reset(&state.display_image, "Display image");
    let source = Arc::downgrade(image);
    if !display_image
        .as_ref()
        .is_some_and(|cached| cached.source.ptr_eq(&source))
    {
        *display_image = None;
        let start = Instant::now();
        let gpu_image = processor.upload(image, request.adjustments.global.is_raw_image);
        let size = wgpu::Extent3d {
            width: gpu_image.width,
            height: gpu_image.height,
            depth_or_array_layers: 1,
        };
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Display Texture"),
            size,
            mip_level_count: size.max_mips(wgpu::TextureDimension::D2),
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        wait_for_gpu(device, None)?;
        log::info!(
            "[render_to_display] uploaded {}x{} and built its blurs in {:?}",
            size.width,
            size.height,
            start.elapsed()
        );
        *display_image = Some(DisplayImage {
            source,
            image: gpu_image,
            output,
        });
    }
    let DisplayImage { image, output, .. } = display_image.as_ref().unwrap();

    let start = Instant::now();
    let submission = processor.render_to_texture(image, request, output);
    if let Some(display) = context
        .display
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
    {
        display.show(device, &output.create_view(&Default::default()));
        display.render(device, &context.queue);
    }
    wait_for_gpu(device, Some(submission))?;
    log::info!(
        "[render_to_display] rendered {}x{} in {:?}",
        image.width,
        image.height,
        start.elapsed()
    );

    if let Some(analytics) = analytics_config {
        // The histogram and waveform need about 1000 px: take the smallest mip that has them.
        let level = (output.width().max(output.height()) / 1024).max(1).ilog2();
        let size = output
            .size()
            .mip_level_size(level, wgpu::TextureDimension::D2);
        let pixels = read_texture(device, &context.queue, output, level, size, 4)?;
        let mip = ImageBuffer::<Rgba<u8>, _>::from_raw(size.width, size.height, pixels)
            .ok_or("Failed to create the analytics image from GPU data")?;
        let _ = analytics.sender.send(crate::AnalyticsJob {
            path: analytics.path,
            image: Arc::new(DynamicImage::ImageRgba8(mip)),
            curve_histogram: processor.read_curve_histogram()?,
            compute_waveform: analytics.compute_waveform,
            active_waveform_channel: analytics.active_waveform_channel,
        });
    }
    Ok(())
}

pub fn process_and_get_dynamic_image(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
) -> Result<DynamicImage, String> {
    process_and_get_dynamic_image_inner(
        context,
        state,
        base_image,
        transform_hash,
        request,
        caller_id,
        RenderOutputPrecision::EightBit,
        None,
    )
}

pub fn process_and_get_dynamic_image_with_precision(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    output_precision: RenderOutputPrecision,
) -> Result<DynamicImage, String> {
    process_and_get_dynamic_image_inner(
        context,
        state,
        base_image,
        transform_hash,
        request,
        caller_id,
        output_precision,
        None,
    )
}

pub fn process_and_get_dynamic_image_with_analytics(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    analytics_config: Option<crate::AnalyticsConfig>,
) -> Result<DynamicImage, String> {
    process_and_get_dynamic_image_inner(
        context,
        state,
        base_image,
        transform_hash,
        request,
        caller_id,
        RenderOutputPrecision::EightBit,
        analytics_config,
    )
}

#[allow(clippy::too_many_arguments)]
fn process_and_get_dynamic_image_inner(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    output_precision: RenderOutputPrecision,
    analytics_config: Option<crate::AnalyticsConfig>,
) -> Result<DynamicImage, String> {
    let start_time = Instant::now();
    let (width, height) = base_image.dimensions();

    let max_dim = context.limits.max_texture_dimension_2d;
    if width > max_dim || height > max_dim {
        log::warn!(
            "Image dimensions ({}x{}) exceed GPU limits ({}). Bypassing GPU processing and returning unprocessed image to prevent a crash. Try upgrading your GPU :)",
            width,
            height,
            max_dim
        );
        return Ok(base_image.clone());
    }

    let processor = lock_processor(context, state)?;
    let processor = processor.as_ref().unwrap();

    let is_raw = request.adjustments.global.is_raw_image;
    let mut cache_lock = lock_or_reset(&state.gpu_image_cache, "GPU image cache");
    let is_cached = cache_lock.as_ref().is_some_and(|cache| {
        cache.transform_hash == transform_hash
            && cache.image.width == width
            && cache.image.height == height
            && cache.image.is_raw == is_raw
    });
    if !is_cached {
        *cache_lock = None;
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(500)),
        });
        *cache_lock = Some(GpuImageCache {
            image: processor.upload(base_image, is_raw),
            transform_hash,
        });
    }
    let image = &cache_lock.as_ref().unwrap().image;

    let (processed_pixels, out_w, out_h) =
        processor.render_to_cpu(image, request, output_precision)?;

    if let Some(analytics) = analytics_config {
        if let RenderedPixels::U8(pixels) = &processed_pixels {
            let pixels_clone = pixels.clone();
            let curve_histogram = processor.read_curve_histogram()?;
            std::thread::spawn(move || {
                if let Some(img_buf) =
                    ImageBuffer::<Rgba<u8>, _>::from_raw(out_w, out_h, pixels_clone)
                {
                    let dynamic_img = DynamicImage::ImageRgba8(img_buf);
                    let _ = analytics.sender.send(crate::AnalyticsJob {
                        path: analytics.path,
                        image: std::sync::Arc::new(dynamic_img),
                        curve_histogram,
                        compute_waveform: analytics.compute_waveform,
                        active_waveform_channel: analytics.active_waveform_channel,
                    });
                }
            });
        } else {
            log::warn!("Skipping analytics for a high-precision CPU readback");
        }
    }

    let duration = start_time.elapsed();
    let fps = 1.0 / duration.as_secs_f64();
    log::info!(
        "[{}] {}x{} processed (ROI: {}x{}) on GPU in {:?} ({:.2} FPS)",
        caller_id,
        width,
        height,
        out_w,
        out_h,
        duration,
        fps
    );

    match processed_pixels {
        RenderedPixels::U8(pixels) => {
            let img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(out_w, out_h, pixels)
                .ok_or("Failed to create 8-bit image buffer from GPU data")?;
            Ok(DynamicImage::ImageRgba8(img_buf))
        }
        RenderedPixels::U16(pixels) => {
            let img_buf = ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(out_w, out_h, pixels)
                .ok_or("Failed to create 16-bit image buffer from GPU data")?;
            Ok(DynamicImage::ImageRgba16(img_buf))
        }
    }
}

/// Heal and Clone patches written without their recipe (patchData) get the one
/// the editor makes when they are painted, with the source picked for them.
fn with_cleanup_recipes(
    original: &DynamicImage,
    is_raw: bool,
    adjustments: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let mut adjustments = adjustments.clone();
    let count = adjustments["aiPatches"].as_array().map_or(0, |p| p.len());
    for i in 0..count {
        let patch = &adjustments["aiPatches"][i];
        let cleanup = patch["subMasks"].as_array().is_some_and(|sub_masks| {
            sub_masks
                .iter()
                .any(|sm| matches!(sm["type"].as_str(), Some("heal" | "clone")))
        });
        if !cleanup || !patch["patchData"].is_null() {
            continue;
        }
        let definition: crate::mask_generation::AiPatchDefinition =
            serde_json::from_value(patch.clone()).map_err(|e| format!("aiPatches[{i}]: {e}"))?;
        let source =
            crate::inpainting::source_image(original, is_raw, &definition.id, &adjustments)?;
        let recipe =
            crate::inpainting::cleanup_patch(&source, &definition, &adjustments, None, None)
                .map_err(|e| format!("aiPatches[{i}]: {e}"))?;
        adjustments["aiPatches"][i]["patchData"] = recipe;
    }
    Ok(adjustments)
}

/// Harness (examples/render.rs): renders `path` with `adjustments` (the editor's
/// JSON) as the editor does with camera profiles on: patches, geometry (warps,
/// crop, rotation) and masks, downscaled to `long_edge` like the preview, then
/// through the full GPU pipeline on a device of its own.
#[doc(hidden)]
pub fn dev_render(
    path: &str,
    adjustments: &serde_json::Value,
    long_edge: u32,
) -> Result<image::RgbImage, String> {
    let settings = crate::app_settings::AppSettings {
        use_camera_profiles: Some(true),
        ..Default::default()
    };
    let is_raw = crate::formats::is_raw_file(path);
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let original =
        crate::image_loader::load_base_image_from_bytes(&bytes, path, false, &settings, None)
            .map_err(|e| e.to_string())?;
    let adjustments = &with_cleanup_recipes(&original, is_raw, adjustments)?;
    let patched = crate::image_loader::composite_patches_on_image(&original, adjustments)
        .map_err(|e| e.to_string())?;
    let (transformed, crop_offset) = crate::adjustment_utils::apply_all_transformations(
        std::borrow::Cow::Borrowed(&patched),
        adjustments,
    );
    // The editor's preview downscale: image's own resize clamps float pixels to 1.0.
    let image = crate::image_processing::downscale_f32_image(&transformed, long_edge, long_edge);
    let scale = image.width() as f32 / transformed.width() as f32;
    drop(transformed);
    drop(patched);

    // Unlike the editor, which logs a broken mask and renders without it, fail
    // with the reason: a hand-written edit should not lose a mask silently.
    let mask_definitions: Vec<crate::mask_generation::MaskDefinition> = adjustments
        .get("masks")
        .map(|m| serde_json::from_value(m.clone()).map_err(|e| format!("masks: {e}")))
        .transpose()?
        .unwrap_or_default();
    let sampled = mask_definitions
        .iter()
        .any(|m| m.requires_warped_image())
        .then(|| crate::mask_generation::build_full_warped_image(&original, is_raw, adjustments).into_owned());
    drop(original);
    let mut mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = Vec::new();
    for def in &mask_definitions {
        mask_bitmaps.extend(crate::mask_generation::generate_mask_bitmap_strict(
            def,
            image.width(),
            image.height(),
            scale,
            (crop_offset.0 * scale, crop_offset.1 * scale),
            sampled.as_ref(),
        )?);
    }
    let (adjustments, camera_profile) =
        crate::camera_profile::render_adjustments(adjustments, path, is_raw, &settings);

    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .map_err(|e| e.to_string())?;
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    let processor = GpuProcessor::new(GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        limits,
        display: Default::default(),
    })?;
    let gpu_image = processor.upload(&image, is_raw as u32);
    let request = RenderRequest {
        adjustments,
        mask_bitmaps: &mask_bitmaps,
        lut: None,
        roi: None,
        camera_profile,
    };
    let (pixels, w, h) =
        processor.render_to_cpu(&gpu_image, request, RenderOutputPrecision::EightBit)?;
    let RenderedPixels::U8(rgba) = pixels else {
        unreachable!()
    };
    let rgba = ImageBuffer::<Rgba<u8>, _>::from_raw(w, h, rgba).ok_or("bad render size")?;
    Ok(DynamicImage::ImageRgba8(rgba).to_rgb8())
}
