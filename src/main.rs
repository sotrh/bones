use std::f32::consts::{PI, TAU};

use bytemuck::{bytes_of, cast_slice};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: glam::Vec3,
    normal: glam::Vec3,
    indices: [i32; 4],
    weights: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Fields {
    mode: u32,
    selected_bone: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    debug: glam::Vec4,
    mv: glam::Mat4,
    mvp: glam::Mat4,
}

#[derive(Debug)]
struct Bone {
    length: f32,
    offset: glam::Vec3,
    orientation: glam::Quat,
}

impl Bone {
    pub fn to_mat4(&self) -> glam::Mat4 {
        glam::Mat4::from_translation(self.offset) * glam::Mat4::from_quat(self.orientation)
    }
}

/// Stores bones in a single list with offsets.
/// Parent of any given bone is the previous bone in the list.
#[derive(Debug)]
struct SimpleSkeleton {
    bones: Vec<Bone>,
    animation_time: f32,
    // In a real application this would likely be keyframes
    pose: Vec<glam::Mat4>,
    inv_bind_pose: Vec<glam::Mat4>,
}

impl SimpleSkeleton {
    pub fn new(bones: Vec<Bone>) -> Self {
        let mut parent_matrix = glam::Mat4::IDENTITY;
        let pose = bones
            .iter()
            .map(|b| {
                let transform = parent_matrix * b.to_mat4();
                let translation = glam::Mat4::from_translation(
                    glam::vec3(0.0, b.length, 0.0),
                );
                parent_matrix = transform * translation;
                transform
            })
            .collect::<Vec<_>>();
        let inv_bind_pose = pose
            .iter()
            .map(|transform| transform.inverse())
            .collect::<Vec<_>>();

        Self {
            bones,
            animation_time: 0.0,
            pose,
            inv_bind_pose,
        }
    }

    pub fn update(&mut self, dt: instant::Duration) {
        self.animation_time += dt.as_secs_f32();
        self.animation_time %= TAU;
        let angle = self.animation_time.sin() * PI / 4.0;
        let axis = glam::vec3(0.0, 0.0, 1.0);
        let mut parent_matrix = glam::Mat4::IDENTITY;
        for (i, bone) in self.bones.iter_mut().enumerate() {
            bone.orientation = glam::Quat::from_axis_angle(axis, angle);

            let transform = parent_matrix * bone.to_mat4();
            let translation = glam::Mat4::from_translation(glam::vec3(
                0.0,
                bone.length,
                0.0,
            ));
            parent_matrix = transform * translation;
            self.pose[i] = transform;
        }
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_visible(false)
        .build(&event_loop)?;

    // Setup data

    let axis = glam::vec3(1.0, 0.0, 0.0);
    let num_bones = 4;
    let length = 2.0 / num_bones as f32;
    let bones = (0..num_bones).map(|_| {
        Bone {
            length,
            offset: glam::Vec3::ZERO,
            orientation: glam::Quat::from_axis_angle(axis, 0.0),
        }
    }).collect::<Vec<_>>();
    let mut skeleton = SimpleSkeleton::new(bones);

    let model = glam::Mat4::from_translation(glam::vec3(0.0, 0.0, 0.0));
    let view = glam::Mat4::look_at_lh(
        glam::vec3(0.0, 2.0, -5.0),
        glam::vec3(0.0, 0.0, 0.0),
        glam::vec3(0.0, 1.0, 0.0),
    );
    let proj = glam::Mat4::perspective_lh(PI / 4.0, 1.0, 0.1, 100.0);
    let mv = view * model;
    let mut uniforms = Uniforms {
        debug: glam::Vec4::ZERO,
        mv: mv,
        mvp: proj * mv,
    };

    // Setup rendering

    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .filter(|a| a.is_surface_supported(&surface))
        .next()
        .expect("There should be a valid adapter");

    println!("Adapter: {:?}", adapter);
    
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            limits: wgpu::Limits::downlevel_defaults(),
            features: wgpu::Features::default(),
        },
        None,
    ))?;

    println!("{:?}", surface.get_supported_formats(&adapter));
    let mut surf_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_supported_formats(&adapter)[0],
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
    };
    surface.configure(&device, &surf_config);

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes_of(&uniforms),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    });
    let pose_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: cast_slice(&skeleton.pose),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });
    let inv_bind_pose_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: cast_slice(&skeleton.inv_bind_pose),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &uniform_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pose_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inv_bind_pose_buffer.as_entire_binding(),
            },
        ],
    });

    let sections = 8;
    let num_rings = 6;
    let width = 0.2;
    let mut vertices = Vec::new();
    let mut skeleton_vertices = Vec::new();
    let mut indices = Vec::new();
    let mut last_p = glam::Vec3::new(0.0, 0.0, 0.0);
    for (bone_index, b) in skeleton.bones.iter().enumerate() {
        let bot = last_p + b.offset;
        let top = bot + glam::vec3(0.0, b.length, 0.0);
        last_p = top;

        let start_index = vertices.len() as u32;
        let bone_index = bone_index as i32;
        let next_bone = (bone_index + 1).min(skeleton.bones.len() as i32 - 1);
        skeleton_vertices.push(Vertex {
            position: bot,
            normal: glam::vec3(0.0, 0.0, -1.0),
            indices: [bone_index, -1, -1, -1],
            weights: [1.0, 0.0, 0.0, 0.0],
        });
        skeleton_vertices.push(Vertex {
            position: top,
            normal: glam::vec3(0.0, 0.0, -1.0),
            indices: [next_bone, -1, -1, -1],
            weights: [1.0, 0.0, 0.0, 0.0],
        });
        vertices.extend((0..num_rings).flat_map(|ring_index| {
            let factor = ring_index as f32 / (num_rings - 1) as f32;
            let p = bot.lerp(top, factor);
            let indices = [bone_index, next_bone, -1, -1];
            let weights = [1.0, 0.0, 0.0, 0.0];
            (0..sections).map(move |section| {
                let theta = section as f32 / sections as f32 * TAU;
                let (s, c) = theta.sin_cos();
                let normal = glam::vec3(c, 0.0, s);
                let position = p + normal * width;
                Vertex {
                    position,
                    normal,
                    indices,
                    weights,
                }
            })
        }));

        // Rings
        for ring in 0..num_rings - 1 {
            let ring_offset = ring as u32 * sections;
            indices.extend((0..sections).flat_map(|si| {
                let bot_l = start_index + si + ring_offset;
                let bot_r = start_index + (si + 1) % sections + ring_offset;
                let top_l = sections + bot_l;
                let top_r = sections + bot_r;
                [bot_l, bot_r, top_r, bot_l, top_r, top_l]
            }));
        }
    }
    let num_indices = indices.len() as u32;
    let num_skeleton_verts = skeleton_vertices.len() as u32;

    let skeleton_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: cast_slice(&skeleton_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as _,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Sint32x4,
            3 => Float32x4,
        ],
    };
    let index_format = wgpu::IndexFormat::Uint32;

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let depth_format = wgpu::TextureFormat::Depth32Float;
    let mut depth_desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: surf_config.width,
            height: surf_config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: depth_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
    };
    let depth_buffer = device.create_texture(&depth_desc);
    let mut depth_view = depth_buffer.create_view(&wgpu::TextureViewDescriptor::default());

    let render_pipeline = create_render_pipeline(
        &device,
        &uniform_layout,
        &shader,
        vertex_layout.clone(),
        Some(depth_format),
        &surf_config,
        wgpu::PrimitiveTopology::TriangleList,
        "vs_main",
        "fs_main",
    );
    let skeleton_pipeline = create_render_pipeline(
        &device,
        &uniform_layout,
        &shader,
        vertex_layout,
        None,
        &surf_config,
        wgpu::PrimitiveTopology::LineList,
        "vs_main",
        "fs_skeleton",
    );

    println!("Finished rendering setup");

    let mut uniforms_dirty = true;
    let mut last_time = instant::Instant::now();
    window.set_visible(true);
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => control_flow.set_exit(),
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            state,
                            ..
                        },
                    ..
                },
            ..
        } => match (key, state == ElementState::Pressed) {
            (VirtualKeyCode::Escape, true) => control_flow.set_exit(),
            (VirtualKeyCode::Space, true) => {
                uniforms.debug.x = (uniforms.debug.x + 1.0) % 2.0;
                uniforms_dirty = true;
            }
            (VirtualKeyCode::Numpad0 | VirtualKeyCode::Key0, true) => {
                uniforms.debug.y = 0.0;
                uniforms_dirty = true;
            }
            (VirtualKeyCode::Numpad1 | VirtualKeyCode::Key1, true) => {
                uniforms.debug.y = 1.0;
                uniforms_dirty = true;
            }
            (VirtualKeyCode::Numpad2 | VirtualKeyCode::Key2, true) => {
                uniforms.debug.y = 2.0;
                uniforms_dirty = true;
            }
            (VirtualKeyCode::Numpad3 | VirtualKeyCode::Key3, true) => {
                uniforms.debug.y = 3.0;
                uniforms_dirty = true;
            }
            _ => (),
        },
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            surf_config.width = size.width;
            surf_config.height = size.height;
            surface.configure(&device, &surf_config);
            depth_desc.size.width = size.width;
            depth_desc.size.height = size.height;
            let depth_buffer = device.create_texture(&depth_desc);
            depth_view = depth_buffer.create_view(&wgpu::TextureViewDescriptor::default());
            uniforms.mvp = glam::Mat4::perspective_lh(
                PI / 4.0,
                size.width as f32 / size.height as f32,
                0.1,
                100.0,
            ) * uniforms.mv;
            uniforms_dirty = true;
        }
        Event::RedrawEventsCleared => window.request_redraw(),
        Event::RedrawRequested(_) => {
            match surface.get_current_texture() {
                Ok(surf_tex) => {
                    if uniforms_dirty {
                        queue.write_buffer(&uniform_buffer, 0, bytes_of(&uniforms));
                        uniforms_dirty = false;
                    }

                    let now = instant::Instant::now();
                    let dt = now - last_time;
                    last_time = now;

                    skeleton.update(dt);
                    queue.write_buffer(&pose_buffer, 0, cast_slice(&skeleton.pose));

                    let view = surf_tex
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    // Mesh
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: true,
                            }),
                            stencil_ops: None,
                        }),
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    rpass.set_index_buffer(index_buffer.slice(..), index_format);
                    rpass.set_bind_group(0, &uniform_bind_group, &[]);
                    rpass.draw_indexed(0..num_indices, 0, 0..1);
                    drop(rpass);

                    // Skeleton
                    if uniforms.debug.x > 0.0 {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: true,
                                },
                            })],
                            depth_stencil_attachment: None,
                        });
                        rpass.set_pipeline(&skeleton_pipeline);
                        rpass.set_vertex_buffer(0, skeleton_vertex_buffer.slice(..));
                        rpass.set_bind_group(0, &uniform_bind_group, &[]);
                        rpass.draw(0..num_skeleton_verts, 0..1);
                    }

                    queue.submit([encoder.finish()]);
                    surf_tex.present()
                }
                Err(wgpu::SurfaceError::Outdated) => {
                    // The surface should be reconfigured in the resize handler
                }
                Err(e) => {
                    eprintln!("{}", e);
                    control_flow.set_exit_with_code(1);
                }
            }
        }
        _ => (),
    });
}

fn create_render_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    vertex_layout: wgpu::VertexBufferLayout,
    depth_format: Option<wgpu::TextureFormat>,
    surf_config: &wgpu::SurfaceConfiguration,
    topology: wgpu::PrimitiveTopology,
    vs_entry: &str,
    fs_entry: &str,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&uniform_layout],
        push_constant_ranges: &[],
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: vs_entry,
            buffers: &[vertex_layout],
        },
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: depth_format.map(|f| wgpu::DepthStencilState {
            format: f,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: fs_entry,
            targets: &[Some(wgpu::ColorTargetState {
                format: surf_config.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
    });
    render_pipeline
}
