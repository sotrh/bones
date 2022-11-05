use std::{
    f32::consts::{PI, TAU},
    io::Write,
};

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
                parent_matrix = transform
                    * glam::Mat4::from_translation(
                        b.orientation.mul_vec3(glam::vec3(0.0, b.length, 0.0)),
                    );
                transform
            })
            .collect::<Vec<_>>();
        let inv_bind_pose = pose
            .iter()
            .map(|transform| transform.inverse())
            .collect::<Vec<_>>();

        Self {
            bones,
            pose,
            inv_bind_pose,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_visible(false)
        .build(&event_loop)?;

    // Setup data

    let bones = vec![
        Bone {
            length: 1.0,
            offset: glam::vec3(0.0, 0.0, 0.0),
            orientation: glam::Quat::IDENTITY,
        },
        Bone {
            length: 1.0,
            offset: glam::vec3(0.0, 0.0, 0.0),
            orientation: glam::Quat::IDENTITY,
        },
        Bone {
            length: 1.0,
            offset: glam::vec3(0.0, 0.0, 0.0),
            orientation: glam::Quat::IDENTITY,
        },
        Bone {
            length: 1.0,
            offset: glam::vec3(0.0, 0.0, 0.0),
            orientation: glam::Quat::IDENTITY,
        },
    ];
    let skeleton = SimpleSkeleton::new(bones);

    let model = glam::Mat4::from_translation(glam::vec3(0.0, 0.0, 1.0));
    let mut uniforms = Uniforms {
        debug: glam::Vec4::ZERO,
        mv: model,
        mvp: model,
    };

    // Setup rendering

    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .next()
        .expect("There should be a valid adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            limits: wgpu::Limits::downlevel_defaults(),
            features: wgpu::Features::default(),
        },
        None,
    ))?;

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
    let num_rings = 4;
    let width = 0.2;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut last_p = glam::Vec3::new(0.0, 0.0, 0.0);
    for (bone_index, b) in skeleton.bones.iter().enumerate() {
        let bot = last_p + b.offset;
        let top = bot + glam::vec3(0.0, b.length, 0.0);
        last_p = top;

        let start_index = vertices.len() as u32;
        vertices.extend((0..num_rings).flat_map(|ring_index| {
            let factor = match ring_index {
                0 => 0.0,
                1 => 0.1,
                2 => 0.9,
                3 => 1.0,
                _ => unreachable!(),
            };
            let p = bot.lerp(top, factor);
            (0..sections).map(move |section| {
                let theta = section as f32 / sections as f32 * TAU;
                let (s, c) = theta.sin_cos();
                let normal = glam::vec3(c, 0.0, s);
                let position = p + normal * width;
                let indices = [bone_index as i32, bone_index as i32 + 1, -1, -1];
                let weights = [1.0 - factor, factor, 0.0, 0.0];
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
    {
        let mut debug_txt = std::fs::File::create("target/indices.csv")?;
        for chunk in indices.chunks(3) {
            writeln!(&mut debug_txt, "{}, {}, {}", chunk[0], chunk[1], chunk[2])?;
        }
    }
    {
        let mut debug_txt = std::fs::File::create("target/vertex_indices.csv")?;
        writeln!(&mut debug_txt, "V, I0, I1, I2, I3, W0, W1, W2, W3")?;
        for (i, v) in vertices.iter().enumerate() {
            writeln!(
                &mut debug_txt,
                "{}, {}, {}, {}, {}, {}, {}, {}, {}",
                i,
                v.indices[0],
                v.indices[1],
                v.indices[2],
                v.indices[3],
                v.weights[0],
                v.weights[1],
                v.weights[2],
                v.weights[3],
            )?;
        }
    }
    let num_indices = indices.len() as u32;

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
            entry_point: "vs_main",
            buffers: &[vertex_layout],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surf_config.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
    });

    println!("Finished rendering setup");

    let mut uniforms_dirty = true;
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
        }
        Event::RedrawEventsCleared => window.request_redraw(),
        Event::RedrawRequested(_) => {
            match surface.get_current_texture() {
                Ok(surf_tex) => {
                    if uniforms_dirty {
                        queue.write_buffer(&uniform_buffer, 0, bytes_of(&uniforms));
                        uniforms_dirty = false;
                    }
                    let view = surf_tex
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
