mod shape;
mod export;
mod world;
mod mesh;

use std::collections::HashSet;
use std::fs;
use std::ops::Mul;
use std::path::PathBuf;
use std::time::Instant;
use ash::vk;
use ash::vk::{BufferUsageFlags, DeviceSize, ImageAspectFlags, ImageSubresourceLayers, Offset3D, PushConstantRange, ShaderStageFlags, WriteDescriptorSet};
use bytemuck::{Pod, Zeroable};
use cen::app::App;
use cen::app::app::AppConfig;
use cen::graphics::pipeline_store::{PipelineConfig, PipelineKey};
use cen::graphics::Renderer;
use cen::graphics::renderer::RenderComponent;
use cen::vulkan::{Buffer, CommandBuffer, DescriptorSetLayout, Image};
use crate::export::parser::{Parseable, Parser};
use glam::{DVec2, EulerRot, Mat4, Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use vsvg::{PageSize, Unit};
use crate::mesh::gen_dodecahedron;
use crate::shape::Line;

struct Mesh {
    vb: Buffer,
    ib: Buffer,
}

struct Rend {
    bloom_pass: BloomPass,
    descriptorset: DescriptorSetLayout,
    pipeline: PipelineKey,
    image: Image,
    meshes: Vec<Mesh>,
    start_time: Instant,
}

#[derive(Pod, Zeroable, Debug)]
#[repr(C)]
#[derive(Copy)]
#[derive(Clone)]
struct PushConstants {
    transform: [f32; 16],
    color: [f32; 4],
    edge_count: i32,
    time: f32,
}

struct BloomPass {
    pub pipeline: PipelineKey,
    pub ds_layout: DescriptorSetLayout,
    pub image: Image,
}

impl BloomPass {
    fn new(renderer: &mut Renderer) -> BloomPass {

        // Image
        let image = Image::new(
            &renderer.device,
            &mut renderer.allocator,
            renderer.swapchain.get_extent().width,
            renderer.swapchain.get_extent().height,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
        );

        // Transition image
        let mut image_command_buffer = CommandBuffer::new(&renderer.device, &renderer.command_pool);
        image_command_buffer.begin();
        {
            renderer.transition_image(&image_command_buffer, image.handle(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::AccessFlags::empty(), vk::AccessFlags::empty());
        }
        image_command_buffer.end();
        renderer.device.submit_single_time_command(renderer.queue, &image_command_buffer);

        // Layout
        let layout_bindings = &[
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE ),
        ];
        let ds_layout = DescriptorSetLayout::new_push_descriptor(
            &renderer.device,
            layout_bindings
        );

        // Pipeline
        let pipeline = renderer.pipeline_store().insert(PipelineConfig {
            shader_path: "shaders/bloom.comp".into(),
            descriptor_set_layouts: vec![
                ds_layout.clone(),
            ],
            push_constant_ranges: vec![],
            macros: Default::default(),
        }).expect("Failed to create pipeline");

        BloomPass {
            image,
            ds_layout,
            pipeline
        }
    }
}

impl Rend {
    fn new(renderer: &mut Renderer) -> Self {

        // Image
        let image = Image::new(
            &renderer.device,
            &mut renderer.allocator,
            renderer.swapchain.get_extent().width,
            renderer.swapchain.get_extent().height,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
        );

        // Transition image
        let mut image_command_buffer = CommandBuffer::new(&renderer.device, &renderer.command_pool);
        image_command_buffer.begin();
        {
            renderer.transition_image(&image_command_buffer, image.handle(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::AccessFlags::empty(), vk::AccessFlags::empty());
        }
        image_command_buffer.end();
        renderer.device.submit_single_time_command(renderer.queue, &image_command_buffer);

        let meshes= vec![
            Self::load_mesh(renderer, PathBuf::from("Dodecahedron.off")),
            Self::load_mesh(renderer, PathBuf::from("Icosahedron.off")),
            // Self::load_mesh(renderer, PathBuf::from("Icosidodecahedron.off")),
        ];

        // Layout
        let layout_bindings = &[
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE )
        ];
        let descriptorset = DescriptorSetLayout::new_push_descriptor(
            &renderer.device,
            layout_bindings
        );
        let push_constants = vec![
            PushConstantRange::default()
                .offset(0)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(size_of::<PushConstants>() as u32)
        ];

        // Pipeline
        let pipeline = renderer.pipeline_store().insert(PipelineConfig {
            shader_path: "shaders/lines.comp".into(),
            descriptor_set_layouts: vec![
                descriptorset.clone(),
            ],
            push_constant_ranges: push_constants,
            macros: Default::default(),
        }).expect("Failed to create pipeline");

        let bloom_pass = BloomPass::new(renderer);

        Self {
            image,
            meshes,
            descriptorset,
            pipeline,
            start_time: Instant::now(),
            bloom_pass,
        }
    }

    pub fn transform(&self, t: f32) -> Mat4 {
        let model = Mat4::from_euler(EulerRot::XYZ, 0.02 * t, 0.03 * t, 0.);
        let view = Mat4::look_at_rh(Vec3::new(1.0, 0.5, 0.5) * 1.3, Vec3::new(0., 0., 0.), Vec3::new(0., 1., 0.));
        let proj = Mat4::perspective_lh(0.8f32, 1., 0.001, 500.);
        proj.mul(view).mul(model)
    }


    pub fn load_mesh(renderer: &mut Renderer, path: PathBuf) -> Mesh {

        let off_file: String = fs::read_to_string(path).unwrap();
        let mesh = off_rs::parse(
            off_file.as_str(),
            Default::default()
        ).expect("Failed to parse off file");

        let mut vertex_buffer = Buffer::new(
            &renderer.device,
            &mut renderer.allocator,
            MemoryLocation::CpuToGpu,
            (mesh.vertices.len() * size_of::<Vec4>()) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST
        );

        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        mesh.faces.iter().for_each(|f| {
            for a in 0..f.vertices.len() {
                let mut i1 = f.vertices[a] as u32;
                let mut i2 = f.vertices[ (a + 1) % f.vertices.len() ] as u32;
                if i2 < i1 {
                    let q = i1;
                    i1 = i2;
                    i2 = q;
                }
                edges.insert((i1, i2) );
            }
        });

        let mut index_buffer = Buffer::new(
            &renderer.device,
            &mut renderer.allocator,
            MemoryLocation::CpuToGpu,
            (edges.len() * 2 * size_of::<u32>()) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST
        );

        let (_, vert_mem, _) = unsafe { vertex_buffer.mapped().align_to_mut::<Vec4>() };
        for i in 0..mesh.vertices.len() {
            let v = mesh.vertices[i].position;
            vert_mem[ i ] = Vec4::new(v.x, v.y, v.z, 0.).mul( 0.1 );
        }

        let (_, index_mem, _) = unsafe { index_buffer.mapped().align_to_mut::<u32>() };
        let mut i: usize = 0;
        for edge in edges {
                index_mem[ i ] = edge.0;
                i += 1;
                index_mem[ i ] = edge.1;
                i += 1;
        }

        Mesh {
            vb: vertex_buffer,
            ib: index_buffer,
        }
    }

    pub fn export(&mut self) -> Vec<(Vec<i32>, Vec<Vec4>)> {
        self.meshes.iter_mut().map(|m| {
            let vertices: Vec<Vec4> = unsafe { m.vb.mapped().align_to_mut::<Vec4>() }.1.into();
            let indices: Vec<i32> = unsafe { m.ib.mapped().align_to_mut::<i32>() }.1.into();
            (indices, vertices)
        }).collect()
    }
}

impl RenderComponent for Rend {
    fn render(&self, renderer: &mut Renderer, command_buffer: &mut CommandBuffer, swapchain_image: &vk::Image) {

        // Transition the render to a source
        renderer.transition_image(
            &command_buffer,
            &self.image.handle(),
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::NONE,
            vk::AccessFlags::TRANSFER_WRITE
        );

        unsafe {
            renderer.device.handle().cmd_clear_color_image(
                command_buffer.handle(),
                *self.image.handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0]
                },
                &[vk::ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }]
            );
        }

        // Transition the render to a source
        renderer.transition_image(
            &command_buffer,
            &self.image.handle(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_WRITE
        );

        let mut i = 0;
        for m in &self.meshes {
            let color = match i {
                0 => Vec4::new(1.0, 0.0, 0.0, 1.0),
                1 => Vec4::new(1.0, 1.0, 1.0, 1.0),
                2 => Vec4::new(0.0, 0.0, 1.0, 1.0),
                _ => Vec4::new(1.0, 1.0, 1.0, 1.0)
            };

            // Render
            let compute = renderer.pipeline_store().get(self.pipeline).unwrap();
            command_buffer.bind_pipeline(&compute);

            let image_bindings = [self.image.binding(vk::ImageLayout::GENERAL)];
            let image_write_descriptor_set = WriteDescriptorSet::default()
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&image_bindings);

            let index_bindings = [m.ib.binding()];
            let index_write_descriptor_set = WriteDescriptorSet::default()
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&index_bindings);

            let vertex_bindings = [m.vb.binding()];
            let vertex_write_descriptor_set = WriteDescriptorSet::default()
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&vertex_bindings);

            command_buffer.bind_push_descriptor(
                &compute,
                0,
                &[image_write_descriptor_set, index_write_descriptor_set, vertex_write_descriptor_set]
            );

            let edge_count = (m.ib.size as f32 / size_of::<u32>() as f32 / 2.0 ) as i32;
            let time = Instant::now().duration_since(self.start_time).as_secs_f32();
            let push_constants = PushConstants {
                transform: self.transform(time).to_cols_array(),
                color: color.to_array(),
                edge_count,
                time
            };

            command_buffer.push_constants(
                &compute,
                ShaderStageFlags::COMPUTE,
                0,
                &bytemuck::cast_slice(std::slice::from_ref(&push_constants))
            );

            command_buffer.dispatch(500, 500, 1 );

            i += 1;
        }

        // Transition the bloom to general
        renderer.transition_image(
            &command_buffer,
            &self.bloom_pass.image.handle(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::NONE,
            vk::AccessFlags::SHADER_WRITE
        );

        // Bloom pass
        let bloom = renderer.pipeline_store().get(self.bloom_pass.pipeline).unwrap();
        command_buffer.bind_pipeline(&bloom);

        let image_bindings = [self.image.binding(vk::ImageLayout::GENERAL)];
        let image_write_descriptor_set = WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_bindings);

        let image_bindings2 = [self.bloom_pass.image.binding(vk::ImageLayout::GENERAL)];
        let image_write_descriptor_set2 = WriteDescriptorSet::default()
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_bindings2);

        command_buffer.bind_push_descriptor(
            &bloom,
            0,
            &[image_write_descriptor_set, image_write_descriptor_set2]
        );

        command_buffer.dispatch(500, 500, 1 );

        // Transition the render to a source
        renderer.transition_image(
            &command_buffer,
            &self.image.handle(),
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ
        );

        // Transition the out to a source
        renderer.transition_image(
            &command_buffer,
            &self.bloom_pass.image.handle(),
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ
        );

        // Transition the swapchain image
        renderer.transition_image(
            &command_buffer,
            &swapchain_image,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::NONE,
            vk::AccessFlags::TRANSFER_WRITE
        );

        // Copy to the swapchain
        unsafe {

            // Use a blit, as a copy doesn't synchronize properly to the swapchain on MoltenVK
            renderer.device.handle().cmd_blit_image(
                command_buffer.handle(),
                *self.bloom_pass.image.handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                *swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_offsets([
                        Offset3D::default(),
                        Offset3D::default().x(self.image.width as i32).y(self.image.height as i32).z(1)
                    ])
                    .dst_offsets([
                        Offset3D::default(),
                        Offset3D::default().x(self.image.width as i32).y(self.image.height as i32).z(1)
                    ])
                    .src_subresource(
                        ImageSubresourceLayers::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(0)
                    )
                    .dst_subresource(
                        ImageSubresourceLayers::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(0)
                    )
                ],
                vk::Filter::NEAREST,
            );
        }

        // Transfer back to default states
        renderer.transition_image(
            &command_buffer,
            &swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::NONE
        );

        // Transition the render image back
        renderer.transition_image(
            &command_buffer,
            &self.image.handle(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::NONE
        );

    }
}

fn export(indices: &Vec<i32>, vertices: &Vec<Vec4>, path: PathBuf, transform: Mat4) {
    // Export the data
    let page_size = PageSize::Custom(200., 200., Unit::Mm);

    let scale = 1150.;
    let offset = DVec2::new(page_size.to_pixels().0, page_size.to_pixels().1) / 2.;

    let mut objects: Vec<Box<dyn Parseable>> = Vec::new();
    for i in 0..indices.len()/2 {
        let p1 = transform.mul(Vec4::from(vertices[indices[i * 2 + 0] as usize]));
        let p2 = transform.mul(Vec4::from(vertices[indices[i * 2 + 1] as usize]));
        objects.push(Box::new(
            Line {
                p1: DVec2::new(p1.x as f64, p1.y as f64) * scale + offset,
                p2: DVec2::new(p2.x as f64, p2.y as f64) * scale + offset,
            }
        ));
    }
    Parser::parse(&objects, page_size, path);
}

fn main() {

    // Run the renderer
    let mut app = App::new(AppConfig {
        width: 1000,
        height: 1000,
        vsync: false,
        log_fps: true,
    });

    let mut rend = Rend::new(app.renderer());

    app.run(&rend);

    let time = Instant::now().duration_since(rend.start_time).as_secs_f32();
    let trans = rend.transform(time);

    let mut i = 0;
    rend.export().iter().for_each(|m| {
        export(&m.0, &m.1, PathBuf::from(format!("target/path{:?}.svg", i)), trans);
        i += 1;
    });
}
