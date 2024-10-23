mod shape;
mod export;
mod world;

use std::ops::Mul;
use std::time::Instant;
use ash::vk;
use ash::vk::{BufferUsageFlags, ImageAspectFlags, ImageSubresourceLayers, Offset3D, PushConstantRange, ShaderStageFlags, WriteDescriptorSet};
use bytemuck::{Pod, Zeroable};
use cen::app::App;
use cen::app::app::AppConfig;
use cen::graphics::pipeline_store::{PipelineConfig, PipelineKey};
use cen::graphics::Renderer;
use cen::graphics::renderer::RenderComponent;
use cen::vulkan::{Buffer, CommandBuffer, DescriptorSetLayout, Image};
use crate::export::parser::{Parseable, Parser};
use glam::{DMat4, DVec2, DVec3, DVec4, EulerRot, Mat4, Vec2, Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use vsvg::{PageSize, Unit};
use crate::shape::Line;

fn gen_mesh() -> (Vec<DVec3>, Vec<(usize, usize)>) {

    let phi = (1. + f64::sqrt(5.) ) / 2.;

    let a = 0.5;
    let b = 0.5 * 1. / phi;
    let c = 0.5 * (2. - phi);

    let vertices: Vec<DVec3> = vec![
        DVec3::new( c, 0.,  a),
        DVec3::new(-c,  0.,  a),
        DVec3::new(-b,  b,  b),
        DVec3::new( 0.,  a,  c),
        DVec3::new( b,  b,  b),
        DVec3::new( b, -b,  b),
        DVec3::new( 0., -a,  c),
        DVec3::new(-b, -b,  b),
        DVec3::new( c,  0., -a),
        DVec3::new(-c,  0., -a),
        DVec3::new(-b, -b, -b),
        DVec3::new( 0., -a, -c),
        DVec3::new( b, -b, -b),
        DVec3::new( b,  b, -b),
        DVec3::new( 0.,  a, -c),
        DVec3::new(-b,  b, -b),
        DVec3::new( a,  c,  0.),
        DVec3::new(-a,  c,  0.),
        DVec3::new(-a, -c,  0.),
        DVec3::new( a, -c,  0.)
    ];

    let lines: Vec<(usize,usize)> = vec![
        (  0,  1 ),
        (  0,  4 ),
        (  0,  5 ),
        (  1,  2 ),
        (  1,  7 ),
        (  2,  3 ),
        (  2, 17 ),
        (  3,  4 ),
        (  3, 14 ),
        (  4, 16 ),
        (  5,  6 ),
        (  5, 19 ),
        (  6,  7 ),
        (  6, 11 ),
        (  7, 18 ),
        (  8,  9 ),
        (  8, 12 ),
        (  8, 13 ),
        (  9, 10 ),
        (  9, 15 ),
        ( 10, 11 ),
        ( 10, 18 ),
        ( 11, 12 ),
        ( 12, 19 ),
        ( 13, 14 ),
        ( 13, 16 ),
        ( 14, 15 ),
        ( 15, 17 ),
        ( 16, 19 ),
        ( 17, 18 )
    ];

    (vertices, lines)
}

struct Rend {
    descriptorset: DescriptorSetLayout,
    pipeline: PipelineKey,
    image: Image,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    start_time: Instant,
}

struct PushConstants {
    transform: Mat4
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

        let mut vertex_buffer = Buffer::new(
            &renderer.device,
            &mut renderer.allocator,
            MemoryLocation::CpuToGpu,
            100 * 4,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST
        );

        let index_buffer = Buffer::new(
            &renderer.device,
            &mut renderer.allocator,
            MemoryLocation::CpuToGpu,
            100 * 4,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST
        );

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

        Self {
            image,
            vertex_buffer,
            index_buffer,
            descriptorset,
            pipeline,
            start_time: Instant::now()
        }
    }

    pub fn transform(&self) -> Mat4 {
        let current_time = self.start_time.elapsed().as_secs_f32();
        let model = Mat4::from_euler(EulerRot::XYZ, 0.1 * current_time, 0.03 * current_time, 0.);
        let view = Mat4::look_at_rh(Vec3::new(1.0, 0.5, 1.0) * 1.3, Vec3::new(0., 0., 0.), Vec3::new(0., 1., 0.));
        let proj = Mat4::perspective_lh(1.8f32, 1., 0.001, 500.);
        proj.mul(view).mul(model)
    }


    pub fn update_mesh(&mut self) {

        let mesh = gen_mesh();

        let (_, vert_mem, _) = unsafe { self.vertex_buffer.mapped().align_to_mut::<Vec4>() };
        for i in 0..mesh.0.len() {
            let v = mesh.0[i];
            vert_mem[ i ] = Vec4::new(v.x as f32, v.y as f32, v.z as f32, 0.);
        }


        let (_, index_mem, _) = unsafe { self.index_buffer.mapped().align_to_mut::<i32>() };
        for i in 0..mesh.1.len() {
            index_mem[ i * 2 + 0 ] = mesh.1[i].0 as i32;
            index_mem[ i * 2 + 1 ] = mesh.1[i].1 as i32;
        }
    }

    pub fn export(&mut self) -> (Vec<i32>, Vec<Vec4>) {
        let vertices: Vec<Vec4> = unsafe { self.vertex_buffer.mapped().align_to_mut::<Vec4>() }.1.into();
        let indices: Vec<i32> = unsafe { self.index_buffer.mapped().align_to_mut::<i32>() }.1.into();

        (indices, vertices)
    }
}

impl RenderComponent for Rend {
    fn render(&self, renderer: &mut Renderer, command_buffer: &mut CommandBuffer, swapchain_image: &vk::Image) {

        // Render
        let compute = renderer.pipeline_store().get(self.pipeline).unwrap();
        command_buffer.bind_pipeline(&compute);

        let image_bindings = [self.image.binding(vk::ImageLayout::GENERAL)];
        let image_write_descriptor_set = WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_bindings);

        let index_bindings = [self.index_buffer.binding()];
        let index_write_descriptor_set = WriteDescriptorSet::default()
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&index_bindings);

        let vertex_bindings = [self.vertex_buffer.binding()];
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

        let push_constants = PushConstants {
            transform: self.transform()
        };

        command_buffer.push_constants(
            &compute,
            ShaderStageFlags::COMPUTE,
            0,
            &bytemuck::cast_slice(std::slice::from_ref(&push_constants.transform.to_cols_array()))
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

            renderer.device.handle().cmd_clear_color_image(
                command_buffer.handle(),
                *swapchain_image,
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

            // Use a blit, as a copy doesn't synchronize properly to the swapchain on MoltenVK
            renderer.device.handle().cmd_blit_image(
                command_buffer.handle(),
                *self.image.handle(),
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

fn main() {

    // Run the renderer
    let mut app = App::new(AppConfig {
        width: 1000,
        height: 1000,
        vsync: true,
        log_fps: false,
    });

    let mut rend = Rend::new(app.renderer());
    rend.update_mesh();

    app.run(&rend);

    let trans = rend.transform();

    let (indices, vertices) = rend.export();

    // Export the data
    let page_size = PageSize::Custom(100., 100., Unit::Mm);

    let scale = 350.;
    let offset = DVec2::new(page_size.to_pixels().0, page_size.to_pixels().1) / 2.;

    let mut objects: Vec<Box<dyn Parseable>> = Vec::new();
    for i in 0..indices.len()/2 {
        let p1 = trans.mul(Vec4::from(vertices[indices[i * 2 + 0] as usize]));
        let p2 = trans.mul(Vec4::from(vertices[indices[i * 2 + 1] as usize]));
        objects.push(Box::new(
            Line {
                p1: DVec2::new(p1.x as f64, p1.y as f64) * scale + offset,
                p2: DVec2::new(p2.x as f64, p2.y as f64) * scale + offset,
            }
        ));
    }
    Parser::parse(&objects, page_size);
}
