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
use glam::{DVec2, EulerRot, Mat4, Vec2, Vec3, Vec4};
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

#[derive(Clone)]
struct Pentagon {
    pub pos: Vec2,
    pub rot: f32,
    pub weight: i32,
}

struct Growth {
    pentagons: Vec<Pentagon>,
}

impl Growth {

    pub fn new() -> Self {
        let PI = std::f32::consts::PI;
        Growth {
            pentagons: vec![Pentagon {
                pos: Vec2::new(0.0, 0.0),
                rot: PI * 1.5 / 5.,
                weight: 0
            }],
        }
    }

    pub fn export_star(&self) -> (Vec<Vec2>, Vec<i32>) {
        let mut verts = vec![];
        let mut indices = vec![];

        let PI = std::f32::consts::PI;
        for pent in &self.pentagons {
            if ( pent.weight + 1 ) % 2 == 0 {
                continue;
            }

            let p1 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 0. / 5. ), f32::sin(pent.rot + 2. * PI * 0. / 5. ));
            let p2 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 1. / 5. ), f32::sin(pent.rot + 2. * PI * 1. / 5. ));
            let p3 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 2. / 5. ), f32::sin(pent.rot + 2. * PI * 2. / 5. ));
            let p4 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 3. / 5. ), f32::sin(pent.rot + 2. * PI * 3. / 5. ));
            let p5 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 4. / 5. ), f32::sin(pent.rot + 2. * PI * 4. / 5. ));

            let s = 0.4f32;
            let i1 = pent.pos + s * Vec2::new(f32::cos(pent.rot + 2. * PI * 0.5 / 5. ), f32::sin(pent.rot + 2. * PI * 0.5 / 5. ));
            let i2 = pent.pos + s * Vec2::new(f32::cos(pent.rot + 2. * PI * 1.5 / 5. ), f32::sin(pent.rot + 2. * PI * 1.5 / 5. ));
            let i3 = pent.pos + s * Vec2::new(f32::cos(pent.rot + 2. * PI * 2.5 / 5. ), f32::sin(pent.rot + 2. * PI * 2.5 / 5. ));
            let i4 = pent.pos + s * Vec2::new(f32::cos(pent.rot + 2. * PI * 3.5 / 5. ), f32::sin(pent.rot + 2. * PI * 3.5 / 5. ));
            let i5 = pent.pos + s * Vec2::new(f32::cos(pent.rot + 2. * PI * 4.5 / 5. ), f32::sin(pent.rot + 2. * PI * 4.5 / 5. ));

            let offset = verts.len() as i32;
            verts.push(p1);
            verts.push(i1);
            verts.push(p2);
            verts.push(i2);
            verts.push(p3);
            verts.push(i3);
            verts.push(p4);
            verts.push(i4);
            verts.push(p5);
            verts.push(i5);

            indices.push(offset + 0);
            indices.push(offset + 1);
            indices.push(offset + 1);
            indices.push(offset + 2);
            indices.push(offset + 2);
            indices.push(offset + 3);
            indices.push(offset + 3);
            indices.push(offset + 4);
            indices.push(offset + 4);
            indices.push(offset + 5);
            indices.push(offset + 5);
            indices.push(offset + 6);
            indices.push(offset + 6);
            indices.push(offset + 7);
            indices.push(offset + 7);
            indices.push(offset + 8);
            indices.push(offset + 8);
            indices.push(offset + 9);
            indices.push(offset + 9);
            indices.push(offset + 0);
        }

        (verts, indices)
    }

    pub fn export(&self) -> (Vec<Vec2>, Vec<i32>) {
        let mut verts = vec![];
        let mut indices = vec![];

        let PI = std::f32::consts::PI;
        for pent in &self.pentagons {
            let p1 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 0. / 5. ), f32::sin(pent.rot + 2. * PI * 0. / 5. ));
            let p2 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 1. / 5. ), f32::sin(pent.rot + 2. * PI * 1. / 5. ));
            let p3 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 2. / 5. ), f32::sin(pent.rot + 2. * PI * 2. / 5. ));
            let p4 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 3. / 5. ), f32::sin(pent.rot + 2. * PI * 3. / 5. ));
            let p5 = pent.pos + Vec2::new(f32::cos(pent.rot + 2. * PI * 4. / 5. ), f32::sin(pent.rot + 2. * PI * 4. / 5. ));

            let offset = verts.len() as i32;
            verts.push(p1);
            verts.push(p2);
            verts.push(p3);
            verts.push(p4);
            verts.push(p5);

            indices.push(offset + 0);
            indices.push(offset + 1);
            indices.push(offset + 1);
            indices.push(offset + 2);
            indices.push(offset + 2);
            indices.push(offset + 3);
            indices.push(offset + 3);
            indices.push(offset + 4);
            indices.push(offset + 4);
            indices.push(offset + 0);
        }

        (verts, indices)
    }

    fn close_enough(a: &Vec2, b: &Vec2) -> bool {
        return f32::abs(a.x - b.x) < 0.01f32 && f32::abs(a.y - b.y) < 0.01f32;
    }

    fn segments_connect(a: &Vec2, b: &Vec2, c: &Vec2, d: &Vec2) -> bool {
        // println!("{:?} {:?} {:?} {:?}", a, b, c, d);
        if ( Self::close_enough(a, c) && Self::close_enough(b, d) ) || ( Self::close_enough(b, c) && Self::close_enough(a, d) ) {
            return true;
        }
        return false;
    }

    // Check if two lines intersect and find the intersection point
    fn segments_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
        let p = a;
        let q = c;
        let r = Vec2 {
            x: b.x - a.x,
            y: b.y - a.y,
        };
        let s = Vec2 {
            x: d.x - c.x,
            y: d.y - c.y,
        };

        let denominator = r.x * s.y - r.y * s.x;

        if denominator == 0.0 {
            // Lines are parallel or collinear
            return false;
        }

        let t = ((q.x - p.x) * s.y - (q.y - p.y) * s.x) / denominator;
        let u = ((q.x - p.x) * r.y - (q.y - p.y) * r.x) / denominator;

        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
            // Intersection point
            true
        } else {
            // Lines do not intersect within the segments
            false
        }
    }

    fn get_segment(pos: &Vec2, rot: &f32, segment: u32, scale: f32) -> (Vec2, Vec2) {
        let PI = std::f32::consts::PI;
        let p1 = pos + scale * Vec2::new(f32::cos(rot + 2. * PI * segment as f32 / 5. ), f32::sin(rot + 2. * PI * segment as f32 / 5. ));
        let p2 = pos + scale * Vec2::new(f32::cos(rot + 2. * PI * (segment + 1) as f32 / 5. ), f32::sin(rot + 2. * PI * (segment + 1) as f32 / 5. ));
        (p1, p2)
    }

    /**
     * Fill in each pentagon with smaller subpentagons
     */
    pub fn recurse(&mut self) {
        let mut new_pents = vec![];

        let PI = std::f32::consts::PI;
        let length = f32::cos(2. * PI * 1. / 10. ) * 2.;
        for pent in &self.pentagons {
            let scaled_pos = pent.pos * length * length;

            new_pents.push(
                Pentagon {
                    pos: scaled_pos,
                    rot: pent.rot + PI / 5.,
                    weight: pent.weight,
                }
            );
            for i in 0..5 {
                new_pents.push(
                    Pentagon {
                        pos: scaled_pos + length * Vec2::new(f32::cos(pent.rot + 2. * PI * i as f32 / 5. ), f32::sin(pent.rot + 2. * PI * i as f32 / 5. )),
                        rot: pent.rot,
                        weight: pent.weight + 1,
                    }
                );
            }
        }

        self.pentagons = new_pents;
    }

    pub fn fill_holes(&mut self) {

        let current_pents = self.pentagons.clone();

        let PI = std::f32::consts::PI;
        let length = f32::cos(2. * PI * 1. / 10. ) * 2.;

        let max_dist = self.pentagons.iter().map(|p| {
            p.pos.length()
        }).max_by(|a,b| a.total_cmp(b)).unwrap();

        for pent_index in 0..current_pents.len() {
            let pent = &current_pents[pent_index];
            for edge in 0..5 {
                let mut rot = pent.rot + 2. * PI * ( edge - 3 ) as f32 / 5.0;
                rot += 2. * PI * 0.5 / 5.0;
                let pos = Vec2::new(pent.pos.x + length * f32::cos( rot ), pent.pos.y + length * f32::sin(rot));

                // Check if pos goes out of restricted area
                if( pos.length() < max_dist ) {
                    self.add_pentagon(pent_index as i32, edge);
                }

            }
        }
    }

    pub fn add_pentagon(&mut self, index: i32, edge: i32) -> Result<(), &'static str> {
        let PI = std::f32::consts::PI;

        let length = f32::cos(2. * PI * 1. / 10. ) * 2.;

        // Get the pentagon of interest
        let last_pent = &self.pentagons[index as usize];
        let new_weigth = last_pent.weight + 1;

        let mut rot = last_pent.rot + 2. * PI * ( edge - 3 ) as f32 / 5.0;
        rot += 2. * PI * 0.5 / 5.0;
        let pos = Vec2::new(last_pent.pos.x + length * f32::cos( rot ), last_pent.pos.y + length * f32::sin(rot));

        // Check intersection
        for pent in &mut self.pentagons {
            for main_segment in 0..5 {
                let dif = pent.pos - pos;
                if dif.x * dif.x + dif.y + dif.y > 2.0 {
                    continue;
                }

                let segment1 = Self::get_segment(&pos, &rot, main_segment, 0.95);
                for s in 0..5 {
                    let segment2 = Self::get_segment(&pent.pos, &pent.rot, s, 0.95);
                    if Self::segments_intersect(segment1.0, segment1.1, segment2.0, segment2.1) {
                        return Err("");
                    }
                }
            }
        }

        self.pentagons.push(Pentagon{
            pos,
            rot,
            weight: new_weigth,
        });

        Ok(())
    }
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
            // Self::load_mesh(renderer, PathBuf::from("Dodecahedron.off")),
            // Self::load_mesh(renderer, PathBuf::from("Icosahedron.off")),
            // Self::load_mesh(renderer, PathBuf::from("Icosidodecahedron.off")),
            Self::load_scene(renderer)
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
        // let model = Mat4::from_euler(EulerRot::XYZ, 0.02 * t, 0.03 * t, 0.);
        // let view = Mat4::look_at_rh(Vec3::new(1.0, 0.5, 0.5) * 1.3, Vec3::new(0., 0., 0.), Vec3::new(0., 1., 0.));
        // let proj = Mat4::perspective_lh(0.8f32, 1., 0.001, 500.);
        // proj.mul(view).mul(model)
        // Mat4::IDENTITY
        Mat4::from_scale(Vec3::splat(0.2f32))
    }

    pub fn load_scene(renderer: &mut Renderer) -> Mesh {

        let mut growth = Growth::new();
        let mut min = 0;
        for i in 0..5 {
            let mut added = false;
            for p in min..growth.pentagons.len() {
                for edge in 0..5 {
                    match growth.add_pentagon(p as i32, edge) {
                        Ok(_) => {
                            added = true;
                            break;
                        }
                        Err(_) => {
                            if edge == 4 {
                                // This smallest pentagon to check
                                min = p;
                            }
                        }
                    }
                }
                if added {
                    break;
                }
            }
        }

        growth.recurse();
        growth.fill_holes();
        growth.recurse();
        growth.fill_holes();
        // growth.recurse();
        // growth.fill_holes();

        let (vertices, indices) = growth.export_star();

        let mut vertex_buffer = Buffer::new(
            &renderer.device,
            &mut renderer.allocator,
            MemoryLocation::CpuToGpu,
            (vertices.len() * size_of::<Vec4>()) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST
        );

        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        indices.chunks(2).for_each(|chunk| {
            if chunk.len() < 2 {
                return;
            }
            edges.insert((chunk[0] as u32, chunk[1] as u32) );
        });

        let mut index_buffer = Buffer::new(
            &renderer.device,
            &mut renderer.allocator,
            MemoryLocation::CpuToGpu,
            (edges.len() * 2 * size_of::<u32>()) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST
        );

        let (_, vert_mem, _) = unsafe { vertex_buffer.mapped().align_to_mut::<Vec4>() };
        for i in 0..vertices.len() {
            let v = vertices[i];
            vert_mem[ i ] = Vec4::new(v.x, v.y, 0., 0.).mul( 0.1 );
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
