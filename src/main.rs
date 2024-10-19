mod shape;
mod export;
mod world;

use std::ops::Mul;
use ash::vk;
use ash::vk::{ImageAspectFlags, ImageLayout};
use cen::app::App;
use cen::app::app::AppConfig;
use cen::graphics::Renderer;
use cen::graphics::renderer::RenderComponent;
use cen::vulkan::CommandBuffer;
use crate::export::parser::{Parser};
use glam::{DMat4, DVec2, DVec3, DVec4, EulerRot, Mat4};
use rand::random;
use vsvg::{PageSize, Unit};
use crate::world::{WorldGen};

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

fn transform(p: DVec3) -> DVec2 {
    let model = DMat4::from_euler(EulerRot::XYZ, 0.5, 0.3, 0.3);
    let view = DMat4::look_at_rh(DVec3::new(1.0, 0.5, 1.0) * 1.3, DVec3::new(0., 0., 0.), DVec3::new(0., 1., 0.));
    let proj = DMat4::perspective_lh(70., 1., 0.001, 500.);
    let q = proj.mul(view).mul(model).mul(DVec4::new(p.x, p.y, p.z, 1.));
    DVec2::new(q.x, q.y) / q.w
}

struct Rend {

}

impl RenderComponent for Rend {
    fn render(&self, renderer: &mut Renderer, command_buffer: &mut CommandBuffer, swapchain_image: &vk::Image) {

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

        unsafe {
            renderer.device.handle().cmd_clear_color_image(
                command_buffer.handle(),
                *swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 1.0, 0.0, 1.0]
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
    }
}

fn main() {
    let page_size = PageSize::Custom(100., 100., Unit::Mm);

    let mut world = WorldGen::new();

    let mesh = gen_mesh();

    let scale = 300.;
    let offset = DVec2::new(page_size.to_pixels().0, page_size.to_pixels().1) / 2.;

    for l in mesh.1 {
        let p1 = transform(mesh.0[l.0]);
        let p2 = transform(mesh.0[l.1]);

        world.add_line(
            p1 * scale + offset,
            p2 * scale + offset,
        );
    }

    let objects = world.convert();

    let app = App::new(AppConfig {
        width: 1000,
        height: 1000,
        vsync: true,
        log_fps: false,
    });
    app.run(Box::new(Rend{}));

    Parser::parse(&objects, page_size);
}
