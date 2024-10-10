mod shape;
mod export;
mod world;

use crate::export::parser::{Parser};
use glam::DVec2;
use rand::random;
use crate::world::{WorldGen};

fn main() {

    let mut world = WorldGen::new();

    let scale = 350.;
    world.add_line(DVec2::new(random::<f64>() * scale, random::<f64>() * scale), DVec2::new(random::<f64>() * scale, random::<f64>() * scale));

    for _ in 0..10 {
        world.add_line(
            DVec2::new(random::<f64>() * scale, random::<f64>() * scale),
            DVec2::new(random::<f64>() * scale, random::<f64>() * scale)
        );
    }

    let objects = world.convert();
    Parser::parse(&objects);
}
