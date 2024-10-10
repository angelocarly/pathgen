mod shape;
mod export;
mod World;

use crate::export::parser::{Parseable, Parser};
use crate::shape::Exx;
use glam::DVec2;
use rand::random;
use crate::World::{Line, WorldGen};

fn main() {

    let mut world = WorldGen::new();

    let scale = 350.;
    let line = Line {
        p1: DVec2::new(random::<f64>() * scale, random::<f64>() * scale),
        p2: DVec2::new(random::<f64>() * scale, random::<f64>() * scale),
    };
    world.add_line(line);

    let mut cut_count = 0;
    while cut_count < 2 {
        let l = Line {
            p1: DVec2::new(random::<f64>() * scale, random::<f64>() * scale),
            p2: DVec2::new(random::<f64>() * scale, random::<f64>() * scale),
        };
        if world.add_line(l) {
            cut_count += 1;
        }
    }

    Parser::parse(world.objects());
}
