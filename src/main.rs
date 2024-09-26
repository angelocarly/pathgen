use std::backtrace;
use crate::export::parser::{Parseable, Parser};
use glam::Vec2;
use rand::random;
use svg::open;
use crate::shape::{Exx, LineStrip, QuadraticBezier};

mod shape;
mod export;

fn main() {

    let basepos = Vec2::new(10.0, 10.0);

    let mut objects : Vec<Box<dyn Parseable>> = Vec::new();

    let scale = 50.;
    let exx = Exx {
        p1: basepos + Vec2::new(random::<f32>() * scale, random::<f32>() * scale),
        p2: basepos + Vec2::new(random::<f32>() * scale, random::<f32>() * scale),
        p3: basepos + Vec2::new(random::<f32>() * scale, random::<f32>() * scale),
    };

    objects.push(Box::new(exx));

    Parser::parse(objects);
}
