mod shape;
mod export;

use crate::export::parser::{Parseable, Parser};
use crate::shape::Exx;
use glam::DVec2;
use rand::random;

fn main() {
    let basepos = DVec2::new(10.0, 10.0);

    let mut objects: Vec<Box<dyn Parseable>> = Vec::new();

    let scale = 50.;
    let exx = Exx {
        p1: basepos + DVec2::new(random::<f64>() * scale, random::<f64>() * scale),
        p2: basepos + DVec2::new(random::<f64>() * scale, random::<f64>() * scale),
        p3: basepos + DVec2::new(random::<f64>() * scale, random::<f64>() * scale),
    };

    objects.push(Box::new(exx));

    Parser::parse(objects);
}
