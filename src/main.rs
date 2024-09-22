use crate::export::parser::{Parseable, Parser};
use glam::Vec2;
use crate::shape::Rect;

mod shape;
mod export;

fn main() {

    let r = Rect {
        p1: Vec2::new(0.0, 0.0),
        p2: Vec2::new(1.0, 1.0),
    };

    let r2 = Rect {
        p1: Vec2::new(5.0, 10.0),
        p2: Vec2::new(1.0, 1.0),
    };

    let mut objects : Vec<Box<dyn Parseable>> = Vec::new();
    objects.push(Box::new(r));
    objects.push(Box::new(r2));

    Parser::parse(objects);
}
