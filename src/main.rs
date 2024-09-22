use crate::export::parser::{Parseable, Parser};
use glam::Vec2;
use crate::shape::Rect;

mod shape;
mod export;

fn main() {

    let size = Vec2::new(1.0, 1.0);

    let mut objects : Vec<Box<dyn Parseable>> = Vec::new();

    for i in 0..10 {
        let r = Rect {
            pos: Vec2::new(1.0, 1.0) * i as f32,
            size: size,
        };
        objects.push(Box::new(r));
    }

    Parser::parse(objects);
}
