use crate::export::parser::{Parseable, Parser};
use glam::Vec2;
use crate::shape::{Line, Rect};

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

        let l = Line {
            p1: Vec2::new(1.0, 2.0) * i as f32,
            p2: Vec2::new(1.0, 2.0) * i as f32 + Vec2::new( 10.0, 0.0 ),
        };
        objects.push(Box::new(l));
    }

    Parser::parse(objects);
}
