use glam::Vec2;
use crate::parser::Parser;
use crate::shape::Rect;

mod shape;
mod parser;

fn main() {

    let r = Rect {
        p1: Vec2::new(0.0, 0.0),
        p2: Vec2::new(1.0, 1.0),
    };

    Parser::parse(r);
}
