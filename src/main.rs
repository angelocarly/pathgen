use crate::export::parser::{Parseable, Parser};
use glam::Vec2;
use crate::shape::{Line, Rect};

mod shape;
mod export;

fn main() {

    let size = Vec2::new(1.0, 1.0);

    let mut objects : Vec<Box<dyn Parseable>> = Vec::new();

    let basepos = Vec2::new(10.0, 4.0);

    for x in 0..12 {
        for y in 0..18 {
            let l = Line {
                p1: basepos + Vec2::new(x as f32 * 4. - ( x * y ) as f32 * 0.2, y as f32 * 2.3),
                p2: basepos + Vec2::new(x as f32 * 4. - ( x * y ) as f32 * 0.2, y as f32 * 2.3) + Vec2::new(1.0, 1.0),
            };
            objects.push(Box::new(l));
        }
    }

    Parser::parse(objects);
}
