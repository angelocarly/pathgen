use crate::export::parser::{Parseable, Parser};
use glam::Vec2;
use rand::random;
use crate::shape::{LineStrip, QuadraticBezier};

mod shape;
mod export;

fn main() {

    let basepos = Vec2::new(10.0, 10.0);

    let mut lines = Vec::new();
    let line_count = 30;
    let elements = 10;
    for l in 0..line_count {
        let mut points = Vec::new();
        for x in 0..elements {
            points.push(Vec2::new(basepos.x + x as f32 * 6., basepos.y + l as f32));
        }
        lines.push(LineStrip {
            points
        });
    }

    for x in 0..elements {
        let offset: Vec2 = Vec2::new(random::<f32>(), random::<f32>()) * 2.0f32 - 1.0f32;
        let mut i = 0;
        for l in &mut lines {
            l.points.get_mut(x).unwrap().x += offset.x * i as f32 * 0.5;
            l.points.get_mut(x).unwrap().y += offset.y * i as f32 * 0.5;
            i += 1;
        }
    }

    let mut objects : Vec<Box<dyn Parseable>> = Vec::new();
    for line in lines {
        objects.push(Box::new(line));
    }

    Parser::parse(objects);
}
