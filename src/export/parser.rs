use glam::Vec2;
use svg::node::element::Path;

use svg::{Document};
use crate::shape::{Line, Rect};

pub trait Parseable {
    fn parse(&self) -> Path;
}

fn param(input: Vec2) -> (f32, f32) {
    (input.x, input.y)
}

impl Parseable for Rect {
    fn parse(&self) -> Path {
        let data = svg::node::element::path::Data::new()
            .move_to(param(self.pos))
            .line_to(param(self.pos + Vec2::new(self.size.x, 0.0)))
            .line_to(param(self.pos + Vec2::new(self.size.x, self.size.y )))
            .line_to(param(self.pos + Vec2::new(0.0, self.size.y)))
            .close();

        Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 0.3f32)
            .set("d", data)
    }
}

impl Parseable for Line {
    fn parse(&self) -> Path {
        let data = svg::node::element::path::Data::new()
            .move_to(param(self.p1))
            .line_to(param(self.p2));

        Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 0.3f32)
            .set("d", data)
    }
}

pub struct Parser {
}

impl Parser {

    pub fn parse(paths: Vec<Box<dyn Parseable>>) {
        let mut document = Document::new()
            .set("viewBox", (0, 0, 70, 70));

        for p in paths {
            document = document.add(p.parse());
        };

        svg::save("target/path.svg", &document).unwrap();
    }
}