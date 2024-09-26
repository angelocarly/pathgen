use glam::Vec2;
use svg::node::element::Path;

use svg::{Document};
use crate::shape::{Line, LineStrip, QuadraticBezier, Rect};

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

impl Parseable for QuadraticBezier {
    fn parse(&self) -> Path {
        let data = svg::node::element::path::Data::new()
            .move_to(param(self.p1))
            .quadratic_curve_by((
                self.c.x, self.c.y,
                self.p2.x, self.p2.y
            ));

        Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 0.3f32)
            .set("d", data)
    }
}

impl Parseable for LineStrip {
    fn parse(&self) -> Path {
        let mut data = svg::node::element::path::Data::new()
            .move_to(param(self.points[0]));

        for p in &self.points {
            data = data.line_to(param(*p));
        }

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

        let mut info_group = svg::node::element::Group::new()
            .set("id", "info");

        info_group = info_group.add(svg::node::element::Rectangle::new()
            .set("x", 0)
            .set("y", 0)
            .set("width", 70)
            .set("height", 70)
            .set("stroke-width", 0)
            .set("fill", "white"));

        document = document.add(info_group);

        let mut path_group = svg::node::element::Group::new()
            .set("id", "paths");
        for p in paths {
            path_group = path_group.add(p.parse());
        };

        document = document.add(path_group);

        svg::save("target/path.svg", &document).unwrap();
    }
}