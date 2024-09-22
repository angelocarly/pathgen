use glam::Vec2;
use svg::Node;
use svg::node::element::Path;
use crate::parser::Parseable;

fn param(input: Vec2) -> (f32, f32) {
    (input.x, input.y)
}

pub struct Rect {
    pub p1: Vec2,
    pub p2: Vec2,
}

impl Parseable for Rect {
    fn parse(&self) -> Path {
        let data = svg::node::element::path::Data::new()
            .move_to(param(self.p1))
            .line_to(param(self.p1 + Vec2::new(self.p2.x, 0.0)))
            .line_to(param(self.p1 + Vec2::new(self.p2.x, self.p2.y )))
            .line_to(param(self.p1 + Vec2::new(0.0, self.p2.y)))
            .close();

        Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 3)
            .set("d", data)
    }
}