use glam::Vec2;
use svg::node::element::Path;
use crate::export::parser::Parseable;

fn param(input: Vec2) -> (f32, f32) {
    (input.x, input.y)
}

pub struct Rect {
    pub pos: Vec2,
    pub size: Vec2,
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
            .set("stroke-width", 3)
            .set("d", data)
    }
}