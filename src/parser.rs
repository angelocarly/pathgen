use glam::Vec2;
use crate::shape::Rect;

use svg::Document;
use svg::node::element::Path;
use svg::node::element::path::Data;

pub struct Parser {
}

fn param(input: Vec2) -> (f32, f32) {
    (input.x, input.y)
}

impl Parser {

    pub fn parse(input: Rect) {
        let data = Data::new()
            .move_to(param(input.p1))
            .line_to(param(input.p1 + Vec2::new(input.p2.x, 0.0)))
            .line_to(param(input.p1 + Vec2::new(input.p2.x, input.p2.y )))
            .line_to(param(input.p1 + Vec2::new(0.0, input.p2.y)))
            .close();

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 3)
            .set("d", data);

        let document = Document::new()
            .set("viewBox", (0, 0, 70, 70))
            .add(path);

        svg::save("image.svg", &document).unwrap();
    }
}