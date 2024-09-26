use std::cmp::min;
use glam::Vec2;
use svg::node::element::Path;
use crate::export::parser::Parseable;

pub struct Rect {
    pub pos: Vec2,
    pub size: Vec2,
}

pub struct Line {
    pub p1: Vec2,
    pub p2: Vec2,
}

pub struct QuadraticBezier {
    pub p1: Vec2,
    pub c: Vec2,
    pub p2: Vec2,
}

pub struct LineStrip {
    pub points: Vec<Vec2>,
}

pub struct Exx {
    pub p1: Vec2,
    pub p2: Vec2,
    pub p3: Vec2,
}

impl Parseable for Exx {
    fn parse(&self) -> Path {
        let mut data = svg::node::element::path::Data::new()
            .move_to((self.p1.x, self.p1.y))
            .line_to((self.p2.x, self.p2.y))
            .line_to((self.p3.x, self.p3.y));

        // Normals from p2 out
        let n1 = (self.p2 - self.p1).normalize();
        let n2 = (self.p2 - self.p3).normalize();

        let len = 1000.;
        let l1 = f32::min(len, (self.p2 - self.p1).length());
        let q1 = self.p2 - n1 * l1;

        let l2 = f32::min(len, (self.p2 - self.p3).length());
        let q2 = self.p2 - n2 * l2;
        data = data
            .move_to((q1.x, q1.y))
            .quadratic_curve_to((
                self.p2.x, self.p2.y,
                q2.x, q2.y
            ));

        svg::node::element::Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 0.3f32)
            .set("d", data)
    }
}