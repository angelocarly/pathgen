use glam::{DVec2};
use vsvg::DocumentTrait;
use vsvg::exports::kurbo;
use crate::export::parser::Parseable;

pub struct Exx {
    pub p1: DVec2,
    pub p2: DVec2,
    pub p3: DVec2,
}

impl Parseable for Exx {
    fn parse(&self, doc: &mut vsvg::Document) {

        // push a path to layer 1
        doc.push_path(1, vec![
            (self.p1.x, self.p1.y),
            (self.p2.x, self.p2.y),
            (self.p3.x, self.p3.y)
        ]);

        // Normals from p2 out
        let n1 = (self.p2 - self.p1).normalize();
        let n2 = (self.p2 - self.p3).normalize();

        let len = 0.7;
        let l1 = (self.p2 - self.p1).length() * len;
        let q1 = self.p2 - n1 * l1;

        let l2 = (self.p2 - self.p3).length() * len;
        let q2 = self.p2 - n2 * l2;
        let path = kurbo::QuadBez::new((q1.x, q1.y), (self.p2.x, self.p2.y), (q2.x, q2.y));
        doc.push_path(1, path)
    }
}