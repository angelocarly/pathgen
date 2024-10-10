use std::fmt::format;
use glam::{DVec2};
use vsvg::{DocumentTrait, PathTrait};
use vsvg::exports::kurbo;
use vsvg::exports::kurbo::Shape;
use crate::export::parser::Parseable;

pub struct Exx {
    pub p1: DVec2,
    pub p2: DVec2,
    pub p3: DVec2,
}

impl Parseable for Exx {
    fn parse(&self, doc: &mut vsvg::Document) {

        // Normals from p2 out
        let n1 = (self.p2 - self.p1).normalize();
        let n2 = (self.p2 - self.p3).normalize();

        doc.push_path(1, vec![
            (self.p1.x, self.p1.y),
            (self.p2.x, self.p2.y),
            (self.p3.x, self.p3.y),
        ]);

        for len in 0..30 {

            let l1 = f64::min((self.p2 - self.p1).length(), len as f64);
            let q1 = self.p2 - n1 * l1;

            let l2 = f64::min((self.p2 - self.p3).length(), len as f64);
            let q2 = self.p2 - n2 * l2;

            let mut path = vsvg::Path::from_svg(
                format!(
                    "M {} {} Q {} {} {} {}",
                    q1.x, q1.y,
                    self.p2.x, self.p2.y,
                    q2.x, q2.y,
                ).as_str()).unwrap();
            // path.metadata_mut().stroke_width = 0.1;

            doc.push_path(1, path)
        }
    }
}