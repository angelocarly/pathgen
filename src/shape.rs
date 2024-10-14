use std::fmt::format;
use glam::{DVec2};
use rand::random;
use vsvg::{DocumentTrait, PathTrait};
use vsvg::exports::usvg::Node::Path;
use crate::export::parser::Parseable;

pub struct Exx {
    pub p1: DVec2,
    pub p2: DVec2,
    pub p3: DVec2,
}

impl Parseable for Exx {
    fn parse(&self, doc: &mut vsvg::Document) {

        if self.p2 == self.p3 || self.p1 == self.p2 { return; }

        // Normals from p2 out
        let n1 = (self.p2 - self.p1).normalize();
        let n2 = (self.p2 - self.p3).normalize();

        // doc.push_path(1, vec![
        //     (self.p1.x, self.p1.y),
        //     (self.p2.x, self.p2.y),
        //     (self.p3.x, self.p3.y),
        // ]);

        let len = 50.;
        let iterations = 7;
        for i in 0..iterations {

            let offset = len * (i as f64 / iterations as f64);
            let l1 = f64::min((self.p2 - self.p1).length(), offset);
            let q1 = self.p2 - n1 * l1;

            let l2 = f64::min((self.p2 - self.p3).length(), offset);
            let q2 = self.p2 - n2 * l2;

            if l1 == (self.p2 - self.p1).length() && l2 == (self.p2 - self.p3).length() {
                break;
            }

            println!("Adding path ({},{}) ({},{}) ({},{})", q1.x, q1.y, self.p2.x, self.p2.y, q2.x, q2.y);

            let path = vsvg::Path::from_svg(
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

#[derive(Clone)]
pub struct Line {
    pub p1: DVec2,
    pub p2: DVec2,
}

impl Parseable for Line {
    fn parse(&self, doc: &mut vsvg::Document) {
        let mut path = vsvg::Path::from_svg(format!(
            "M {} {} L {} {}",
            self.p1.x, self.p1.y,
            self.p2.x, self.p2.y
        ).as_str()).unwrap();

        path.metadata_mut().color = vsvg::Color::new(random::<u8>(), random::<u8>(), random::<u8>(), 255);
        doc.push_path(1, path );
    }
}

#[derive(Clone)]
pub struct Point {
    pub p1: DVec2,
}

impl Parseable for Point {
    fn parse(&self, doc: &mut vsvg::Document) {
        let r = 3.;
        doc.push_path(1, vec![
            (self.p1.x - r, self.p1.y - r),
            (self.p1.x + r, self.p1.y - r),
            (self.p1.x + r, self.p1.y + r),
            (self.p1.x - r, self.p1.y + r),
            (self.p1.x - r, self.p1.y - r),
        ]);
    }
}
