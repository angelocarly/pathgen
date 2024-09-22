use svg::node::element::Path;
use glam::Vec2;
use crate::shape::Rect;

use svg::{Document, Node};

pub trait Parseable {
    fn parse(&self) -> Path;
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

        svg::save("image.svg", &document).unwrap();
    }
}