use svg::node::element::Path;

use svg::{Document};

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

        svg::save("target/image.svg", &document).unwrap();
    }
}