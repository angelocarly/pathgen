use vsvg::{DocumentTrait, LayerTrait, PageSize, Unit};

pub trait Parseable {
    fn parse(&self, doc: &mut vsvg::Document);
}

pub struct Parser {
}

impl Parser {

    pub fn parse(paths: Vec<Box<dyn Parseable>>) {

        let mut doc = vsvg::Document::new_with_page_size(PageSize::Custom(30., 30., Unit::Mm));

        /* == Layers == */
        let mut layer = vsvg::Layer::default();
        layer.metadata_mut().name = Option::from("Layer 2".to_string());

        for p in paths {
            p.parse(&mut doc);
        };

        // save to SVG
        doc.to_svg_file("target/path.svg").unwrap();
    }
}