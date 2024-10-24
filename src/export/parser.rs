use std::path::PathBuf;
use vsvg::{DocumentTrait, LayerTrait, PageSize};

pub trait Parseable {
    fn parse(&self, doc: &mut vsvg::Document);
}

pub struct Parser {
}

impl Parser {

    pub fn parse(paths: &[Box<dyn Parseable>], page_size: PageSize, path: PathBuf) {

        let mut doc = vsvg::Document::new_with_page_size(page_size);

        /* == Layers == */
        let mut layer = vsvg::Layer::default();
        layer.metadata_mut().name = Option::from("Layer 2".to_string());

        for p in paths {
            p.parse(&mut doc);
        };

        // save to SVG
        doc.to_svg_file(path).unwrap();
    }
}