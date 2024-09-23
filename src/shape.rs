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
