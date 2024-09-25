use glam::Vec2;

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