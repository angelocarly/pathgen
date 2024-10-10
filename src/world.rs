use std::cmp::Ordering;
use crate::DVec2;
use crate::export::parser::Parseable;
use crate::shape::{Exx, Line};
/*
 * Intersection of two lines functions
 */
fn intersect_graph_lines(a1: f64, b1: f64, a2: f64, b2: f64) -> Option<DVec2> {
    if a1 == a2 {
        return None
    }

    let x = (b2 - b1) / (a1 - a2);
    let y = a1 * x + b1;

    Some(DVec2::new(x, y))
}

/*
 * Intersection of two lines defined by two points each
 */
fn intersect_lines(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> Option<DVec2> {
    let a1 = (b.y - a.y) / (b.x - a.x);
    let b1 = a.y - a1 * a.x;

    let a2 = (d.y - c.y) / (d.x - c.x);
    let b2 = c.y - a2 * c.x;

    if let Some(p) = intersect_graph_lines(a1, b1, a2, b2) {

        if p.x >= f64::min(a.x,b.x) && p.x <= f64::max(a.x,b.x)
            && p.y >= f64::min(a.y,b.y) && p.y <= f64::max(a.y,b.y)
            && p.x >= f64::min(c.x,d.x) && p.x <= f64::max(c.x,d.x)
            && p.y >= f64::min(c.y,d.y) && p.y <= f64::max(c.y,d.y) {

            return Some(p)
        }
    }

    None
}

pub struct WorldGen {
    vertices: Vec<Vertex>,
    lines: Vec< VLine >
}

#[derive(Debug)]
pub struct Vertex {
    pub p: DVec2
}

#[derive(PartialEq, Debug)]
pub struct VLine {
    pub i1: usize,
    pub i2: usize,
}


impl WorldGen {

    pub(crate) fn new() -> WorldGen {
        WorldGen {
            vertices: Vec::new(),
            lines: Vec::new(),
        }
    }

    pub fn add_line(&mut self, p1: DVec2, p2: DVec2) -> bool {
        let mut intersect = false;
        let mut line_vertices = Vec::new();

        let i1 = self.vertices.len();
        self.vertices.push(Vertex{ p: p1 });
        line_vertices.push(i1);
        let i2 = self.vertices.len();
        self.vertices.push(Vertex{ p: p2 });
        line_vertices.push(i2);

        let mut add_lines = Vec::new();

        // Split all existing lines
        for l in self.lines.iter_mut() {
            let v1 = &self.vertices[l.i1];
            let v2 = &self.vertices[l.i2];
            if let Some(p) = intersect_lines(p1, p2, v1.p, v2.p) {

                let x1 = self.vertices.len();
                self.vertices.push(Vertex {p});
                line_vertices.push(x1);

                // Resize the current line to the split
                let end = l.i2;
                l.i2 = x1;

                add_lines.push(VLine {
                    i1: x1,
                    i2: end,
                });

                intersect = true
            }
        }

        // Add the new lines to the lines
        self.lines.append(&mut add_lines);

        // Create the new line
        line_vertices.sort_by(|a, b| {
            let v1 = &self.vertices[*a];
            let v2 = &self.vertices[*b];

            if v1.p.x > v2.p.x {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });

        for i in 0..(line_vertices.len() - 1) {
            self.lines.push(VLine {
                i1: line_vertices[i],
                i2: line_vertices[i + 1],
            })
        }

        intersect
    }

    pub fn convert(&self) -> Vec<Box<dyn Parseable>> {
        let mut objects: Vec<Box<dyn Parseable>> = Vec::new();

        let mut drawn = Vec::new();

        for i in 0..self.vertices.len() {

            let mut joins = Vec::new();

            for li in 0..self.lines.len() {
                let line = &self.lines[li];

                if line.i1 == i { joins.push(line.i2) }
                if line.i2 == i { joins.push(line.i1) }
            }

            println!("{}", joins.len());

            // Draw all vertices
            //objects.push(Box::new(Point {
             //   p1: self.vertices[i].p,
            //}));

            // Draw all for now
            match joins.len() {
                1 => {
                    objects.push(Box::new(Line {
                        p1: self.vertices[joins[0]].p,
                        p2: self.vertices[i].p,
                    }));
                },
                4 => {
                    let p1 = self.vertices[joins[0]].p;
                    let p2 = self.vertices[joins[1]].p;
                    let v= self.vertices[i].p;
                    let r1 = (p1.y - v.y) / (v.x - p1.x);
                    let r2 = (p2.y - v.y) / (v.x - p2.x);

                    // if j0 and j1 are in line
                    if f64::abs(r1 - r2) < 0.1 {
                        // Case 1
                        objects.push(Box::new(Exx{
                            p1: self.vertices[joins[0]].p,
                            p2: self.vertices[i].p,
                            p3: self.vertices[joins[2]].p,
                        }));
                        objects.push(Box::new(Exx{
                            p1: self.vertices[joins[1]].p,
                            p2: self.vertices[i].p,
                            p3: self.vertices[joins[3]].p,
                        }));
                    } else {
                        // Case 2
                        objects.push(Box::new(Exx{
                            p1: self.vertices[joins[0]].p,
                            p2: self.vertices[i].p,
                            p3: self.vertices[joins[1]].p,
                        }));
                        objects.push(Box::new(Exx{
                            p1: self.vertices[joins[2]].p,
                            p2: self.vertices[i].p,
                            p3: self.vertices[joins[3]].p,
                        }));
                    }
                },
                _ => {}
            }

            drawn.append(&mut joins);

        }
        objects
    }
}