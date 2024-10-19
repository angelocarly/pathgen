use std::cmp::Ordering;
use crate::DVec2;
use crate::export::parser::Parseable;
use crate::shape::{Exx, Line, Point};
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
    vertices: Vec<DVec2>,
    lines: Vec< VLine >
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
        let intersect = false;
        let mut line_vertices = Vec::new();

        let i1;
        if !self.vertices.contains(&p1) {
            i1 = self.vertices.len();
            self.vertices.push(p1);
        } else {
            i1 = self.vertices.iter().position(|&r| { r == p1}).unwrap();
        }
        let i2;
        if !self.vertices.contains(&p2) {
            i2 = self.vertices.len();
            self.vertices.push(p2);
        } else {
            i2 = self.vertices.iter().position(|&r| { r == p2}).unwrap();
        }

        line_vertices.push(i1);
        line_vertices.push(i2);

        let mut add_lines = Vec::new();

        // Split all existing lines
        for l in self.lines.iter_mut() {
            // let v1 = &self.vertices[l.i1];
            // let v2 = &self.vertices[l.i2];
            // if let Some(p) = intersect_lines(p1, p2, *v1, *v2) {
            //
            //     let x1 = self.vertices.len();
            //     self.vertices.push(p);
            //     line_vertices.push(x1);
            //
            //     // Resize the current line to the split
            //     let end = l.i2;
            //     l.i2 = x1;
            //
            //     add_lines.push(VLine {
            //         i1: x1,
            //         i2: end,
            //     });
            //
            //     intersect = true
            // }
        }

        // Add the new lines to the lines
        self.lines.append(&mut add_lines);

        // Create the new line
        line_vertices.sort_by(|a, b| {
            let v1 = &self.vertices[*a];
            let v2 = &self.vertices[*b];

            if v1.x > v2.x {
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

        println!("Vertices: {:?}", self.vertices.len());
        println!("Lines: {:?}", self.lines.len());

        // for l in self.lines.iter() {
        //     objects.push(Box::new(Line {
        //         p1: self.vertices[l.i1],
        //         p2: self.vertices[l.i2],
        //     }));
        // }

        for i in 0..self.vertices.len() {

            let mut joins = Vec::new();

            for li in 0..self.lines.len() {
                let line = &self.lines[li];

                if line.i1 == i { joins.push(line.i2) }
                if line.i2 == i { joins.push(line.i1) }
            }

            // Draw all vertices
            // objects.push(Box::new(Point {
            //    p1: self.vertices[i],
            // }));

            // Draw all for now
            match joins.len() {
                2 => {
                    let p1 = self.vertices[joins[0]];
                    let p2 = self.vertices[joins[1]];
                    let v= self.vertices[i];

                    let r1 = (p1.y - v.y) / (v.x - p1.x);
                    let r2 = (p2.y - v.y) / (v.x - p2.x);


                    // if j0 and j1 are not continuous
                    if f64::abs(r1 - r2) > 0.01 {
                        // Case 1
                        objects.push(Box::new(Exx{
                            p1: self.vertices[joins[0]],
                            p2: self.vertices[i],
                            p3: self.vertices[joins[1]],
                        }));
                    }
                },
                3 => {
                    let p1 = self.vertices[joins[0]];
                    let p2 = self.vertices[joins[1]];
                    let p3 = self.vertices[joins[2]];
                    let v= self.vertices[i];

                    let r1 = (p1.y - v.y) / (v.x - p1.x);
                    let r2 = (p2.y - v.y) / (v.x - p2.x);
                    let r3 = (p3.y - v.y) / (v.x - p3.x);

                    // if j0 and j1 are not continuous
                    if f64::abs(r1 - r2) > 0.01 {
                        // Case 1
                        objects.push(Box::new(Exx{
                            p1: p1,
                            p2: v,
                            p3: p2,
                        }));
                    }

                    // if j0 and j1 are not continuous
                    if f64::abs(r1 - r3) > 0.01 {
                        // Case 2
                        objects.push(Box::new(Exx{
                            p1: p1,
                            p2: v,
                            p3: p3,
                        }));
                    }

                    // if j0 and j1 are not continuous
                    if f64::abs(r2 - r3) > 0.01 {
                        // Case 3
                        objects.push(Box::new(Exx{
                            p1: p2,
                            p2: v,
                            p3: p3,
                        }));
                    }
                },
                _ => {}
            }
        }
        objects
    }
}