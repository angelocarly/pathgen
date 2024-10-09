mod shape;
mod export;

use crate::export::parser::{Parseable, Parser};
use crate::shape::Exx;
use glam::DVec2;
use rand::random;

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

        if( p.x >= f64::min(a.x,b.x) && p.x <= f64::max(a.x,b.x)
            && p.y >= f64::min(a.y,b.y) && p.y <= f64::max(a.y,b.y)
            && p.x >= f64::min(c.x,d.x) && p.x <= f64::max(c.x,d.x)
            && p.y >= f64::min(c.y,d.y) && p.y <= f64::max(c.y,d.y) ) {

            return Some(p)
        }
    }

    None
}

fn main() {
    let basepos = DVec2::new(10.0, 10.0);

    let mut objects: Vec<Box<dyn Parseable>> = Vec::new();

    let scale = 100.;
    let a = basepos + DVec2::new(random::<f64>() * 100., random::<f64>() * 100.);
    let b = basepos + DVec2::new(random::<f64>() * 100., random::<f64>() * 100.);
    let c = basepos + DVec2::new(random::<f64>() * 100., random::<f64>() * 100.);
    let d = basepos + DVec2::new(random::<f64>() * 100., random::<f64>() * 100.);

    if let Some(r) = intersect_lines(a, b, c, d) {
        objects.push(Box::new(Exx{
            p1: a,
            p2: r,
            p3: c
        }));
        objects.push(Box::new(Exx{
            p1: b,
            p2: r,
            p3: d
        }));
    } else {
        if let Some(r) = intersect_lines(a, c, b, d) {
            objects.push(Box::new(Exx {
                p1: a,
                p2: r,
                p3: b
            }));
            objects.push(Box::new(Exx {
                p1: c,
                p2: r,
                p3: d
            }));
        }
    }

    Parser::parse(objects);
}
