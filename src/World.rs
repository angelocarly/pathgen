use crate::random;
use crate::DVec2;
use crate::export::parser::Parseable;
use crate::shape::Exx;
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

pub struct Line<'a> {
    pub p1: DVec2,
    pub p2: DVec2,
    s1: Option<Intersect<'a>>,
    s2: Option<Intersect<'a>>,
}

pub struct Intersect<'a> {
    p: DVec2,
    l1: &'a Line<'a>,
    l2: &'a Line<'a>,
    l3: &'a Line<'a>,
    l4: &'a Line<'a>,
}

pub struct WorldGen<'a> {
    lines: Vec<Line<'a>>,
    intersects: Vec<Intersect<'a>>
}

impl WorldGen {

    pub(crate) fn new() -> WorldGen {
        WorldGen {
            lines: Vec::new(),
            intersects: Vec::new(),
        }
    }

    pub fn add_line(&mut self, line: Line) -> bool {
        assert!(self.lines.len() > 0);

        let mut intersect = false;

        for i in self.lines.len() {
            let l = self.lines[i];

            if let Some(p) = intersect_lines(line.p1, line.p2, l.p1, l.p2) {

                let l1 = Line { p1: line.p1, p2: p, s1: None, s2: None };
                let l2 = Line { p1: p, p2: line.p2, s1: None, s2: None };
                let l3 = Line { p1: l.p1, p2: p, s1: None, s2: None };
                let l4 = Line { p1: p, p2: l.p2, s1: None, s2: None };

                // Update intersects

                self.lines.remove(l);
                i -= 1;

                self.intersects.push(Intersect {
                    p,
                    l1: &l1,
                    l2: &l2,
                    l3: &l3,
                    l4: &l4,
                });

                self.lines.push(l1);
                self.lines.push(l2);
                self.lines.push(l3);
                self.lines.push(l4);

                intersect = true;
            }
        }

        if intersect {
            self.lines.push(line);
        }

        intersect
    }

    pub fn convert(&self) -> Vec<Box<dyn Parseable>> {
        let mut objects: Vec<Box<dyn Parseable>> = Vec::new();
        objects.push(Box::new(Exx{
            p1: line.p1,
            p2: r,
            p3: l.p1
        }));
        objects.push(Box::new(Exx{
            p1: line.p2,
            p2: r,
            p3: l.p2
        }));
        objects
    }
}