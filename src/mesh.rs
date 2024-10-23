use glam::{Vec3};

enum Topology {
    Lines
}

/*
 * Raw path data, used for mesh alterations
 */
pub struct Path {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vec3>,
    pub topology: Topology
}

pub fn gen_dodecahedron() -> Path {

    let phi = (1. + f32::sqrt(5.) ) / 2.;

    let a = 0.5;
    let b = 0.5 * 1. / phi;
    let c = 0.5 * (2. - phi);

    let vertices: Vec<Vec3> = vec![
        Vec3::new( c, 0.,  a),
        Vec3::new(-c,  0.,  a),
        Vec3::new(-b,  b,  b),
        Vec3::new( 0.,  a,  c),
        Vec3::new( b,  b,  b),
        Vec3::new( b, -b,  b),
        Vec3::new( 0., -a,  c),
        Vec3::new(-b, -b,  b),
        Vec3::new( c,  0., -a),
        Vec3::new(-c,  0., -a),
        Vec3::new(-b, -b, -b),
        Vec3::new( 0., -a, -c),
        Vec3::new( b, -b, -b),
        Vec3::new( b,  b, -b),
        Vec3::new( 0.,  a, -c),
        Vec3::new(-b,  b, -b),
        Vec3::new( a,  c,  0.),
        Vec3::new(-a,  c,  0.),
        Vec3::new(-a, -c,  0.),
        Vec3::new( a, -c,  0.)
    ];

    let indices: Vec<u32> = vec![
         0,  1,
         0,  4,
         0,  5,
         1,  2,
         1,  7,
         2,  3,
         2, 17,
         3,  4,
         3, 14,
         4, 16,
         5,  6,
         5, 19,
         6,  7,
         6, 11,
         7, 18,
         8,  9,
         8, 12,
         8, 13,
         9, 10,
         9, 15,
        10, 11,
        10, 18,
        11, 12,
        12, 19,
        13, 14,
        13, 16,
        14, 15,
        15, 17,
        16, 19,
        17, 18
    ];

    Path {
        indices,
        vertices,
        topology: Topology::Lines
    }
}
