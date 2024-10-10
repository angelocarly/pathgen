use crate::export::parser::Parseable;

pub struct WorldGen {
    objects: Vec<Box<dyn Parseable>>
}

impl WorldGen {

    pub(crate) fn new() -> WorldGen {
        WorldGen {
            objects: Vec::new()
        }
    }

    fn add_line() {

    }

    pub fn objects(&self) -> &Vec<Box<dyn Parseable>> {
        &self.objects
    }
}