use crate::engine::{Val, topological_sort};

mod engine;

fn main() {
    let a = Val::named(1., "a");
    let b = Val::named(2., "b");
    let mut c = a + b;
    c.name = String::from("c");

    let e = Val::named(4., "e");
    let f = Val::named(5., "f");
    let mut g = e * f;
    g.name = String::from("g");

    let mut h = c * g;
    h.name = String::from("h");
    *h.grad.borrow_mut() = 1.0;
    h.backward();

    let topo = topological_sort(&h);
    for v in topo.iter() {
        println!("{}", v);
    }
}
