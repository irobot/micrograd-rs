use engine::Node;
use nn::Neuron;

use crate::engine::topological_sort;

mod engine;
mod nn;

fn main() {
    let a = Node::named(1., "a");
    let b = Node::named(2., "b");
    let c = (a + b).name("c");

    let e = Node::named(4., "e");
    let f = Node::named(5., "f");
    let g = (e * f).name("g");

    let h = (c * g).name("h");

    let mut i = h.relu();
    i.mark_output();
    i.backward();

    let topo = topological_sort(&i);
    for v in topo.iter() {
        println!("{}", v);
    }

    let neuron = Neuron::with_n_inputs(3);
    println!("{}", neuron);
}
