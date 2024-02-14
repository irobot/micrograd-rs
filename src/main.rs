use engine::Node;

use crate::{
    engine::topological_sort,
    serialization::{read_json_mlp, read_json_training_data},
};

mod data;
mod engine;
mod nn;
mod serialization;

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

    let examples: Vec<Vec<f64>> = vec![
        [2.0, 3.0, -1.],
        [3.0, -1., 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.],
    ]
    .iter()
    .map(|example| Vec::from(example))
    .collect();
    let y = Vec::from([1., -1., -1., 1.0]);
    let expectations: Vec<Vec<f64>> = y.iter().map(|gt| vec![*gt]).collect();

    println!("Simple example {:?}\n{:?}", examples, expectations);

    let training_set = data::TrainingSet(
        examples
            .iter()
            .zip(expectations.iter())
            .map(|(i, o)| data::TrainingSetItem {
                i: i.clone(),
                o: o.clone(),
            })
            .collect(),
    );

    if let Ok(mut mlp) = nn::MLP::new(3, vec![4, 4, 1]) {
        nn::train(&mut mlp, training_set, None, 30);
    };

    let mut mlp = read_json_mlp("mlp_init.json");
    // let mut mlp = MLP::new(2, vec![16, 16, 1]).unwrap();

    let training_set = read_json_training_data("moons.json");
    nn::train(&mut mlp, training_set, Some(100), 100);
}
