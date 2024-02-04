use std::{fmt::Display, rc::Rc};

use rand::random;

use crate::engine::{Data, Node};

pub struct Neuron {
    inputs: Vec<Node>,
    weights: Vec<Node>,
    bias: Node,
    output: Node,
}

pub fn make_random_nodes(count: usize) -> Vec<Node> {
    (0..count)
        .map(|_| Node::new(random()))
        .into_iter()
        .collect()
}

type Nonlinearity = Option<fn(n: &Node) -> Node>;

impl Neuron {
    pub fn new(
        inputs: &Vec<Node>,
        weights: Vec<Node>,
        bias: Node,
        nonlinearity: Nonlinearity,
    ) -> Neuron {
        let linear_output = inputs
            .iter()
            .zip(weights.iter())
            .map(|(input, weight)| input * weight)
            .reduce(|acc, v| acc + v)
            .unwrap()
            + &bias;

        let output = if let Some(nonlin) = nonlinearity {
            nonlin(&linear_output)
        } else {
            linear_output
        };

        Neuron {
            inputs: inputs.clone(),
            weights,
            bias,
            output,
        }
    }

    pub fn parameters(&self) -> Vec<Node> {
        [self.weights.clone(), vec![self.bias.clone()]].concat()
    }

    pub fn with_n_inputs(input_count: usize, nonlinearity: Nonlinearity) -> Neuron {
        let inputs = make_random_nodes(input_count);
        let weights = make_random_nodes(input_count);
        let bias = Node::new(random());
        Neuron::new(&inputs, weights, bias, nonlinearity)
    }

    #[allow(dead_code)]
    pub fn from_inputs(inputs: &Vec<Node>, nonlinearity: Nonlinearity) -> Neuron {
        let weights = make_random_nodes(inputs.len());
        let bias = Node::new(random());
        Neuron::new(inputs, weights, bias, nonlinearity)
    }

    pub fn set_inputs(&self, inputs: Vec<Data>) {
        for (input, data) in self.inputs.iter().zip(inputs) {
            input.set(data);
        }
    }
}

fn short_format_value(v: &Node) -> String {
    format!("[{} | {}]\n", v.data(), v.grad())
}

fn format_values(vals: &Vec<Node>) -> String {
    let mut result = String::new();
    for v in vals.iter() {
        result.push_str(short_format_value(v).as_str());
    }
    result
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Neuron: inputs\n{} weights\n{}\nbias {}\nout {}",
            format_values(&self.inputs),
            format_values(&self.weights),
            short_format_value(&self.bias),
            short_format_value(&self.output)
        )
    }
}

struct Layer(Vec<Rc<Neuron>>);

impl Layer {
    pub fn new(inputs: &Vec<Node>, nout: usize, nonlin: Nonlinearity) -> Layer {
        Layer(
            (0..nout)
                .map(|_| Rc::new(Neuron::from_inputs(inputs, nonlin)))
                .collect(),
        )
    }

    pub fn outputs(&self) -> Vec<Node> {
        self.0.iter().map(|neuron| neuron.output.clone()).collect()
    }

    pub fn parameters(&self) -> Vec<Node> {
        let mut result = vec![];
        for n in self.0.iter() {
            result.append(&mut n.parameters());
        }
        result
    }
}

pub struct MLP {
    inputs: Vec<Node>,
    outputs: Vec<Node>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Result<MLP, &'static str> {
        if nin < 1 {
            return Err("At least one input is needed in the first layer of the MLP!");
        }

        if nouts.len() < 1 || nouts[0] < 1 {
            return Err("One or more output layers are needed");
        }

        let inputs = (0..nin).map(|_| Node::new(0.)).collect();
        let mut layers = vec![Layer::new(&inputs, nouts[0], Some(Node::relu))];

        for i in 1..nouts.len() {
            let nout = nouts[i];
            let nonlin: Nonlinearity = if i == (nouts.len() - 1) {
                None
            } else {
                Some(Node::relu)
            };
            layers.push(Layer::new(&layers[i - 1].outputs(), nout, nonlin));
        }

        let outputs = layers.last().unwrap().outputs();

        Ok(MLP { inputs, outputs })
    }
}

pub fn print_graph(n: &Node) {
    println!("{} {}", n.data(), n.expr);
    for c in n.expr.get_inputs() {
        print_graph(c);
    }
}

pub fn train(mlp: &mut MLP, max_iterations: usize) {
    let examples = vec![
        [2.0, 3.0, -1.],
        [3.0, -1., 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.],
    ];
    let expectations = vec![[1.], [-1.], [-1.], [1.0]];

    let goals: Vec<Node> = mlp.outputs.iter().map(|_| Node::default()).collect();

    let mut score = mlp
        .outputs
        .iter()
        .zip(goals.iter())
        .map(|(o, g)| (o + -g) * (o + -g))
        .reduce(|a, b| a + b)
        .unwrap();

    score.mark_output();

    for _i in 0..max_iterations {
        let mut loss = 0.0;
        for (example, expected) in examples.iter().zip(&expectations) {
            for (input_node, input) in mlp.inputs.iter().zip(example) {
                input_node.set(*input);
            }
            for (goal_node, goal) in goals.iter().zip(expected) {
                goal_node.set(*goal);
            }

            score.forward();
            loss += score.data();
            score.backward();
            score.update();
        }
        println!("Iteration loss: {}", loss);
    }
}

#[cfg(test)]
mod test {
    use crate::{engine::Node, nn::print_graph};

    use super::Neuron;

    #[test]
    fn test_neuron_output() {
        let n = Neuron::with_n_inputs(3, None);
        let weights = n.weights.iter().map(|n| n.data());
        let bias = n.bias.data();
        let inputs = n.inputs.iter().map(|n| n.data());
        let computed_output = n.output.data();

        let actual_output = weights
            .zip(inputs)
            .map(|(a, b)| a * b)
            .reduce(|acc, v| acc + v)
            .unwrap()
            + bias;

        assert_eq!(computed_output, actual_output);
    }

    #[test]
    fn test_neuron_update() {
        let inputs = (1..=3).map(|_| Node::default()).collect();
        let weights = (2..=4).map(|v| Node::new(f64::from(v))).collect();
        let bias = Node::new(0.);

        let mut n = Neuron::new(&inputs, weights, bias, Some(Node::relu));
        n.output.mark_output();
        n.set_inputs(vec![1., 2., 3.]);

        // assert_eq!(inputs[0].data(), 1.);
        n.output.forward();
        print_graph(&n.output);
        assert_eq!(n.output.data(), 20.);
        let computed_output = n.output.data();
        assert_eq!(computed_output, 20.);

        n.output.backward();
        n.output.update();
        assert_eq!(inputs[0].grad(), 2.);
        assert_eq!(inputs[0].data(), -1.);
    }
}
