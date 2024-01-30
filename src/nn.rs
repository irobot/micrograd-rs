use std::fmt::Display;

use rand::random;

use crate::engine::Node;

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

impl Neuron {
    pub fn new(inputs: Vec<Node>, weights: Vec<Node>, bias: Node) -> Neuron {
        let output = inputs
            .iter()
            .zip(weights.iter())
            .map(|(input, weight)| input * weight)
            .reduce(|acc, v| acc + v)
            .unwrap()
            + &bias;

        Neuron {
            inputs,
            weights,
            bias,
            output,
        }
    }

    pub fn with_n_inputs(input_count: usize) -> Neuron {
        let inputs = make_random_nodes(input_count);
        let weights = make_random_nodes(input_count);
        let bias = Node::new(random());
        Neuron::new(inputs, weights, bias)
    }

    #[allow(dead_code)]
    pub fn from_inputs(inputs: Vec<Node>) -> Neuron {
        let weights = make_random_nodes(inputs.len());
        let bias = Node::new(random());
        Neuron::new(inputs, weights, bias)
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
            "Neuron: inputs\n{}weights\n{}\nbias {}\nout{}",
            format_values(&self.inputs),
            format_values(&self.weights),
            short_format_value(&self.bias),
            short_format_value(&self.output)
        )
    }
}

#[cfg(test)]
mod test {
    use crate::engine::Node;

    use super::Neuron;

    #[test]
    fn test_neuron_output() {
        let n = Neuron::with_n_inputs(3);
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
        let node0 = Node::new(1.);
        let inputs = vec![node0.clone(), Node::new(2.), Node::new(3.)];
        let weights = vec![Node::new(2.), Node::new(3.), Node::new(4.)];
        let bias = Node::new(0.);

        let mut n = Neuron::new(inputs, weights, bias);
        let computed_output = n.output.data();
        assert_eq!(computed_output, 20.);

        n.output.mark_output();
        n.output.backward();
        n.output.update();
        assert_eq!(node0.grad(), 2.);
        assert_eq!(node0.data(), 3.);
    }
}
