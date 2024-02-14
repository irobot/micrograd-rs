use std::fs::File;
use std::io::{Error, Write};
use std::iter::Sum;
use std::{fmt::Display, rc::Rc};

use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

use crate::data::{TrainingSet, TrainingSetItem};
use crate::engine::{topological_sort, Data, Node, Value};

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<Node>,
    bias: Node,
}

pub fn make_random_nodes(count: usize, node_type: &str) -> Vec<Node> {
    let uniform = Uniform::from(-1.0..1.0);
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let v = rng.sample(uniform);
            Node::make(Value::new(v), "", node_type)
        })
        .into_iter()
        .collect()
}

pub type Nonlinearity = Option<fn(n: &Node) -> Node>;

impl Neuron {
    pub fn new(weights: Vec<Node>, bias: Node) -> Neuron {
        Neuron { weights, bias }
    }

    pub fn add_inputs(&self, inputs: &Vec<Node>, nonlinearity: Nonlinearity) -> Node {
        let weighted: Vec<(&Node, &Node)> = inputs.iter().zip(self.weights.iter()).collect();

        let linear_output = weighted
            .iter()
            .map(|(input, weight)| *weight * *input)
            .reduce(|acc, v| acc + v)
            .unwrap()
            + &self.bias;

        if let Some(nonlin) = nonlinearity {
            nonlin(&linear_output)
        } else {
            linear_output
        }
    }

    pub fn parameters(&self) -> Vec<Node> {
        [self.weights.clone(), vec![self.bias.clone()]].concat()
    }

    pub fn with_n_inputs(input_count: usize) -> Neuron {
        Neuron::from_inputs(&make_random_nodes(input_count, "inp"))
    }

    #[allow(dead_code)]
    pub fn from_inputs(inputs: &Vec<Node>) -> Neuron {
        let weights = make_random_nodes(inputs.len(), "w");
        let bias = make_random_nodes(1, "b")[0].clone();
        Neuron::new(weights, bias)
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
            "Neuron: weights\n{}\nbias {}",
            format_values(&self.weights),
            short_format_value(&self.bias),
        )
    }
}

#[derive(Debug)]
pub struct Layer(pub Vec<Rc<Neuron>>);

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Layer {
        Layer(
            (0..nout)
                .map(|_| Rc::new(Neuron::with_n_inputs(nin)))
                .collect(),
        )
    }

    pub fn add_inputs(&self, inputs: &Vec<Node>, nonlin: Nonlinearity) -> Vec<Node> {
        self.0
            .iter()
            .map(|n| n.add_inputs(inputs, nonlin))
            .collect()
    }

    pub fn parameters(&self) -> Vec<Node> {
        self.0.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Result<MLP, &'static str> {
        if nin < 1 {
            return Err("At least one input is needed in the first layer of the MLP!");
        }

        if nouts.len() < 1 || nouts[0] < 1 {
            return Err("One or more output layers are needed");
        }

        let mut layers = vec![Layer::new(nin, nouts[0])];

        for i in 1..nouts.len() {
            let nout = nouts[i];
            layers.push(Layer::new(nouts[i - 1], nout));
        }

        Ok(MLP { layers })
    }

    pub fn add_inputs(&self, inputs: &Vec<Data>) -> Vec<Node> {
        let mut prev_inputs = inputs.iter().map(|i| Node::new(*i)).collect();
        let last_layer_idx = self.layers.len() - 1;
        let mut idx = 0;
        for layer in self.layers.iter() {
            let nonlin: Nonlinearity = if idx == last_layer_idx {
                None
            } else {
                Some(Node::relu)
            };
            idx += 1;
            prev_inputs = layer.add_inputs(&prev_inputs, nonlin);
        }
        prev_inputs
    }

    pub fn parameters(&self) -> Vec<Node> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

fn print_expr_json(node: &Node) -> String {
    let op = node.expr.op_name();
    let inputs = node.expr.get_inputs();
    let ids = inputs
        .iter()
        .map(|n| format!("{},", n.id))
        .reduce(|a, b| a + &b)
        .unwrap_or_default();
    format!("{{ op: '{}', c: [{}] }}", op, ids)
}

fn print_node_json(node: &Node) -> String {
    format!(
        "{{ id: {}, v: {}, g: {}, expr: {} }},",
        node.id,
        node.data(),
        node.grad(),
        print_expr_json(node)
    )
}

#[allow(dead_code)]
pub fn print_graph_json(node: &Node) {
    let sorted = topological_sort(node);
    let mut result = String::from("");
    for n in sorted.iter().rev() {
        result.push_str(&print_node_json(n));
    }
    println!("[{}]", result);
}

fn print_expr_dot(node: &Node) -> String {
    let inputs = node.expr.get_inputs();
    let ids = inputs
        .iter()
        .map(|n| format!("{} ", n.id))
        .reduce(|a, b| a + &b)
        .unwrap_or_default();
    format!("{{ {} }}", ids)
}

fn print_node_dot(node: &Node) -> String {
    let op_name = node.expr.op_name();
    let op = if op_name == "[leaf]" { "" } else { op_name };
    let shape = match node.node_type.as_str() {
        "op" => "trapezium",
        "w" => "box",
        "inp" => "house",
        "o" => "square",
        _ => "oval",
    };
    format!(
        "{} [label=\"{} {} {}&#92;nd {:.2} | g {:?}\", shape=\"{}\"]\n{} -> {}\n",
        node.id,
        node.name,
        node.node_type,
        op,
        node.data(),
        node.grad(),
        shape,
        node.id,
        print_expr_dot(node)
    )
}

pub fn format_graph_dot(node: &Node) -> String {
    let sorted = topological_sort(node);
    let mut result = String::from("");
    for n in sorted.iter().rev() {
        result.push_str(&print_node_dot(n));
    }
    format!("digraph \"graph\" {{{}}}", result)
}

#[allow(dead_code)]
pub fn write_graph_to_file(node: &Node) -> Result<(), Error> {
    let dot = format_graph_dot(node);
    let path = "graph.dot";
    let mut output = File::create(path).unwrap();
    let res = write!(output, "{}", dot);
    res
}

#[allow(dead_code)]
pub fn print_graph(node: &Node) {
    let sorted = topological_sort(node);
    let mut result = String::from("");
    for n in sorted.iter().rev() {
        result.push_str(&print_node_json(n));
    }
    println!("[{}]", result);
}

pub fn train(
    mlp: &mut MLP,
    training_set: TrainingSet,
    batch_size: Option<usize>,
    max_iterations: usize,
) {
    let mut rng = thread_rng();
    let full_training_set = training_set
        .0
        .iter()
        .map(|ti| ti)
        .collect::<Vec<&TrainingSetItem>>();

    let parameters = mlp.parameters();
    println!("Number of parameters: {}", parameters.len());

    // L2 regularization
    // @note In contrast with the original, the reg-loss nodes
    // are built only once, outside of the training loop.
    // Reg-loss is still being evaluated at every training step.
    let alpha = 1e-4;
    let reg_loss_0 = parameters.iter().map(|p| p * p).sum::<Node>();
    let mut reg_loss = reg_loss_0 * alpha;
    reg_loss.name = "RLoss".to_string();

    for i in 0..max_iterations {
        let ts = if let Some(bs) = batch_size {
            training_set
                .0
                .choose_multiple(&mut rng, bs)
                .collect::<Vec<&TrainingSetItem>>()
        } else {
            full_training_set.clone()
        };

        let learning_rate = 1.0 - 0.9 * (i as f64) / (100. as f64);
        let mut data_losses = vec![];

        for item in ts.iter() {
            let output_node = &mlp.add_inputs(&item.i)[0];
            let ground_truth = item.o[0];

            // svm "max-margin" loss
            // @note to self - this relies on the fact that the expected output is always
            // either 1 or -1, I think. So ideally expected output * actual output = 1
            let loss = (-(output_node * Node::new(ground_truth)) + 1.).relu();
            data_losses.push(loss);
        }

        let data_loss = Node::sum(data_losses.into_iter()) * (1. / ts.len() as f64);
        let mut total_loss = &reg_loss + &data_loss;
        total_loss.mark_output();
        total_loss.forward();

        total_loss.zero_grad();
        total_loss.backward();
        for p in parameters.iter() {
            p.node.update(learning_rate);
        }
        println!(
            "{} total loss: {}, reg loss: {}, data loss: {}",
            i,
            total_loss.data(),
            reg_loss.data(),
            data_loss.data()
        );
    }

    // mlp.outputs[0].mark_output();
    // write_graph_to_file(&mlp.outputs[0]).ok();
    // write_graph_to_file(&total_loss).ok();
}

#[cfg(test)]
mod test {
    use crate::{engine::Node, nn::print_graph};

    use super::Neuron;

    #[test]
    fn test_neuron_output() {
        let weights: Vec<Node> = vec![1., 2., 3.].iter().map(|v| Node::new(*v)).collect();
        let bias = Node::new(4.);
        let n = Neuron::new(weights.clone(), bias.clone());
        let inputs = vec![5., 6., 7.].iter().map(|v| Node::new(*v)).collect();
        let computed_output = n.add_inputs(&inputs, None);

        let actual_output = weights
            .iter()
            .zip(inputs)
            .map(|(a, b)| a * b)
            .reduce(|acc, v| acc + v)
            .unwrap()
            + &bias;

        assert_eq!(computed_output.data(), actual_output.data());
    }

    #[test]
    fn test_neuron_update() {
        let weights = (2..=4).map(|v| Node::new(f64::from(v))).collect();
        let bias = Node::new(0.);

        let n = Neuron::new(weights, bias);
        let inputs = vec![Node::new(1.), Node::new(2.), Node::new(3.)];
        let mut output = n.add_inputs(&inputs, Some(Node::relu));
        output.mark_output();

        output.forward();
        print_graph(&output);
        assert_eq!(output.data(), 20.);
        let computed_output = output.data();
        assert_eq!(computed_output, 20.);

        output.backward();
        output.update(1.);
        assert_eq!(inputs[0].grad(), 2.);
        assert_eq!(inputs[0].data(), -1.);
    }
}
