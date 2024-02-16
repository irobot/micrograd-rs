use std::{collections::HashMap, rc::Rc};

use crate::nn::Neuron;
use crate::{
    data::{
        ExprInit, GraphInit, LayerInit, MLPInit, NeuronInit, NodeInit, OpInit, TrainingSet,
        ValueInit,
    },
    engine::{Expr, NeuronId, Node, Value},
    nn::{Layer, MLP},
};

// Translate serialized ids to materialized `Value` ids
pub type IdMap = HashMap<u32, Node>;

pub trait FromInit<I> {
    fn from_init(init: &I, id_map: &mut IdMap) -> Self;
}

impl FromInit<ValueInit> for Node {
    fn from_init(init: &ValueInit, id_map: &mut IdMap) -> Self {
        let node = Node::from_value(Value::from_neuron_parameter(init.data, None));
        id_map.insert(init.id, node.clone());
        node
    }
}

impl FromInit<ExprInit> for Node {
    fn from_init(init: &ExprInit, id_map: &mut IdMap) -> Self {
        let n = |id: u32| {
            id_map
                .get(&id)
                .unwrap_or_else(|| panic!("missing node id {}", id))
                .clone()
        };
        let op = match init.op {
            OpInit::Add(l, r) => Expr::Add(n(l), n(r)),
            OpInit::Mul(l, r) => Expr::Mul(n(l), n(r)),
            OpInit::Pow(l, r) => Expr::Mul(n(l), n(r)),
            OpInit::Relu(l) => Expr::Relu(n(l)),
            OpInit::Neg(l) => Expr::Mul(n(l), Node::new(-1.)),
        };
        let node = Node::from_op(Box::new(op));
        id_map.insert(init.id, node.clone());
        node
    }
}

impl FromInit<NodeInit> for Node {
    fn from_init(init: &NodeInit, id_map: &mut IdMap) -> Self {
        let node = match init {
            NodeInit::Value(vi) => Node::from_init(vi, id_map),
            NodeInit::Expr(ei) => Node::from_init(ei, id_map),
        };
        node
    }
}

impl FromInit<GraphInit> for Node {
    fn from_init(init: &GraphInit, id_map: &mut IdMap) -> Self {
        let nodes: Vec<Node> = init.0.iter().map(|n| Node::from_init(n, id_map)).collect();
        nodes.last().unwrap().clone()
    }
}

impl FromInit<NeuronInit> for Neuron {
    fn from_init(neuron_init: &NeuronInit, id_map: &mut IdMap) -> Neuron {
        let mut inputs = vec![];
        let mut weights = vec![];
        for wi in neuron_init.weights.iter() {
            if let Some(input) = id_map.get(&wi.input_node_id) {
                inputs.push(input.clone());
            } else {
                let input = Node::new(0.);
                id_map.insert(wi.input_node_id, input.clone());
                inputs.push(input);
            }

            let node = Node::from_init(&wi.value, id_map);
            weights.push(node);
        }

        let bias = Node::from_init(&neuron_init.bias, id_map);
        Neuron::new(weights, bias)
    }
}

impl FromInit<LayerInit> for Layer {
    fn from_init(layer_init: &LayerInit, id_map: &mut IdMap) -> Layer {
        Layer(
            layer_init
                .neurons
                .iter()
                .map(|ni| Rc::new(Neuron::from_init(ni, id_map)))
                .collect(),
        )
    }
}

fn get_data(node_init: &NodeInit, label: &str) -> f64 {
    match node_init {
        NodeInit::Value(vi) => vi.data,
        NodeInit::Expr(_) => panic!("Expected a Value for {}, not an expression", label),
    }
}

fn from_neuron_init(nid: NeuronId, neuron_init: &NeuronInit) -> Neuron {
    let weights = neuron_init
        .weights
        .iter()
        .map(|wi| {
            Node::from_value(Value::from_neuron_parameter(
                get_data(&wi.value, "weight"),
                Some(nid),
            ))
        })
        .collect();
    let bias = Node::from_value(Value::from_neuron_parameter(
        neuron_init.bias.data,
        Some(nid),
    ));
    Neuron::new(weights, bias)
}

pub fn from_layer_init(layer_idx: usize, layer_init: &LayerInit) -> Layer {
    let mut neurons = vec![];
    for idx in 0..layer_init.neurons.len() {
        let ni = &layer_init.neurons[idx];
        neurons.push(Rc::new(from_neuron_init((layer_idx, idx), ni)));
    }

    Layer(neurons)
}

pub fn from_mlp_init(mlp_init: &MLPInit) -> MLP {
    let mut layers = vec![];
    let layer_count = mlp_init.layers.len();
    for layer_idx in 0..layer_count {
        let layer = from_layer_init(layer_idx, &mlp_init.layers[layer_idx]);
        layers.push(layer);
    }

    MLP { layers }
}

#[allow(dead_code)]
pub fn read_json_graph(filename: &str) -> Node {
    let file = std::fs::File::open(filename).expect("file should open read only");
    let graph_init: GraphInit = serde_json::from_reader(file).expect("file should be proper JSON");
    let mut id_map = IdMap::new();
    Node::from_init(&graph_init, &mut id_map)
}

pub fn read_json_mlp(filename: &str) -> MLP {
    let file = std::fs::File::open(filename).expect("file should open read only");
    let mlp_init = serde_json::from_reader(file).expect("file should be proper JSON");
    from_mlp_init(&mlp_init)
}

pub fn read_json_training_data(filename: &str) -> TrainingSet {
    let file = std::fs::File::open(filename).expect("file should open read only");
    let training_set: TrainingSet =
        serde_json::from_reader(file).expect("file should be proper JSON");
    training_set
}
