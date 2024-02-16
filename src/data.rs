use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ValueInit {
    pub id: u32,
    pub data: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum NodeInit {
    Value(ValueInit),
    Expr(ExprInit)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExprInit {
    pub id: u32,
    pub op: OpInit,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum OpInit {
    // tuples are node init ids
    Add(u32, u32),
    Mul(u32, u32),
    Pow(u32, u32),
    Relu(u32),
    Neg(u32),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphInit(pub Vec<NodeInit>);

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WeightInit {
    pub value: NodeInit, 
    pub input_node_id: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NeuronInit {
    pub bias: ValueInit,
    pub weights: Vec<WeightInit>,
    pub nonlin: Option<NodeInit>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LayerInit {
    pub neurons: Vec<NeuronInit>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MLPInit {
    pub layers: Vec<LayerInit>,
    // input node ids
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
}


#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingSetItem {
    // Example inputs
    pub i: Vec<f64>,
    // Expected output(s)
    pub o: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingSet(pub Vec<TrainingSetItem>);