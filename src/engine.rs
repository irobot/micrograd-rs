use std::{
    cell::RefCell,
    collections::HashSet,
    fmt,
    ops,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

type Data = f64;

pub struct Value {
    pub id: usize,
    pub data: Data,
    pub grad: RefCell<Data>,
    pub op: Expr,
}

#[derive(Clone)]
pub struct Node {
    pub node: Rc<Value>,
    pub name: String,
}

pub enum Expr {
    Leaf,
    Const(Node),
    Add(Node, Node),
    Mul(Node, Node),
    Relu(Node),
}

pub trait Op: Backprop {
    fn get_inputs(&self) -> Vec<&Node>;
}

pub trait Backprop: std::fmt::Display {
    fn backward(&self, grad: Data);
}

impl Backprop for Expr {
    fn backward(&self, grad: Data) {
        match self {
            Expr::Add(left, right) => {
                left.add_grad(grad);
                right.add_grad(grad);
            }
            Expr::Mul(left, right) => {
                left.add_grad(grad * right.data());
                right.add_grad(grad * left.data());
            }
            Expr::Relu(input) => {
                let d =  if input.data() >= 0. { 1.} else { 0. };
                input.add_grad(grad * d)
            },
            Expr::Const(_) => (),
            Expr::Leaf => (),
        }
    }
}

impl Op for Expr {
    fn get_inputs(&self) -> Vec<&Node> {
        match self {
            Expr::Add(left, right) => vec![left, right],
            Expr::Mul(left, right) => vec![left, right],
            Expr::Relu(input) => vec![input],
            Expr::Const(_) => vec![],
            Expr::Leaf => vec![],
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Relu(input) => write!(f, "= ReLU({})", input.data()),
            Expr::Add(left, right) => write!(f, "= {} + {}", left.data(), right.data()),
            Expr::Mul(left, right) => write!(f, "= {} * {}", left.data(), right.data()),
            Expr::Const(v) => write!(f, "{}", v.node.data),
            Expr::Leaf => Result::Ok(()),
        }
    }
}

fn relu(v: Data) -> Data {
    if v > 0. {
        v
    } else {
        0.
    }
}

pub fn topological_sort(v: &Node) -> Vec<&Node> {
    let mut topo: Vec<&Node> = vec![];
    let mut visited = HashSet::<usize>::new();
    let mut stack: Vec<&Node> = vec![];

    stack.push(v);
    visited.insert(v.node.id);

    while stack.len() > 0 {
        // last is guaranteed to be Some
        // due to the condition in the line above
        let head = *stack.last().unwrap();
        let inputs = head.node.op.get_inputs();
        let next = inputs.iter().find(|c| !visited.contains(&c.node.id));
        if let Some(h) = next {
            stack.push(h);
            visited.insert(h.node.id);
            continue;
        }
        topo.push(&head);
        stack.pop();
    }
    topo
}

static OBJECT_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl Value {
    pub fn make(data: Data, op: Expr) -> Value {
        let id = OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst);
        Value {
            id,
            data,
            grad: RefCell::new(0.),
            op,
        }
    }

    pub fn new(v: Data) -> Value {
        Value::make(v, Expr::Leaf)
    }

    pub fn from_op(v: Data, op: Expr) -> Value {
        Value::make(v, op)
    }

    pub fn add_grad(&self, delta_grad: Data) {
        *(self.grad.borrow_mut()) += delta_grad;
    }

    pub fn set_grad(&self, grad: Data) {
        *self.grad.borrow_mut() = grad;
    }

    pub fn backward(&self) {
        self.op.backward(*(self.grad.borrow()));
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grad = self.node.grad.borrow();
        write!(
            f,
            "{} => Value({} {}, grad={})",
            self.name, self.node.data, self.node.op, grad
        )
    }
}

impl Default for Node {
    fn default() -> Self {
        Self {
            node: Rc::new(Value::new(0.)),
            name: Default::default(),
        }
    }
}

impl Node {
    pub fn new(v: Data) -> Node {
        Node::from(Value::new(v))
    }

    pub fn from(v: Value) -> Node {
        let name = v.id.to_string();
        Node {
            node: Rc::new(v),
            name,
        }
    }

    pub fn named(v: Data, name: &str) -> Node {
        Node {
            node: Rc::new(Value::new(v)),
            name: name.to_string(),
        }
    }

    pub fn name(&mut self, name: &str) -> Node {
        Node {
            node: self.node.clone(),
            name: name.to_string(),
        }
    }

    pub fn data(&self) -> Data {
        self.node.data
    }

    pub fn relu(&self) -> Node {
        let data = self.data();
        let name = self.node.id.to_string();
        Node {
            node: Rc::new(Value::from_op(relu(data), Expr::Relu(self.clone()))),
            name,
        }
    }

    fn add_grad(&self, delta_grad: Data) {
        self.node.add_grad(delta_grad)
    }

    pub fn backward(&self) {
        let topo = topological_sort(self);
        for v in topo.iter() {
            v.node.set_grad(0.);
        }
        self.node.set_grad(1.);
        for v in topo.iter().rev() {
            v.node.backward();
        }
    }
}

impl ops::Mul for Node {
    type Output = Node;
    fn mul(self, _rhs: Node) -> Node {
        &self * &_rhs
    }
}

impl ops::Mul<Node> for &Node {
    type Output = Node;
    fn mul(self, _rhs: Node) -> Node {
        self * &_rhs
    }
}

impl ops::Mul<&Node> for Node {
    type Output = Node;
    fn mul(self, _rhs: &Node) -> Node {
        &self * _rhs
    }
}

impl ops::Mul for &Node {
    type Output = Node;
    fn mul(self, _rhs: &Node) -> Node {
        Node::from(Value::from_op(
            self.data() * _rhs.data(),
            Expr::Mul(self.clone(), _rhs.clone()),
        ))
    }
}

impl ops::Mul<f64> for &Node {
    type Output = Node;
    fn mul(self, _rhs: f64) -> Node {
        self * &Node::from(Value::new(_rhs))
    }
}

impl ops::Mul<f64> for Node {
    type Output = Node;
    fn mul(self, _rhs: f64) -> Node {
        &self * _rhs
    }
}


impl ops::Add<Node> for Node {
    type Output = Node;
    fn add(self, _rhs: Node) -> Node {
        &self + &_rhs
    }
}

impl ops::Add<Node> for &Node {
    type Output = Node;
    fn add(self, _rhs: Node) -> Node {
        self + &_rhs
    }
}

impl ops::Add<&Node> for Node {
    type Output = Node;
    fn add(self, _rhs: &Node) -> Node {
        &self + _rhs
    }
}

impl ops::Add for &Node {
    type Output = Node;
    fn add(self, _rhs: &Node) -> Node {
        let left = self;
        let right = _rhs;
        Node::from(Value::from_op(
            left.data() + right.data(),
            Expr::Add(left.clone(), right.clone()),
        ))
    }
}

impl ops::Add<f64> for &Node {
    type Output = Node;
    fn add(self, _rhs: f64) -> Node {
        self + &Node::from(Value::new(_rhs))
    }
}

impl ops::Add<f64> for Node {
    type Output = Node;
    fn add(self, _rhs: f64) -> Node {
        &self + _rhs
    }
}

#[cfg(test)]
mod test {

    use crate::engine::{topological_sort, Node};

    #[test]
    fn test_topo_sort() {
        let a = Node::named(1., "a");
        let b = Node::named(2., "b");
        let c = (a + b).name("c");

        let d = Node::named(3., "d");
        let e = Node::named(4., "e");
        let f = (d * e).name("f");
        let g = (f + c).name("g");

        let topo = topological_sort(&g);
        let mut order = String::from("");
        for v in topo.iter() {
            order.push_str(&v.name);
        }
        assert_eq!(order, "defabcg");
    }

    #[test]
    fn test_scalar_add() {
        assert_eq!((Node::new(1.) + 2.).data(), 3.);
    }

    #[test]
    fn test_scalar_mul() {
        assert_eq!((Node::new(7.) * 6.).data(), 42.);
    }

    #[test]
    fn test_relu_pos() {
        assert_eq!(Node::new(10.).relu().data(), 10.);
    }

    #[test]
    fn test_relu_neg() {
        assert_eq!(Node::new(-10.).relu().data(), 0.);
    }

    #[test]
    fn test_sanity_check() {
        let x = Node::named(-4.0, "x");
        let z = ((&x * 2.0).name("z0") + (&x + 2.).name("z1")).name("z");
        let q = (&z.relu() + (&z * &x).name("q1")).name("q");
        let h = ((&z * &z).relu()).name("h");
        let y = ((h + &q).name("y0") + (q * &x).name("y1")).name("y");
        y.backward();

        // forward pass went well
        assert_eq!(y.data(), -20.);
        // backward pass went well
        assert_eq!(*(x.node.grad.borrow()), 46.);
    }
}
