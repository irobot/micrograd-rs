use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Display},
    ops::{self, Deref},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

type Data = f64;

pub struct Value {
    pub id: usize,
    pub data: RefCell<Data>,
    pub grad: RefCell<Data>,
    pub expr: Box<dyn Backprop>,
}

#[derive(Clone)]
pub struct Node {
    pub node: Rc<Value>,
    pub name: String,
    pub train_state: Option<Vec<Node>>,
}

pub enum Expr {
    Leaf,
    Const(Node),
    Add(Node, Node),
    Mul(Node, Node),
    Relu(Node),
}

pub trait Backprop: Display {
    fn backward(&self, grad: Data);
    fn get_inputs(&self) -> Vec<&Node>;
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
                let d = if input.data() >= 0. { 1. } else { 0. };
                input.add_grad(grad * d)
            }
            Expr::Const(_) => (),
            Expr::Leaf => (),
        }
    }
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
            Expr::Const(v) => write!(f, "{}", v.data()),
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
    visited.insert(v.id);

    while stack.len() > 0 {
        // last is guaranteed to be Some
        // due to the condition in the line above
        let head = *stack.last().unwrap();
        let inputs = head.expr.get_inputs();
        let next = inputs.iter().find(|c| !visited.contains(&c.id));
        if let Some(h) = next {
            stack.push(h);
            visited.insert(h.id);
            continue;
        }
        topo.push(&head);
        stack.pop();
    }
    topo
}

static OBJECT_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl Value {
    pub fn make(data: Data, expr: Box<dyn Backprop>) -> Value {
        let id = OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst);
        Value {
            id,
            data: RefCell::new(data),
            grad: RefCell::new(0.),
            expr,
        }
    }

    pub fn new(v: Data) -> Value {
        Value::make(v, Box::new(Expr::Leaf))
    }

    pub fn from_op(v: Data, op: Box<dyn Backprop>) -> Value {
        Value::make(v, op)
    }

    pub fn grad(&self) -> Data {
        *(self.grad.borrow())
    }

    pub fn add_grad(&self, delta_grad: Data) {
        *(self.grad.borrow_mut()) += delta_grad;
    }

    pub fn set_grad(&self, grad: Data) {
        *self.grad.borrow_mut() = grad;
    }

    pub fn data(&self) -> Data {
        *self.data.borrow()
    }

    pub fn update(&self) {
        *(self.data.borrow_mut()) += self.grad();
    }

    pub fn backward(&self) {
        self.expr.backward(self.grad());
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grad = self.grad.borrow();
        write!(
            f,
            "{} => Value({} {}, grad={})",
            self.name,
            self.data(),
            self.expr,
            grad
        )
    }
}

impl Default for Node {
    fn default() -> Self {
        Self {
            node: Rc::new(Value::new(0.)),
            name: Default::default(),
            train_state: None,
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
            train_state: None,
        }
    }

    pub fn named(v: Data, name: &str) -> Node {
        Node {
            node: Rc::new(Value::new(v)),
            name: name.to_string(),
            train_state: None,
        }
    }

    pub fn name(&mut self, name: &str) -> Node {
        Node {
            node: self.node.clone(),
            name: name.to_string(),
            train_state: None,
        }
    }

    pub fn constant(&self) -> Node {
        Node {
            node: Rc::new(Value::from_op(
                self.data(),
                Box::new(Expr::Const(self.clone())),
            )),
            name: self.id.to_string(),
            train_state: None,
        }
    }

    pub fn relu(&self) -> Node {
        Node {
            node: Rc::new(Value::from_op(
                relu(self.data()),
                Box::new(Expr::Relu(self.clone())),
            )),
            name: self.id.to_string(),
            train_state: None,
        }
    }

    pub fn backward(&mut self) {
        if self.train_state.is_none() {
            return;
        }
        let topo = self.train_state.as_deref().unwrap();
        for v in topo.iter() {
            v.set_grad(0.);
        }
        self.set_grad(1.);
        for v in topo.iter().rev() {
            v.node.backward();
        }
    }

    pub fn update(&self) {
        let topo = self.train_state.as_deref().unwrap();
        for v in topo.iter() {
            v.node.update();
        }
    }

    pub fn mark_output(&mut self) {
        let sorted = topological_sort(self);
        self.train_state = Some(sorted.iter().map(|n| n.to_owned().clone()).collect());
    }
}

impl Deref for Node {
    type Target = Rc<Value>;
    fn deref(&self) -> &Self::Target {
        &self.node
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
            Box::new(Expr::Mul(self.clone(), _rhs.clone())),
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
        Node::from(Value::from_op(
            self.data() + _rhs.data(),
            Box::new(Expr::Add(self.clone(), _rhs.clone())),
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
        let mut y = ((h + &q).name("y0") + (q * &x).name("y1")).name("y");
        y.mark_output();
        y.backward();

        // forward pass went well
        assert_eq!(y.data(), -20.);
        // backward pass went well
        assert_eq!(*(x.grad.borrow()), 46.);
    }
}
