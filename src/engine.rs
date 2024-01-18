use std::{
    cell::RefCell,
    collections::HashSet,
    fmt, ops,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

type Data = f64;

pub struct Val {
    pub id: usize,
    pub name: String,
    pub data: Data,
    pub grad: RefCell<Data>,
    pub op: Box<dyn Op>,
}

pub trait Op: Backprop {
    fn get_inputs(&self) -> Vec<&Val>;
}

pub trait Backprop: std::fmt::Display {
    fn backward(&self, grad: Data);
}

struct Add {
    left: Rc<Val>,
    right: Rc<Val>,
}

impl Backprop for Add {
    fn backward(&self, grad: Data) {
        self.left.add_grad(grad);
        self.right.add_grad(grad);
    }
}

impl Op for Add {
    fn get_inputs(&self) -> Vec<&Val> {
        vec![self.left.as_ref(), self.right.as_ref()]
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ld = self.left.data;
        let rd = self.right.data;
        write!(f, "= {} + {}", ld, rd)
    }
}

struct Mul {
    left: Rc<Val>,
    right: Rc<Val>,
}

impl Backprop for Mul {
    fn backward(&self, grad: Data) {
        self.left.add_grad(grad * self.right.data);
        self.right.add_grad(grad * self.left.data);
    }
}

impl Op for Mul {
    fn get_inputs(&self) -> Vec<&Val> {
        vec![self.left.as_ref(), self.right.as_ref()]
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ld = self.left.data;
        let rd = self.right.data;
        write!(f, "= {} * {}", ld, rd)
    }
}

struct Relu {
    value: Rc<Val>
}

fn relu(v: Data) -> Data {
    if v > 0. { v } else { 0. }
}

impl Backprop for Relu {
    fn backward(&self, grad: Data) {
        self.value.add_grad(grad * relu(self.value.data));
    }
}

impl Op for Relu {
    fn get_inputs(&self) -> Vec<&Val> {
        vec![self.value.as_ref()]
    }
}

impl fmt::Display for Relu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "= ReLU({})", self.value.data)
    }
}

struct Noop {}

impl Backprop for Noop {
    fn backward(&self, _: Data) {
        ()
    }
}

impl Op for Noop {
    fn get_inputs(&self) -> Vec<&Val> {
        vec![]
    }
}

const NOOP: Noop = Noop {};

impl fmt::Display for Noop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

pub fn topological_sort(v: &Val) -> Vec<&Val> {
    let mut topo: Vec<&Val> = vec![];
    let mut visited = HashSet::<usize>::new();
    let mut stack: Vec<&Val> = vec![];

    stack.push(v);
    visited.insert(v.id);

    while stack.len() > 0 {
        // last is guaranteed to be Some
        // due to the condition in the line above
        let head = *stack.last().unwrap();
        let inputs = head.op.get_inputs(); 
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

impl Val {
    pub fn make(
        v: Data,
        name: Option<String>,
        op: Box<dyn Op>,
    ) -> Val {
        let id = OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst);
        Val {
            id,
            name: name.unwrap_or(id.to_string()),
            data: v,
            grad: RefCell::new(0.),
            op,
        }
    }

    pub fn new(v: Data) -> Val {
        Val::make(v, None, Box::new(NOOP))
    }

    pub fn named(v: Data, name: &str) -> Val {
        Val::make(v, Some(String::from(name)), Box::new(NOOP))
    }

    pub fn from_op(v: Data, op: Box<dyn Op>) -> Val {
        Val::make(v, None, op)
    }

    pub fn add_grad(&self, delta_grad: Data) {
        *(self.grad.borrow_mut()) += delta_grad;
    }

    pub fn relu(self) -> Val {
        let data = self.data;
        Val::from_op(relu(data), Box::new(Relu { value: Rc::new(self)}))
    }

    pub fn backward(&self) {
        let topo = topological_sort(self);

        for v in topo.iter().rev() {
            v.op.backward(*(v.grad.borrow()));
        }
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grad = self.grad.borrow();
        write!(
            f,
            "{} => Value({} {}, grad={})",
            self.name, self.data, self.op, grad
        )
    }
}

impl ops::Mul for Val {
    type Output = Val;

    fn mul(self, _rhs: Val) -> Val {
        let left = Rc::new(self);
        let right = Rc::new(_rhs);
        Val::from_op(
            left.data * right.data,
            Box::new(Mul { left, right }),
        )
    }
}

impl ops::Mul<f64> for Val {
    type Output = Val;

    fn mul(self, _rhs: f64) -> Val {
        self * Val::new(_rhs)
    }
}

impl ops::Add for Val {
    type Output = Val;

    fn add(self, _rhs: Val) -> Val {
        let left = Rc::new(self);
        let right = Rc::new(_rhs);
        Val::from_op(
            left.data + right.data,
            Box::new(Add { left, right }),
        )
    }
}

impl ops::Add<f64> for Val {
    type Output = Val;

    fn add(self, _rhs: f64) -> Val {
        self + Val::new(_rhs)
    }
}

#[cfg(test)]
mod test {

    use std::rc::Rc;

    use crate::engine::{topological_sort, Val};

    #[test]
    fn test_topo_sort() {
        println!("Hi!");

        let a = Val::named(1., "a");
        let b = Val::named(2., "b");
        let mut c = a + b;
        c.name = String::from("c");

        let d = Val::named(3., "d");
        let e = Val::named(4., "e");
        let mut f = d * e;
        f.name = String::from("f");
        let mut g = f + c;
        g.name = String::from("g");

        let topo = topological_sort(&g);
        let mut order = String::from("");
        for v in topo.iter() {
            order.push_str(&v.name);
        }
        assert_eq!(order, "defabcg");
    }

    #[test]
    fn test_scalar_add() {
        assert_eq!((Val::new(1.) + 2.).data, 3.);
    }

    #[test]
    fn test_scalar_mul() {
        assert_eq!((Val::new(7.) * 6.).data, 42.);
    }

    #[test]
    fn test_relu_pos() {
        assert_eq!(Val::new(10.).relu().data, 10.);
    }

    #[test]
    fn test_relu_neg() {
        assert_eq!(Val::new(-10.).relu().data, 0.);
    }

    #[test]
    fn test_sanity_check() {

        // let xrc = Rc::new(Val::new(-4.0));
        // let x = xrc.clone();
        // let z = *x * 2.0 + *x + 2.;
        // let q = z.relu() + z * *x;
        // let h = (z * z).relu();
        // let y = h + q + q * *x;
        // y.backward();

        // // forward pass went well
        // assert_eq!(y.data, 5.);
        // // backward pass went well
        // assert_eq!(*(x.grad.borrow()), 5.);
    }
}
