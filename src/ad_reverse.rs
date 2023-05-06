use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Add, Div, Index, Mul, Neg, Sub},
    ptr,
};

use crate::diffable_ops::{
    AddOp, BinaryOp, BinaryRevOp, CropOp, Diffable, DivOp, EqOp, ExpOp, ExpandOp, LogOp, MaxOp,
    MulOp, PadOp, PermuteOp, PowOp, ReshapeOp, SubOp, SumOp, UnaryOp, UnaryRevOp,
};

/// Reverse AD implementation.

enum TracedOp<'t, TTensor: 't> {
    Var,
    Unary(Box<dyn UnaryRevOp<TTensor> + 't>, usize),
    BinaryDA(Box<dyn BinaryRevOp<TTensor> + 't>, usize),
    BinaryDB(Box<dyn BinaryRevOp<TTensor> + 't>, usize),
    Binary(Box<dyn BinaryRevOp<TTensor> + 't>, usize, usize),
}

impl<T> Debug for TracedOp<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TracedOp::Var => write!(f, "Var"),
            TracedOp::Unary(_, i) => write!(f, "Unary(_, {i})"),
            TracedOp::Binary(_, i, j) => write!(f, "Binary(_, {i}, {j})"),
            TracedOp::BinaryDA(_, i) => write!(f, "BinaryL(_, {i})"),
            TracedOp::BinaryDB(_, i) => write!(f, "BinaryR(_, {i})"),
        }
    }
}

#[derive(Debug)]
pub struct Trace<'t, TTensor> {
    trace: RefCell<Vec<TracedOp<'t, TTensor>>>,
}

impl<'t, TTensor> Trace<'t, TTensor> {
    #[must_use]
    pub fn new() -> Self {
        Trace {
            trace: RefCell::new(vec![]),
        }
    }

    fn push_op(&self, primal: TTensor, node: TracedOp<'t, TTensor>) -> Reverse<'_, 't, TTensor> {
        let mut trace = self.trace.borrow_mut();
        let index = trace.len();
        trace.push(node);
        Reverse::Reverse(self, primal, index)
    }

    pub fn var(&self, primal: TTensor) -> Reverse<'_, 't, TTensor> {
        self.push_op(primal, TracedOp::Var)
    }
}

impl<T: Default> Default for Trace<'_, T> {
    fn default() -> Self {
        Trace::new()
    }
}

#[derive(Clone)] //needed for higher order derivatives
pub enum Reverse<'a, 't, T> {
    Lift(T),
    Reverse(&'a Trace<'t, T>, T, usize),
}

impl<T> Reverse<'_, '_, T> {
    pub fn lift(x: T) -> Self {
        Reverse::Lift(x)
    }

    fn into_primal(self) -> T {
        match self {
            Reverse::Lift(x) | Reverse::Reverse(_, x, _) => x,
        }
    }

    pub fn primal(&self) -> &T {
        match self {
            Reverse::Lift(x) | Reverse::Reverse(_, x, _) => x,
        }
    }
}

impl<T: Debug> Debug for Reverse<'_, '_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reverse::Lift(x) => write!(f, "Lift({x:?})"),
            Reverse::Reverse(_, x, i) => write!(f, "Reverse(_, {x:?}, {i})"),
        }
    }
}

impl<T: Default> Default for Reverse<'_, '_, T> {
    fn default() -> Self {
        Reverse::Lift(T::default())
    }
}

impl<T: PartialEq> PartialEq for Reverse<'_, '_, T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Lift(l0), Self::Lift(r0)) => l0 == r0,
            (Self::Reverse(_, l1, _), Self::Reverse(_, r1, _)) => l1 == r1,
            _ => false,
        }
    }
}

impl<'t, T: Diffable> Reverse<'_, 't, T> {
    fn unary<O: UnaryOp<T, Args = TArgs> + 't, TArgs: ?Sized>(&self, args: &TArgs) -> Self {
        let (op, primal) = O::f(self.primal(), args);
        match self {
            Reverse::Lift(_) => Reverse::Lift(primal),
            Reverse::Reverse(trace, _, tan) => {
                let op = TracedOp::Unary(Box::new(op), *tan);
                trace.push_op(primal, op)
            }
        }
    }

    fn binary<O: BinaryOp<T> + 't>(&self, rhs: &Self) -> Self {
        let (op, primal) = O::f(self.primal(), rhs.primal());
        match (self, rhs) {
            (Reverse::Lift(_), Reverse::Lift(_)) => Reverse::Lift(primal),
            (Reverse::Lift(_), Reverse::Reverse(trace, _, idx)) => {
                let op = TracedOp::BinaryDB(Box::new(op), *idx);
                trace.push_op(primal, op)
            }
            (Reverse::Reverse(trace, _, idx), Reverse::Lift(_)) => {
                let op = TracedOp::BinaryDA(Box::new(op), *idx);
                trace.push_op(primal, op)
            }
            (Reverse::Reverse(left_trace, _, left), Reverse::Reverse(right_trace, _, right)) => {
                assert!(ptr::eq(*left_trace, *right_trace), "traces must be the same - likely perturbation confusion. Are lifts in the right place?");
                let op = TracedOp::Binary(Box::new(op), *left, *right);
                left_trace.push_op(primal, op)
            }
        }
    }
}
impl<T: Clone + Diffable> Diffable for Reverse<'_, '_, T> {
    fn log(&self) -> Self {
        self.unary::<LogOp<T>, _>(&())
    }

    fn exp(&self) -> Self {
        self.unary::<ExpOp<T>, _>(&())
    }

    fn add(&self, rhs: &Self) -> Self {
        self.binary::<AddOp>(rhs)
    }

    fn sub(&self, rhs: &Self) -> Self {
        self.binary::<SubOp>(rhs)
    }

    fn mul(&self, rhs: &Self) -> Self {
        self.binary::<MulOp<T>>(rhs)
    }

    fn div(&self, rhs: &Self) -> Self {
        self.binary::<DivOp<T>>(rhs)
    }

    fn pow(&self, rhs: &Self) -> Self {
        self.binary::<PowOp<T>>(rhs)
    }

    fn eq(&self, other: &Self) -> Self {
        self.binary::<EqOp>(other)
    }

    fn sum(&self, axes: &[usize]) -> Self {
        self.unary::<SumOp, _>(axes)
    }

    fn max(&self, axes: &[usize]) -> Self {
        self.unary::<MaxOp<T>, _>(axes)
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        self.unary::<ReshapeOp, _>(shape)
    }

    fn permute(&self, dims: &[usize]) -> Self {
        self.unary::<PermuteOp, _>(dims)
    }

    fn expand(&self, shape: &[usize]) -> Self {
        self.unary::<ExpandOp, _>(shape)
    }

    fn pad(&self, padding: &[(usize, usize)]) -> Self {
        self.unary::<PadOp, _>(padding)
    }

    fn crop(&self, limits: &[(usize, usize)]) -> Self {
        self.unary::<CropOp, _>(limits)
    }

    fn zeros_like(&self) -> Self {
        Reverse::lift(self.primal().zeros_like())
    }

    fn ones_like(&self) -> Self {
        Reverse::lift(self.primal().ones_like())
    }

    fn shape(&self) -> &[usize] {
        self.primal().shape()
    }
}

crate::tensor::impl_bin_op!(Add, add, Reverse<'a, 't, T: Diffable + Clone>);
crate::tensor::impl_bin_op!(Sub, sub, Reverse<'a, 't, T: Diffable + Clone>);
crate::tensor::impl_bin_op!(Mul, mul, Reverse<'a, 't, T: Diffable + Clone>);
crate::tensor::impl_bin_op!(Div, div, Reverse<'a, 't, T: Diffable + Clone>);

impl<'a, 't, T: Clone + Diffable> Reverse<'a, 't, T> {
    fn neg(&self) -> Self {
        self.zeros_like().sub(self)
    }
}

crate::tensor::impl_un_op!(Neg, neg, Reverse<'a, 't, T: Diffable + Clone>);

#[derive(Debug)]
struct Grad<T> {
    grad: Option<Vec<T>>,
}

#[derive(Debug)]
struct Adjoints<T> {
    adjoints: Vec<Option<T>>,
}

impl<T: Diffable + Clone> Adjoints<T> {
    fn new(len: usize) -> Self {
        Adjoints {
            adjoints: vec![None; len],
        }
    }
    fn update(&mut self, idx: usize, dfda: T) {
        self.adjoints[idx] = self.adjoints[idx]
            .as_ref()
            .map(|c| c.add(&dfda))
            .or(Some(dfda));
    }
    fn pop(&mut self) {
        self.adjoints.pop();
    }
}

impl<T> Index<usize> for Adjoints<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        self.adjoints[idx].as_ref().unwrap()
    }
}

impl<T: Diffable + Clone> Grad<T> {
    pub fn of(result: &Reverse<T>) -> Self {
        match result {
            Reverse::Reverse(trace, primal, var) => {
                let trace = trace.trace.borrow();
                let mut adjoints = Adjoints::new(var + 1);
                adjoints.adjoints[*var] = Some(primal.ones_like());

                // backpropagate
                for i in (0..=*var).rev() {
                    if adjoints.adjoints[i].is_none() {
                        // no gradient to propagate - this node makes no contribution.
                        continue;
                    }
                    let node = &trace[i];
                    match node {
                        TracedOp::Var => {
                            // vars are always at the start of the trace (see vjp)
                            // and don't contribute to each other. So we can stop now.
                            break;
                        }
                        TracedOp::Unary(op, a) => adjoints.update(*a, op.df_dfda(&adjoints[i])),
                        TracedOp::Binary(op, a, b) => {
                            adjoints.update(*a, op.df_dfda(&adjoints[i]));
                            adjoints.update(*b, op.df_dfdb(&adjoints[i]));
                        }
                        TracedOp::BinaryDA(op, a) => {
                            adjoints.update(*a, op.df_dfda(&adjoints[i]));
                        }
                        TracedOp::BinaryDB(op, b) => {
                            adjoints.update(*b, op.df_dfdb(&adjoints[i]));
                        }
                    }
                    adjoints.pop();
                }

                Self {
                    grad: Some(
                        adjoints
                            .adjoints
                            .into_iter()
                            // zeros is correct, but we don't know the shape.
                            // Perhaps grad should contain options as well?
                            .map(|x| x.unwrap_or(primal.zeros_like()))
                            .collect(),
                    ),
                }
            }
            Reverse::Lift(_) => Self {
                // signal that the result was a constant, so we have no trace.
                grad: None,
            },
        }
    }

    pub fn get(&self, vars: &[&Reverse<T>]) -> Vec<T> {
        vars.iter()
            .map(|rev| match rev {
                Reverse::Reverse(_, p, var) => self
                    .grad
                    .as_ref()
                    .map_or(p.zeros_like(), |v| v[*var].clone()),
                Reverse::Lift(x) => x.zeros_like(),
            })
            .collect()
    }
}

fn wrap_slice<'a, 't, TTensor: Clone>(
    primal: &[&TTensor],
    trace: &'a Trace<'t, TTensor>,
) -> Vec<Reverse<'a, 't, TTensor>> {
    primal.iter().map(|&ati| trace.var(ati.clone())).collect()
}

/// Compute the result and the vector-Jacobian product of a function at the given point.
#[allow(clippy::missing_panics_doc)]
pub fn vjp1<'t, TTensor: Diffable + Clone + 't, F>(f: F, at: &TTensor) -> (TTensor, TTensor)
where
    for<'a> F: Fn(&'a Reverse<'a, 't, TTensor>) -> Reverse<'a, 't, TTensor>,
{
    let trace = Trace::new();

    let owned_vars = wrap_slice(&[at], &trace);
    let vars: Vec<_> = owned_vars.iter().collect();

    let result = f(vars[0]);

    let grad = Grad::of(&result);
    (
        result.into_primal(),
        // the unwrap is "fine" here - this should only panic if
        // something is seriously wrong with the implementation.
        grad.get(&vars).into_iter().next().unwrap(),
    )
}

// Compute the result and the vector-Jacobian product of a function at the given points.
pub fn vjpn<'t, TTensor: Diffable + Clone + 't, F>(f: F, at: &[&TTensor]) -> (TTensor, Vec<TTensor>)
where
    for<'a> F: Fn(&'a [&'a Reverse<'a, 't, TTensor>]) -> Reverse<'a, 't, TTensor>,
{
    let trace = Trace::new();

    let owned_vars = wrap_slice(at, &trace);
    let vars: Vec<_> = owned_vars.iter().collect();

    let result = f(&vars);

    let grad = Grad::of(&result);
    (result.into_primal(), grad.get(&vars))
}
