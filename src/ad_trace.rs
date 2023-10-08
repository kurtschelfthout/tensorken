use crate::ad_ops::{BinaryDiffOp, UnaryDiffOp};
use std::cell::RefCell;
use std::fmt::Debug;

pub(crate) enum TracedOp<'t, TTensor: 't> {
    Var,
    Unary(Box<dyn UnaryDiffOp<TTensor> + 't>, usize),
    BinaryDA(Box<dyn BinaryDiffOp<TTensor> + 't>, usize),
    BinaryDB(Box<dyn BinaryDiffOp<TTensor> + 't>, usize),
    Binary(Box<dyn BinaryDiffOp<TTensor> + 't>, usize, usize),
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

    pub(crate) fn push_op(&self, node: TracedOp<'t, TTensor>) -> usize {
        let mut trace = self.trace.borrow_mut();
        let index = trace.len();
        trace.push(node);
        index
    }

    pub fn var(&self) -> usize {
        self.push_op(TracedOp::Var)
    }

    pub(crate) fn borrow(&self) -> std::cell::Ref<'_, Vec<TracedOp<'t, TTensor>>> {
        self.trace.borrow()
    }
}

impl<T: Default> Default for Trace<'_, T> {
    fn default() -> Self {
        Trace::new()
    }
}
