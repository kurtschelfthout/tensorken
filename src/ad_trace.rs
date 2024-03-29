use crate::ad_ops::{BinaryDiffOp, UnaryDiffOp};
use std::cell::RefCell;
use std::fmt::Debug;

pub(crate) enum TracedOp<T> {
    Var,
    Unary(Box<dyn UnaryDiffOp<T>>, usize),
    BinaryDA(Box<dyn BinaryDiffOp<T>>, usize),
    BinaryDB(Box<dyn BinaryDiffOp<T>>, usize),
    Binary(Box<dyn BinaryDiffOp<T>>, usize, usize),
}

impl<T> Debug for TracedOp<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Var => write!(f, "Var"),
            Self::Unary(_, i) => write!(f, "Unary(_, {i})"),
            Self::Binary(_, i, j) => write!(f, "Binary(_, {i}, {j})"),
            Self::BinaryDA(_, i) => write!(f, "BinaryL(_, {i})"),
            Self::BinaryDB(_, i) => write!(f, "BinaryR(_, {i})"),
        }
    }
}

#[derive(Debug)]
pub struct Trace<T> {
    trace: RefCell<Vec<TracedOp<T>>>,
}

impl<T> Trace<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            trace: RefCell::new(vec![]),
        }
    }

    pub(crate) fn push_op(&self, node: TracedOp<T>) -> usize {
        let mut trace = self.trace.borrow_mut();
        let index = trace.len();
        trace.push(node);
        index
    }

    pub fn var(&self) -> usize {
        self.push_op(TracedOp::Var)
    }

    pub(crate) fn borrow(&self) -> std::cell::Ref<'_, Vec<TracedOp<T>>> {
        self.trace.borrow()
    }
}

impl<T: Default> Default for Trace<T> {
    fn default() -> Self {
        Self::new()
    }
}
