use tensorken::{Cpu32, Diffable, DiffableExt};

/// A macro to print the result of an expression and the expression itself.
macro_rules! do_example {
    ($e:expr) => {
        println!(">>> {}", stringify!($e));
        let result = $e;
        println!("{result}");
    };
    ($e:expr, $debug:literal) => {
        println!(">>> {}", stringify!($e));
        let result = $e;
        println!("{result:?}");
    };
}

/// A macro to print the result of an expression, the expression itself,
/// and bind the result to a variable.
macro_rules! let_example {
    ($t:ident, $e:expr) => {
        println!(">>> {}", stringify!(let $t = $e));
        let $t = $e;
        println!("{}", $t);
    };
    ($t:ident, $e:expr, $debug:literal) => {
        println!(">>> {}", stringify!(let $t = $e));
        let $t = $e;
        println!("{:?}", $t);
    };
}

type Tr = Cpu32;

fn main() {
    let_example!(t, Tr::linspace(1.0, 20.0, 20_u8).reshape(&[4, 5]));
    do_example!(t.sum(&[0]));
    do_example!(t.crop(&[(0, 4), (0, 1)]));
    do_example!(t.sum(&[]));
}
