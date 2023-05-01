use tensorken::tensor::Cpu32;

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
    // how to make an eye
    let_example!(dim, 3);
    do_example!(&Tr::eye(dim));
    let_example!(t, &Tr::scalar(1.0));
    let_example!(t, t.pad(&[(0, dim)]));
    let_example!(t, t.reshape(&[1, dim + 1]));
    let_example!(t, t.expand(&[dim, dim + 1]));
    let_example!(t, t.reshape(&[dim * (dim + 1)]));
    let_example!(t, t.crop(&[(0, dim * dim)]));
    let_example!(t, t.reshape(&[dim, dim]));
}
