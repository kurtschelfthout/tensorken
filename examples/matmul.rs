use tensorken::{Cpu32, Diffable, DiffableExt, Shape};

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
    // how to multiply matrices, the hard way
    let_example!(l, Tr::linspace(0.0, 11.0, 12_u16).reshape(&[3, 4]));
    let_example!(r, Tr::linspace(12.0, 23.0, 12_u16).reshape(&[4, 3]));
    do_example!(&l.matmul(&r));

    // left's shape from [..., m, n] to [..., m, 1, n]
    let_example!(s, l.shape(), true);
    let_example!(
        l_shape,
        [&s[..s.ndims() - 1], &[1, s[s.ndims() - 1]]].concat(),
        true
    );
    let_example!(l, l.reshape(&l_shape));

    // right's shape from [..., n, o] to [..., 1, o, n]
    let_example!(s, r.shape(), true);
    let_example!(
        r_shape,
        [&s[..s.ndims() - 2], &[1], &s[s.ndims() - 2..]].concat(),
        true
    );
    let_example!(
        r,
        r.reshape(&r_shape)
            .transpose(r_shape.ndims() - 1, r_shape.ndims() - 2)
    );

    // after multiply: [..., m, o, n]
    let_example!(prod, &l * &r);
    // after sum:      [..., m, o, 1]
    let_example!(sum, prod.sum(&[prod.shape().ndims() - 1]));
    // after reshape:  [..., m, o]
    let_example!(s, sum.shape(), true);
    do_example!(sum.reshape(&s[..s.ndims() - 1]));
}
