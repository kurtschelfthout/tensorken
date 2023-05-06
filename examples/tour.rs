use tensorken::{Cpu32, CpuRawTensor, Diffable, DiffableExt, IndexValue, Tensor};

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
    do_example!(Tensor::<CpuRawTensor<f32>>::new(
        &[3, 2],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ));

    // unary operations
    let_example!(t, &Tr::new(&[3, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));

    do_example!(t.exp());
    do_example!(t.log());

    // binary operations
    let_example!(t1, &Tr::new(&[2, 2], &[0.0, 1.0, 2.0, 3.0]));
    let_example!(t2, &Tr::new(&[2, 2], &[6.0, 7.0, 8.0, 9.0]));

    do_example!(t1 + t2);
    do_example!(t1 * t2);
    do_example!(t1.matmul(t2));

    // broadcasting
    let_example!(t1, &Tr::new(&[6], &[2.0, 1.0, 4.0, 2.0, 8.0, 4.0]));
    let_example!(s1, &Tr::scalar(2.0));
    do_example!((t1.shape(), s1.shape()), true);
    do_example!(t1 + s1);

    let_example!(t1, &Tr::new(&[3, 2], &[2.0, 1.0, 4.0, 2.0, 8.0, 4.0]));
    let_example!(t2, &Tr::new(&[1, 2], &[10.0, 100.0]));
    do_example!(t1 + t2);
    let_example!(t3, &Tr::new(&[3, 1], &[10.0, 100.0, 1000.]));
    do_example!(t1 + t3);

    let_example!(t1, &Tr::new(&[2, 3], &[2.0, 1.0, 4.0, 2.0, 8.0, 4.0]));
    let_example!(s1, &Tr::scalar(2.0));
    do_example!((t1.shape(), s1.shape()), true);
    do_example!(t1 + s1);

    let_example!(t1, &Tr::new(&[3, 2], &[2.0, 1.0, 4.0, 2.0, 8.0, 4.0]));
    let_example!(t2, &Tr::new(&[2], &[10.0, 100.0]));
    do_example!((t1.shape(), t2.shape()), true);
    do_example!(t1 + t2);

    // reduce operations
    let_example!(t, &Tr::new(&[4], &[0.0, 1.0, 2.0, 3.0]));
    do_example!(t.sum(&[0]));

    let_example!(t, &Tr::new(&[2, 2], &[0.0, 1.0, 2.0, 3.0]));
    do_example!(t.sum(&[0, 1]));
    do_example!(t.sum(&[0]));
    do_example!(t.sum(&[1]));

    // movement ops/slicing and dicing
    let_example!(t, &Tr::new(&[1, 2, 2], &[0.0, 1.0, 2.0, 3.0]));
    do_example!(t.expand(&[5, 2, 2]));

    let_example!(t, &Tr::new(&[3, 2], &[2.0, 1.0, 4.0, 2.0, 8.0, 4.0]));
    do_example!(t.crop(&[(0, 2), (1, 2)]));

    let_example!(t, &Tr::new(&[3, 2], &[2.0, 1.0, 4.0, 2.0, 8.0, 4.0]));
    do_example!(t.pad(&[(1, 2), (1, 3)]));

    let_example!(t, &Tr::new(&[2, 2], &[0.0, 1.0, 2.0, 3.0]));
    do_example!(t.at(1));
    do_example!(t.at(&[1, 0]));

    let_example!(t, Tr::linspace(0.0, 23.0, 24));
    let_example!(t6x4, t.reshape(&[6, 4]));
    let_example!(t3x8, t6x4.reshape(&[3, 8]));

    do_example!(t3x8.permute(&[1, 0]));

    // broadcasting with matmul in more than 2 dimensions boggles the mind
    let_example!(t1, &Tr::linspace(1.0, 36.0, 36).reshape(&[3, 2, 2, 3]));
    let_example!(t2, &Tr::linspace(37.0, 72.0, 36).reshape(&[3, 2, 3, 2]));
    do_example!(t1.matmul(t2));
    let_example!(t3, &Tr::linspace(39.0, 72.0, 12).reshape(&[2, 3, 2]));
    do_example!(t1.matmul(t3));
}
