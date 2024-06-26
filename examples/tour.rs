use tensorken::{hd, tl, Cpu32, CpuBool, CpuI32, Ellipsis, NewAxis, Sym};

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
    do_example!(Cpu32::new(&[3, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));

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
    do_example!(t.reshape(&[1, 6]));
    do_example!(t.pad(&[(1, 2), (1, 3)]));

    let_example!(t, &Tr::linspace(1.0, 24.0, 24u8).reshape(&[2, 3, 4]));
    do_example!(t.ix1(1));
    do_example!(t.ix2(1, 0));
    do_example!(t.ix2(..tl(0), Ellipsis));
    // hd counts from the front, tl from the back
    do_example!(t.ix2(Ellipsis, hd(1)..tl(1)));
    // invert the range to flip. First bound is still inclusive, second exclusive.
    do_example!(t.ix2(Ellipsis, tl(1)..hd(1)));
    // add some new axes
    do_example!(t.ix4(NewAxis, Ellipsis, NewAxis, NewAxis));

    let_example!(t, Tr::linspace(0.0, 23.0, 24_u8));
    let_example!(t6x4, t.reshape(&[6, 4]));
    let_example!(t3x8, t6x4.reshape(&[3, 8]));

    do_example!(t3x8.permute(&[1, 0]));

    // broadcasting with matmul in more than 2 dimensions boggles the mind
    let_example!(t1, &Tr::linspace(1.0, 36.0, 36_u8).reshape(&[3, 2, 2, 3]));
    let_example!(t2, &Tr::linspace(37.0, 72.0, 36_u8).reshape(&[3, 2, 3, 2]));
    do_example!(t1.matmul(t2));
    let_example!(t3, &Tr::linspace(39.0, 72.0, 12_u8).reshape(&[2, 3, 2]));
    do_example!(t1.matmul(t3));

    // a subset of operations (checked by the type system) are possible with i32
    let_example!(ti1, &CpuI32::new(&[2, 2], &[0, 1, 2, 3]));
    let_example!(ti2, &CpuI32::new(&[2, 2], &[4, 5, 6, 7]));
    do_example!(ti1 + ti2);
    do_example!(ti1 * ti2);
    do_example!(ti1.matmul(ti2));

    // an even smaller subset is possible with bool
    let_example!(tb1, &CpuBool::new(&[2, 2], &[true, false, true, false]));
    let_example!(tb2, &CpuBool::new(&[2, 2], &[false, true, false, true]));
    do_example!(tb2.reshape(&[4]));

    let ti1 = &Sym::new(&[2, 2], &[0, 1, 2, 3]);
    let ti2 = &Sym::new(&[2, 2], &[4, 5, 6, 7]);
    println!("{:?}", ti1 + ti2);
    println!("{:?}", ti1.matmul(ti2));
}
