use tensorken::{Axes, Cpu32, CpuI32};

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
type TrI = CpuI32;

fn main() {
    let_example!(t, Tr::linspace(1., 15.0, 15u8).reshape(&[5, 3]));
    let_example!(i, TrI::new(&[2], &[2, 0]));

    // we want: t.oidx(i, ..)

    // first convert the indexing tensor i to a one hot tensor i_one_hot
    let_example!(
        i_range,
        TrI::new(&[5, 1], (0..5).collect::<Vec<_>>().as_slice())
    );
    let_example!(i_one_hot, i.eq(&i_range).cast::<f32>());

    // then multiply the one hot tensor with the original tensor
    // but first reshape a bit so the right dimensions are broadcasted
    let_example!(t, t.reshape(&[5, 1, 3]));
    let_example!(i_one_hot, i_one_hot.reshape(&[5, 2, 1]));
    let_example!(mul_result, t.mul(&i_one_hot));

    // then sum reduce the result, and remove the remaining original dimension
    let_example!(sum_result, mul_result.sum(&[0]).squeeze(&Axes::Axis(0)));
}
