use tensorken::{hd, tl, Axes, CpuBool, CpuI32, Ellipsis, NewAxis};

/// A macro to print the result of an expression and the expression itself.
macro_rules! with_shapes {
    ($t:expr, $e:expr) => {
        println!(">>> {}", stringify!($e));
        let input_shape = $t.shape();
        let result: CpuI32 = $e;
        let result_shape = result.shape();
        println!("{:?} -> {:?}", input_shape, result_shape);
        println!("{result}");
    };
}

/// A macro to print the result of an expression and the expression itself.
macro_rules! do_example {
    ($e:expr) => {
        println!(">>> {}", stringify!($e));
        let result = $e;
        println!("{result}");
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
}

type TrI = CpuI32;
type TrB = CpuBool;

fn main() {
    let_example!(t, TrI::new(&[4, 3, 2], &(1..25).collect::<Vec<_>>()));

    // intro
    // slicing
    do_example!(t.vix3(1..3, 2.., ..));
    // slicing and int indexing
    let_example!(i, TrI::new(&[2], &[2, 1]));
    do_example!(t.vix3(&i, &i, 1));

    // slicing and masking
    let_example!(b, TrB::new(&[4], &[false, false, true, false]));
    do_example!(t.vix3(&b, &i, 1..));

    // basic indexing
    do_example!(t.ix3(0, 1, 2).to_scalar());
    with_shapes!(t, t.ix1(1));

    do_example!(t
        .ix3(t.shape()[0] - 1, t.shape()[1] - 1, t.shape()[2] - 1)
        .to_scalar());
    do_example!(t.ix3(tl(0), tl(0), tl(0)).to_scalar());

    with_shapes!(t, t.ix3(1, .., ..));
    with_shapes!(t, t.ix3(.., .., 2));

    with_shapes!(t, t.ix2(1, Ellipsis));
    with_shapes!(t, t.ix2(Ellipsis, 2));

    with_shapes!(t, t.ix3(1..3, 1..2, ..));
    with_shapes!(t, t.ix3(..3, 1.., ..));
    with_shapes!(t, t.ix3(..tl(0), ..tl(1), ..));
    with_shapes!(t, t.ix3(hd(3)..hd(1), hd(2)..hd(1), ..));

    with_shapes!(t, t.ix4(.., .., NewAxis, ..));

    with_shapes!(t, t.ix4(0, 1.., NewAxis, ..tl(0)));

    // fancy indexing with int tensors
    let_example!(i, TrI::new(&[2], &[1, 2]));
    with_shapes!(t, t.oix1(&i));

    let_example!(i, TrI::new(&[4], &[3, 0, 2, 1]));
    with_shapes!(t, t.oix1(&i));

    let_example!(i, TrI::new(&[4], &[0, 0, 1, 1]));
    with_shapes!(t, t.oix1(&i));

    let_example!(i, TrI::new(&[5], &[1; 5]));
    with_shapes!(t, t.oix1(&i));

    let_example!(i, TrI::new(&[2, 2], &[0, 1, 1, 0]));
    with_shapes!(t, t.oix1(&i));

    let_example!(i, TrI::new(&[2, 2], &[0, 1, 1, 0]));
    with_shapes!(t, t.oix3(.., .., &i));

    // oix
    with_shapes!(t, t.ix3(1..3, 1..2, ..));
    let_example!(i1, TrI::new(&[2], &[1, 2]));
    let_example!(i2, TrI::new(&[1], &[1]));
    let_example!(i3, TrI::new(&[2], &[0, 1]));
    with_shapes!(t, t.oix3(&i1, &i2, &i3));

    let_example!(i1, TrI::new(&[2], &[3, 0]));
    let_example!(i2, TrI::new(&[2], &[2, 0]));
    let_example!(i3, TrI::new(&[2], &[1, 0]));
    with_shapes!(t, t.oix3(&i1, &i2, &i3));

    let_example!(i1, TrI::new(&[2, 2], &[3, 3, 0, 0]));
    let_example!(i2, TrI::new(&[2], &[2, 0]));
    let_example!(i3, TrI::new(&[2, 2], &[1, 0, 1, 0]));
    with_shapes!(t, t.oix3(&i1, &i2, &i3));

    // vix
    let_example!(t, TrI::new(&[3, 3], &(1..10).collect::<Vec<_>>()));
    let_example!(i1, TrI::new(&[2], &[0, 2]));
    let_example!(i2, TrI::new(&[2], &[0, 2]));
    with_shapes!(t, t.oix2(&i1, &i2));
    with_shapes!(t, t.vix2(&i1, &i2));

    let_example!(i1, TrI::new(&[2, 2], &[0, 0, 2, 2]));
    let_example!(i2, TrI::new(&[2, 2], &[0, 2, 0, 2]));
    with_shapes!(t, t.vix2(&i1, &i2));

    let_example!(t, TrI::new(&[4, 3, 2], &(1..25).collect::<Vec<_>>()));
    let_example!(i1, TrI::new(&[2], &[0, 2]));
    let_example!(i2, TrI::new(&[2], &[0, 1]));
    with_shapes!(t, t.vix3(&i1, .., &i2));
    with_shapes!(t, t.oix3(&i1, .., &i2));

    // masks
    let_example!(t, TrI::new(&[4, 3, 2], &(1..25).collect::<Vec<_>>()));
    let_example!(i, TrB::new(&[4], &[false, false, true, false]));
    with_shapes!(t, t.oix1(&i));

    let_example!(i, t.eq(&TrI::new(&[2], &[1, 2])));
    with_shapes!(t, t.oix1(&i));

    // basic and fancy indexing compose
    let_example!(t, TrI::new(&[4, 3, 2], &(1..25).collect::<Vec<_>>()));
    let_example!(i1, TrI::new(&[2], &[0, 2]));
    let_example!(i2, TrI::new(&[2], &[0, 1]));
    with_shapes!(t, t.vix3(&i1, ..2, &i2));
    with_shapes!(t, t.vix3(.., ..2, ..).vix3(&i1, .., &i2));

    // oix

    // one dim
    let_example!(t, TrI::new(&[4], &(1..5).collect::<Vec<_>>()));
    let_example!(i, TrI::new(&[2], &[2, 0]));

    // first convert the indexing tensor i to a one hot tensor i_one_hot
    let_example!(
        i_range,
        TrI::new(&[4, 1], (0..4).collect::<Vec<_>>().as_slice())
    );
    let_example!(i_one_hot, i.eq(&i_range).cast::<i32>());

    // then multiply the one hot tensor with the original tensor
    // but first reshape a bit so the right dimensions are broadcasted
    let_example!(t, t.reshape(&[4, 1]));

    let_example!(mul_result, t.mul(&i_one_hot));

    // then sum reduce the result, and remove the remaining original dimension
    let_example!(sum_result, mul_result.sum(&[0]).squeeze(&Axes::Axis(0)));

    // two dim
    // we want: t.oix(i, ..)
    let_example!(t, TrI::new(&[4, 2], &(1..9).collect::<Vec<_>>()));
    let_example!(i, TrI::new(&[2], &[2, 0]));

    // first convert the indexing tensor i to a one hot tensor i_one_hot
    let_example!(
        i_range,
        TrI::new(&[4, 1], (0..4).collect::<Vec<_>>().as_slice())
    );
    let_example!(i_one_hot, i.eq(&i_range).cast::<i32>());

    // then multiply the one hot tensor with the original tensor
    // but first reshape a bit so the right dimensions are broadcasted
    let_example!(t, t.reshape(&[4, 1, 2]));
    let_example!(i_one_hot, i_one_hot.reshape(&[4, 2, 1]));
    let_example!(mul_result, t.mul(&i_one_hot));

    // then sum reduce the result, and remove the remaining original dimension
    let_example!(sum_result, mul_result.sum(&[0]).squeeze(&Axes::Axis(0)));

    // vix
    // we want: t.vix(i, j)
    let_example!(t, TrI::new(&[4, 6], &(1..25).collect::<Vec<_>>()));
    let_example!(i, TrI::new(&[2], &[2, 0]));
    let_example!(j, TrI::new(&[2], &[1, 0]));

    // first convert the indexing tensor i to a one hot tensor i_one_hot
    let_example!(
        i_range,
        TrI::new(&[4], (0..4).collect::<Vec<_>>().as_slice())
    );
    let_example!(i, i.reshape(&[2, 1]));
    // now i_one_hot has shape [2, 4]
    let_example!(i_one_hot, i.eq(&i_range).cast::<i32>());

    // multiply the one hot tensor with the original tensor
    // but first reshape so the right dimensions are broadcasted
    let_example!(i_one_hot, i_one_hot.reshape(&[2, 4, 1]));
    let_example!(mul_result, t.mul(&i_one_hot));

    // then sum reduce the result, and remove the remaining original dimension
    let_example!(t, mul_result.sum(&[1]).squeeze(&Axes::Axis(1)));

    // then do the same for j
    let_example!(
        j_range,
        TrI::new(&[6], (0..6).collect::<Vec<_>>().as_slice())
    );
    let_example!(j, j.reshape(&[2, 1]));
    // j_one_hot has shape [2,6]...
    let_example!(j_one_hot, j.eq(&j_range).cast::<i32>());

    // the same shape as t! So we can just multiply...
    let_example!(mul_result, t.mul(&j_one_hot));
    // and sum:
    let_example!(sum_result, mul_result.sum(&[1]).squeeze(&Axes::Axis(1)));

    // masks
    let_example!(
        b,
        TrB::new(&[2, 3], &[false, false, true, false, true, false])
    );
    let_example!(i_b, TrI::new(&[2], &[2, 4]));
}
