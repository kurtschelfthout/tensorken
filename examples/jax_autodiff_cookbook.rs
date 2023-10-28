use rand::{rngs::StdRng, SeedableRng};
use tensorken::{
    grad1, grad2, jacfwd, jacrev, value_and_grad2, Cpu32, Diffable, DiffableExt, Forward, Reverse,
    TensorLike, TensorLikeRef,
};

type Tr = Cpu32;

// Following JAX's Autodiff Cookbook https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

#[allow(non_snake_case)]
fn main() {
    // ## Gradients
    // ### Starting with `grad`

    let s = Tr::scalar(2.0);
    let dr = grad1(|t| t.tanh(), &s);
    print!("dr: {dr}");

    let ddr = grad1(|t| grad1(|t| t.tanh(), t), &s);
    print!("ddr: {ddr}");

    let dddr = grad1(|t| grad1(|t| grad1(|t| t.tanh(), t), t), &s);
    print!("dddr: {dddr}");

    // # Outputs probability of a label being true.
    fn predict<'t, T>(W: &'t T, b: &'t T, inputs: &T) -> T
    where
        T: TensorLike<'t>,
        //for<'s> &'s T: TensorLikeRef<T>,
    {
        (inputs.dot(W) + b).sigmoid()
    }

    // Build a toy dataset.
    let inputs = Tr::new(
        &[4, 3],
        &[
            0.52, 1.12, 0.77, //
            0.88, -1.08, 0.15, //
            0.52, 0.06, -1.30, //
            0.74, -2.49, 1.39,
        ],
    );
    let targets = Tr::new(&[4], &[1.0, 1.0, 0.0, 1.0]);

    let key = 0;
    let mut rng = StdRng::seed_from_u64(key);
    let W = Tr::randn(&[3], &mut rng);
    // TODO in JAX, the shape of a scalar is &[] not &[1].
    let b = Tr::randn(&[1], &mut rng);

    let prediction = predict(&W, &b, &inputs);
    println!("prediction: {prediction}");

    // Training loss is the negative log-likelihood of the training examples.
    fn loss<'t, T>(W: &'t T, b: &'t T, inputs: &T, targets: &'t T) -> T
    where
        T: TensorLike<'t>,
        for<'s> &'s T: TensorLikeRef<T>,
    {
        let prediction = predict(W, b, inputs);
        let label_probs = &prediction * targets
            + (&prediction.ones_like() - &prediction) * (targets.ones_like() - targets);
        -label_probs.log().sum(&[0])
    }

    // Differentiate loss wrt W
    let W_grad = grad1(
        |W| {
            loss(
                W,
                &Reverse::lift(&b),
                &Reverse::lift(&inputs),
                &Reverse::lift(&targets),
            )
        },
        &W,
    );
    print!("W_grad: {W_grad}");

    // Differentiate loss wrt b
    let b_grad = grad1(
        |b| {
            loss(
                &Reverse::lift(&W),
                b,
                &Reverse::lift(&inputs),
                &Reverse::lift(&targets),
            )
        },
        &b,
    );
    print!("b_grad: {b_grad}");

    // Differentiate loss wrt W and b - should give the same answer
    let (W_grad, b_grad) = grad2(
        |W, b| loss(W, b, &Reverse::lift(&inputs), &Reverse::lift(&targets)),
        &W,
        &b,
    );
    print!("W_grad: {W_grad}");
    print!("b_grad: {b_grad}");

    // ### Differentiating with respect to nested lists, tuples, and dicts
    // TODO no support for other container types in Tensorken atm

    // ### Evaluate a function and its gradient using `value_and_grad`

    let (loss_value, _Wb_grad) = value_and_grad2(
        |W, b| loss(W, b, &Reverse::lift(&inputs), &Reverse::lift(&targets)),
        &W,
        &b,
    );
    print!("loss value: {loss_value}");
    print!("loss value: {}", loss(&W, &b, &inputs, &targets));

    // ### Checking against numerical differences
    let eps = Tr::scalar(1e-4);
    let half_eps = &eps / Tr::scalar(2.);
    let b_grad_numerical = (loss(&W, &(&b + &half_eps), &inputs, &targets)
        - loss(&W, &(&b - &half_eps), &inputs, &targets))
        / &eps;
    print!("b_grad_numerical {}", b_grad_numerical);
    print!("b_grad_autodiff {}", b_grad);

    // TODO implement a numerical gradient checker

    // ### Hessian-vector products with `grad`-of-`grad`

    // ### Jacobians and Hessians using jacfwd and jacrev

    let J = jacfwd(
        |W| predict(W, &Forward::lift(&b), &Forward::lift(&inputs)),
        &W,
    );
    print!("jacfwd result, with shape {:?}", J.shape());
    print!("{}", &J);

    let J = jacrev(
        |W| predict(W, &Reverse::lift(&b), &Reverse::lift(&inputs)),
        &W,
    );
    print!("jacrev result, with shape {:?}", J.shape());
    print!("{}", &J);

    let hessian = jacfwd(
        |W| {
            jacrev(
                |W| {
                    predict(
                        W,
                        &Reverse::lift(&Forward::lift(&b)),
                        &Reverse::lift(&Forward::lift(&inputs)),
                    )
                },
                W,
            )
        },
        &W,
    );
    println!("hessian result, with shape {:?}", hessian.shape());
    println!("{}", &hessian);
}
