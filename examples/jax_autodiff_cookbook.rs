use rand::{rngs::StdRng, SeedableRng};
use tensorken::{
    grad1, grad2, value_and_grad2, Cpu32, DiffableExt, Reverse, TensorLike, TensorLikeRef,
};

type Tr = Cpu32;

// Following JAX's Autodiff Cookbook https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

#[allow(non_snake_case)]
fn main() {
    // ## Gradients
    // ### Starting with `grad`

    let s = Tr::scalar(2.0);
    let dr = grad1(|t| t.tanh(), &s);
    println!("dr: {dr}");

    let ddr = grad1(|t| grad1(|t| t.tanh(), t), &s);
    println!("ddr: {ddr}");

    let dddr = grad1(|t| grad1(|t| grad1(|t| t.tanh(), t), t), &s);
    println!("dddr: {dddr}");

    // # Outputs probability of a label being true.
    fn predict<'t, T>(W: &'t T, b: &'t T, inputs: &T) -> T
    where
        T: TensorLike<'t>,
        // for<'s> &'s T: TensorLikeRef<T>,
    {
        // TODO JAX uses dot here, which does something special with the last axis,
        // depending on the shape of the arguments.
        (inputs.matmul(W) + b).sigmoid()
    }

    // Build a toy dataset.
    let inputs = Tr::new(
        &[4, 3],
        &[
            0.52, 1.12, 0.77, 0.88, -1.08, 0.15, 0.52, 0.06, -1.30, 0.74, -2.49, 1.39,
        ],
    );
    let targets = Tr::new(&[4], &[1.0, 1.0, 0.0, 1.0]);

    // Training loss is the negative log-likelihood of the training examples.
    fn loss<'t, T>(W: &'t T, b: &'t T, inputs: &T, targets: &'t T) -> T
    where
        T: TensorLike<'t>,
        for<'s> &'s T: TensorLikeRef<T>,
    {
        let preds = predict(W, b, inputs);
        let label_probs =
            &preds * targets + (&preds.ones_like() - &preds) * (targets.ones_like() - targets);
        -label_probs.log().sum(&[0, 1])
    }

    let key = 0;
    let mut rng = StdRng::seed_from_u64(key);
    // TODO since we use matmul, not dot, we have to make this a matrix explicitly.
    let W = Tr::randn(&[3, 1], &mut rng);
    // TODO in JAX, the shape of a scalar is &[] not &[1].
    let b = Tr::randn(&[1], &mut rng);

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
    println!("W_grad: {W_grad}");

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
    println!("b_grad: {b_grad}");

    // Differentiate loss wrt W and b
    let (W_grad, b_grad) = grad2(
        |W, b| loss(W, b, &Reverse::lift(&inputs), &Reverse::lift(&targets)),
        &W,
        &b,
    );
    println!("W_grad: {W_grad}");
    println!("b_grad: {b_grad}");

    // ### Differentiating with respect to nested lists, tuples, and dicts
    // TODO no support for other container types in Tensorken atm

    // ### Evaluate a function and its gradient using `value_and_grad`

    let (loss_value, _Wb_grad) = value_and_grad2(
        |W, b| loss(W, b, &Reverse::lift(&inputs), &Reverse::lift(&targets)),
        &W,
        &b,
    );
    println!("loss value: {loss_value}");
    println!("loss value: {}", loss(&W, &b, &inputs, &targets));

    // ### Checking against numerical differences
    // TODO implement a numerical gradient checker

    // ### Hessian-vector products with `grad`-of-`grad`

    // ### Jacobians and Hessians using jacfwd and jacrev
}
