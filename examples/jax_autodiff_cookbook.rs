use rand::{rngs::StdRng, SeedableRng};
use tensorken::{
    diff1, grad1, grad2, jacfwd, jacrev, jvpn, value_and_grad2, vjpn, Cpu32, Diffable, DiffableExt,
    Forward, Reverse, TensorLike, TensorLikeRef,
};

type Tr = Cpu32;

// Following JAX's Autodiff Cookbook https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

#[allow(non_snake_case)]
fn main() {
    // ## Gradients
    // ### Starting with `grad`

    let p = Tr::scalar(2.0);
    let df = grad1(|x| x.tanh(), &p);
    print!("df: {df}");

    let ddf = grad1(|x| grad1(|x| x.tanh(), x), &p);
    print!("ddf: {ddf}");

    let dddf = grad1(|x| grad1(|x| grad1(|x| x.tanh(), x), x), &p);
    print!("dddf: {dddf}");

    // Outputs probability of a label being true.
    fn predict<'t, T>(w: &'t T, b: &'t T, inputs: &T) -> T
    where
        T: TensorLike<'t>,
        //for<'s> &'s T: TensorLikeRef<T>,
    {
        (inputs.dot(w) + b).sigmoid()
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
    let w = Tr::randn(&[3], &mut rng);
    // TODO in JAX, the shape of a scalar is &[] not &[1].
    let b = Tr::randn(&[1], &mut rng);

    let prediction = predict(&w, &b, &inputs);
    println!("prediction: {prediction}");

    // Training loss is the negative log-likelihood of the training examples.
    fn loss<'t, T>(w: &'t T, b: &'t T, inputs: &T, targets: &'t T) -> T
    where
        T: TensorLike<'t>,
        for<'s> &'s T: TensorLikeRef<T>,
    {
        let prediction = predict(w, b, inputs);
        let label_probs = &prediction * targets
            + (&prediction.ones_like() - &prediction) * (targets.ones_like() - targets);
        -label_probs.log().sum(&[0])
    }
    let l = loss(&w, &b, &inputs, &targets);
    print!("loss: {l}");

    // Differentiate loss wrt weights
    let w_grad = grad1(
        |w| {
            loss(
                w,
                &Reverse::lift(&b),
                &Reverse::lift(&inputs),
                &Reverse::lift(&targets),
            )
        },
        &w,
    );
    print!("w_grad: {w_grad}");

    // Differentiate loss wrt b
    let b_grad = grad1(
        |b| {
            loss(
                &Reverse::lift(&w),
                b,
                &Reverse::lift(&inputs),
                &Reverse::lift(&targets),
            )
        },
        &b,
    );
    print!("b_grad: {b_grad}");

    // Differentiate loss wrt W and b - should give the same answer
    let (w_grad, b_grad) = grad2(
        |w, b| loss(w, b, &Reverse::lift(&inputs), &Reverse::lift(&targets)),
        &w,
        &b,
    );
    print!("w_grad: {w_grad}");
    print!("b_grad: {b_grad}");

    let new_w = &w - &w_grad;
    let new_b = &b - &b_grad;
    let new_prediction = predict(&new_w, &new_b, &inputs);
    let new_loss = loss(&new_w, &new_b, &inputs, &targets);
    print!("new_prediction: {new_prediction}");
    print!("new_loss: {new_loss}");

    // ### Differentiating with respect to nested lists, tuples, and dicts
    // TODO no support for other container types in Tensorken atm

    // ### Evaluate a function and its gradient using `value_and_grad`

    let (loss_value, (w_grad, b_grad)) = value_and_grad2(
        |w, b| loss(w, b, &Reverse::lift(&inputs), &Reverse::lift(&targets)),
        &w,
        &b,
    );
    print!("loss: {loss_value}, w_grad: {w_grad}, b_grad: {b_grad}");
    print!("loss (direct): {}", loss(&w, &b, &inputs, &targets));

    // ### Checking against numerical differences
    // step size for finite differences
    let eps = Tr::scalar(1e-4);
    let half_eps = &eps / Tr::scalar(2.);
    let b_grad_numerical = (loss(&w, &(&b + &half_eps), &inputs, &targets)
        - loss(&w, &(&b - &half_eps), &inputs, &targets))
        / &eps;
    print!("b_grad_numerical {}", b_grad_numerical);
    print!("b_grad_autodiff {}", b_grad);

    // TODO implement a numerical gradient checker

    // ### Hessian-vector products with `grad`-of-`grad`

    // ### Jacobians and Hessians using jacfwd and jacrev

    let J = jacfwd(
        |w| predict(w, &Forward::lift(&b), &Forward::lift(&inputs)),
        &w,
    );
    println!("jacfwd result, with shape {:?}", J.shape());
    print!("{}", &J);

    let J = jacrev(
        |w| predict(w, &Reverse::lift(&b), &Reverse::lift(&inputs)),
        &w,
    );
    println!("jacrev result, with shape {:?}", J.shape());
    print!("{}", &J);

    let deriv = grad1(
        |w| predict(w, &Reverse::lift(&b), &Reverse::lift(&inputs)),
        &w,
    );
    println!("deriv result, with shape {:?}", deriv.shape());
    print!("{}", &deriv);
    print!("sum of Jacobian's columns \n{}", &J.sum(&[0]));

    let hessian = jacfwd(
        |w| {
            jacrev(
                |w| {
                    predict(
                        w,
                        &Reverse::lift(&Forward::lift(&b)),
                        &Reverse::lift(&Forward::lift(&inputs)),
                    )
                },
                w,
            )
        },
        &w,
    );
    println!("hessian result, with shape {:?}", hessian.shape());
    println!("{}", &hessian);

    // ## JVPs in JAX code

    let v = Tr::randn(w.shape(), &mut rng);
    // Push forward the vector `v` along `f` evaluated at `w`
    let (y, u) = jvpn(
        |w| predict(&w[0], &Forward::lift(&b), &Forward::lift(&inputs)),
        &[&w],
        &[&v],
    );
    println!("y: {y}, u: {u}");

    // ## VJPs in JAX code

    // Pull back the covector `u` along `f` evaluated at `w`
    let (y, pullback) = vjpn(
        |w| predict(&w[0], &Reverse::lift(&b), &Reverse::lift(&inputs)),
        &[&w],
    );
    let u = Tr::randn(y.shape(), &mut rng);
    let v = pullback.call(&u);
    println!("y: {y}, v: {}", &v[0]);

    let p = Tr::scalar(2.0);
    let df = diff1(|x| x.tanh(), &p);
    print!("df: {df}");

    let ddf = diff1(|x| diff1(|x| x.tanh(), x), &p);
    print!("ddf: {ddf}");

    let dddf = diff1(|x| diff1(|x| diff1(|x| x.tanh(), x), x), &p);
    print!("dddf: {dddf}");
}
