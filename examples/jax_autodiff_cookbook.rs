use rand::{rngs::StdRng, SeedableRng};

use tensorken::{
    diff1, grad1, grad2, jacfwd, jacrev, jvpn, num::Float, value_and_grad2, vjpn, Cpu32,
    DiffableOps, Tensor,
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
    fn predict<T, E: Float, I: DiffableOps<Repr<E> = T>>(
        w: &Tensor<T, E, I>,
        b: &Tensor<T, E, I>,
        inputs: &Tensor<T, E, I>,
    ) -> Tensor<T, E, I> {
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
    let b = Tr::randn(&[1], &mut rng);

    let prediction = predict(&w, &b, &inputs);
    println!("prediction: {prediction}");

    // Training loss is the negative log-likelihood of the training examples.
    fn loss<T, E: Float, I: DiffableOps<Repr<E> = T>>(
        w: &Tensor<T, E, I>,
        b: &Tensor<T, E, I>,
        inputs: &Tensor<T, E, I>,
        targets: &Tensor<T, E, I>,
    ) -> Tensor<T, E, I> {
        let prediction = predict(w, b, inputs);
        let label_probs = &prediction * targets
            + (&prediction.ones_like() - &prediction) * (targets.ones_like() - targets);
        -label_probs.log().sum(&[0])
    }
    let l = loss(&w, &b, &inputs, &targets);
    print!("loss: {l}");

    // Differentiate loss wrt weights
    let w_grad = grad1(
        |w| loss(w, &b.lift_rev(), &inputs.lift_rev(), &targets.lift_rev()),
        &w,
    );
    print!("w_grad: {w_grad}");

    // Differentiate loss wrt b
    let b_grad = grad1(
        |b| loss(&w.lift_rev(), b, &inputs.lift_rev(), &targets.lift_rev()),
        &b,
    );
    print!("b_grad: {b_grad}");

    // Differentiate loss wrt W and b - should give the same answer
    let (w_grad, b_grad) = grad2(
        |w, b| loss(w, b, &inputs.lift_rev(), &targets.lift_rev()),
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
    // No support for other container types in Tensorken

    // ### Evaluate a function and its gradient using `value_and_grad`

    let (loss_value, (w_grad, b_grad)) = value_and_grad2(
        |w, b| loss(w, b, &inputs.lift_rev(), &targets.lift_rev()),
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

    // Here they compare with a numerical gradient checker

    // ### Hessian-vector products with `grad`-of-`grad`

    // ### Jacobians and Hessians using jacfwd and jacrev

    let J = jacfwd(|w| predict(w, &b.lift_fwd(), &inputs.lift_fwd()), &w);
    println!("jacfwd result, with shape {:?}", J.shape());
    print!("{}", &J);

    let J = jacrev(|w| predict(w, &b.lift_rev(), &inputs.lift_rev()), &w);
    println!("jacrev result, with shape {:?}", J.shape());
    print!("{}", &J);

    let deriv = grad1(|w| predict(w, &b.lift_rev(), &inputs.lift_rev()), &w);
    println!("deriv result, with shape {:?}", deriv.shape());
    print!("{}", &deriv);
    print!("sum of Jacobian's columns \n{}", &J.sum(&[0]));

    let hessian = jacfwd(
        |w| {
            jacrev(
                |w| predict(w, &b.lift_fwd().lift_rev(), &inputs.lift_fwd().lift_rev()),
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
        |w| predict(&w[0], &b.lift_fwd(), &inputs.lift_fwd()),
        &[&w],
        &[&v],
    );
    println!("y: {y}, u: {u}");

    // ## VJPs in JAX code

    // Pull back the covector `u` along `f` evaluated at `w`
    let (y, pullback) = vjpn(|w| predict(&w[0], &b.lift_rev(), &inputs.lift_rev()), &[&w]);
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
