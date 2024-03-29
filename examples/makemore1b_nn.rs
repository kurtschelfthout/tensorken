#![allow(non_snake_case)]

extern crate tensorken;

use std::{
    collections::{HashMap, HashSet},
    io::Read,
    path::{Path, PathBuf},
};

use rand::{distributions::WeightedIndex, prelude::Distribution, rngs::StdRng, SeedableRng};
use tensorken::{num::Float, value_and_grad1, Axes, DiffableOps, Tensor, Wgpu};

// This example shows the first half of the first of Karpathy's from zero-to-hero tutorials on makemomre.
// It builds a bigram, character-level language model from a set of names.
// The second part needs neural networks, so Tensorken does not support it yet.
// I more or less follow this notebook: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb

// Get the path to this source file, using the file! macro
fn get_path() -> &'static Path {
    let path = file!();
    Path::new(path).parent().unwrap()
}

// read the names.txt file in the current directory, containing newline separated names, and return a vector of lowercase strings.
fn read_names() -> Vec<String> {
    let mut names_file = PathBuf::from(get_path());
    names_file.push("../data/names.txt");
    let mut file = std::fs::File::open(names_file).unwrap();

    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let mut names = vec![];
    for line in contents.lines() {
        names.push(line.to_ascii_lowercase());
    }
    names
}

// change this to Cpu32 to run on CPU. But it's very very slow.
type Tr = Wgpu<f32>;
type TrI32 = Wgpu<i32>;

fn main() {
    // read and show some stats on names
    let names = read_names();
    println!("names: {:?}", &names[..10]);

    // Moving on to neural networks, instead of counting explicitly for our bigram model.
    // First, create a training set.
    // The training set is a list of pairs of one-hot encoded bigram indices.
    // xs: [5, 27]
    // ys: [5, 27]
    let (xs, ys, itos) = create_training_set(&names);

    // one hot encodings of the training set
    let xenc = xs.one_hot(27_usize).cast::<f32>();
    let yenc = ys.one_hot(27_usize).cast::<f32>();

    // randomly generated weights
    let mut rng = StdRng::seed_from_u64(2_147_483_647);
    // mut because we'll update W as we optimize.
    let mut W = Tr::randn(&[27, 27], &mut rng);

    // run the neural net - just a single linear layer + softmax. No bias.
    fn predict<T, E: Float, I: DiffableOps<Repr<E> = T>>(
        W: &Tensor<T, E, I>,
        xenc: &Tensor<T, E, I>,
    ) -> Tensor<T, E, I> {
        let logits = xenc.matmul(W);
        let counts = logits.exp();
        &counts / &counts.sum(&[1]) // 5, 27
    }

    // let probs = predict(&W, &xenc);
    // println!("probs\n{probs:.3}");

    // Let's calculate the loss.
    fn loss<T, I: DiffableOps<Repr<f32> = T>>(
        probs: &Tensor<T, f32, I>,
        yenc: &Tensor<T, f32, I>,
    ) -> Tensor<T, f32, I> {
        // the max here is a trick to get a columns vector of only the "correct" probabilities, as given by yenc
        // Karpathy does this with indexing one by one, or using pytorch's tensor indexing notation: probs[torch.arange(5), ys]
        let ps = (yenc * probs).max(&[1]).squeeze(&Axes::All);
        let nlls = -ps.log();
        nlls.sum(&[0]) / Tensor::scalar(nlls.shape()[0] as f32)
    }

    // let l = loss(&probs, &yenc);
    // println!("loss={l}");

    // Now let's do the same thing, but with autodiff.
    let rev_xenc = xenc.lift_rev();
    let rev_yenc = yenc.lift_rev();

    // now let's put that all together and keep learning in a loop
    for k in 0..200 {
        // forward and backward pass
        let (l, g) = value_and_grad1(
            |W| {
                let probs = predict(W, &rev_xenc);
                loss(&probs, &rev_yenc)
            },
            &W,
        );
        println!("iter={k} loss={l}");
        // update
        W = &W - &g * Tr::scalar(50.0);
        W = W.realize();
    }

    // and finally we can sample some names from the model
    let mut rng = StdRng::seed_from_u64(2_147_483_647);
    for _ in 0..50 {
        let mut out = vec![];
        let mut ix = 0;
        loop {
            let xenc = TrI32::full(&[1], ix as i32).one_hot(27_usize).cast::<f32>();
            let probs = predict(&W, &xenc).squeeze(&Axes::All);
            // print!("probs\n{probs:.3}");
            let dist = WeightedIndex::new(probs.ravel()).unwrap();
            ix = dist.sample(&mut rng);
            out.push(itos[&ix]);
            if ix == 0 {
                break;
            }
        }
        println!("{}", out.iter().collect::<String>());
    }
}

#[allow(clippy::cast_precision_loss)]
fn create_training_set(names: &[String]) -> (TrI32, TrI32, HashMap<usize, char>) {
    // this first part is the same as `tensor_bigram`.

    // Get all the unique characters in names.
    let mut chars: Vec<_> = names
        .iter()
        .flat_map(|name| name.chars())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    chars.sort_unstable();

    // Create a mapping from characters to indices, leaving room for the start/end token '.'
    let mut stoi: HashMap<_, _> = chars.iter().enumerate().map(|(i, c)| (*c, i + 1)).collect();
    stoi.insert('.', 0);

    // reverse index
    let itos: HashMap<_, _> = stoi.iter().map(|(c, i)| (*i, *c)).collect();

    let mut xs = vec![];
    let mut ys = vec![];
    for name in names.iter() {
        let chars: Vec<_> = format!(".{name}.").chars().collect();
        for bigram in chars.windows(2) {
            let ix1 = stoi[&bigram[0]];
            let ix2 = stoi[&bigram[1]];
            //println!("{bigram:?}");
            xs.push(ix1 as i32);
            ys.push(ix2 as i32);
        }
    }
    let xs = TrI32::new(&[xs.len()], &xs);
    let ys = TrI32::new(&[ys.len()], &ys);
    //println!("xs\n{xs}ys\n{ys}");
    (xs, ys, itos)
}
