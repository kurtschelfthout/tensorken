#![warn(clippy::pedantic)]

extern crate tensorken;

use std::{
    collections::{HashMap, HashSet},
    f32::consts::E,
    io::Read,
    path::{Path, PathBuf},
};

use prettytable::{Cell, Row, Table};
use rand::{distributions::WeightedIndex, prelude::Distribution, rngs::StdRng, SeedableRng};
use tensorken::{
    raw_tensor_cpu::CpuRawTensor,
    tensor::{IndexValue, Tensor},
};

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
    names_file.push("names.txt");
    let mut file = std::fs::File::open(names_file).unwrap();

    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let mut names = Vec::new();
    for line in contents.lines() {
        names.push(line.to_ascii_lowercase());
    }
    names
}

fn main() {
    // read and show some stats on names
    let names = read_names();
    println!("names: {:?}", &names[..10]);
    println!(
        "names len: {}, min len: {}, max len: {}",
        names.len(),
        names.iter().map(String::len).min().unwrap(),
        names.iter().map(String::len).max().unwrap()
    );

    // Preamble: explaining what character-level bigrams are.
    // A bit about start and end tokens.
    dict_bigram(&names);

    // Now let's do a character-level bigram model, using tensors.
    // Introducing basic concepts like loss, and working with tensors.
    // Here's where Karpathy imports torch. Lame! Instead, we'll use Tensorken.
    tensor_bigram(&names);
}

fn dict_bigram(names: &[String]) {
    // create a hashmap from bigrams to number of instances of that bigram in names.
    // Tokens are letters, with added start < and end > tokens.
    // Karpathy uses <S> and <E>, which is a bit awkward since Rust splits strings into chars.
    let mut bigram_counts = std::collections::HashMap::new();
    for name in names.iter() {
        let name = format!("<{name}>");
        for bigram in name.chars().collect::<Vec<_>>().windows(2) {
            let bigram = bigram.iter().collect::<String>();
            *bigram_counts.entry(bigram).or_insert(0) += 1;
        }
    }

    // show the bigram_counts hashmap, sorted by count.
    // (on ties, these are in a slightly different order than in Karpathy's notebook)
    let mut bigrams_sorted_by_count: Vec<_> = bigram_counts.into_iter().collect();
    bigrams_sorted_by_count.sort_by(|a, b| b.1.cmp(&a.1));
    println!("bigram_counts: {:?}", &bigrams_sorted_by_count[..10]);
}

type Cpu = CpuRawTensor<f32>;
// type Gpu<'a> = WgpuRawTensor<'a, f32>;

// Candidate for addition to Tensorken.
fn pretty_print_bigram(tensor: &Tensor<Cpu>, itos: &HashMap<usize, char>, prec: usize) {
    let mut table = Table::new();
    for row in 0..tensor.shape()[0] {
        let mut table_row = Row::empty();
        for col in 0..tensor.shape()[1] {
            table_row.add_cell(Cell::new(&format!(
                "{}{}\n{:.prec$}",
                itos[&row],
                itos[&col],
                tensor.at(&[row, col])
            )));
        }
        table.add_row(table_row);
    }
    table.printstd();
}

// Candidate for addition to Tensorken.
fn multinouilli_sample(tensor: &Tensor<Cpu>, row: usize, rng: &mut StdRng) -> usize {
    // Purely on a rand usage basis, I should only make the WeightedIndex once per row.
    // Also, all this copying out should not be necessary. Maybe contiguous tensors could
    // have a method that returns a slice of the underlying data? Or an iterator, more generally?
    let weights = tensor.at(row).ravel();
    let dist = WeightedIndex::new(weights).unwrap();
    dist.sample(rng)
}

#[allow(clippy::cast_precision_loss)]
fn tensor_bigram(names: &[String]) {
    // Get all the unique characters in names.
    let mut chars = names
        .iter()
        .flat_map(|name| name.chars())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    chars.sort_unstable();
    println!("chars: {chars:?}");

    // Create a mapping from characters to indices, leaving room for the start/end token '.'
    let mut stoi = chars
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i + 1))
        .collect::<HashMap<_, _>>();
    stoi.insert('.', 0);
    // reverse index
    let itos = stoi
        .iter()
        .map(|(c, i)| (*i, *c))
        .collect::<std::collections::HashMap<_, _>>();
    println!("stoi: {stoi:#?} \nitos: {itos:#?}");

    // Create a tensor to store bigram counts.
    // This is a mutable tensor, so it doesn't make sense to use any other backend.
    let mut bigrams_mut = Tensor::<Cpu>::zeros(&[27, 27]).to_tensor_mut();

    for name in names {
        let chars: Vec<_> = format!(".{name}.").chars().collect();
        for bigram in chars.windows(2) {
            let i = stoi[&bigram[0]];
            let j = stoi[&bigram[1]];
            bigrams_mut[&[i, j]] += 1.0;
        }
    }

    let bigrams = bigrams_mut.to_tensor::<Cpu>();

    // I could import a plotting library, but I'll just pretty print the tensor.
    // No heatmaps for us...I spent a bit of time looking for a terminal-based heatmap :) but came up empty.
    // At least can read all the numbers, which is not the case for the heatmap!
    pretty_print_bigram(&bigrams, &itos, 0);

    // "smoothing": add 1 so no count is 0. That way, our model will never predict 0 probability.
    let bigrams = &bigrams + Tensor::scalar(1.0);

    // normalize to get probabilities.
    // Note - broadcasting in play here. We're dividing a 27x27 tensor by a 27x1 tensor.
    let bigrams = &bigrams / &bigrams.sum(&[1]);
    pretty_print_bigram(&bigrams, &itos, 4);

    // Now let's do some predictions.
    let mut rng = StdRng::seed_from_u64(2_147_483_647);
    for _ in 0..5 {
        let mut out = Vec::new();
        let mut ix = 0;
        loop {
            ix = multinouilli_sample(&bigrams, ix, &mut rng);
            out.push(itos[&ix]);
            if ix == 0 {
                break;
            }
        }
        println!("{}", out.iter().collect::<String>());
    }

    // now measure the loss.
    // Don't run this on the GPU - it is extremely slow, because
    // of all the memory transfers involved in at calls.
    let mut log_likelihood = 0.0;
    let mut n = 0;
    for name in names {
        let chars: Vec<_> = format!(".{name}.").chars().collect();
        for bigram in chars.windows(2) {
            let i = stoi[&bigram[0]];
            let j = stoi[&bigram[1]];
            let prob = bigrams.at(&[i, j]);
            let log_prob = prob.log(E);
            log_likelihood += log_prob;
            n += 1;
        }
    }
    let n = n as f32;
    println!("negative log likelihood: {}", -log_likelihood / n);
}
