use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{DiffableOps, Shape, Tensor, ToCpu, num::Elem};

#[derive(Debug, Serialize, Deserialize)]
struct TensorPayload<E> {
    shape: Vec<usize>,
    values: Vec<E>,
}

impl<E: Elem> TensorPayload<E> {
    fn new<T, I: ToCpu<Repr<E> = T>>(tensor: &Tensor<T, E, I>) -> TensorPayload<E> {
        TensorPayload {
            shape: tensor.shape().to_vec(),
            values: tensor.ravel(),
        }
    }

    fn tensor_from_json_payload<T, I: DiffableOps<Repr<E> = T>>(&self) -> Tensor<T, E, I> {
        let expected = self.shape.size();
        let actual = self.values.len();
        assert!(
            expected == actual,
            "shape implies {expected} values, but payload contains {actual} values"
        );
        Tensor::new(&self.shape, &self.values)
    }
}

/// Serialize a tensor to JSON and write it to a file path.
///
/// # Panics
/// if file creation or serialization fails.
pub fn tensor_to_json_file<T, E: Elem + Serialize, I: ToCpu<Repr<E> = T>>(
    tensor: &Tensor<T, E, I>,
    path: impl AsRef<Path>,
) {
    let payload = TensorPayload::new(tensor);
    let file = File::create(path).expect("file creation failed");
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &payload).expect("json serialization failed");
}

/// Read JSON from a file path and deserialize it into a tensor.
///
/// # Panics
/// if file reading or deserialization fails, or if the shape and values length do not match.
pub fn tensor_from_json_file<T, E: Elem + DeserializeOwned, I: DiffableOps<Repr<E> = T>>(
    path: impl AsRef<Path>,
) -> Tensor<T, E, I> {
    let file = File::open(path).expect("file open failed");
    let reader = BufReader::new(file);
    let payload: TensorPayload<E> =
        serde_json::from_reader(reader).expect("json deserialization failed");
    payload.tensor_from_json_payload()
}
