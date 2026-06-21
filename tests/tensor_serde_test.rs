use std::{
    env, fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use tensorken::{Cpu32, tensor_from_json_file, tensor_to_json_file};

fn temp_file_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    env::temp_dir().join(format!(
        "tensorken_{name}_{}_{}.json",
        std::process::id(),
        nanos
    ))
}

#[test]
fn test_tensor_json_file_round_trip() {
    let t = Cpu32::linspace(1.0, 12.0, 12_u8)
        .reshape(&[2, 2, 3])
        .permute(&[2, 0, 1]);
    let path = temp_file_path("round_trip");

    tensor_to_json_file(&t.to_cpu(), &path);
    let loaded: Cpu32 = tensor_from_json_file(&path);

    assert_eq!(loaded.shape(), t.shape());
    assert_eq!(loaded.ravel(), t.ravel());

    let _ = fs::remove_file(path);
}
