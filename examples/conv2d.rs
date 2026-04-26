use tensorken::Wgpu32;

fn ofshape(s: &[usize]) -> Wgpu32 {
    let prod: usize = s.iter().product();
    Wgpu32::linspace(0., prod as f32 - 1.0, prod as u16).reshape(s)
}

fn main() {
    let im = ofshape(&[2, 3, 4, 5]);
    let ker = ofshape(&[5, 3, 2, 2]);
    println!("input\n{im}{ker}\nconv\n{}", im.conv2d(&ker));
}
