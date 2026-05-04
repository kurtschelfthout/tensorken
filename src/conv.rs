/// A struct for cross-correlation parameters.
///
/// Given an image shape and kernel, a cross-correlation can be thought of as
/// a sparse matrix mapping the flattened input image to the flattened output
/// image and whose nonzero elements are taken from the kernel. For example,
/// consider the following 2x2 kernel:
///
/// ```text
/// [[1, 2],
///  [3, 4]]
/// ```
///
/// With this kernel and `CorrelateOpts::default()`, the output of `correlate()`
/// on a 3x3 image is a 2x2 image, and the flattened output is the same as
/// multiplying the following matrix with the flattened input image:
///
/// ```text
/// [[1, 2, 0, 3, 4, 0, 0, 0, 0],
///  [0, 1, 2, 0, 3, 4, 0, 0, 0],
///  [0, 0, 0, 1, 2, 0, 3, 4, 0],
///  [0, 0, 0, 0, 1, 2, 0, 3, 4]]
/// ```
///
/// `CorrelateOpts` can represent a wide variety of such sparse matrices.
/// `iter_tensor_index` provides a way to iterate over the elements that are
/// taken from the kernel.
#[derive(Copy, Clone, Debug)]
pub struct CorrelateOpts<const N: usize> {
    /// The spacing within the image at which the kernel will be applied. Has
    /// the effect of downsampling the input image.
    pub stride: [usize; N],
    /// The spacing of individual kernel elements.
    pub dilation: [usize; N],
    /// The spacing of image elements. Has the effect of upsampling the image.
    pub fill: [usize; N],
    /// Padding added to the input (before fill)
    pub padding: [(usize, usize); N],
}

impl<const N: usize> CorrelateOpts<N> {
    /// Returns whether these options are valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.stride.is_empty()
            && self.stride.iter().all(|x| *x > 0)
            && self.dilation.iter().all(|x| *x > 0)
            && self.fill.iter().all(|x| *x > 0)
    }

    /// Returns the number of kernel dimensions represented by this value, or
    /// `None` if invalid.
    #[must_use]
    pub fn kernel_dim(&self) -> Option<usize> {
        self.is_valid().then_some(self.stride.len())
    }

    /// Calculate the output shape for these operands according to the rules of
    /// `RawTensor::correlate`
    ///
    /// # Errors
    ///
    /// If the shapes are inconsistent with the rules.
    pub fn output_shape(&self, im: &[usize], ker: &[usize]) -> Result<Vec<usize>, String> {
        if im.len() != N + 2 {
            return Err(format!("invalid image shape for correlate(): {im:?}"));
        }
        if ker.len() != N + 2 {
            return Err(format!("invalid kernel shape for correlate(): {ker:?}"));
        }

        let batch = im[0];
        let oc = ker[0];
        let ic = ker[1];
        if im[1] != ic {
            return Err(format!("image and kernel mismatch in iC: {im:?} {ker:?}"));
        }

        let im0 = &im[2..];
        let ker0 = &ker[2..];

        let mut out = Vec::with_capacity(N + 2);
        out.push(batch);
        out.push(oc);

        for i in 0..N {
            let stride = self.stride[i];
            let dilation = self.dilation[i];
            let fill = self.fill[i];
            let (pl, pr) = self.padding[i];

            // size of filled and padded image and dilated kernel
            let is = (im0[i] - 1) * fill + 1 + pl + pr;
            let ks = (ker0[i] - 1) * dilation + 1;

            // now we count the number of positive integers `x` that
            // index a valid kernel position, i.e. where
            //   stride * x + ks <= is
            //   stride * x <= is - ks
            //   x <= (is - ks) / stride
            //   x < (is - ks) / stride + 1
            out.push((is - ks) / stride + 1);
        }

        Ok(out)
    }

    fn assert_can_transpose(&self) {
        // While in principle we can compute the parameters for the transpose,
        // the math is tricky and I haven't gotten around to figuring it out yet.
        assert!(self.stride.iter().all(|x| *x == 1));
        assert!(self.dilation.iter().all(|x| *x == 1));
        assert!(self.fill.iter().all(|x| *x == 1));
        assert!(self.padding.iter().all(|(a, b)| *a == 0 && *b == 0));
    }

    #[must_use]
    pub fn for_image_transpose(&self, _ker: &[usize]) -> Self {
        self.assert_can_transpose();
        Self {
            stride: [1; N],
            dilation: [1; N],
            fill: [1; N],
            padding: [(0, 0); N],
        }
    }

    #[must_use]
    pub fn for_kernel_transpose(&self, ker: &[usize]) -> Self {
        self.assert_can_transpose();
        Self {
            stride: [1; N],
            dilation: [1; N],
            fill: [1; N],
            padding: std::array::from_fn(|i| (ker[i] - 1, ker[i] - 1)),
        }
    }

    /// Given an output tensor index, return an iterator over image and kernel
    /// indices that should be dotted to calculate the output.
    #[must_use]
    pub fn iter_tensor_index(
        &self,
        im: &[usize],
        ker: &[usize],
        out: &[usize],
    ) -> CorrelateIndexIterator<N> {
        CorrelateIndexIterator::new(self, im, ker, out)
    }
}

impl<const N: usize> Default for CorrelateOpts<N> {
    fn default() -> Self {
        Self {
            stride: [1; N],
            dilation: [1; N],
            fill: [1; N],
            padding: [(0, 0); N],
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CorrelateIndex<const N: usize>(isize, [isize; N]);

impl<const N: usize> CorrelateIndex<N> {
    fn zero() -> Self {
        Self(0, [0; N])
    }

    fn from_usize_slice(x: &[usize]) -> Self {
        assert!(x.len() == N + 1);
        let mut out = Self::zero();
        for (i, x) in x.iter().enumerate() {
            *out.place_mut(N - i) = x.cast_signed();
        }
        out
    }

    fn place(&self, i: usize) -> isize {
        match i {
            i if i > N => panic!("out of bounds"),
            i if i == N => self.0,
            _ => self.1[N - 1 - i],
        }
    }

    fn place_mut(&mut self, i: usize) -> &mut isize {
        match i {
            i if i > N => panic!("out of bounds"),
            i if i == N => &mut self.0,
            _ => &mut self.1[N - 1 - i],
        }
    }

    fn increment(&mut self, m: &Self) {
        for i in 0..=N {
            let x = self.place_mut(i);
            if *x + 1 >= m.place(i) && N > i {
                *x = 0;
            } else {
                *x += 1;
                break;
            }
        }
    }

    fn contains(&self, other: &Self) -> bool {
        (0..=N).all(|i| {
            let x = self.place(i);
            let y = other.place(i);
            0 <= y && y < x
        })
    }

    fn binary<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(isize, isize) -> isize,
    {
        let mut out = Self::zero();
        for i in 0..=N {
            *out.place_mut(i) = f(self.place(i), other.place(i));
        }
        out
    }

    fn add(&self, other: &Self) -> Self {
        self.binary(other, |a, b| a + b)
    }

    fn mul(&self, other: &Self) -> Self {
        self.binary(other, |a, b| a * b)
    }

    fn try_divide(&self, other: &Self) -> Option<Self> {
        let mut out = Self::zero();
        for i in 0..=N {
            let a = self.place(i);
            let b = other.place(i);
            if a % b == 0 {
                *out.place_mut(i) = a / b;
            } else {
                return None;
            }
        }
        Some(out)
    }

    fn iter_usize(self) -> impl Iterator<Item = usize> {
        (0..=N).rev().map(move |i| self.place(i).cast_unsigned())
    }
}

#[test]
fn test_correlate_index_increment() {
    let mut x: CorrelateIndex<1> = CorrelateIndex::zero();
    let m: CorrelateIndex<1> = CorrelateIndex(2, [3]);

    assert!(m.contains(&x));
    x.increment(&m);
    assert_eq!(x, CorrelateIndex(0, [1]));
    assert!(m.contains(&x));
    x.increment(&m);
    assert_eq!(x, CorrelateIndex(0, [2]));
    assert!(m.contains(&x));
    x.increment(&m);
    assert_eq!(x, CorrelateIndex(1, [0]));
    assert!(m.contains(&x));
    x.increment(&m);
    assert_eq!(x, CorrelateIndex(1, [1]));
    assert!(m.contains(&x));
    x.increment(&m);
    assert_eq!(x, CorrelateIndex(1, [2]));
    assert!(m.contains(&x));
    x.increment(&m);
    assert_eq!(x, CorrelateIndex(2, [0]));
    assert!(!m.contains(&x));
}

#[derive(Debug)]
pub struct CorrelateIndexIterator<const N: usize> {
    batch: usize,
    channel: usize,
    dilation: CorrelateIndex<N>,
    fill: CorrelateIndex<N>,
    im_start: CorrelateIndex<N>,
    im_mod: CorrelateIndex<N>,
    ker_mod: CorrelateIndex<N>,
    next: CorrelateIndex<N>,
}

// In the two dimensional case, [b, oc, oy, ox] maps to starting indices
// [b, 0, iy, ix] and [oc, 0, ky, kx], and we just iterate over all valid
// values of the last 3 indices.

impl<const N: usize> CorrelateIndexIterator<N> {
    fn new(opts: &CorrelateOpts<N>, im: &[usize], ker: &[usize], out: &[usize]) -> Self {
        use std::array::from_fn;

        assert_eq!(out.len(), N + 2);

        let im_start: Vec<isize> = (0..N)
            .map(|i| {
                let pad = opts.padding[i].0.cast_signed();
                let fill = opts.fill[i].cast_signed();
                let out = out[i + 2].cast_signed();
                let stride = opts.stride[i].cast_signed();
                -pad * fill + out * stride
            })
            .collect();

        Self {
            batch: out[0],
            channel: out[1],
            dilation: CorrelateIndex(1, from_fn(|i| opts.dilation[i].cast_signed())),
            fill: CorrelateIndex(1, from_fn(|i| opts.fill[i].cast_signed())),
            im_start: CorrelateIndex(0, from_fn(|i| im_start[i])),
            im_mod: CorrelateIndex::from_usize_slice(&im[1..]),
            ker_mod: CorrelateIndex::from_usize_slice(&ker[1..]),
            next: CorrelateIndex::zero(),
        }
    }
}

impl<const N: usize> Iterator for CorrelateIndexIterator<N> {
    type Item = (Vec<usize>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.ker_mod.contains(&self.next) {
                return None;
            }

            let im_filled_next = self.next.mul(&self.dilation).add(&self.im_start);
            let ker_next = self.next;
            self.next.increment(&self.ker_mod);

            if let Some(im_next) = im_filled_next.try_divide(&self.fill) {
                if self.im_mod.contains(&im_next) {
                    let mut im_at = vec![self.batch];
                    let mut ker_at = vec![self.channel];
                    im_at.extend(im_next.iter_usize());
                    ker_at.extend(ker_next.iter_usize());
                    return Some((im_at, ker_at));
                }
            }
        }
    }
}
