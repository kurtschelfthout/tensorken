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
    /// The padding behavior
    pub padding: CorrelatePad<N>,
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
        let padding = self.padding.as_pad(self.dilation, ker0);

        let mut out = Vec::with_capacity(N + 2);
        out.push(batch);
        out.push(oc);

        for i in 0..N {
            let stride = self.stride[i];
            let dilation = self.dilation[i];
            let fill = self.fill[i];
            let (pl, pr) = padding[i];

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
        assert_eq!(self.padding, CorrelatePad::Valid);
    }

    #[must_use]
    pub fn for_image_transpose(&self) -> Self {
        self.assert_can_transpose();
        Self::default()
    }

    #[must_use]
    pub fn for_kernel_transpose(&self) -> Self {
        self.assert_can_transpose();
        Self::default().pad_full()
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

    /// Returns these options with the given stride on all axes
    #[must_use]
    pub fn with_stride(self, stride: usize) -> Self {
        Self {
            stride: [stride; N],
            dilation: self.dilation,
            fill: self.fill,
            padding: self.padding,
        }
    }

    /// Returns these options with the given dilation on all axes
    #[must_use]
    pub fn with_dilation(self, dilation: usize) -> Self {
        Self {
            stride: self.stride,
            dilation: [dilation; N],
            fill: self.fill,
            padding: self.padding,
        }
    }

    /// Returns these options with the given fill on all axes
    #[must_use]
    pub fn with_fill(self, fill: usize) -> Self {
        Self {
            stride: self.stride,
            dilation: self.dilation,
            fill: [fill; N],
            padding: self.padding,
        }
    }

    /// Returns these options with the padding behavior set to `Same`
    #[must_use]
    pub fn pad_same(self) -> Self {
        Self {
            stride: self.stride,
            dilation: self.dilation,
            fill: self.fill,
            padding: CorrelatePad::Same,
        }
    }

    /// Returns these options with the padding behavior set to `Full`
    #[must_use]
    pub fn pad_full(self) -> Self {
        Self {
            stride: self.stride,
            dilation: self.dilation,
            fill: self.fill,
            padding: CorrelatePad::Full,
        }
    }
}

impl<const N: usize> Default for CorrelateOpts<N> {
    fn default() -> Self {
        Self {
            stride: [1; N],
            dilation: [1; N],
            fill: [1; N],
            padding: CorrelatePad::Valid,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CorrelatePad<const N: usize> {
    /// No padding
    Valid,

    /// Padding is applied so that the input and output images have the same
    /// shape, and kernels are "centered" on each element of the input. For N*N
    /// kernel where N is odd, this is simply floor(N/2), e.g. for a 5x5 kernel
    /// this is a padding of (2, 2) on each axis. When N is even, the
    /// "remainder" padding is arbitrarily chosen to go at the *end* of each
    /// axis, e.g. for a 2x2 kernel, this will be a padding of (0, 1) on each
    /// axis. If this is undesirable, the `Pad` variant can be used instead.
    Same,

    /// Padding is applied so that the convolution uses all kernel positions
    /// with at least one non-padding input. For example, for a 5x5 kernel, this
    /// applies a padding of (4, 4) on each axis.
    Full,

    /// Specify an explicit padding. The output shape will depend on this, the
    /// image shape, and the kernel shape.
    Pad([(usize, usize); N]),
}

impl<const N: usize> CorrelatePad<N> {
    /// Returns the amount of actual padding to apply, given the kernel. Only
    /// the last N dimensions are used, and returns a vec with N elements.
    fn as_pad(&self, dilation: [usize; N], ker_shape: &[usize]) -> [(usize, usize); N] {
        use std::array::from_fn;
        let s = &ker_shape[ker_shape.len() - N..];
        let s = move |i| (s[i] - 1) * dilation[i] + 1;

        match self {
            CorrelatePad::Valid => [(0, 0); N],
            CorrelatePad::Same => from_fn(|i| ((s(i) - 1) / 2, s(i) / 2)),
            CorrelatePad::Full => from_fn(|i| (s(i) - 1, s(i) - 1)),
            CorrelatePad::Pad(x) => x.as_array().copied().unwrap(),
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

        let im_start: Vec<isize> = {
            opts.padding
                .as_pad(opts.dilation, ker)
                .into_iter()
                .enumerate()
                .map(|(i, (pad, _))| {
                    let pad = pad.cast_signed();
                    let fill = opts.fill[i].cast_signed();
                    let out = out[i + 2].cast_signed();
                    let stride = opts.stride[i].cast_signed();
                    -pad * fill + out * stride
                })
                .collect()
        };

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
