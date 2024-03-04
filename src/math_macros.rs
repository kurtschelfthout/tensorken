// A couple of macros to help with implementing Add, Sub, Mul, Div and Neg for all
// combinations of owned/borrowed arguments.

macro_rules! impl_bin_op {
    // Lots of complexity to deal with optional lifetimes, parameters and bounds.
    // See https://stackoverflow.com/a/61189128/72211 for some explanation.
    // Add          , add         , Reverse,    , <      'a or T :  B  + C                       >
    ($op_trait:ident, $op_fn:ident, $name:ident $(< $( $ps:tt $( : $pb:tt $(+ $pbb:tt )* )?  ),+ >)? ) => {
        impl$(< $( $ps $( : $pb $(+ $pbb )* )?  ),+ >)? $op_trait for $name$(< $( $ps ),+ >)? where T::Elem: Num{
            type Output = Self;

            fn $op_fn(self, rhs: Self) -> Self::Output {
                DiffableExt::$op_fn(&self, &rhs)
            }
        }

        impl$(< $( $ps $( : $pb $(+ $pbb )* )?  ),+ >)? $op_trait for &$name$(< $( $ps ),+ >)? where T::Elem: Num{
            type Output = $name$(< $( $ps ),+ >)?;

            fn $op_fn(self, rhs: Self) -> Self::Output {
                DiffableExt::$op_fn(self, rhs)
            }
        }

        impl$(< $( $ps $( : $pb $(+ $pbb )* )?  ),+ >)? $op_trait<&$name$(< $( $ps ),+ >)?> for $name$(< $( $ps ),+ >)? where T::Elem: Num{
            type Output = Self;

            fn $op_fn(self, rhs: &$name$(< $( $ps ),+ >)?) -> Self::Output {
                DiffableExt::$op_fn(&self, rhs)
            }
        }

        impl$(< $( $ps $( : $pb $(+ $pbb )* )?  ),+ >)? $op_trait<$name$(< $( $ps ),+ >)?> for &$name$(< $( $ps ),+ >)? where T::Elem: Num{
            type Output = $name$(< $( $ps ),+ >)?;

            fn $op_fn(self, rhs: $name$(< $( $ps ),+ >)?) -> Self::Output {
                DiffableExt::$op_fn(self, &rhs)
            }
        }
    };
}

pub(crate) use impl_bin_op;

macro_rules! impl_un_op {
    ($op_trait:ident, $op_fn:ident, $name:ident $(< $( $ps:tt $( : $pb:tt $(+ $pbb:tt )* )?  ),+ >)? ) => {

        impl$(< $( $ps $( : $pb $(+ $pbb )* )?  ),+ >)? $op_trait for $name$(< $( $ps ),+ >)? where T::Elem: Num{
            type Output = Self;

            fn $op_fn(self) -> Self::Output {
                DiffableExt::$op_fn(&self)
            }
        }

        impl$(< $( $ps $( : $pb $(+ $pbb )* )?  ),+ >)? $op_trait for &$name$(< $( $ps ),+ >)? where T::Elem: Num{
            type Output = $name$(< $( $ps ),+ >)?;

            fn $op_fn(self) -> Self::Output {
                DiffableExt::$op_fn(self)
            }
        }
    }
}

pub(crate) use impl_un_op;
