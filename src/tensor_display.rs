use std::{
    fmt::{Display, Formatter},
    sync::LazyLock,
};

use prettytable::{Cell, Table, format};

use crate::{CpuRawTensor, Tensor, ToCpu, num::Bool, raw_tensor_cpu::CpuRawTensorImpl};

static FORMAT_TENSOR: LazyLock<format::TableFormat> = LazyLock::new(|| {
    format::FormatBuilder::new()
        .column_separator(' ')
        .borders('│')
        .separators(
            &[format::LinePosition::Top],
            format::LineSeparator::new(' ', ' ', '┌', '┐'),
        )
        .separators(
            &[format::LinePosition::Bottom],
            format::LineSeparator::new(' ', ' ', '└', '┘'),
        )
        .padding(1, 1)
        .build()
});

/// Returns a reference to the prettytable format for displaying tensors.
fn get_pretty_format() -> &'static format::TableFormat {
    &FORMAT_TENSOR
}

static FORMAT_TENSOR_SINGLE_LINE: LazyLock<format::TableFormat> = LazyLock::new(|| {
    format::FormatBuilder::new()
        .column_separator(' ')
        .left_border('[')
        .right_border(']')
        .padding(1, 0)
        .build()
});

fn get_single_line_format() -> &'static format::TableFormat {
    &FORMAT_TENSOR_SINGLE_LINE
}

// static mut FORMAT_TENSOR_NUMPY: Option<format::TableFormat> = None;
// static INIT_FORMAT_TENSOR_NUMPY: Once = Once::new();

// fn get_numpy_format() -> &'static format::TableFormat {
//     unsafe {
//         INIT_FORMAT_TENSOR_NUMPY.call_once(|| {
//             FORMAT_TENSOR_NUMPY = Some(
//                 format::FormatBuilder::new()
//                     .column_separator(' ')
//                     .left_border('[')
//                     .right_border(']')
//                     .separators(
//                         &[format::LinePosition::Top, format::LinePosition::Bottom],
//                         format::LineSeparator::new(' ', ' ', ' ', ' '),
//                     )
//                     .padding(1, 0)
//                     .build(),
//             );
//         });
//         return FORMAT_TENSOR_NUMPY.as_ref().unwrap();
//     }
// }

fn create_table<E: Bool + Display>(
    tensor: &Tensor<CpuRawTensor<E>, E, CpuRawTensorImpl>,
    table: &mut Table,
    precision: Option<usize>,
) {
    let shape = tensor.shape();

    if shape.len() == 2 {
        table.set_format(if shape[0] == 1 {
            *get_single_line_format()
        } else {
            *get_pretty_format()
        });
        for r in 0..shape[0] {
            let row = table.add_empty_row();
            for c in 0..shape[1] {
                if let Some(precision) = precision {
                    row.add_cell(Cell::new(&format!(
                        "{:.precision$}",
                        tensor.ix2(r, c).to_scalar(),
                    )));
                } else {
                    row.add_cell(Cell::new(&format!("{}", tensor.ix2(r, c).to_scalar())));
                }
            }
        }
    } else {
        table.set_format(*get_pretty_format());
        for r in 0..shape[0] {
            let row = table.add_empty_row();
            for c in 0..shape[1] {
                let mut table = Table::new();
                let tensor = tensor.ix2(r, c);
                create_table(&tensor, &mut table, precision);
                row.add_cell(Cell::new(&format!("{table}")));
            }
        }
    }
}

impl<T, E: Bool + Display, I: ToCpu<Repr<E> = T>> Display for Tensor<T, E, I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let cpu = if self.shape().len().is_multiple_of(2) {
            self.to_cpu()
        } else {
            self.reshape(&[&[1], self.shape()].concat()).to_cpu()
        };

        let mut table = Table::new();
        create_table(&cpu, &mut table, f.precision());
        write!(f, "{table}")
    }
}
