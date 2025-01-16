use core::fmt;

use super::block::MBlockData;
use super::func::{MFuncData, MLabel};
use super::inst::MInstData;
use super::regs::{RegKind, VReg};
use crate::infra::linked_list::LinkedListContainer;
use crate::infra::storage::GenericArena;

/// The raw data of the machine code.
/// e.g. the data section, the bss section.
pub enum RawData {
    /// Bytes of the data, declared in the data section.
    Bytes(Vec<u8>),
    /// Zero-initialized bytes of the data, declared in the bss section.
    ///
    /// The field is the size of the zero-initialized data.
    Bss(usize),
}

/// The context of the machine code.
#[derive(Default)]
pub struct MContext {
    /// The arena of the instructions.
    pub(super) insts: GenericArena<MInstData>,

    /// The arena of the blocks.
    pub(super) blocks: GenericArena<MBlockData>,

    /// The arena of the functions.
    pub(super) funcs: GenericArena<MFuncData>,

    /// raw data sections
    raw_data: Vec<(MLabel, RawData)>,

    /// The counter of the virtual registers.
    vreg_counter: u32,

    /// The architecture string.
    arch: String,
}

impl MContext {
    /// Create a new machine code context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new virtual register.
    pub fn new_vreg(&mut self, kind: RegKind) -> VReg {
        let vreg = VReg::new(self.vreg_counter, kind);
        self.vreg_counter += 1;
        vreg
    }

    /// Add a piece of raw data to the context.
    pub fn add_raw_data(&mut self, label: impl Into<MLabel>, data: RawData) {
        self.raw_data.push((label.into(), data));
    }

    /// Set the architecture string to `arch`.
    pub fn set_arch(&mut self, arch: impl Into<String>) {
        self.arch = arch.into();
    }

    /// Get the architecture string.
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Display the machine code context.
    pub fn display(&self) -> DisplayMContext {
        DisplayMContext { mctx: self }
    }
}

pub struct DisplayMContext<'a> {
    mctx: &'a MContext,
}

impl fmt::Display for DisplayMContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The architecture attribute.
        writeln!(f, "\t.attribute arch, \"{}\"", self.mctx.arch())?;

        // The text section, generating the code.
        writeln!(f, "\t.text")?;
        for func_data in self.mctx.funcs.iter() {
            let func = func_data.self_ptr();

            writeln!(f, "\t.globl {}", func.label(self.mctx))?;

            // Skip the external functions.
            if func.is_external(self.mctx) {
                continue;
            }

            writeln!(f, "\t.align 1")?;
            writeln!(f, "\t.type {}, @function", func.label(self.mctx))?;
            writeln!(f, "{}:", func.label(self.mctx))?;

            for block in func.iter(self.mctx) {
                writeln!(f, "{}:", block.label(self.mctx))?;
                for inst in block.iter(self.mctx) {
                    writeln!(f, "\t{}", inst.display(self.mctx))?;
                }
            }

            writeln!(f)?;
        }

        // The data section, generating the data.
        for (label, raw_data) in self.mctx.raw_data.iter() {
            writeln!(f, "\t.type {}, @object", label)?;
            match raw_data {
                RawData::Bytes(bytes) => {
                    writeln!(f, "\t.data")?;
                    writeln!(f, "\t.global {}", label)?;
                    writeln!(f, "\t.align 2")?;
                    writeln!(f, "{}:", label)?;
                    for byte in bytes.iter() {
                        writeln!(f, "\t.byte {}", byte)?;
                    }
                    writeln!(f)?;
                }
                RawData::Bss(size) => {
                    writeln!(f, "\t.bss")?;
                    writeln!(f, "\t.global {}", label)?;
                    writeln!(f, "\t.align 2")?;
                    writeln!(f, "{}:", label)?;
                    writeln!(f, "\t.zero {}", size)?;
                    writeln!(f)?;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
