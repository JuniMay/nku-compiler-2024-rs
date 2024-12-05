//! The operands in the backend framework.

use super::regs::Reg;
use crate::ir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MOperandKind {
    Undef,
    Mem(MemLoc),
    /// Keep a register for all immediate values. Used for better codegen
    /// quality.
    ///
    /// Another solution is codegen in postorder (like Cranelift), but due to
    /// historical reasons, we use this solution.
    Imm(Reg, i64),
    Reg(Reg),
}

/// The operand in the assembly, this is made copy-able for convenience.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MOperand {
    /// Tracker for the corresponding type in IR.
    pub(super) ty: ir::Ty,
    /// Kind of this operand.
    pub(super) kind: MOperandKind,
}

/// A memory location
///
/// # Stack Layout
///
/// The stack layout of a function is as follows:
///
/// ```text
///              param by stack #n
///                     ...          -----> maybe depends on the calling convention
///              param by stack #0
///       high +-------------------+ <-- frame pointer
///            |  Saved Registers  |
///        |   +- - - - - - - - - -+
///        |   | (maybe alignment) |
///        |   +- - - - - - - - - -+ <-- start of local slots
///        |   |                   |
///  grow  |   |  Local Variables  |
///        |   |                   |
///        V   +- - - - - - - - - -+
///            |  for arg passing  |
///       low  +-------------------+ <-- stack pointer
///                     ...
/// ```
///
/// We can index local variables/slots with frame pointer or stack pointer. The
/// benefit of using stack pointer is that we can allocate fp as a general
/// purpose register. But either way, we need to modify the offset of the slots
/// after register allocation.
///
/// Actually, the space for arg passing can be dynamically allocated, but it
/// will generate more add/sub instructions. So we can reserve a fixed space
/// (with maximum required space) for arg passing at the prologue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemLoc {
    /// The memory location is a register + offset.
    RegOffset { base: Reg, offset: i64 },
    /// The memory location is a stack slot.
    ///
    /// Usually, the offset is based on frame pointer, after generating all
    /// instruction and finishing register allocation, this kind of location
    /// should be converted to [MemLoc::RegOffset], with an explicit base
    /// register, e.g., frame pointer.
    ///
    /// The offset represents the relative offset to the start of local slots,
    /// and all the slots should be modified after all the instructions are
    /// generated, and the registers are allocated.
    ///
    /// A scratch register should be reserved if the target needs a place to
    /// hold offset. e.g., for RISC-V, if the immediate cannot fit in the
    /// offset field of load/store, a scratch register will be needed to load
    /// the immediate first.
    ///
    /// XXX: if we index local slots with stack pointer, and place the offset
    /// after the saved registers we can determine the offset from sp before
    /// register allocation, and the scratch register is no longer needed.
    Slot { offset: i64 },
    /// An incoming parameter.
    ///
    /// The offset is relative to the frame pointer, i.e., the start of the
    /// saved registers.
    Incoming { offset: i64 },
}
