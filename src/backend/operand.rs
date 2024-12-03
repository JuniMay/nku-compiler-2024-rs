//! The operands in the backend framework.

use std::fmt;

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

/// The kind of a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegKind {
    /// The general purpose register.
    General,
}

/// The register.
/// 
/// It can be either a physical register or a virtual register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    P(PReg),
    V(VReg),
}

impl Reg {
    /// Get the kind of the register.
    pub fn kind(&self) -> RegKind {
        match self {
            Reg::P(preg) => preg.kind(),
            Reg::V(vreg) => vreg.kind(),
        }
    }

    /// Check if the register is a physical register.
    pub fn is_preg(&self) -> bool { matches!(self, Reg::P(_)) }

    /// Check if the register is a virtual register.
    pub fn is_vreg(&self) -> bool { matches!(self, Reg::V(_)) }
}

/// The physical register.
///
/// Cranelift uses a bit-encoded representation, but here just separate the
/// register number and the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(u8, RegKind);

impl PReg {
    /// Create a new physical register.
    pub const fn new(num: u8, kind: RegKind) -> Self { Self(num, kind) }

    /// Get the number of the register.
    pub const fn num(&self) -> u8 { self.0 }

    /// Get the kind of the register.
    pub const fn kind(&self) -> RegKind { self.1 }
}

/// The virtual register.
///
/// Let's hope the number of virtual registers does not exceed [u32::MAX].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(u32, RegKind);

impl VReg {
    /// Create a new virtual register.
    pub fn new(num: u32, kind: RegKind) -> Self { Self(num, kind) }

    /// Get the number of the register.
    pub fn num(&self) -> u32 { self.0 }

    /// Get the kind of the register.
    pub fn kind(&self) -> RegKind { self.1 }
}

impl From<VReg> for Reg {
    fn from(vreg: VReg) -> Self { Self::V(vreg) }
}

impl From<PReg> for Reg {
    fn from(preg: PReg) -> Self { Self::P(preg) }
}

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            match self.1 {
                RegKind::General => "$r",
            },
            self.0
        )
    }
}
