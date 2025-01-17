use std::fmt::{self, write};

use super::block::MBlock;
use super::context::MContext;
use super::func::MLabel;
use super::imm::Imm12;
use super::operand::MemLoc;
use super::regs::{Reg, RegKind};
use crate::infra::linked_list::LinkedListNode;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

/// The data of the machine instruction.
pub struct MInstData {
    kind: MInstKind,
    next: Option<MInst>,
    prev: Option<MInst>,
    parent: Option<MBlock>,
}

/// The machine instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MInst(GenericPtr<MInstData>);

impl MInst {
    pub fn new(mctx: &mut MContext, kind: MInstKind) -> Self {
        mctx.alloc_with(|inst| MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        })
    }
}

/// Kinds of machine instructions.
///
/// The instructions are classified with its format. This classification is
/// derived from cranelift.
pub enum MInstKind {
    /// ALU instructions with two registers (rd and rs) and an immediate.
    AluRRI {
        op: AluOpRRI,
        rd: Reg,
        rs: Reg,
        imm: Imm12,
    },
    /// ALU instructions with three registers (rd, and two rs-s).
    AluRRR {
        op: AluOpRRR,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },
    /// FPU instructions with two registers (rd, and two rs-s).
    FpuRRR {
        op: FpuOpRRR,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },
    /// Load instructions.
    Load { op: LoadOp, rd: Reg, loc: MemLoc },
    /// Store instructions.
    Store { op: StoreOp, rs: Reg, loc: MemLoc },
    /// Load immediate pseudo instruction.
    Li { rd: Reg, imm: u64 },
    /// Load address pseudo instruction.
    La { rd: Reg, loc: String },
    /// Jump instructions.
    J {
        target: MBlock,
        rd: Reg,
        rs: Option<Reg>,
    },
    /// Call instructions.
    Call { target: MLabel },
    /// FPU move instructions.
    FpuMove { op: FpuMoveOp, rd: Reg, rs: Reg },
    /// Branch instructions.
    Branch {
        op: BranchOp,
        rs1: Reg,
        rs2: Reg,
        target: MBlock,
    },
}

#[derive(Copy, Clone)]
pub enum LoadOp {
    // 伪指令
    La,
    // 基本指令
    Lb,
    Lh,
    Lw,
    Ld,
    Lbu,
    Lhu,
    Lwu,
    Flw,
    Fld,
}

impl fmt::Display for LoadOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LoadOp::La => write!(f, "la"),
            LoadOp::Lb => write!(f, "lb"),
            LoadOp::Lh => write!(f, "lh"),
            LoadOp::Lw => write!(f, "lw"),
            LoadOp::Ld => write!(f, "ld"),
            LoadOp::Lbu => write!(f, "lbu"),
            LoadOp::Lhu => write!(f, "lhu"),
            LoadOp::Lwu => write!(f, "lwu"),
            LoadOp::Flw => write!(f, "flw"),
            LoadOp::Fld => write!(f, "fld"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum StoreOp {
    Sb,
    Sh,
    Sw,
    Sd,
    Fsw,
    Fsd,
}

impl fmt::Display for StoreOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StoreOp::Sb => write!(f, "sb"),
            StoreOp::Sh => write!(f, "sh"),
            StoreOp::Sw => write!(f, "sw"),
            StoreOp::Sd => write!(f, "sd"),
            StoreOp::Fsw => write!(f, "fsw"),
            StoreOp::Fsd => write!(f, "fsd"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum AluOpRRI {
    Addi,
    Addiw,
    Slli,
    Slliw,
    Srli,
    Srliw,
    Srai,
    Sraiw,
    Xori,
    Ori,
    Andi,
    Slti,
    Sltiu,
}

impl fmt::Display for AluOpRRI {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AluOpRRI::Addi => write!(f, "addi"),
            AluOpRRI::Addiw => write!(f, "addiw"),
            AluOpRRI::Slli => write!(f, "slli"),
            AluOpRRI::Slliw => write!(f, "slliw"),
            AluOpRRI::Srli => write!(f, "srli"),
            AluOpRRI::Srliw => write!(f, "srliw"),
            AluOpRRI::Srai => write!(f, "srai"),
            AluOpRRI::Sraiw => write!(f, "sraiw"),
            AluOpRRI::Xori => write!(f, "xori"),
            AluOpRRI::Ori => write!(f, "ori"),
            AluOpRRI::Andi => write!(f, "andi"),
            AluOpRRI::Slti => write!(f, "slti"),
            AluOpRRI::Sltiu => write!(f, "sltiu"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum AluOpRRR {
    // rv64gc
    Add,
    Addw,
    Sub,
    Subw,
    Sll,
    Sllw,
    Srl,
    Srlw,
    Sra,
    Sraw,
    Xor,
    Or,
    And,
    Mul,
    Mulw,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divw,
    Divu,
    Divuw,
    Rem,
    Remw,
    Remu,
    Remuw,
    Rew,
    // comp
    Slt,
    Sltu,
}

impl fmt::Display for AluOpRRR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AluOpRRR::Add => write!(f, "add"),
            AluOpRRR::Addw => write!(f, "addw"),
            AluOpRRR::Sub => write!(f, "sub"),
            AluOpRRR::Subw => write!(f, "subw"),
            AluOpRRR::Sll => write!(f, "sll"),
            AluOpRRR::Sllw => write!(f, "sllw"),
            AluOpRRR::Srl => write!(f, "srl"),
            AluOpRRR::Srlw => write!(f, "srlw"),
            AluOpRRR::Sra => write!(f, "sra"),
            AluOpRRR::Sraw => write!(f, "sraw"),
            AluOpRRR::Xor => write!(f, "xor"),
            AluOpRRR::Or => write!(f, "or"),
            AluOpRRR::And => write!(f, "and"),
            AluOpRRR::Slt => write!(f, "slt"),
            AluOpRRR::Sltu => write!(f, "sltu"),
            AluOpRRR::Mul => write!(f, "mul"),
            AluOpRRR::Mulw => write!(f, "mulw"),
            AluOpRRR::Mulh => write!(f, "mulh"),
            AluOpRRR::Mulhsu => write!(f, "mulhsu"),
            AluOpRRR::Mulhu => write!(f, "mulhu"),
            AluOpRRR::Div => write!(f, "div"),
            AluOpRRR::Divw => write!(f, "divw"),
            AluOpRRR::Divu => write!(f, "divu"),
            AluOpRRR::Divuw => write!(f, "divuw"),
            AluOpRRR::Rem => write!(f, "rem"),
            AluOpRRR::Remw => write!(f, "remw"),
            AluOpRRR::Remu => write!(f, "remu"),
            AluOpRRR::Remuw => write!(f, "remuw"),
            AluOpRRR::Rew => write!(f, "rew"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum FpuOpRRR {
    FaddS,
    FaddD,
    FsubS,
    FsubD,
    FmulS,
    FmulD,
    FdivS,
    FdivD,
    FCmp { op: FpuCompareOp },
}

impl fmt::Display for FpuOpRRR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FpuOpRRR::FaddS => write!(f, "fadd.s"),
            FpuOpRRR::FaddD => write!(f, "fadd.d"),
            FpuOpRRR::FsubS => write!(f, "fsub.s"),
            FpuOpRRR::FsubD => write!(f, "fsub.d"),
            FpuOpRRR::FmulS => write!(f, "fmul.s"),
            FpuOpRRR::FmulD => write!(f, "fmul.d"),
            FpuOpRRR::FdivS => write!(f, "fdiv.s"),
            FpuOpRRR::FdivD => write!(f, "fdiv.d"),
            FpuOpRRR::FCmp { op } => write!(f, "{}", op),
        }
    }
}

#[derive(Copy, Clone)]
pub enum CompareOp {
    Slt,
    Sltu,
    Seqz,
    Snez,
    Sgtz,
    Sltz,
}

#[derive(Copy, Clone)]
pub enum FpuCompareOp {
    FeqS,
    FeqD,
    FltS,
    FltD,
    FleS,
    FleD,
}

impl fmt::Display for FpuCompareOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FpuCompareOp::FeqS => write!(f, "feq.s"),
            FpuCompareOp::FeqD => write!(f, "feq.d"),
            FpuCompareOp::FltS => write!(f, "flt.s"),
            FpuCompareOp::FltD => write!(f, "flt.d"),
            FpuCompareOp::FleS => write!(f, "fle.s"),
            FpuCompareOp::FleD => write!(f, "fle.d"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum FpuMoveOp {
    FmvXS,
    FmvXD,
    FmvSX,
    FmvDX,
    FmvWX,
    FmvXW,
}

impl fmt::Display for FpuMoveOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FpuMoveOp::FmvXS => write!(f, "fmv.x.s"),
            FpuMoveOp::FmvXD => write!(f, "fmv.x.s"),
            FpuMoveOp::FmvSX => write!(f, "fmv.s.x"),
            FpuMoveOp::FmvDX => write!(f, "fmv.d.x"),
            FpuMoveOp::FmvWX => write!(f, "fmv.w.x"),
            FpuMoveOp::FmvXW => write!(f, "fmv.x.w"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum BranchOp {
    Beq,
    Bne,
    Blt,
    Bge,
    Bgt,
    Ble,
    Bltu,
    Bgeu,
}

impl fmt::Display for BranchOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BranchOp::Beq => write!(f, "beq"),
            BranchOp::Bne => write!(f, "bne"),
            BranchOp::Blt => write!(f, "blt"),
            BranchOp::Bge => write!(f, "bge"),
            BranchOp::Bgt => write!(f, "bgt"),
            BranchOp::Ble => write!(f, "ble"),
            BranchOp::Bltu => write!(f, "bltu"),
            BranchOp::Bgeu => write!(f, "bgeu"),
        }
    }
}

pub struct DisplayMInst<'a> {
    mctx: &'a MContext,
    inst: MInst,
}

impl MInst {
    pub fn kind(self, mctx: &MContext) -> &MInstKind {
        &self.deref(mctx).kind
    }

    pub fn kind_mut(self, mctx: &mut MContext) -> &mut MInstKind {
        &mut self.deref_mut(mctx).kind
    }

    pub fn display(self, mctx: &MContext) -> DisplayMInst {
        DisplayMInst { mctx, inst: self }
    }

    // XXX: These instruction creation methods are just for demonstration.
    // You can refactor them as you need.

    /// Create a new `li` instruction.
    ///
    /// imm: The immediate value.
    ///
    /// Returns (inst, rd).
    pub fn li(mctx: &mut MContext, imm: u64) -> (Self, Reg) {
        let rd = mctx.new_vreg(RegKind::General).into();
        let kind = MInstKind::Li { rd, imm };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        let inst = mctx.alloc(data);
        (inst, rd)
    }

    /// Create a new `load` instruction.
    ///
    /// op: LoadOp
    /// loc: The memory location.
    ///
    /// Returns (inst, rd).
    pub fn load(mctx: &mut MContext, op: LoadOp, loc: MemLoc) -> (Self, Reg) {
        let rd = mctx.new_vreg(RegKind::General).into();
        let kind = MInstKind::Load { op, rd, loc };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        let inst = mctx.alloc(data);
        (inst, rd)
    }

    /// Create a new `store` instruction.
    ///
    /// op: StoreOp
    /// rs: The source register.
    /// loc: The memory location.
    ///
    /// Returns the instruction.
    pub fn store(mctx: &mut MContext, op: StoreOp, rs: Reg, loc: MemLoc) -> Self {
        let kind = MInstKind::Store { op, rs, loc };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    /// Create a new `alu_rrr` instruction.
    ///
    /// op: AluOpRRI
    /// rs1: The first source register.
    /// rs2: The second source register.
    ///
    /// Returns (inst, rd).
    pub fn alu_rrr(mctx: &mut MContext, op: AluOpRRR, rs1: Reg, rs2: Reg) -> (Self, Reg) {
        let rd = mctx.new_vreg(RegKind::General).into();
        let kind = MInstKind::AluRRR { op, rd, rs1, rs2 };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        let inst = mctx.alloc(data);
        (inst, rd)
    }

    /// Create a new `alu_rri` instruction.
    ///
    /// op: AluOpRRI
    /// rs: The source register.
    /// imm: The immediate value.
    ///
    /// Returns (inst, rd).
    pub fn alu_rri(mctx: &mut MContext, op: AluOpRRI, rs: Reg, imm: Imm12) -> (Self, Reg) {
        let rd = mctx.new_vreg(RegKind::General).into();
        let kind = MInstKind::AluRRI { op, rd, rs, imm };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        let inst = mctx.alloc(data);
        (inst, rd)
    }

    /// Create a new `alu_rri` instruction using raw values.
    /// This is useful when you want to create an instruction without allocating
    /// a new register.
    ///
    /// op: AluOpRRI
    /// rd: The destination register.
    /// rs: The source register.
    /// imm: The immediate value.
    ///
    /// Returns the instruction.
    pub fn raw_alu_rri(mctx: &mut MContext, op: AluOpRRI, rd: Reg, rs: Reg, imm: Imm12) -> Self {
        let kind = MInstKind::AluRRI { op, rd, rs, imm };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    /// Creatr a new `jump` instruction.
    ///
    /// target: The target block.
    ///
    /// Returns the instruction.
    pub fn j(mctx: &mut MContext, rs: Option<Reg>, target: MBlock) -> Self {
        let rd = mctx.new_vreg(RegKind::General).into();
        let kind = MInstKind::J { rd, rs, target };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    // TODO: add more instruction creation methods as you need.

    /// Create a new `fpu_rrr` instruction.
    ///
    /// op: FpuOpRRI
    /// rs1: The first source register.
    /// rs2: The second source register.
    ///
    /// Returns (inst, rd).
    pub fn fpu_rrr(mctx: &mut MContext, op: FpuOpRRR, rs1: Reg, rs2: Reg) -> (Self, Reg) {
        let rd = mctx.new_vreg(RegKind::Float).into();
        let kind = MInstKind::FpuRRR { op, rd, rs1, rs2 };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        let inst = mctx.alloc(data);
        (inst, rd)
    }

    /// Create a new `compare` instruction.
    ///    
    /// op: CompareOp
    /// rs1: The first source register.
    /// rs2: The second source register.
    ///
    /// Returns the instruction.
    pub fn fpu_move(mctx: &mut MContext, op: FpuMoveOp, rd: Reg, rs: Reg) -> Self {
        let kind = MInstKind::FpuMove { op, rd, rs };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    /// Create a new `call` instruction.
    ///
    /// target: The target block.
    ///
    /// Returns the instruction.
    pub fn call(mctx: &mut MContext, target: MLabel) -> Self {
        let kind = MInstKind::Call { target };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    /// Create a new `load address` instruction.
    pub fn la(mctx: &mut MContext, loc: &String) -> (Self, Reg) {
        let rd = mctx.new_vreg(RegKind::General).into();
        let kind = MInstKind::La { rd, loc: loc.clone() };
        let data = MInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        (mctx.alloc(data), rd)
    }

    pub fn replace(&mut self, mctx: &mut MContext, new_kind: MInstKind) {
        // 获取 `MInst` 的可变引用
        let data = self.deref_mut(mctx);
        // 替换 `kind`
        data.kind = new_kind;
    }

}

impl fmt::Display for DisplayMInst<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inst.deref(self.mctx).kind {
            MInstKind::Li { rd, imm } => write!(f, "li {}, {}", rd, imm),
            MInstKind::La { rd, loc } => write!(f, "la {}, {}", rd, loc),
            MInstKind::Load { op, rd, loc } => {
                let slot = match loc {
                    MemLoc::RegOffset { base, offset } => {
                        format!("{}({})", offset, base)
                    }
                    MemLoc::Slot { offset } => {
                        format!("{}(??? SLOT)", offset)
                    }
                    MemLoc::Incoming { offset } => {
                        format!("{}(??? INCOMING)", offset)
                    }
                };
                write!(f, "{} {}, {}", op, rd, slot)
            }
            MInstKind::Store { op, rs, loc } => {
                let slot = match loc {
                    MemLoc::RegOffset { base, offset } => {
                        format!("{}({})", offset, base)
                    }
                    MemLoc::Slot { offset } => {
                        format!("{}(??? SLOT)", offset)
                    }
                    MemLoc::Incoming { offset } => {
                        format!("{}(??? INCOMING)", offset)
                    }
                };
                write!(f, "{} {}, {}", op, rs, slot)
            }
            MInstKind::AluRRR { op, rd, rs1, rs2 } => write!(f, "{} {}, {}, {}", op, rd, rs1, rs2),
            MInstKind::AluRRI { op, rd, rs, imm } => write!(f, "{} {}, {}, {}", op, rd, rs, imm),
            MInstKind::J { target, rd, rs } => {
                if let Some(rs) = rs {
                    write!(f, "jalr {}, {}, {}", rd, rs, &target.label(self.mctx).to_string()[1..])
                } else {
                    write!(f, "jal {}, {}", rd, &target.label(self.mctx).to_string()[1..])
                }
            }
            MInstKind::FpuRRR { op, rd, rs1, rs2 } => write!(f, "{} {}, {}, {}", op, rd, rs1, rs2),
            MInstKind::Call { target } => write!(f, "call {}", target),
            MInstKind::FpuMove { op, rd, rs } => write!(f, "{} {}, {}", op, rd, rs),
            MInstKind::Branch {
                op,
                rs1,
                rs2,
                target,
            } => write!(f, "{} {}, {}, {}", op, rs1, rs2, &target.label(self.mctx).to_string()[1..]),
        }
    }
}

impl ArenaPtr for MInst {
    type Arena = MContext;
    type Data = MInstData;
}

impl LinkedListNode for MInst {
    type Container = MBlock;
    type Ctx = MContext;

    fn next(self, ctx: &Self::Ctx) -> Option<Self> {
        self.deref(ctx).next
    }

    fn prev(self, ctx: &Self::Ctx) -> Option<Self> {
        self.deref(ctx).prev
    }

    fn set_next(self, arena: &mut Self::Ctx, next: Option<Self>) {
        self.deref_mut(arena).next = next;
    }

    fn set_prev(self, arena: &mut Self::Ctx, prev: Option<Self>) {
        self.deref_mut(arena).prev = prev;
    }

    fn container(self, ctx: &Self::Ctx) -> Option<Self::Container> {
        self.deref(ctx).parent
    }

    fn set_container(self, arena: &mut Self::Ctx, container: Option<Self::Container>) {
        self.deref_mut(arena).parent = container;
    }
}

impl Arena<MInst> for MContext {
    fn alloc_with<F>(&mut self, f: F) -> MInst
    where
        F: FnOnce(MInst) -> MInstData,
    {
        MInst(self.insts.alloc_with(|p| f(MInst(p))))
    }

    fn try_deref(&self, ptr: MInst) -> Option<&MInstData> {
        self.insts.try_deref(ptr.0)
    }

    fn try_deref_mut(&mut self, ptr: MInst) -> Option<&mut MInstData> {
        self.insts.try_deref_mut(ptr.0)
    }

    fn try_dealloc(&mut self, ptr: MInst) -> Option<MInstData> {
        self.insts.try_dealloc(ptr.0)
    }
}

impl MInstKind {
    pub fn extract_mem_loc_mut(&mut self) -> Option<&mut MemLoc> {
        match self {
            MInstKind::Load { loc, .. } => Some(loc),
            MInstKind::Store { loc, .. } => Some(loc),
            _ => None,
        }
    }
}
