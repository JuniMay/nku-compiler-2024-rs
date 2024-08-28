use std::fmt;

use super::imm::Imm12;
use super::lower::RvLowerSpec;
use super::regs;
use crate::backend::block::MBlock;
use crate::backend::context::MContext;
use crate::backend::func::{MFunc, MLabel};
use crate::backend::inst::{DisplayMInst, MInst};
use crate::backend::lower::{LowerConfig, MemLoc};
use crate::backend::regs::Reg;
use crate::backend::{PReg, RegKind};
use crate::infra::linked_list::LinkedListNode;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub struct RvInstData {
    kind: RvInstKind,
    next: Option<RvInst>,
    prev: Option<RvInst>,
    parent: Option<MBlock<RvInst>>,
}

impl Clone for RvInstData {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            next: None,
            prev: None,
            parent: None,
        }
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct RvInst(GenericPtr<RvInstData>);

impl RvInst {
    pub fn kind(self, mctx: &MContext<Self>) -> &RvInstKind { &self.deref(mctx).kind }

    pub fn build_li(mctx: &mut MContext<Self>, rd: Reg, imm: impl Into<u64>) -> Self {
        let kind = RvInstKind::Li {
            rd,
            imm: imm.into(),
        };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn build_alu_rri(
        mctx: &mut MContext<Self>,
        op: AluOpRRI,
        rd: Reg,
        rs: Reg,
        imm: Imm12,
    ) -> Self {
        let kind = RvInstKind::AluRRI { op, rd, rs, imm };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn build_alu_rrr(
        mctx: &mut MContext<Self>,
        op: AluOpRRR,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    ) -> Self {
        let kind = RvInstKind::AluRRR { op, rd, rs1, rs2 };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn build_load(mctx: &mut MContext<Self>, op: LoadOp, rd: Reg, loc: MemLoc) -> Self {
        let kind = RvInstKind::Load { op, rd, loc };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn build_la(mctx: &mut MContext<Self>, rd: Reg, label: MLabel) -> Self {
        let kind = RvInstKind::La { rd, label };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn build_load_addr(mctx: &mut MContext<Self>, rd: Reg, loc: MemLoc) -> Self {
        let kind = RvInstKind::LoadAddr { rd, loc };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn store(mctx: &mut MContext<Self>, op: StoreOp, src: Reg, loc: MemLoc) -> Self {
        let kind = RvInstKind::Store { op, src, loc };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn ret(mctx: &mut MContext<Self>) -> Self {
        let kind = RvInstKind::Ret;
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn call(mctx: &mut MContext<Self>, func: MFunc<Self>, arg_regs: Vec<PReg>) -> Self {
        let kind = RvInstKind::Call { func, arg_regs };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn j(mctx: &mut MContext<Self>, block: MBlock<Self>) -> Self {
        let kind = RvInstKind::J { block };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn br(
        mctx: &mut MContext<Self>,
        op: BrOp,
        rs1: Reg,
        rs2: Reg,
        block: MBlock<Self>,
    ) -> Self {
        let kind = RvInstKind::Br {
            op,
            rs1,
            rs2,
            block,
        };
        let data = RvInstData {
            kind,
            next: None,
            prev: None,
            parent: None,
        };
        mctx.alloc(data)
    }

    pub fn li(mctx: &mut MContext<Self>, imm: impl Into<u64>) -> (Self, Reg) {
        let rd: Reg = mctx.new_vreg(RegKind::General).into();
        let inst = Self::build_li(mctx, rd, imm);
        (inst, rd)
    }

    pub fn alu_rri(mctx: &mut MContext<Self>, op: AluOpRRI, rs: Reg, imm: Imm12) -> (Self, Reg) {
        let rd: Reg = mctx.new_vreg(RegKind::General).into();
        let inst = Self::build_alu_rri(mctx, op, rd, rs, imm);
        (inst, rd)
    }

    pub fn alu_rrr(mctx: &mut MContext<Self>, op: AluOpRRR, rs1: Reg, rs2: Reg) -> (Self, Reg) {
        let rd: Reg = mctx.new_vreg(RegKind::General).into();
        let inst = Self::build_alu_rrr(mctx, op, rd, rs1, rs2);
        (inst, rd)
    }

    pub fn load(mctx: &mut MContext<Self>, op: LoadOp, loc: MemLoc) -> (Self, Reg) {
        let rd = match op {
            LoadOp::Lb
            | LoadOp::Lh
            | LoadOp::Lw
            | LoadOp::Ld
            | LoadOp::Lbu
            | LoadOp::Lhu
            | LoadOp::Lwu => mctx.new_vreg(RegKind::General).into(),
        };
        let inst = Self::build_load(mctx, op, rd, loc);
        (inst, rd)
    }

    pub fn la(mctx: &mut MContext<Self>, label: MLabel) -> (Self, Reg) {
        let rd: Reg = mctx.new_vreg(RegKind::General).into();
        let inst = Self::build_la(mctx, rd, label);
        (inst, rd)
    }

    pub fn load_addr(mctx: &mut MContext<Self>, loc: MemLoc) -> (Self, Reg) {
        let rd: Reg = mctx.new_vreg(RegKind::General).into();
        let inst = Self::build_load_addr(mctx, rd, loc);
        (inst, rd)
    }
}

#[derive(Clone)]
pub enum RvInstKind {
    Li {
        rd: Reg,
        imm: u64,
    },
    AluRRI {
        op: AluOpRRI,
        rd: Reg,
        rs: Reg,
        imm: Imm12,
    },
    AluRRR {
        op: AluOpRRR,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },
    Load {
        op: LoadOp,
        rd: Reg,
        loc: MemLoc,
    },
    Store {
        op: StoreOp,
        src: Reg,
        loc: MemLoc,
    },
    Ret,
    Call {
        func: MFunc<RvInst>,
        arg_regs: Vec<PReg>,
    },
    J {
        block: MBlock<RvInst>,
    },
    Br {
        op: BrOp,
        rs1: Reg,
        rs2: Reg,
        block: MBlock<RvInst>,
    },
    La {
        rd: Reg,
        label: MLabel,
    },
    /// Load the address of a memory location into a register.
    ///
    /// Because we don't know the place of local slots in the memory, so we must
    /// create such a forward reference to load the address of a local slot.
    /// After generating all code (and register allocation), we should modify
    /// all the [MemLoc::Slot] into [MemLoc::RegOffset] and make this into a
    /// addi or li + add sequence.
    LoadAddr {
        rd: Reg,
        loc: MemLoc,
    },
}

pub struct DisplayRvInst<'a> {
    mctx: &'a MContext<RvInst>,
    inst: RvInst,
}

impl<'a> DisplayMInst<'a> for RvInst {
    type Display = DisplayRvInst<'a>;

    fn display(self, mctx: &'a MContext<Self>) -> Self::Display {
        DisplayRvInst { mctx, inst: self }
    }
}

impl<'a> fmt::Display for DisplayRvInst<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use RvInstKind as Ik;

        match &self.inst.deref(self.mctx).kind {
            Ik::Li { rd, imm } => {
                if *imm < 4096 {
                    write!(f, "li {}, {}", regs::display(*rd), imm)
                } else {
                    write!(f, "li {}, {:#018x}", regs::display(*rd), imm)
                }
            }
            Ik::AluRRI { op, rd, rs, imm } => {
                write!(
                    f,
                    "{} {}, {}, {}",
                    op,
                    regs::display(*rd),
                    regs::display(*rs),
                    imm
                )
            }
            Ik::AluRRR { op, rd, rs1, rs2 } => {
                write!(
                    f,
                    "{} {}, {}, {}",
                    op,
                    regs::display(*rd),
                    regs::display(*rs1),
                    regs::display(*rs2)
                )
            }
            Ik::Load { op, rd, loc } => {
                let slot = match loc {
                    MemLoc::RegOffset { base, offset } => {
                        format!("{}({})", offset, regs::display(*base))
                    }
                    MemLoc::Slot { offset } => {
                        format!("{}(??? SLOT)", offset)
                    }
                    MemLoc::Incoming { offset } => {
                        format!("{}(??? INCOMING)", offset)
                    }
                };
                write!(f, "{} {}, {}", op, regs::display(*rd), slot)
            }
            Ik::Store { op, src, loc } => {
                let slot = match loc {
                    MemLoc::RegOffset { base, offset } => {
                        format!("{}({})", offset, regs::display(*base))
                    }
                    MemLoc::Slot { offset } => {
                        format!("{}(??? SLOT)", offset)
                    }
                    MemLoc::Incoming { offset } => {
                        format!("{}(??? INCOMING)", offset)
                    }
                };
                write!(f, "{} {}, {}", op, regs::display(*src), slot)
            }
            Ik::Ret => write!(f, "ret"),
            Ik::Call { func, .. } => write!(f, "call {}", func.label(self.mctx)),
            Ik::J { block } => write!(f, "j {}", block.label(self.mctx)),
            Ik::Br {
                op,
                rs1,
                rs2,
                block,
            } => {
                write!(
                    f,
                    "{} {}, {}, {}",
                    op,
                    regs::display(*rs1),
                    regs::display(*rs2),
                    block.label(self.mctx)
                )
            }
            Ik::La { rd, label } => write!(f, "la {}, {}", regs::display(*rd), label),
            Ik::LoadAddr { rd, loc } => {
                // unreachable!("load addr should be converted to li + add or addi")
                let slot = match loc {
                    MemLoc::RegOffset { base, offset } => {
                        format!("{}({} REG_OFFSET)", offset, regs::display(*base))
                    }
                    MemLoc::Slot { offset } => {
                        format!("{}(??? SLOT)", offset)
                    }
                    MemLoc::Incoming { offset } => {
                        format!("{}(??? INCOMING)", offset)
                    }
                };
                write!(f, "LOAD_ADDR??? {}, {}", regs::display(*rd), slot)
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BrOp {
    Beq,
    Bne,
    Blt,
    Bge,
    Bltu,
    Bgeu,
}

impl fmt::Display for BrOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BrOp::Beq => write!(f, "beq"),
            BrOp::Bne => write!(f, "bne"),
            BrOp::Blt => write!(f, "blt"),
            BrOp::Bge => write!(f, "bge"),
            BrOp::Bltu => write!(f, "bltu"),
            BrOp::Bgeu => write!(f, "bgeu"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum AluOpRRR {
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
    Slt,
    Sltu,
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
pub enum LoadOp {
    Lb,
    Lh,
    Lw,
    Ld,
    Lbu,
    Lhu,
    Lwu,
}

impl fmt::Display for LoadOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LoadOp::Lb => write!(f, "lb"),
            LoadOp::Lh => write!(f, "lh"),
            LoadOp::Lw => write!(f, "lw"),
            LoadOp::Ld => write!(f, "ld"),
            LoadOp::Lbu => write!(f, "lbu"),
            LoadOp::Lhu => write!(f, "lhu"),
            LoadOp::Lwu => write!(f, "lwu"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum StoreOp {
    Sb,
    Sh,
    Sw,
    Sd,
}

impl fmt::Display for StoreOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StoreOp::Sb => write!(f, "sb"),
            StoreOp::Sh => write!(f, "sh"),
            StoreOp::Sw => write!(f, "sw"),
            StoreOp::Sd => write!(f, "sd"),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Frm {
    Rne,
    Rtz,
    Rdn,
    Rup,
    Rmm,
    Dyn,
}

impl fmt::Display for Frm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Frm::Rne => write!(f, ", rne"),
            Frm::Rtz => write!(f, ", rtz"),
            Frm::Rdn => write!(f, ", rdn"),
            Frm::Rup => write!(f, ", rup"),
            Frm::Rmm => write!(f, ", rmm"),
            Frm::Dyn => write!(f, ""), // not sure if this is correct
        }
    }
}

impl MInst for RvInst {
    type S = RvLowerSpec;

    fn from_ptr(ptr: GenericPtr<Self::Data>) -> Self { Self(ptr) }

    fn ptr(self) -> GenericPtr<Self::Data> { self.0 }

    fn uses(self, mctx: &MContext<Self>, config: &LowerConfig) -> Vec<Reg> {
        use RvInstKind as Ik;

        match &self.deref(mctx).kind {
            Ik::Li { .. } => vec![],
            Ik::AluRRI { rs, .. } => vec![*rs],
            Ik::AluRRR { rs1, rs2, .. } => vec![*rs1, *rs2],
            Ik::Load { loc, .. } => match loc {
                MemLoc::RegOffset { base, .. } => vec![*base],
                MemLoc::Slot { .. } | MemLoc::Incoming { .. } => {
                    if config.omit_frame_pointer {
                        vec![regs::sp().into()]
                    } else {
                        vec![regs::fp().into()]
                    }
                }
            },
            Ik::Store { src, loc, .. } => {
                let mut uses = vec![*src];
                match loc {
                    MemLoc::RegOffset { base, .. } => uses.push(*base),
                    // XXX: Slot will use a frame pointer to index local variables
                    // but if it is omitted, it will fall back to the stack pointer
                    MemLoc::Slot { .. } | MemLoc::Incoming { .. } => {
                        if config.omit_frame_pointer {
                            uses.push(regs::sp().into());
                        } else {
                            uses.push(regs::fp().into());
                        }
                    }
                }
                uses
            }
            Ik::Ret => vec![],
            Ik::Call { arg_regs, .. } => arg_regs.iter().map(|r| (*r).into()).collect(),
            Ik::J { .. } => vec![],
            Ik::Br { rs1, rs2, .. } => vec![*rs1, *rs2],
            Ik::La { .. } => vec![],
            Ik::LoadAddr { loc, .. } => match loc {
                MemLoc::RegOffset { base, .. } => vec![*base],
                MemLoc::Slot { .. } | MemLoc::Incoming { .. } => {
                    if config.omit_frame_pointer {
                        vec![regs::sp().into()]
                    } else {
                        vec![regs::fp().into()]
                    }
                }
            },
        }
    }

    fn defs(self, mctx: &MContext<Self>, _config: &LowerConfig) -> Vec<Reg> {
        use RvInstKind as Ik;

        match &self.deref(mctx).kind {
            Ik::Li { rd, .. } => vec![*rd],
            Ik::AluRRI { rd, .. } => vec![*rd],
            Ik::AluRRR { rd, .. } => vec![*rd],
            Ik::Load { rd, .. } => vec![*rd],
            Ik::Store { .. } => vec![],
            Ik::Ret => vec![],
            Ik::Call { .. } => regs::RETURN_REGS.iter().map(|r| (*r).into()).collect(),
            Ik::J { .. } => vec![],
            Ik::Br { .. } => vec![],
            Ik::La { rd, .. } => vec![*rd],
            Ik::LoadAddr { rd, .. } => vec![*rd],
        }
    }

    fn clobbers(self, mctx: &MContext<Self>, _config: &LowerConfig) -> Vec<Reg> {
        use RvInstKind as Ik;

        match &self.deref(mctx).kind {
            Ik::Li { rd, .. } => vec![*rd],
            Ik::AluRRI { rd, .. } => vec![*rd],
            Ik::AluRRR { rd, .. } => vec![*rd],
            Ik::Load { rd, .. } => vec![*rd],
            Ik::Store { .. } => vec![],
            Ik::Ret => vec![],
            Ik::Call { .. } => regs::CALLER_SAVED_REGS
                .iter()
                .map(|r| (*r).into())
                .collect(),
            Ik::J { .. } => vec![],
            Ik::Br { .. } => vec![],
            Ik::La { rd, .. } => vec![*rd],
            Ik::LoadAddr { rd, .. } => vec![*rd],
        }
    }

    fn all_regs(self, mctx: &MContext<Self>, _config: &LowerConfig) -> Vec<Reg> {
        use RvInstKind as Ik;

        match &self.deref(mctx).kind {
            Ik::Li { rd, .. } => vec![*rd],
            Ik::AluRRI { rd, rs, .. } => vec![*rd, *rs],
            Ik::AluRRR { rd, rs1, rs2, .. } => vec![*rd, *rs1, *rs2],
            Ik::Load { rd, loc, .. } => match loc {
                MemLoc::RegOffset { base, .. } => vec![*rd, *base],
                MemLoc::Slot { .. } | MemLoc::Incoming { .. } => vec![*rd],
            },
            Ik::Store { src, loc, .. } => {
                let mut regs = vec![*src];
                match loc {
                    MemLoc::RegOffset { base, .. } => regs.push(*base),
                    MemLoc::Slot { .. } | MemLoc::Incoming { .. } => {}
                }
                regs
            }
            Ik::Ret => vec![],
            Ik::Call { .. } => vec![],
            Ik::J { .. } => vec![],
            Ik::Br { rs1, rs2, .. } => vec![*rs1, *rs2],
            Ik::La { rd, .. } => vec![*rd],
            Ik::LoadAddr { rd, .. } => vec![*rd],
        }
    }

    fn is_terminator(self, mctx: &MContext<Self>) -> bool {
        use RvInstKind as Ik;

        matches!(&self.deref(mctx).kind, Ik::J { .. } | Ik::Br { .. })
    }

    fn succs(self, mctx: &MContext<Self>) -> Vec<MBlock<Self>> {
        use RvInstKind as Ik;

        if let Ik::J { block } = &self.deref(mctx).kind {
            vec![*block]
        } else if let Ik::Br { block, .. } = &self.deref(mctx).kind {
            vec![*block]
        } else {
            vec![]
        }
    }

    fn adjust_offset<F>(self, mctx: &mut MContext<Self>, f: F, _config: &LowerConfig)
    where
        F: FnOnce(MemLoc) -> Option<MemLoc>,
    {
        use RvInstKind as Ik;

        let (old_loc, new_loc) = match self.kind(mctx) {
            Ik::Load { loc, .. } => (*loc, f(*loc)),
            Ik::Store { loc, .. } => (*loc, f(*loc)),
            Ik::LoadAddr { loc, .. } => (*loc, f(*loc)),
            Ik::Li { .. }
            | Ik::AluRRI { .. }
            | Ik::AluRRR { .. }
            | Ik::Ret
            | Ik::Call { .. }
            | Ik::J { .. }
            | Ik::Br { .. }
            | Ik::La { .. } => return,
        };

        if new_loc.is_none() {
            // no modification is needed
            return;
        }

        let new_loc = match (old_loc, new_loc.unwrap()) {
            (MemLoc::Slot { .. } | MemLoc::Incoming { .. }, MemLoc::RegOffset { base, offset }) => {
                if Imm12::try_from_i64(offset).is_none() {
                    // li t0, offset
                    let t0 = regs::t0();
                    let li = Self::build_li(mctx, t0.into(), offset as u64);
                    self.insert_before(mctx, li).unwrap();
                    // add t0, base, t0
                    let add = Self::build_alu_rrr(mctx, AluOpRRR::Add, t0.into(), base, t0.into());
                    self.insert_before(mctx, add).unwrap();
                    MemLoc::RegOffset {
                        base: t0.into(),
                        offset: 0,
                    }
                } else {
                    MemLoc::RegOffset { base, offset }
                }
            }
            _ => unreachable!(),
        };

        match &mut self.deref_mut(mctx).kind {
            Ik::Load { loc, .. } => *loc = new_loc,
            Ik::Store { loc, .. } => *loc = new_loc,
            Ik::LoadAddr { rd, .. } => {
                // we need to remove this instruction and replace with addi or add
                match new_loc {
                    MemLoc::RegOffset { base, offset } => {
                        let rd = *rd;
                        let addi = Self::build_alu_rri(
                            mctx,
                            AluOpRRI::Addi,
                            rd,
                            base,
                            Imm12::try_from_i64(offset).unwrap(),
                        );
                        self.insert_before(mctx, addi).unwrap();
                        self.remove(mctx);
                    }
                    MemLoc::Slot { .. } | MemLoc::Incoming { .. } => unreachable!(),
                }
            }
            Ik::Li { .. }
            | Ik::AluRRI { .. }
            | Ik::AluRRR { .. }
            | Ik::Ret
            | Ik::Call { .. }
            | Ik::J { .. }
            | Ik::Br { .. }
            | Ik::La { .. } => unreachable!(),
        }
    }

    fn replace_reg(self, mctx: &mut MContext<Self>, from: Reg, to: Reg) {
        use RvInstKind as Ik;

        match &mut self.deref_mut(mctx).kind {
            Ik::Li { rd, .. } => {
                if *rd == from {
                    *rd = to;
                }
            }
            Ik::AluRRI { rd, rs, .. } => {
                if *rd == from {
                    *rd = to;
                }
                if *rs == from {
                    *rs = to;
                }
            }
            Ik::AluRRR { rd, rs1, rs2, .. } => {
                if *rd == from {
                    *rd = to;
                }
                if *rs1 == from {
                    *rs1 = to;
                }
                if *rs2 == from {
                    *rs2 = to;
                }
            }
            Ik::Load { rd, loc, .. } => {
                if *rd == from {
                    *rd = to;
                }
                match loc {
                    MemLoc::RegOffset { base, .. } => {
                        if *base == from {
                            *base = to;
                        }
                    }
                    MemLoc::Slot { .. } | MemLoc::Incoming { .. } => {}
                }
            }
            Ik::Store { src, loc, .. } => {
                if *src == from {
                    *src = to;
                }
                match loc {
                    MemLoc::RegOffset { base, .. } => {
                        if *base == from {
                            *base = to;
                        }
                    }
                    MemLoc::Slot { .. } | MemLoc::Incoming { .. } => {}
                }
            }
            Ik::Ret => {}
            Ik::Call { .. } => {}
            Ik::J { .. } => {}
            Ik::Br { rs1, rs2, .. } => {
                if *rs1 == from {
                    *rs1 = to;
                }
                if *rs2 == from {
                    *rs2 = to;
                }
            }
            Ik::La { rd, .. } => {
                if *rd == from {
                    *rd = to;
                }
            }
            Ik::LoadAddr { rd, loc, .. } => {
                if *rd == from {
                    *rd = to;
                }
                match loc {
                    MemLoc::RegOffset { base, .. } => {
                        if *base == from {
                            *base = to;
                        }
                    }
                    MemLoc::Slot { .. } | MemLoc::Incoming { .. } => {}
                }
            }
        }
    }
}

impl ArenaPtr for RvInst {
    type Arena = MContext<Self>;
    type Data = RvInstData;
}

impl LinkedListNode for RvInst {
    type Container = MBlock<Self>;
    type Ctx = MContext<Self>;

    fn next(self, ctx: &Self::Ctx) -> Option<Self> { self.deref(ctx).next }

    fn prev(self, ctx: &Self::Ctx) -> Option<Self> { self.deref(ctx).prev }

    fn set_next(self, arena: &mut Self::Ctx, next: Option<Self>) {
        self.deref_mut(arena).next = next;
    }

    fn set_prev(self, arena: &mut Self::Ctx, prev: Option<Self>) {
        self.deref_mut(arena).prev = prev;
    }

    fn container(self, ctx: &Self::Ctx) -> Option<Self::Container> { self.deref(ctx).parent }

    fn set_container(self, arena: &mut Self::Ctx, container: Option<Self::Container>) {
        self.deref_mut(arena).parent = container;
    }
}
