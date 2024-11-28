use std::fmt;

use crate::infra::{
    linked_list::LinkedListNode,
    storage::{Arena, ArenaPtr, GenericPtr},
};

use super::{block::MBlock, context::MContext, imm::Imm12, operand::Reg};

pub struct MInstData {
    kind: MInstKind,
    next: Option<MInst>,
    prev: Option<MInst>,
    parent: Option<MBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MInst(GenericPtr<MInstData>);

pub enum MInstKind {
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
    // TODO: add more instructions as you need.
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

pub struct DisplayMInst<'a> {
    mctx: &'a MContext,
    inst: MInst,
}

impl MInst {
    pub fn display(self, mctx: &MContext) -> DisplayMInst {
        DisplayMInst { mctx, inst: self }
    }
}

impl<'a> fmt::Display for DisplayMInst<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!("implement display for machine instruction")
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
