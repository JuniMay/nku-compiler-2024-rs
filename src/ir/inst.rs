use std::collections::HashMap;
use std::fmt;

use super::block::Block;
use super::context::Context;
use super::def_use::Operand;
use super::ty::Ty;
use super::value::Value;
use crate::infra::linked_list::LinkedListNode;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub enum IntCmpCond {
    Eq,
    Ne,
    Slt,
    Sle,
}

impl fmt::Display for IntCmpCond {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntCmpCond::Eq => write!(f, "eq"),
            IntCmpCond::Ne => write!(f, "ne"),
            IntCmpCond::Slt => write!(f, "slt"),
            IntCmpCond::Sle => write!(f, "sle"),
        }
    }
}

pub enum IntBinaryOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    ICmp { cond: IntCmpCond },
}

impl fmt::Display for IntBinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntBinaryOp::Add => write!(f, "add"),
            IntBinaryOp::Sub => write!(f, "sub"),
            IntBinaryOp::Mul => write!(f, "mul"),
            IntBinaryOp::SDiv => write!(f, "sdiv"),
            IntBinaryOp::UDiv => write!(f, "udiv"),
            IntBinaryOp::SRem => write!(f, "srem"),
            IntBinaryOp::URem => write!(f, "urem"),
            IntBinaryOp::Shl => write!(f, "shl"),
            IntBinaryOp::LShr => write!(f, "lshr"),
            IntBinaryOp::AShr => write!(f, "ashr"),
            IntBinaryOp::And => write!(f, "and"),
            IntBinaryOp::Or => write!(f, "or"),
            IntBinaryOp::Xor => write!(f, "xor"),
            IntBinaryOp::ICmp { cond } => write!(f, "icmp {}", cond),
        }
    }
}

pub enum CastOp {
    Zext,
    Sext,
    Trunc,
}

impl fmt::Display for CastOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CastOp::Zext => write!(f, "zext"),
            CastOp::Sext => write!(f, "sext"),
            CastOp::Trunc => write!(f, "trunc"),
        }
    }
}

pub enum InstKind {
    Alloca { ty: Ty },
    Phi,
    Load,
    Store,
    GetElementPtr { bound_ty: Ty },
    Call,
    Br,
    CondBr,
    Ret,
    IntBinary { op: IntBinaryOp },
    Cast { op: CastOp },
}

pub enum OperandList {
    Operands {
        operands: Vec<Operand<Value>>,
    },
    PhiNode {
        incoming: HashMap<Block, Operand<Value>>,
    },
}

impl Default for OperandList {
    fn default() -> Self {
        OperandList::Operands {
            operands: Vec::new(),
        }
    }
}

pub struct InstData {
    /// Pointer to the instruction itself.
    self_ptr: Inst,
    /// The kind of the instruction.
    kind: InstKind,
    /// The type of the instruction result, void if the instruction does not
    /// produce a result.
    ty: Ty,
    /// The operands of the instruction.
    ///
    /// This can either be a list of values or a phi node (with predecessor ->
    /// value mapping).
    operands: OperandList,
    /// The successors of the instruction.
    ///
    /// Only branch instructions have successors.
    successors: Vec<Operand<Block>>,
    // Linked list pointers.
    next: Option<Inst>,
    prev: Option<Inst>,
    container: Option<Block>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Inst(GenericPtr<InstData>);

impl Inst {
    fn new(ctx: &mut Context, kind: InstKind, ty: Ty) -> Self {
        ctx.alloc_with(|self_ptr| InstData {
            self_ptr,
            kind,
            ty,
            operands: OperandList::default(),
            successors: Vec::new(),
            next: None,
            prev: None,
            container: None,
        })
    }

    /// Get the operand at the given index.
    ///
    /// # Panics
    ///
    /// - Panics if the instruction is a phi node.
    /// - Panics if the index is out of bounds.
    fn operand(&self, ctx: &Context, index: usize) -> Value {
        match self.try_deref(ctx).unwrap().operands {
            OperandList::Operands { ref operands } => operands[index].used(),
            OperandList::PhiNode { .. } => panic!("not an operand"),
        }
    }

    /// Get the incoming value from the given block.
    ///
    /// # Panics
    ///
    /// - Panics if the instruction is not a phi node.
    /// - Panics if the block is not in the incoming list.
    fn incoming(&self, ctx: &Context, block: Block) -> Value {
        match self.try_deref(ctx).unwrap().operands {
            OperandList::Operands { .. } => panic!("not a phi node"),
            OperandList::PhiNode { ref incoming } => incoming[&block].used(),
        }
    }

    /// Get the successor at the given index.
    ///
    /// # Panics
    ///
    /// - Panics if the index is out of bounds (or the instruction is not a
    ///   branch instruction).
    fn successor(&self, ctx: &Context, index: usize) -> Block {
        self.try_deref(ctx).unwrap().successors[index].used()
    }
}

impl ArenaPtr for Inst {
    type Arena = Context;
    type Data = InstData;
}

impl Arena<Inst> for Context {
    fn alloc_with<F>(&mut self, f: F) -> Inst
    where
        F: FnOnce(Inst) -> InstData,
    {
        Inst(self.insts.alloc_with(|ptr| f(Inst(ptr))))
    }

    fn try_dealloc(&mut self, ptr: Inst) -> Option<InstData> { self.insts.try_dealloc(ptr.0) }

    fn try_deref(&self, ptr: Inst) -> Option<&InstData> { self.insts.try_deref(ptr.0) }

    fn try_deref_mut(&mut self, ptr: Inst) -> Option<&mut InstData> {
        self.insts.try_deref_mut(ptr.0)
    }
}

impl LinkedListNode for Inst {
    type Container = Block;
    type Ctx = Context;

    fn next(self, ctx: &Self::Ctx) -> Option<Self> { self.try_deref(ctx).unwrap().next }

    fn prev(self, ctx: &Self::Ctx) -> Option<Self> { self.try_deref(ctx).unwrap().prev }

    fn container(self, ctx: &Self::Ctx) -> Option<Self::Container> {
        self.try_deref(ctx).unwrap().container
    }

    fn set_next(self, ctx: &mut Self::Ctx, next: Option<Self>) {
        self.try_deref_mut(ctx).unwrap().next = next;
    }

    fn set_prev(self, ctx: &mut Self::Ctx, prev: Option<Self>) {
        self.try_deref_mut(ctx).unwrap().prev = prev;
    }

    fn set_container(self, ctx: &mut Self::Ctx, container: Option<Self::Container>) {
        self.try_deref_mut(ctx).unwrap().container = container;
    }
}
