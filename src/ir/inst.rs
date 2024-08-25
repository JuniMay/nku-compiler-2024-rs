use super::block::Block;
use super::context::Context;
use super::ty::Ty;
use crate::infra::linked_list::LinkedListNode;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub enum IntCmpCond {
    Eq,
    Ne,
    Slt,
    Sle,
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

pub enum CastOp {
    Zext,
    Sext,
    Trunc,
}

pub enum InstKind {
    Alloca,
    Phi,
    Load,
    Store,
    GetElementPtr,
    Call,
    Br,
    CondBr,
    Ret,
    IntBinary { op: IntBinaryOp },
    Cast { op: CastOp },
}

pub struct InstData {
    /// Pointer to the instruction itself.
    self_ptr: Inst,
    /// The kind of the instruction.
    kind: InstKind,
    /// The type of the instruction result, void if the instruction does not
    /// produce a result.
    ty: Ty,

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
            next: None,
            prev: None,
            container: None,
        })
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
