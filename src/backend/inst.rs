use super::block::MBlock;
use super::context::MContext;
use super::lower::{LowerConfig, MemLoc};
use super::regs::Reg;
use super::LowerSpec;
use crate::infra::linked_list::LinkedListNode;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub trait MInst:
    ArenaPtr<Arena = MContext<Self>> + LinkedListNode<Container = MBlock<Self>, Ctx = MContext<Self>>
{
    type S: LowerSpec<I = Self>;

    fn from_ptr(ptr: GenericPtr<Self::Data>) -> Self;

    fn ptr(self) -> GenericPtr<Self::Data>;

    fn uses(self, mctx: &MContext<Self>, config: &LowerConfig) -> Vec<Reg>;

    /// Return registers defined by this instruction.
    /// Note for call instruction, all return registers are defined.
    fn defs(self, mctx: &MContext<Self>, config: &LowerConfig) -> Vec<Reg>;

    /// Return registers that may be clobbered (edited) by this instruction.
    /// Note for call instruction, all caller-saved registers are clobbered.
    fn clobbers(self, mctx: &MContext<Self>, config: &LowerConfig) -> Vec<Reg>;

    /// Return all registers that are written in assembly of this instruction.
    fn all_regs(self, mctx: &MContext<Self>, config: &LowerConfig) -> Vec<Reg>;

    fn is_terminator(self, mctx: &MContext<Self>) -> bool;

    fn succs(self, mctx: &MContext<Self>) -> Vec<MBlock<Self>>;

    fn adjust_offset<F>(self, mctx: &mut MContext<Self>, f: F, config: &LowerConfig)
    where
        F: FnOnce(MemLoc) -> Option<MemLoc>;

    fn replace_reg(self, mctx: &mut MContext<Self>, from: Reg, to: Reg);

    fn remove(self, mctx: &mut MContext<Self>) {
        self.unlink(mctx);
        mctx.try_dealloc(self).unwrap();
    }
}

pub trait DisplayMInst<'a>: MInst {
    type Display: std::fmt::Display + 'a;

    fn display(self, mctx: &'a MContext<Self>) -> Self::Display;
}

impl<I> Arena<I> for MContext<I>
where
    I: MInst,
{
    fn alloc_with<F>(&mut self, f: F) -> I
    where
        F: FnOnce(I) -> I::Data,
    {
        I::from_ptr(self.insts.alloc_with(|p| f(I::from_ptr(p))))
    }

    fn try_deref(&self, ptr: I) -> Option<&I::Data> { self.insts.try_deref(ptr.ptr()) }

    fn try_deref_mut(&mut self, ptr: I) -> Option<&mut I::Data> {
        self.insts.try_deref_mut(ptr.ptr())
    }

    fn try_dealloc(&mut self, ptr: I) -> Option<I::Data> { self.insts.try_dealloc(ptr.ptr()) }
}
