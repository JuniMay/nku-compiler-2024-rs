use std::hash::Hash;

use super::context::MContext;
use super::func::{MFunc, MLabel};
use super::inst::MInst;
use crate::infra::linked_list::{LinkedListContainer, LinkedListNode};
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub struct MBlockData {
    /// The label of the block.
    label: MLabel,

    /// The head machine instruction of the block.
    head: Option<MInst>,

    /// The tail machine instruction of the block.
    tail: Option<MInst>,

    /// The next block in the function.
    next: Option<MBlock>,

    /// The previous block in the function.
    prev: Option<MBlock>,

    /// The parent function of the block.
    parent: Option<MFunc>,
}

/// A machine code block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MBlock(GenericPtr<MBlockData>);

impl MBlock {
    /// Create a new machine code block.
    pub fn new(mctx: &mut MContext, label: impl Into<MLabel>) -> Self {
        mctx.alloc(MBlockData {
            label: label.into(),
            head: None,
            tail: None,
            next: None,
            prev: None,
            parent: None,
        })
    }

    /// Get the label of the block.
    pub fn label(self, arena: &MContext) -> &MLabel {
        &self.deref(arena).label
    }

    /// Get the instructions size of the block.
    pub fn size(self, arena: &MContext) -> usize {
        let mut size = 0;
        for _ in self.iter(arena) {
            size += 1;
        }
        size
    }

    /// Remove the block from the context.
    pub fn remove(self, arena: &mut MContext) {
        self.unlink(arena);
        arena.try_dealloc(self).unwrap();
    }
}

impl Hash for MBlock {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl ArenaPtr for MBlock {
    type Arena = MContext;
    type Data = MBlockData;
}

impl Arena<MBlock> for MContext {
    fn alloc_with<F>(&mut self, f: F) -> MBlock
    where
        F: FnOnce(MBlock) -> MBlockData,
    {
        MBlock(self.blocks.alloc_with(|p| f(MBlock(p))))
    }

    fn try_deref(&self, ptr: MBlock) -> Option<&MBlockData> {
        self.blocks.try_deref(ptr.0)
    }

    fn try_deref_mut(&mut self, ptr: MBlock) -> Option<&mut MBlockData> {
        self.blocks.try_deref_mut(ptr.0)
    }

    fn try_dealloc(&mut self, ptr: MBlock) -> Option<MBlockData> {
        self.blocks.try_dealloc(ptr.0)
    }
}

impl LinkedListContainer<MInst> for MBlock {
    type Ctx = MContext;

    fn head(self, ctx: &Self::Ctx) -> Option<MInst> {
        self.deref(ctx).head
    }

    fn tail(self, ctx: &Self::Ctx) -> Option<MInst> {
        self.deref(ctx).tail
    }

    fn set_head(self, ctx: &mut Self::Ctx, head: Option<MInst>) {
        self.deref_mut(ctx).head = head;
    }

    fn set_tail(self, ctx: &mut Self::Ctx, tail: Option<MInst>) {
        self.deref_mut(ctx).tail = tail;
    }
}

impl LinkedListNode for MBlock {
    type Container = MFunc;
    type Ctx = MContext;

    fn next(self, ctx: &Self::Ctx) -> Option<Self> {
        self.deref(ctx).next
    }

    fn prev(self, ctx: &Self::Ctx) -> Option<Self> {
        self.deref(ctx).prev
    }

    fn set_next(self, ctx: &mut Self::Ctx, next: Option<Self>) {
        self.deref_mut(ctx).next = next;
    }

    fn set_prev(self, ctx: &mut Self::Ctx, prev: Option<Self>) {
        self.deref_mut(ctx).prev = prev;
    }

    fn container(self, ctx: &Self::Ctx) -> Option<Self::Container> {
        self.deref(ctx).parent
    }

    fn set_container(self, ctx: &mut Self::Ctx, container: Option<Self::Container>) {
        self.deref_mut(ctx).parent = container;
    }
}
