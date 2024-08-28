use std::hash::Hash;

use super::context::MContext;
use super::func::{MFunc, MLabel};
use super::inst::MInst;
use crate::infra::linked_list::{LinkedListContainer, LinkedListNode};
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub struct MBlockData<I> {
    label: MLabel,

    head: Option<I>,
    tail: Option<I>,

    next: Option<MBlock<I>>,
    prev: Option<MBlock<I>>,

    parent: Option<MFunc<I>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MBlock<I>(GenericPtr<MBlockData<I>>);

impl<I> MBlock<I>
where
    I: MInst,
{
    pub fn new(mctx: &mut MContext<I>, label: impl Into<MLabel>) -> Self {
        mctx.alloc(MBlockData {
            label: label.into(),
            head: None,
            tail: None,
            next: None,
            prev: None,
            parent: None,
        })
    }

    pub fn label(self, arena: &MContext<I>) -> &MLabel { &self.deref(arena).label }

    pub fn size(self, arena: &MContext<I>) -> usize {
        let mut size = 0;
        for _ in self.iter(arena) {
            size += 1;
        }
        size
    }

    pub fn remove(self, arena: &mut MContext<I>) {
        self.unlink(arena);
        arena.try_dealloc(self).unwrap();
    }
}

impl<I> Hash for MBlock<I> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.0.hash(state); }
}

impl<I> ArenaPtr for MBlock<I>
where
    I: MInst,
{
    type Arena = MContext<I>;
    type Data = MBlockData<I>;
}

impl<I> Arena<MBlock<I>> for MContext<I>
where
    I: MInst,
{
    fn alloc_with<F>(&mut self, f: F) -> MBlock<I>
    where
        F: FnOnce(MBlock<I>) -> MBlockData<I>,
    {
        MBlock(self.blocks.alloc_with(|p| f(MBlock(p))))
    }

    fn try_deref(&self, ptr: MBlock<I>) -> Option<&MBlockData<I>> { self.blocks.try_deref(ptr.0) }

    fn try_deref_mut(&mut self, ptr: MBlock<I>) -> Option<&mut MBlockData<I>> {
        self.blocks.try_deref_mut(ptr.0)
    }

    fn try_dealloc(&mut self, ptr: MBlock<I>) -> Option<MBlockData<I>> {
        self.blocks.try_dealloc(ptr.0)
    }
}

impl<I> LinkedListContainer<I> for MBlock<I>
where
    I: MInst,
{
    type Ctx = MContext<I>;

    fn head(self, ctx: &Self::Ctx) -> Option<I> { self.deref(ctx).head }

    fn tail(self, ctx: &Self::Ctx) -> Option<I> { self.deref(ctx).tail }

    fn set_head(self, ctx: &mut Self::Ctx, head: Option<I>) { self.deref_mut(ctx).head = head; }

    fn set_tail(self, ctx: &mut Self::Ctx, tail: Option<I>) { self.deref_mut(ctx).tail = tail; }
}

impl<I> LinkedListNode for MBlock<I>
where
    I: MInst,
{
    type Container = MFunc<I>;
    type Ctx = MContext<I>;

    fn next(self, ctx: &Self::Ctx) -> Option<Self> { self.deref(ctx).next }

    fn prev(self, ctx: &Self::Ctx) -> Option<Self> { self.deref(ctx).prev }

    fn set_next(self, ctx: &mut Self::Ctx, next: Option<Self>) { self.deref_mut(ctx).next = next; }

    fn set_prev(self, ctx: &mut Self::Ctx, prev: Option<Self>) { self.deref_mut(ctx).prev = prev; }

    fn container(self, ctx: &Self::Ctx) -> Option<Self::Container> { self.deref(ctx).parent }

    fn set_container(self, ctx: &mut Self::Ctx, container: Option<Self::Container>) {
        self.deref_mut(ctx).parent = container;
    }
}
