use std::collections::HashSet;

use super::context::Context;
use super::def_use::{Usable, User};
use super::func::Func;
use super::inst::Inst;
use crate::infra::linked_list::{LinkedListContainer, LinkedListNode};
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr, Idx};

pub struct BlockData {
    self_ptr: Block,

    users: HashSet<User<Block>>,

    next: Option<Block>,
    prev: Option<Block>,

    head: Option<Inst>,
    tail: Option<Inst>,

    container: Option<Func>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Block(GenericPtr<BlockData>);

impl Block {
    pub fn new(ctx: &mut Context) -> Self {
        ctx.alloc_with(|self_ptr| BlockData {
            self_ptr,
            users: HashSet::new(),
            next: None,
            prev: None,
            head: None,
            tail: None,
            container: None,
        })
    }

    /// Get the name of the block.
    pub fn name(self, _ctx: &Context) -> String {
        // We use the arena index directly as the block number. This is not a good way
        // to number blocks in a real compiler, but only for debugging purposes.
        format!("%{}", self.0.index())
    }
}

impl ArenaPtr for Block {
    type Arena = Context;
    type Data = BlockData;
}

impl Arena<Block> for Context {
    fn alloc_with<F>(&mut self, f: F) -> Block
    where
        F: FnOnce(Block) -> BlockData,
    {
        Block(self.blocks.alloc_with(|ptr| f(Block(ptr))))
    }

    fn try_dealloc(&mut self, ptr: Block) -> Option<BlockData> { self.blocks.try_dealloc(ptr.0) }

    fn try_deref(&self, ptr: Block) -> Option<&BlockData> { self.blocks.try_deref(ptr.0) }

    fn try_deref_mut(&mut self, ptr: Block) -> Option<&mut BlockData> {
        self.blocks.try_deref_mut(ptr.0)
    }
}

impl LinkedListContainer<Inst> for Block {
    type Ctx = Context;

    fn head(self, ctx: &Self::Ctx) -> Option<Inst> {
        self.try_deref(ctx).expect("invalid pointer").head
    }

    fn tail(self, ctx: &Self::Ctx) -> Option<Inst> {
        self.try_deref(ctx).expect("invalid pointer").tail
    }

    fn set_head(self, ctx: &mut Self::Ctx, head: Option<Inst>) {
        self.try_deref_mut(ctx).expect("invalid pointer").head = head;
    }

    fn set_tail(self, ctx: &mut Self::Ctx, tail: Option<Inst>) {
        self.try_deref_mut(ctx).expect("invalid pointer").tail = tail;
    }
}

impl LinkedListNode for Block {
    type Container = Func;
    type Ctx = Context;

    fn next(self, ctx: &Self::Ctx) -> Option<Self> {
        self.try_deref(ctx).expect("invalid pointer").next
    }

    fn prev(self, ctx: &Self::Ctx) -> Option<Self> {
        self.try_deref(ctx).expect("invalid pointer").prev
    }

    fn container(self, ctx: &Self::Ctx) -> Option<Self::Container> {
        self.try_deref(ctx).expect("invalid pointer").container
    }

    fn set_next(self, ctx: &mut Self::Ctx, next: Option<Self>) {
        self.try_deref_mut(ctx).expect("invalid pointer").next = next;
    }

    fn set_prev(self, ctx: &mut Self::Ctx, prev: Option<Self>) {
        self.try_deref_mut(ctx).expect("invalid pointer").prev = prev;
    }

    fn set_container(self, ctx: &mut Self::Ctx, container: Option<Self::Container>) {
        self.try_deref_mut(ctx).expect("invalid pointer").container = container;
    }
}

impl Usable for Block {
    fn users(self, arena: &Self::Arena) -> impl IntoIterator<Item = User<Self>> {
        self.try_deref(arena).unwrap().users.iter().copied()
    }

    fn insert_user(self, arena: &mut Self::Arena, user: User<Self>) {
        self.try_deref_mut(arena).unwrap().users.insert(user);
    }

    fn remove_user(self, arena: &mut Self::Arena, user: User<Self>) {
        self.try_deref_mut(arena).unwrap().users.remove(&user);
    }
}
