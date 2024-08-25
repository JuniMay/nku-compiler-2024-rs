use super::block::Block;
use super::context::Context;
use super::ty::Ty;
use super::value::Value;
use crate::infra::linked_list::LinkedListContainer;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub struct FuncData {
    name: String,
    params: Vec<Value>,
    ret_ty: Ty,

    head: Option<Block>,
    tail: Option<Block>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Func(GenericPtr<FuncData>);

impl ArenaPtr for Func {
    type Arena = Context;
    type Data = FuncData;
}

impl Arena<Func> for Context {
    fn alloc_with<F>(&mut self, f: F) -> Func
    where
        F: FnOnce(Func) -> FuncData,
    {
        Func(self.funcs.alloc_with(|ptr| f(Func(ptr))))
    }

    fn try_dealloc(&mut self, ptr: Func) -> Option<FuncData> { self.funcs.try_dealloc(ptr.0) }

    fn try_deref(&self, ptr: Func) -> Option<&FuncData> { self.funcs.try_deref(ptr.0) }

    fn try_deref_mut(&mut self, ptr: Func) -> Option<&mut FuncData> {
        self.funcs.try_deref_mut(ptr.0)
    }
}

impl LinkedListContainer<Block> for Func {
    type Ctx = Context;

    fn head(self, ctx: &Self::Ctx) -> Option<Block> { self.try_deref(ctx).unwrap().head }

    fn tail(self, ctx: &Self::Ctx) -> Option<Block> { self.try_deref(ctx).unwrap().tail }

    fn set_head(self, ctx: &mut Self::Ctx, head: Option<Block>) {
        self.try_deref_mut(ctx).unwrap().head = head;
    }

    fn set_tail(self, ctx: &mut Self::Ctx, tail: Option<Block>) {
        self.try_deref_mut(ctx).unwrap().tail = tail;
    }
}
