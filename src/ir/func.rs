use std::fmt;

use super::block::Block;
use super::context::Context;
use super::ty::Ty;
use super::value::Value;
use crate::infra::linked_list::{LinkedListContainer, LinkedListNode};
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub struct FuncData {
    pub(super) self_ptr: Func,
    name: String,
    params: Vec<Value>,
    ret_ty: Ty,

    head: Option<Block>,
    tail: Option<Block>,
    // TODO: Distinguish `define` and `declare`.
    is_define: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Func(GenericPtr<FuncData>);

pub struct DisplayFunc<'ctx> {
    ctx: &'ctx Context,
    func: Func,
}

impl Func {
    pub fn new(ctx: &mut Context, name: String, ret_ty: Ty, is_define: Option<bool>) -> Self {
        ctx.alloc_with(|self_ptr| FuncData {
            self_ptr,
            name,
            params: Vec::new(),
            ret_ty,
            head: None,
            tail: None,
            is_define: is_define.unwrap_or(true),
        })
    }

    pub fn add_param(self, ctx: &mut Context, ty: Ty) -> Value {
        let index = self.deref(ctx).params.len() as u32;
        let param = Value::new_param(ctx, self, ty, index);
        self.deref_mut(ctx).params.push(param);
        param
    }

    pub fn name(self, ctx: &Context) -> &str {
        &self.deref(ctx).name
    }

    pub fn params(self, ctx: &Context) -> &[Value] {
        &self.deref(ctx).params
    }

    pub fn ret_ty(self, ctx: &Context) -> Ty {
        self.deref(ctx).ret_ty
    }

    pub fn head(self, ctx: &Context) -> Option<Block> {
        self.deref(ctx).head
    }

    pub fn remove_block(self, ctx: &mut Context, block: Block) {
        let mut head = self.head(ctx);
        let mut tail = self.tail(ctx);

        if head == Some(block) {
            head = block.next(ctx);
            self.set_head(ctx, head);
        }

        if tail == Some(block) {
            tail = block.prev(ctx);
            self.set_tail(ctx, tail);
        }

        let prev = block.prev(ctx);
        let next = block.next(ctx);

        if let Some(prev) = prev {
            prev.set_next(ctx, next);
        }
        if let Some(next) = next {
            next.set_prev(ctx, prev);
        }

        ctx.try_dealloc(block);
    }

    pub fn display(self, ctx: &Context) -> DisplayFunc {
        DisplayFunc { ctx, func: self }
    }
}

impl fmt::Display for DisplayFunc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.func.deref(self.ctx).is_define {
            write!(
                f,
                "define {} @{}(",
                self.func.ret_ty(self.ctx).display(self.ctx),
                self.func.name(self.ctx)
            )?;

            for (i, param) in self.func.params(self.ctx).iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", param.display(self.ctx, true))?;
            }

            write!(f, ") {{")?;

            for block in self.func.iter(self.ctx) {
                write!(f, "\n{}", block.display(self.ctx))?;
            }

            write!(f, "\n}}")?;
        } else {
            write!(
                f,
                "declare {} @{}(",
                self.func.ret_ty(self.ctx).display(self.ctx),
                self.func.name(self.ctx)
            )?;

            for (i, param) in self.func.params(self.ctx).iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", param.ty(self.ctx).display(self.ctx))?;
            }

            write!(f, ")")?;
        }

        Ok(())
    }
}

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

    fn try_dealloc(&mut self, ptr: Func) -> Option<FuncData> {
        self.funcs.try_dealloc(ptr.0)
    }

    fn try_deref(&self, ptr: Func) -> Option<&FuncData> {
        self.funcs.try_deref(ptr.0)
    }

    fn try_deref_mut(&mut self, ptr: Func) -> Option<&mut FuncData> {
        self.funcs.try_deref_mut(ptr.0)
    }
}

impl LinkedListContainer<Block> for Func {
    type Ctx = Context;

    fn head(self, ctx: &Self::Ctx) -> Option<Block> {
        self.try_deref(ctx).unwrap().head
    }

    fn tail(self, ctx: &Self::Ctx) -> Option<Block> {
        self.try_deref(ctx).unwrap().tail
    }

    fn set_head(self, ctx: &mut Self::Ctx, head: Option<Block>) {
        self.try_deref_mut(ctx).unwrap().head = head;
    }

    fn set_tail(self, ctx: &mut Self::Ctx, tail: Option<Block>) {
        self.try_deref_mut(ctx).unwrap().tail = tail;
    }
}
