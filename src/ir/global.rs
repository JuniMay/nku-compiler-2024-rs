use std::fmt;

use super::{ConstantValue, Context, Ty};
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub struct GlobalData {
    pub(super) self_ptr: Global,
    name: String,
    value: ConstantValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Global(GenericPtr<GlobalData>);

impl Global {
    pub fn new(ctx: &mut Context, name: String, value: ConstantValue) -> Self {
        ctx.alloc_with(|self_ptr| GlobalData {
            self_ptr,
            name,
            value,
        })
    }

    pub fn name(self, ctx: &Context) -> &str { &self.deref(ctx).name }

    pub fn value(self, ctx: &Context) -> &ConstantValue { &self.deref(ctx).value }

    pub fn ty(self, ctx: &Context) -> Ty { self.value(ctx).ty() }
}

pub struct DisplayGlobal<'ctx> {
    ctx: &'ctx Context,
    global: Global,
}

impl Global {
    pub fn display(self, ctx: &Context) -> DisplayGlobal { DisplayGlobal { ctx, global: self } }
}

impl fmt::Display for DisplayGlobal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "@{} = global {}",
            self.global.name(self.ctx),
            self.global.value(self.ctx).to_string(self.ctx, true)
        )
    }
}

impl ArenaPtr for Global {
    type Arena = Context;
    type Data = GlobalData;
}

impl Arena<Global> for Context {
    fn alloc_with<F>(&mut self, f: F) -> Global
    where
        F: FnOnce(Global) -> GlobalData,
    {
        Global(self.globals.alloc_with(|ptr| f(Global(ptr))))
    }

    fn try_dealloc(&mut self, ptr: Global) -> Option<GlobalData> { self.globals.try_dealloc(ptr.0) }

    fn try_deref(&self, ptr: Global) -> Option<&GlobalData> { self.globals.try_deref(ptr.0) }

    fn try_deref_mut(&mut self, ptr: Global) -> Option<&mut GlobalData> {
        self.globals.try_deref_mut(ptr.0)
    }
}
