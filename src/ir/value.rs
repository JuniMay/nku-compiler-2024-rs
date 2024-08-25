use super::context::Context;
use super::func::Func;
use super::inst::Inst;
use super::ty::Ty;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub enum ConstantValue {
    Undef,
    AggregateZero,
    Int1(bool),
    Int8(i8),
    Int32(i32),
    Array { elems: Vec<ConstantValue> },
}

pub enum ValueKind {
    /// The value is the result of an instruction.
    InstResult { inst: Inst },
    /// The value is a function parameter.
    Param { func: Func, ty: Ty, index: u32 },
    /// The value is an invariant constant.
    Constant { ty: Ty, value: ConstantValue },
}

pub struct ValueData {
    self_ptr: Value,
    kind: ValueKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(GenericPtr<ValueData>);

impl Value {
    pub fn undef(ctx: &mut Context, ty: Ty) -> Self {
        ctx.alloc_with(|self_ptr| ValueData {
            self_ptr,
            kind: ValueKind::Constant {
                ty,
                value: ConstantValue::Undef,
            },
        })
    }

    pub fn zeroinitializer(ctx: &mut Context, ty: Ty) -> Self {
        ctx.alloc_with(|self_ptr| ValueData {
            self_ptr,
            kind: ValueKind::Constant {
                ty,
                value: ConstantValue::AggregateZero,
            },
        })
    }

    pub fn i1(ctx: &mut Context, value: bool) -> Self {
        let i1 = Ty::i1(ctx);
        ctx.alloc_with(|self_ptr| ValueData {
            self_ptr,
            kind: ValueKind::Constant {
                ty: i1,
                value: ConstantValue::Int1(value),
            },
        })
    }

    pub fn i8(ctx: &mut Context, value: i8) -> Self {
        let i8 = Ty::i8(ctx);
        ctx.alloc_with(|self_ptr| ValueData {
            self_ptr,
            kind: ValueKind::Constant {
                ty: i8,
                value: ConstantValue::Int8(value),
            },
        })
    }

    pub fn i32(ctx: &mut Context, value: i32) -> Self {
        let i32 = Ty::i32(ctx);
        ctx.alloc_with(|self_ptr| ValueData {
            self_ptr,
            kind: ValueKind::Constant {
                ty: i32,
                value: ConstantValue::Int32(value),
            },
        })
    }
}

impl ArenaPtr for Value {
    type Arena = Context;
    type Data = ValueData;
}

impl Arena<Value> for Context {
    fn alloc_with<F>(&mut self, f: F) -> Value
    where
        F: FnOnce(Value) -> ValueData,
    {
        Value(self.values.alloc_with(|ptr| f(Value(ptr))))
    }

    fn try_dealloc(&mut self, ptr: Value) -> Option<ValueData> { self.values.try_dealloc(ptr.0) }

    fn try_deref(&self, ptr: Value) -> Option<&ValueData> { self.values.try_deref(ptr.0) }

    fn try_deref_mut(&mut self, ptr: Value) -> Option<&mut ValueData> {
        self.values.try_deref_mut(ptr.0)
    }
}
