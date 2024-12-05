use std::collections::HashSet;
use std::fmt;

use super::context::Context;
use super::def_use::{Usable, User};
use super::func::Func;
use super::inst::Inst;
use super::ty::Ty;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr, Idx};

pub enum ConstantValue {
    /// The undefined value.
    Undef { ty: Ty },
    /// `zeroinitializer` constant.
    AggregateZero { ty: Ty },
    /// A boolean constant.
    Int1 { ty: Ty, value: bool },
    /// An 8-bit integer constant.
    Int8 { ty: Ty, value: i8 },
    /// A 32-bit integer constant.
    Int32 { ty: Ty, value: i32 },
    /// An array constant.
    Array { ty: Ty, elems: Vec<ConstantValue> },
    /// Global variables/functions are treated as constants, because their
    /// addresses are immutable.
    GlobalRef {
        /// The pointer type.
        ty: Ty,
        /// The name of the global variable/function.
        name: String,
        /// The type of the value that the global variable/function points to.
        value_ty: Ty,
    },
}

impl ConstantValue {
    pub fn ty(&self) -> Ty {
        match self {
            ConstantValue::Undef { ty } => *ty,
            ConstantValue::AggregateZero { ty } => *ty,
            ConstantValue::Int1 { ty, .. } => *ty,
            ConstantValue::Int8 { ty, .. } => *ty,
            ConstantValue::Int32 { ty, .. } => *ty,
            ConstantValue::Array { ty, .. } => *ty,
            ConstantValue::GlobalRef { ty, .. } => *ty,
        }
    }

    pub fn i1(ctx: &mut Context, value: bool) -> ConstantValue {
        let i1 = Ty::i1(ctx);
        ConstantValue::Int1 { ty: i1, value }
    }

    pub fn i8(ctx: &mut Context, value: i8) -> ConstantValue {
        let i8 = Ty::i8(ctx);
        ConstantValue::Int8 { ty: i8, value }
    }

    pub fn i32(ctx: &mut Context, value: i32) -> ConstantValue {
        let i32 = Ty::i32(ctx);
        ConstantValue::Int32 { ty: i32, value }
    }

    pub fn global_ref(ctx: &mut Context, name: String, value_ty: Ty) -> ConstantValue {
        let ty = Ty::ptr(ctx);
        ConstantValue::GlobalRef { ty, name, value_ty }
    }

    pub fn undef(ctx: &mut Context, ty: Ty) -> ConstantValue { ConstantValue::Undef { ty } }

    pub fn to_string(&self, ctx: &Context, typed: bool) -> String {
        let mut s = if typed {
            format!("{} ", self.ty().display(ctx))
        } else {
            String::new()
        };

        match self {
            ConstantValue::Undef { .. } => s.push_str("undef"),
            ConstantValue::AggregateZero { .. } => s.push_str("zeroinitializer"),
            ConstantValue::Int1 { value, .. } => s.push_str(&value.to_string()),
            ConstantValue::Int8 { value, .. } => s.push_str(&value.to_string()),
            ConstantValue::Int32 { value, .. } => s.push_str(&value.to_string()),
            ConstantValue::Array { elems, .. } => {
                s.push('[');
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&elem.to_string(ctx, true));
                }
                s.push(']');
            }
            ConstantValue::GlobalRef { name, .. } => {
                s.push('@');
                s.push_str(name);
            }
        }

        s
    }
}

pub enum ValueKind {
    /// The value is the result of an instruction.
    InstResult { inst: Inst, ty: Ty },
    /// The value is a function parameter.
    Param { func: Func, ty: Ty, index: u32 },
    /// The value is an invariant constant.
    Constant { value: ConstantValue },
}

pub struct ValueData {
    _self_ptr: Value,
    pub kind: ValueKind,
    /// The user of this value.
    ///
    /// This is only useful when the value is an instruction result or a
    /// function parameter. For constants, the users are not tracked.
    users: HashSet<User<Value>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(GenericPtr<ValueData>);

pub struct DisplayValue<'ctx> {
    ctx: &'ctx Context,
    value: Value,
    with_type: bool,
}

impl<'ctx> fmt::Display for DisplayValue<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.value.try_deref(self.ctx).unwrap().kind {
            ValueKind::InstResult { ty, .. } | ValueKind::Param { ty, .. } => {
                // We use the arena index directly as the value number. This is not a good way
                // to number values in a real compiler, but only for debugging purposes.
                if self.with_type {
                    write!(f, "{} %v{}", ty.display(self.ctx), self.value.0.index())
                } else {
                    write!(f, "%v{}", self.value.0.index())
                }
            }
            ValueKind::Constant { ref value } => {
                write!(f, "{}", value.to_string(self.ctx, self.with_type))
            }
        }
    }
}

impl Value {
    fn new(ctx: &mut Context, kind: ValueKind) -> Self {
        ctx.alloc_with(|self_ptr| ValueData {
            _self_ptr: self_ptr,
            kind,
            users: HashSet::new(),
        })
    }

    pub fn ty(&self, ctx: &Context) -> Ty {
        match self.try_deref(ctx).unwrap().kind {
            ValueKind::InstResult { ty, .. } => ty,
            ValueKind::Param { ty, .. } => ty,
            ValueKind::Constant { ref value } => value.ty(),
        }
    }

    pub(super) fn new_param(ctx: &mut Context, func: Func, ty: Ty, index: u32) -> Self {
        Self::new(ctx, ValueKind::Param { func, ty, index })
    }

    pub(super) fn new_inst_result(ctx: &mut Context, inst: Inst, ty: Ty) -> Self {
        Self::new(ctx, ValueKind::InstResult { inst, ty })
    }

    pub fn display(self, ctx: &Context, with_type: bool) -> DisplayValue {
        DisplayValue {
            ctx,
            value: self,
            with_type,
        }
    }

    pub fn is_param(&self, ctx: &Context) -> bool {
        matches!(self.try_deref(ctx).unwrap().kind, ValueKind::Param { .. })
    }

    pub fn i1(ctx: &mut Context, value: bool) -> Self {
        let value = ConstantValue::i1(ctx, value);
        Self::new(ctx, ValueKind::Constant { value })
    }

    pub fn i8(ctx: &mut Context, value: i8) -> Self {
        let value = ConstantValue::i8(ctx, value);
        Self::new(ctx, ValueKind::Constant { value })
    }

    pub fn i32(ctx: &mut Context, value: i32) -> Self {
        let value = ConstantValue::i32(ctx, value);
        Self::new(ctx, ValueKind::Constant { value })
    }

    pub fn global_ref(ctx: &mut Context, name: String, value_ty: Ty) -> Self {
        let value = ConstantValue::global_ref(ctx, name, value_ty);
        Self::new(ctx, ValueKind::Constant { value })
    }

    pub fn undef(ctx: &mut Context, ty: Ty) -> Self {
        let value = ConstantValue::undef(ctx, ty);
        Self::new(ctx, ValueKind::Constant { value })
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

impl Usable for Value {
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
