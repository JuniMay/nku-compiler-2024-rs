use super::context::Context;
use crate::infra::storage::{Arena, ArenaPtr, UniqueArenaPtr};

/// Internal data of types in IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TyData {
    /// The `void` type.
    Void,
    /// The `i1` type.
    Int1,
    /// The `i8` type.
    Int8,
    /// The `i32` type.
    Int32,
    /// The pointer type.
    Ptr,
    /// The array type.
    Array {
        /// The element type.
        elem: Ty,
        /// The length of the array.
        len: usize,
    },
}

/// A handle to a type in IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ty(UniqueArenaPtr<TyData>);

impl Ty {
    /// Fetch a type representing `void`.
    pub fn void(ctx: &mut Context) -> Self { ctx.alloc(TyData::Void) }

    /// Fetch a type representing `i1`.
    pub fn i1(ctx: &mut Context) -> Self { ctx.alloc(TyData::Int1) }

    /// Fetch a type representing `i8`.
    pub fn i8(ctx: &mut Context) -> Self { ctx.alloc(TyData::Int8) }

    /// Fetch a type representing `i32`.
    pub fn i32(ctx: &mut Context) -> Self { ctx.alloc(TyData::Int32) }

    /// Fetch a type representing a pointer.
    pub fn ptr(ctx: &mut Context) -> Self { ctx.alloc(TyData::Ptr) }

    /// Fetch a type representing an array.
    pub fn array(ctx: &mut Context, elem: Ty, len: usize) -> Self {
        ctx.alloc(TyData::Array { elem, len })
    }

    /// Get the bit width of the type.
    pub fn bitwidth(&self, ctx: &Context) -> usize {
        match self.try_deref(ctx).unwrap() {
            TyData::Void => 0,
            TyData::Int1 => 1,
            TyData::Int8 => 8,
            TyData::Int32 => 32,
            TyData::Ptr => ctx.target.ptr_size as usize * 8,
            TyData::Array { elem, len } => elem.bitwidth(ctx) * len,
        }
    }

    /// Try to dereference the type as an array.
    pub fn as_array(&self, ctx: &Context) -> Option<(Ty, usize)> {
        match self.try_deref(ctx).unwrap() {
            TyData::Array { elem, len } => Some((*elem, *len)),
            _ => None,
        }
    }
}

impl ArenaPtr for Ty {
    type Arena = Context;
    type Data = TyData;
}

impl Arena<Ty> for Context {
    fn alloc_with<F>(&mut self, _: F) -> Ty
    where
        F: FnOnce(Ty) -> TyData,
    {
        panic!("types cannot be allocated with a closure");
    }

    fn alloc(&mut self, data: TyData) -> Ty { Ty(self.tys.alloc(data)) }

    fn try_dealloc(&mut self, ptr: Ty) -> Option<TyData> { self.tys.try_dealloc(ptr.0) }

    fn try_deref(&self, ptr: Ty) -> Option<&TyData> { self.tys.try_deref(ptr.0) }

    fn try_deref_mut(&mut self, ptr: Ty) -> Option<&mut TyData> { self.tys.try_deref_mut(ptr.0) }
}
