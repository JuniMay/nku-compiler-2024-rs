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
    /// The `f32` type.
    Float32,
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

pub struct DisplayTy<'ctx> {
    ctx: &'ctx Context,
    ty: Ty,
}

impl<'ctx> std::fmt::Display for DisplayTy<'ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.ty.try_deref(self.ctx).unwrap() {
            TyData::Void => write!(f, "void"),
            TyData::Int1 => write!(f, "i1"),
            TyData::Int8 => write!(f, "i8"),
            TyData::Int32 => write!(f, "i32"),
            TyData::Float32 => write!(f, "float"),
            TyData::Ptr => write!(f, "ptr"),
            TyData::Array { elem, len } => write!(
                f,
                "[{} x {}]",
                len,
                DisplayTy {
                    ctx: self.ctx,
                    ty: *elem
                }
            ),
        }
    }
}

impl Ty {
    /// Fetch a type representing `void`.
    pub fn void(ctx: &mut Context) -> Self { ctx.alloc(TyData::Void) }

    /// Fetch a type representing `i1`.
    pub fn i1(ctx: &mut Context) -> Self { ctx.alloc(TyData::Int1) }

    /// Fetch a type representing `i8`.
    pub fn i8(ctx: &mut Context) -> Self { ctx.alloc(TyData::Int8) }

    /// Fetch a type representing `i32`.
    pub fn i32(ctx: &mut Context) -> Self { ctx.alloc(TyData::Int32) }

    /// Fetch a type representing `f32`.
    pub fn f32(ctx: &mut Context) -> Self { ctx.alloc(TyData::Float32) }

    /// Fetch a type representing a pointer.
    pub fn ptr(ctx: &mut Context) -> Self { ctx.alloc(TyData::Ptr) }

    /// Fetch a type representing an array.
    pub fn array(ctx: &mut Context, elem: Ty, len: usize) -> Self {
        ctx.alloc(TyData::Array { elem, len })
    }

    pub fn is_void(&self, ctx: &Context) -> bool {
        matches!(self.try_deref(ctx).unwrap(), TyData::Void)
    }

    /// Get the bit width of the type.
    pub fn bitwidth(&self, ctx: &Context) -> usize {
        match self.try_deref(ctx).unwrap() {
            TyData::Void => 0,
            TyData::Int1 => 1,
            TyData::Int8 => 8,
            TyData::Int32 => 32,
            TyData::Float32 => 32,
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

    /// Get the displayable type.
    pub fn display(self, ctx: &Context) -> DisplayTy { DisplayTy { ctx, ty: self } }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ty() {
        let mut ctx = Context::new(8);
        let void = Ty::void(&mut ctx);
        let i1 = Ty::i1(&mut ctx);
        let i8 = Ty::i8(&mut ctx);
        let i32 = Ty::i32(&mut ctx);
        let ptr = Ty::ptr(&mut ctx);
        let arr = Ty::array(&mut ctx, i32, 10);

        assert_eq!(void.bitwidth(&ctx), 0);
        assert_eq!(i1.bitwidth(&ctx), 1);
        assert_eq!(i8.bitwidth(&ctx), 8);
        assert_eq!(i32.bitwidth(&ctx), 32);
        assert_eq!(ptr.bitwidth(&ctx), 64);
        assert_eq!(arr.bitwidth(&ctx), 320);

        assert_eq!(i32.as_array(&ctx), None);
        assert_eq!(arr.as_array(&ctx), Some((i32, 10)));
    }

    #[test]
    fn test_display_ty() {
        let mut ctx = Context::new(8);
        let void = Ty::void(&mut ctx);
        let i1 = Ty::i1(&mut ctx);
        let i8 = Ty::i8(&mut ctx);
        let i32 = Ty::i32(&mut ctx);
        let ptr = Ty::ptr(&mut ctx);
        let arr = Ty::array(&mut ctx, i32, 10);

        assert_eq!(void.display(&ctx).to_string(), "void");
        assert_eq!(i1.display(&ctx).to_string(), "i1");
        assert_eq!(i8.display(&ctx).to_string(), "i8");
        assert_eq!(i32.display(&ctx).to_string(), "i32");
        assert_eq!(ptr.display(&ctx).to_string(), "ptr");
        assert_eq!(arr.display(&ctx).to_string(), "[10 x i32]");
    }
}
