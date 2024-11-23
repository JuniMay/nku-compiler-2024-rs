//! Define the types in AST of SysY language.
//! The types are used in the AST and the symbol table.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::{fmt, hash};

/// The type in AST
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Void,
    /// The boolean type.
    ///
    /// There is no `bool` type, but the intermediate result might be boolean.
    Bool,
    /// The char type.
    Char,
    /// The integer type.
    Int,
    /// The float type.
    Float,
    /// The array type, with the element type and the size.
    Array(Type, usize),
    /// The function type, with params and return type.
    Func(Vec<Type>, Type),
    /// The pointer type.
    Ptr(Type),
    /// The args type.
    Args,
}

// The type in AST
#[derive(Clone, Eq)]
pub struct Type(Rc<TypeKind>);

impl hash::Hash for Type {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl PartialEq for Type {
    // Just compare the pointers
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for Type {
    /// Display the type.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            TypeKind::Void => write!(f, "void"),
            TypeKind::Bool => write!(f, "bool"),
            TypeKind::Char => write!(f, "char"),
            TypeKind::Int => write!(f, "int"),
            TypeKind::Float => write!(f, "float"),
            TypeKind::Array(ty, size) => write!(f, "{}[{:?}]", ty, size),
            TypeKind::Func(params, ret) => write!(
                f,
                "{}({})",
                ret,
                params
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            TypeKind::Ptr(ty) => write!(f, "{}*", ty),
            TypeKind::Args => write!(f, "..."),
        }
    }
}

impl Type {
    thread_local! {
        /// The pool to implement singleton.
        ///
        /// Reference: https://github.com/pku-minic/koopa/blob/master/src/ir/types.rs
        ///
        /// XXX: This is not the only solution. In the implementation of IR, we use
        /// `UniqueArena` to store types.
        static POOL: RefCell<HashMap<TypeKind, Type>> = RefCell::new(HashMap::default());
    }

    /// Create a new type.
    pub fn make(kind: TypeKind) -> Type {
        Self::POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            if let Some(ty) = pool.get(&kind) {
                ty.clone()
            } else {
                let ty = Type(Rc::new(kind.clone()));
                pool.insert(kind, ty.clone());
                ty
            }
        })
    }

    /// Get the kind of the type.
    pub fn kind(&self) -> &TypeKind {
        &self.0
    }

    /// Create a new void type.
    pub fn void() -> Self {
        Self::make(TypeKind::Void)
    }

    /// Create a new boolean type.
    pub fn bool() -> Self {
        Self::make(TypeKind::Bool)
    }

    /// Create a new char type.
    pub fn char() -> Self {
        Self::make(TypeKind::Char)
    }

    /// Create a new integer type.
    pub fn int() -> Self {
        Self::make(TypeKind::Int)
    }

    /// Create a new float type.
    pub fn float() -> Self {
        Self::make(TypeKind::Float)
    }

    /// Create a new array type.
    pub fn array(ty: Type, size: usize) -> Self {
        Self::make(TypeKind::Array(ty, size))
    }

    /// Create a new function type.
    pub fn func(params: Vec<Type>, ret: Type) -> Self {
        Self::make(TypeKind::Func(params, ret))
    }

    /// Create a new pointer type.
    pub fn ptr(ty: Type) -> Self {
        Self::make(TypeKind::Ptr(ty))
    }

    /// Create a new args type.
    pub fn args() -> Self {
        Self::make(TypeKind::Args)
    }

    /// Check if the type is a int type.
    pub fn is_int(&self) -> bool {
        matches!(self.kind(), TypeKind::Int)
    }

    /// Check if the type is a bool type.
    pub fn is_bool(&self) -> bool {
        matches!(self.kind(), TypeKind::Bool)
    }

    /// Check if the type is a void type.
    pub fn is_void(&self) -> bool {
        matches!(self.kind(), TypeKind::Void)
    }

    /// Check if the type is a char type.
    pub fn is_char(&self) -> bool {
        matches!(self.kind(), TypeKind::Char)
    }

    /// Check if the type is a float type.
    pub fn is_float(&self) -> bool {
        matches!(self.kind(), TypeKind::Float)
    }

    /// Check if the type is a array type.
    pub fn is_array(&self) -> bool {
        matches!(self.kind(), TypeKind::Array(_, _))
    }

    /// Check if the type is a function type.
    pub fn is_func(&self) -> bool {
        matches!(self.kind(), TypeKind::Func(_, _))
    }

    /// Check if the type is a pointer type.
    pub fn is_ptr(&self) -> bool {
        matches!(self.kind(), TypeKind::Ptr(_))
    }

    /// Check if the type is a args type.
    pub fn is_args(&self) -> bool {
        matches!(self.kind(), TypeKind::Args)
    }

    /// Get the parameters and return type of a function type.
    ///
    /// # Panics
    ///
    /// - Panics if the type is not a function type.
    pub fn unwrap_func(&self) -> (&[Type], &Type) {
        if let TypeKind::Func(params, ret) = self.kind() {
            (params, ret)
        } else {
            panic!("unwrap_func: not a function type: {}", self);
        }
    }

    /// Get the bytewidth of the type.
    pub fn bytewidth(&self) -> usize {
        match self.kind() {
            TypeKind::Void => 0,
            TypeKind::Bool => 1,
            TypeKind::Char => 1,
            TypeKind::Int => 4,
            TypeKind::Float => 4, //iakke?
            TypeKind::Array(ty, size) => ty.bytewidth() * size,
            TypeKind::Func(_, _) | TypeKind::Ptr(_) => unreachable!(),
            TypeKind::Args => 0,
        }
    }
    pub fn is_equal(&self, other: &Self) -> bool {
        let mut this = self.0.as_ref();
        let mut other = other.0.as_ref();
        loop {
            match (this, other) {
                (TypeKind::Array(t1, _), TypeKind::Array(t2, _))
                | (TypeKind::Array(t1, _), TypeKind::Ptr(t2))
                | (TypeKind::Ptr(t1), TypeKind::Array(t2, _))
                | (TypeKind::Ptr(t1), TypeKind::Ptr(t2)) => {
                    this = t1.0.as_ref();
                    other = t2.0.as_ref();
                }
                (TypeKind::Int, TypeKind::Int)
                | (TypeKind::Char, TypeKind::Char)
                | (TypeKind::Float, TypeKind::Float)
                | (TypeKind::Bool, TypeKind::Bool) => {
                    return true;
                }
                _ => {
                    return this == other;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_display() {
        assert_eq!(Type::void().to_string(), "void");
        assert_eq!(Type::char().to_string(), "char");
        assert_eq!(Type::bool().to_string(), "bool");
        assert_eq!(Type::int().to_string(), "int");
        assert_eq!(Type::float().to_string(), "float");
    }
}
