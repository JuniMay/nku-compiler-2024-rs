use super::block::BlockData;
use super::func::FuncData;
use super::inst::InstData;
use super::ty::TyData;
use super::value::ValueData;
use crate::infra::storage::{GenericArena, UniqueArena};

pub struct TargetInfo {
    /// Pointer size in bytes.
    pub ptr_size: u32,
}

pub struct Context {
    /// Singleton storage for types.
    pub(super) tys: UniqueArena<TyData>,
    /// Storage for blocks.
    pub(super) blocks: GenericArena<BlockData>,
    /// Storage for instructions.
    pub(super) insts: GenericArena<InstData>,
    /// Storage for functions.
    pub(super) funcs: GenericArena<FuncData>,
    /// Storage for values.
    pub(super) values: GenericArena<ValueData>,

    /// Target information.
    pub(super) target: TargetInfo,
}

impl Context {
    pub fn new(ptr_size: u32) -> Self {
        Self {
            tys: UniqueArena::default(),
            blocks: GenericArena::default(),
            insts: GenericArena::default(),
            funcs: GenericArena::default(),
            values: GenericArena::default(),
            target: TargetInfo { ptr_size },
        }
    }
}
