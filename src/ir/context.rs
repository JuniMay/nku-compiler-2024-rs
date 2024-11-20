use std::fmt;

use super::block::BlockData;
use super::func::FuncData;
use super::global::GlobalData;
use super::inst::InstData;
use super::ty::TyData;
use super::value::ValueData;
use super::Func;
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
    /// Storage for global variables.
    pub(super) globals: GenericArena<GlobalData>,

    /// Target information.
    pub(super) target: TargetInfo,
}

impl Default for Context {
    fn default() -> Self { Self::new(4) }
}

impl Context {
    pub fn new(ptr_size: u32) -> Self {
        Self {
            tys: UniqueArena::default(),
            blocks: GenericArena::default(),
            insts: GenericArena::default(),
            funcs: GenericArena::default(),
            values: GenericArena::default(),
            globals: GenericArena::default(),
            target: TargetInfo { ptr_size },
        }
    }

    

    pub fn set_target_info(&mut self, target: TargetInfo) { self.target = target; }

    pub fn funcs(&self) -> impl Iterator<Item = Func> + '_ {
        self.funcs.iter().map(|data| data.self_ptr)
    }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for GlobalData {
            self_ptr: global, ..
        } in self.globals.iter()
        {
            writeln!(f, "{}", global.display(self))?;
        }

        for FuncData { self_ptr: func, .. } in self.funcs.iter() {
            writeln!(f, "{}", func.display(self))?;
        }

        Ok(())
    }
}
