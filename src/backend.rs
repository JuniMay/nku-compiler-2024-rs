mod block;
mod context;
mod func;
mod inst;
mod lower;
mod regs;

pub mod regalloc;

#[cfg(feature = "target-riscv64")]
pub mod riscv64;

pub use block::MBlock;
pub use context::{MContext, RawData};
pub use func::MFunc;
pub use lower::{LowerConfig, LowerContext, LowerSpec};
pub use regs::{PReg, RegKind, VReg};
