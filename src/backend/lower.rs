use std::collections::HashMap;
use std::hash::Hash;

use super::func::MLabel;
use super::inst::MInst;
use super::regs::Reg;
use super::{MBlock, MContext, MFunc, PReg};
use crate::infra::linked_list::{CursorStrategy, LinkedListContainer, LinkedListNode};
use crate::ir;

/// A memory location
///
/// # Stack Layout
///
/// The stack layout of a function is as follows:
///
/// ```text
///              param by stack #n
///                     ...          -----> maybe depends on the calling convention
///              param by stack #0
///       high +-------------------+ <-- frame pointer
///            |  Saved Registers  |
///        |   +- - - - - - - - - -+
///        |   | (maybe alignment) |
///        |   +- - - - - - - - - -+ <-- start of local slots
///        |   |                   |
///  grow  |   |  Local Variables  |
///        |   |                   |
///        V   +- - - - - - - - - -+
///            |  for arg passing  |
///       low  +-------------------+ <-- stack pointer
///                     ...
/// ```
///
/// We can index local variables/slots with frame pointer or stack pointer. The
/// benefit of using stack pointer is that we can allocate fp as a general
/// purpose register. But either way, we need to modify the offset of the slots
/// after register allocation.
///
/// Actually, the space for arg passing can be dynamically allocated, but it
/// will generate more add/sub instructions. So we can reserve a fixed space
/// (with maximum required space) for arg passing at the prologue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemLoc {
    /// The memory location is a register + offset.
    RegOffset { base: Reg, offset: i64 },
    /// The memory location is a stack slot.
    ///
    /// Usually, the offset is based on frame pointer, after generating all
    /// instruction and finishing register allocation, this kind of location
    /// should be converted to [MemLoc::RegOffset], with an explicit base
    /// register, e.g., frame pointer.
    ///
    /// The offset represents the relative offset to the start of local slots,
    /// and all the slots should be modified after all the instructions are
    /// generated, and the registers are allocated.
    ///
    /// A scratch register should be reserved if the target needs a place to
    /// hold offset. e.g., for RISC-V, if the immediate cannot fit in the
    /// offset field of load/store, a scratch register will be needed to load
    /// the immediate first.
    ///
    /// XXX: if we index local slots with stack pointer, and place the offset
    /// after the saved registers we can determine the offset from sp before
    /// register allocation, and the scratch register is no longer needed.
    Slot { offset: i64 },
    /// An incoming parameter.
    ///
    /// The offset is relative to the frame pointer, i.e., the start of the
    /// saved registers.
    Incoming { offset: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MValueKind {
    Mem(MemLoc),
    /// Keep a register for all immediate values. Used for better codegen
    /// quality.
    ///
    /// Another solution is codegen in postorder (like Cranelift), but due to
    /// historical reasons, we use this solution.
    Imm(Reg, i64),
    Reg(Reg),
    Undef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MValue {
    ty: ir::Ty,
    kind: MValueKind,
}

impl MValue {
    pub fn ty(&self) -> ir::Ty { self.ty }

    pub fn kind(&self) -> MValueKind { self.kind }

    pub fn is_undef(&self) -> bool { matches!(self.kind, MValueKind::Undef) }

    pub fn new_reg(ty: ir::Ty, reg: impl Into<Reg>) -> Self {
        Self {
            ty,
            kind: MValueKind::Reg(reg.into()),
        }
    }

    pub fn new_mem(ty: ir::Ty, mem: MemLoc) -> Self {
        Self {
            ty,
            kind: MValueKind::Mem(mem),
        }
    }

    pub fn new_imm(ty: ir::Ty, reg: impl Into<Reg>, imm: i64) -> Self {
        Self {
            ty,
            kind: MValueKind::Imm(reg.into(), imm),
        }
    }

    pub fn new_undef(ty: ir::Ty) -> Self {
        Self {
            ty,
            kind: MValueKind::Undef,
        }
    }
}

#[derive(Clone)]
pub struct LowerConfig {
    pub omit_frame_pointer: bool,
    pub combine_stack_adjustments: bool,
}

impl Default for LowerConfig {
    fn default() -> Self {
        Self {
            omit_frame_pointer: true,
            combine_stack_adjustments: true,
        }
    }
}

pub struct LowerContext<'a, S>
where
    S: LowerSpec,
{
    /// The machine context.
    pub(super) mctx: MContext<S::I>,

    /// The IR context.
    pub(super) ctx: &'a ir::Context,

    /// The mapping from IR value to lowered machine value.
    pub(super) lowered: HashMap<ir::Value, MValue>,
    /// Functions in the machine code.
    ///
    /// Because we want to get the machine function by the symbol when
    /// generating call instruction, so we need to map the IR symbol to mfunc.
    pub funcs: HashMap<ir::Func, MFunc<S::I>>,
    /// Mapping IR block to machine block.
    pub blocks: HashMap<ir::Block, MBlock<S::I>>,
    /// Other global labels, for IR global slots
    ///
    /// This is usually not necessary, because we directly use the IR symbol as
    /// the machine label. However, this two are different types, so it's
    /// better to map them.
    pub labels: HashMap<ir::Global, MLabel>,

    /// The current function and block.
    pub(super) curr_func: Option<MFunc<S::I>>,
    /// The current block.
    pub(super) curr_block: Option<MBlock<S::I>>,

    /// In case the name of blocks are not allocated in the IR.
    label_counter: u32,

    pub(super) config: LowerConfig,
}

pub trait LowerSpec: Sized {
    type I: MInst<S = Self>;

    /// Get the stack alignment of the target.
    ///
    /// # Returns
    ///
    /// The stack alignment in bytes.
    fn stack_align() -> u32;

    /// Get the frame pointer register.
    fn frame_pointer() -> PReg;

    /// Get the stack pointer register.
    fn stack_pointer() -> PReg;

    /// Get the size of a pointer.
    ///
    /// # Returns
    ///
    /// The size of a pointer in bytes.
    fn pointer_size() -> usize;

    /// Get the allocatable registers.
    fn allocatable_regs() -> Vec<PReg>;

    /// Get the non-allocatable registers.
    fn non_allocatable_regs() -> Vec<PReg>;

    /// Get callee-saved registers.
    fn callee_saved_regs() -> Vec<PReg>;

    /// Get caller-saved registers.
    fn caller_saved_regs() -> Vec<PReg>;

    /// Get the aligned size of the stack frame.
    fn total_stack_size(lower: &mut LowerContext<Self>, mfunc: MFunc<Self::I>) -> u64;

    /// Get the return register by the type.
    fn return_reg(ctx: &ir::Context, ty: ir::Ty) -> PReg;

    /// Generate a move instruction.
    ///
    /// The src can be all kinds of values, including register, memory,
    /// immediate, etc. The types of src and dst must be the same when calling
    /// this function.
    fn gen_move(lower: &mut LowerContext<Self>, dst: Reg, src: MValue);

    /// Adjust stack pointer usages.
    ///
    /// The offset of the stack pointer should be adjusted after register
    /// allocation.
    fn gen_sp_adjust(lower: &mut LowerContext<Self>, offset: i64);

    fn gen_int_binary(
        lower: &mut LowerContext<Self>,
        op: ir::IntBinaryOp,
        lhs: MValue,
        rhs: MValue,
        dst_ty: ir::Ty,
    ) -> MValue;

    fn gen_cond_br(lower: &mut LowerContext<Self>, cond: MValue, dst: MBlock<Self::I>);

    fn gen_load(lower: &mut LowerContext<Self>, ty: ir::Ty, mem_loc: MemLoc) -> MValue;

    fn gen_store(lower: &mut LowerContext<Self>, val: MValue, mem_loc: MemLoc);

    fn gen_ret(lower: &mut LowerContext<Self>);

    // TODO: Add more prototypes if needed

    fn gen_func_prologue(lower: &mut LowerContext<Self>, func: MFunc<Self::I>);

    fn gen_func_epilogue(lower: &mut LowerContext<Self>, func: MFunc<Self::I>);

    fn gen_spill_load(lower: &mut LowerContext<Self>, reg: Reg, slot: MemLoc, inst: Self::I);

    fn gen_spill_store(lower: &mut LowerContext<Self>, reg: Reg, slot: MemLoc, inst: Self::I);

    fn display_reg(reg: Reg) -> String;
}

impl<'a, S> LowerContext<'a, S>
where
    S: LowerSpec,
{
    pub fn new(ctx: &'a ir::Context, config: LowerConfig) -> Self {
        Self {
            mctx: MContext::new(),
            ctx,
            lowered: HashMap::default(),
            funcs: HashMap::default(),
            blocks: HashMap::default(),
            labels: HashMap::default(),
            curr_func: None,
            curr_block: None,
            label_counter: 0,
            config,
        }
    }

    pub fn regalloc(&mut self) {
        todo!("implement register allocation");
    }

    pub fn finish(self) -> MContext<S::I> { self.mctx }

    pub fn mctx(&self) -> &MContext<S::I> { &self.mctx }

    pub fn mctx_mut(&mut self) -> &mut MContext<S::I> { &mut self.mctx }

    pub fn lower(&mut self) {
        // firstly, create all functions, globals and blocks
        for func in self.ctx.funcs() {
            let symbol = func.name(self.ctx);
            let label = MLabel::from(symbol.to_string());

            let mfunc = MFunc::new(&mut self.mctx, label);
            self.funcs.insert(func, mfunc);

            for block in func.iter(self.ctx) {
                let mblock = MBlock::new(&mut self.mctx, format!(".{}", block.name(self.ctx)));
                mfunc.push_back(&mut self.mctx, mblock);
                self.blocks.insert(block, mblock);
            }
        }

        // TODO: You also need to translate global slots in IR into data in
        // machine code
    }

    pub fn after_regalloc(&mut self) {
        for func in self.ctx.funcs() {
            let mfunc = self.funcs[&func];

            self.curr_func = Some(mfunc);

            self.curr_block = Some(mfunc.head(&self.mctx).unwrap());
            S::gen_func_prologue(self, mfunc);

            self.curr_block = Some(mfunc.tail(&self.mctx).unwrap());
            S::gen_func_epilogue(self, mfunc);
        }

        // prologue/epilogue might changed the saved regs or other things, so we
        // postpone the offset adjustment to here
        self.adjust_offset();
    }

    fn adjust_offset(&mut self) {
        for func in self.ctx.funcs() {
            let mfunc = self.funcs[&func];

            let storage_size = mfunc.storage_stack_size(&self.mctx) as i64;
            let outgoing_size = mfunc.outgoing_stack_size(&self.mctx) as i64;

            if !self.config.combine_stack_adjustments {
                assert_eq!(outgoing_size, 0);
            }

            let total_stack_size = S::total_stack_size(self, mfunc) as i64;

            let mut cursor = mfunc.cursor(self.mctx(), CursorStrategy::Post);

            while let Some(mblock) = cursor.next(self.mctx()) {
                let mut curr_inst = mblock.head(&self.mctx);

                while let Some(inst) = curr_inst {
                    // because the target might modify the instruction sequence
                    let next_inst = inst.next(&self.mctx);

                    if self.config.omit_frame_pointer {
                        inst.adjust_offset(
                            &mut self.mctx,
                            |mem_loc| match mem_loc {
                                MemLoc::Slot { offset } => Some(MemLoc::RegOffset {
                                    base: S::stack_pointer().into(),
                                    offset: storage_size + outgoing_size + offset,
                                }),
                                MemLoc::Incoming { offset } => Some(MemLoc::RegOffset {
                                    base: S::stack_pointer().into(),
                                    offset: total_stack_size + offset,
                                }),
                                MemLoc::RegOffset { .. } => None,
                            },
                            &self.config,
                        );
                    } else {
                        inst.adjust_offset(
                            &mut self.mctx,
                            |mem_loc| match mem_loc {
                                MemLoc::Slot { offset } => Some(MemLoc::RegOffset {
                                    base: S::frame_pointer().into(),
                                    offset: -(total_stack_size - storage_size - outgoing_size)
                                        + offset, // offset is negative, so add
                                }),
                                MemLoc::Incoming { offset } => Some(MemLoc::RegOffset {
                                    base: S::frame_pointer().into(),
                                    offset,
                                }),
                                MemLoc::RegOffset { .. } => None,
                            },
                            &self.config,
                        );
                    }
                    curr_inst = next_inst;
                }
            }
        }
    }

    fn lower_inst(&mut self, inst: ir::Inst) {
        use ir::InstKind as Ik;

        match inst.kind(self.ctx) {
            Ik::Load => {
                let ty = inst.result(self.ctx).unwrap().ty(self.ctx);
                let ptr = inst.operand(self.ctx, 0);
                let mem_loc = match self.lowered[&ptr].kind() {
                    MValueKind::Reg(reg) => MemLoc::RegOffset {
                        base: reg,
                        offset: 0,
                    },
                    MValueKind::Mem(loc) => loc,
                    MValueKind::Imm(..) => unreachable!(),
                    MValueKind::Undef => {
                        self.lowered
                            .insert(inst.result(self.ctx).unwrap(), MValue::new_undef(ty));
                        return;
                    }
                };

                let mval = S::gen_load(self, ty, mem_loc);
                self.lowered.insert(inst.result(self.ctx).unwrap(), mval);
            }
            Ik::Store => {
                let ptr = inst.operand(self.ctx, 1);
                let mem_loc = match self.lowered[&ptr].kind() {
                    MValueKind::Reg(reg) => MemLoc::RegOffset {
                        base: reg,
                        offset: 0,
                    },
                    MValueKind::Mem(loc) => loc,
                    MValueKind::Imm(..) => unreachable!(),
                    MValueKind::Undef => return,
                };

                let val = inst.operand(self.ctx, 0);
                S::gen_store(self, self.lowered[&val], mem_loc);
            }
            Ik::Br => {}
            Ik::Call => {}
            Ik::Ret => {
                if inst.operand_iter(self.ctx).next().is_some() {
                    let val = inst.operand(self.ctx, 0);
                    let ty = val.ty(self.ctx);
                    let mval = self.lowered[&val];
                    let ret_reg = S::return_reg(self.ctx, ty);
                    S::gen_move(self, ret_reg.into(), mval);
                }
                // `ret` should be generated in the epilogue
            }
            // TODO
            _ => todo!("implement other instructions"),
        }
    }
}
