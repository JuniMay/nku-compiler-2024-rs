//! Target Code Generation.
//!
//! The assembly code is generated here.

use std::collections::HashMap;

use crate::{infra::linked_list::LinkedListContainer, ir};

use super::{
    block::MBlock,
    context::MContext,
    func::{MFunc, MLabel},
    operand::{MOperand, MOperandKind, MemLoc},
};

pub struct CodegenContext<'s> {
    /// The machine code context.
    pub(super) mctx: MContext,
    /// The IR context.
    pub(super) ctx: &'s ir::Context,
    /// The mapping from IR value to machine code operand.
    pub(super) lowered: HashMap<ir::Value, MOperand>,

    /// The function mapping.
    ///
    /// We want to get machine function by the name when generating call
    /// instruction, so simply map the function name to the machine function.
    pub funcs: HashMap<String, MFunc>,

    /// Mapping IR blocks to machine blocks.
    pub blocks: HashMap<ir::Block, MBlock>,

    /// Other global labels, for global variables/constants.
    pub globals: HashMap<String, MLabel>,

    /// The current function.
    pub(super) curr_func: Option<MFunc>,
    /// The current block.
    pub(super) curr_block: Option<MBlock>,

    /// The block label counter.
    ///
    /// In case the name of blocks are not provided, we generate a unique label
    /// as the name.
    label_counter: u32,
}

impl<'s> CodegenContext<'s> {
    pub fn new(ctx: &'s ir::Context) -> Self {
        Self {
            mctx: MContext::default(),
            ctx,
            lowered: HashMap::default(),
            funcs: HashMap::default(),
            blocks: HashMap::default(),
            globals: HashMap::default(),
            curr_func: None,
            curr_block: None,
            label_counter: 0,
        }
    }

    pub fn finish(self) -> MContext {
        self.mctx
    }

    pub fn mctx(&self) -> &MContext {
        &self.mctx
    }

    pub fn mctx_mut(&mut self) -> &mut MContext {
        &mut self.mctx
    }

    pub fn codegen(&mut self) {
        for func in self.ctx.funcs() {
            let name = func.name(self.ctx);
            let label = MLabel::from(name);

            let mfunc = MFunc::new(&mut self.mctx, label);
            self.funcs.insert(name.to_string(), mfunc);

            for block in func.iter(self.ctx) {
                let mblock = MBlock::new(&mut self.mctx, format!(".{}", block.name(self.ctx)));
                let _ = mfunc.push_back(&mut self.mctx, mblock);
                self.blocks.insert(block, mblock);
            }

            // TODO: There are several things to be handled before translating instructions:
            //  1. External functions and corresponding signatures.
            //  2. Global variables/constants.

            for func in self.ctx.funcs() {
                self.curr_func = Some(self.funcs[func.name(self.ctx)]);
                let mfunc = self.curr_func.unwrap();

                // TODO: Incoming parameters can be handled here.

                // XXX: You can use dominance/cfg to generate better assembly.

                for block in func.iter(self.ctx) {
                    self.curr_block = Some(self.blocks[&block]);
                    let mblock = self.curr_block.unwrap();

                    for inst in block.iter(self.ctx) {
                        // TODO: Translate the instruction.
                        match inst.kind(self.ctx) {
                            ir::InstKind::Alloca { ty } => {
                                let size = (ty.bitwidth(self.ctx) + 7) / 8;
                                mfunc.add_storage_stack_size(&mut self.mctx, size as u64);
                                // because the stack grows downward, we need to use negative offset
                                let offset = -(mfunc.storage_stack_size(&self.mctx) as i64);
                                let mem_loc = MemLoc::Slot { offset };
                                let ty = inst.result(self.ctx).unwrap().ty(self.ctx);
                                let mopd = MOperand {
                                    ty,
                                    kind: MOperandKind::Mem(mem_loc),
                                };
                                self.lowered.insert(inst.result(self.ctx).unwrap(), mopd);
                            }
                            ir::InstKind::Store => {
                                todo!()
                            }
                            ir::InstKind::Load => {
                                todo!()
                            }
                            ir::InstKind::IntBinary { op } => {
                                todo!()
                            }
                            ir::InstKind::Br => {
                                todo!()
                            }
                            &ir::InstKind::Ret => {
                                todo!()
                            }
                            _ => todo!("MORE!!!!"),
                        }
                    }
                }
            }
        }
    }

    pub fn regalloc(&mut self) {
        // TODO: Register allocation.
    }

    pub fn after_regalloc(&mut self) {
        // TODO: The stack frame is determined after register allocation, so
        // we need to add instructions to adjust the stack frame.
    }

    pub fn emit(&mut self) {
        // TODO: Emit the assembly code.
    }
}
