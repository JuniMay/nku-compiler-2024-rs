//! Target Code Generation.
//!
//! The assembly code is generated here.

use std::collections::HashMap;

use crate::{infra::linked_list::LinkedListContainer, ir};

use super::{
    block::MBlock,
    context::MContext,
    func::{MFunc, MLabel},
    operand::MOperand,
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

                // TODO: Incoming parameters can be handled here.

                // XXX: You can use dominance/cfg to generate better assembly.

                for block in func.iter(self.ctx) {
                    self.curr_block = Some(self.blocks[&block]);

                    for inst in block.iter(self.ctx) {
                        // TODO: Translate the instruction.
                    }
                }
            }
        }
    }
}
