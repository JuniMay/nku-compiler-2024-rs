//! Target Code Generation.
//!
//! The assembly code is generated here.

use std::collections::HashMap;

use super::block::MBlock;
use super::context::MContext;
use super::func::{MFunc, MLabel};
use super::imm::Imm12;
use super::inst::{AluOpRRI, AluOpRRR, LoadOp, MInst, MInstKind, StoreOp};
use super::operand::{MOperand, MOperandKind, MemLoc};
use super::regs::{self, Reg};
use crate::infra::linked_list::{LinkedListContainer, LinkedListNode};
use crate::infra::storage::ArenaPtr;
use crate::ir::{self, ConstantValue, IntBinaryOp, Ty, Value};

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

    /// Finish the code generation and return the machine code context.
    pub fn finish(self) -> MContext { self.mctx }

    /// Get a reference to the machine code context.
    pub fn mctx(&self) -> &MContext { &self.mctx }

    /// Get a mutable reference to the machine code context.
    pub fn mctx_mut(&mut self) -> &mut MContext { &mut self.mctx }

    /// Do the code generation.
    pub fn codegen(&mut self) {
        // Generate plcaceholders for all the functions and blocks.
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
        }

        // TODO: There are several things to be handled before translating instructions:
        //  1. External functions and corresponding signatures.
        //  2. Global variables/constants.

        // XXX: This is just a demonstration, you may refactor this part entirely.
        for func in self.ctx.funcs() {
            self.curr_func = Some(self.funcs[func.name(self.ctx)]);
            let mfunc = self.curr_func.unwrap();

            // TODO: Incoming parameters can be handled here.

            // XXX: You can use dominance/cfg to generate better assembly.

            // Translate the instructions.
            for block in func.iter(self.ctx) {
                self.curr_block = Some(self.blocks[&block]);
                let mblock = self.curr_block.unwrap();

                for inst in block.iter(self.ctx) {
                    // TODO: Translate the instruction.
                    match inst.kind(self.ctx) {
                        ir::InstKind::Alloca { ty } => {
                            // Allocate space on the stack.
                            let size = (ty.bitwidth(self.ctx) + 7) / 8;
                            mfunc.add_storage_stack_size(&mut self.mctx, size as u64);
                            // Because the stack grows downward, we need to use negative offset.
                            let offset = -(mfunc.storage_stack_size(&self.mctx) as i64);
                            let mem_loc = MemLoc::Slot { offset };
                            let ty = inst.result(self.ctx).unwrap().ty(self.ctx);
                            let mopd = MOperand {
                                ty,
                                kind: MOperandKind::Mem(mem_loc),
                            };
                            // Insert the result into the lowered map.
                            self.lowered.insert(inst.result(self.ctx).unwrap(), mopd);
                        }
                        ir::InstKind::Store => {
                            let val = inst.operand(self.ctx, 0);
                            let ptr = inst.operand(self.ctx, 1);
                            let mem_loc = match self.lowered[&ptr].kind {
                                MOperandKind::Mem(mem_loc) => mem_loc,
                                _ => unreachable!(),
                            };
                            // Here we use a helper function to generate the store instruction.
                            // You can change the implementation of the helper functions as you
                            // like. Or you can also not use helper functions.
                            self.gen_store(val, mem_loc);
                        }
                        ir::InstKind::Load => {
                            let ptr = inst.operand(self.ctx, 0);
                            let mem_loc = match self.lowered[&ptr].kind {
                                MOperandKind::Mem(mem_loc) => mem_loc,
                                // There might be other cases, but i'll just panic for now.
                                _ => unreachable!(),
                            };
                            let ty = inst.result(self.ctx).unwrap().ty(self.ctx);
                            let mopd = self.gen_load(ty, mem_loc);
                            self.lowered.insert(inst.result(self.ctx).unwrap(), mopd);
                        }
                        ir::InstKind::IntBinary { op } => {
                            // TODO: Here's a simple example, you may need to handle more cases.
                            let lhs = inst.operand(self.ctx, 0);
                            let rhs = inst.operand(self.ctx, 1);
                            let mopd = self.gen_int_binary(*op, lhs, rhs);
                            self.lowered.insert(inst.result(self.ctx).unwrap(), mopd);
                        }
                        ir::InstKind::Ret => {
                            // TODO: You can handle multiple return values as you like.
                            if inst.operand_iter(self.ctx).count() == 1 {
                                let val = inst.operand(self.ctx, 0);
                                self.gen_ret_move(val);
                            }
                            // The `ret` should be generated in function
                            // epilogue, after register allocation.
                        }
                        &ir::InstKind::Br => {
                            // You can also encapsulate this into a helper function for cleaner
                            // code.
                            if inst.operand_iter(self.ctx).count() == 0 {
                                // Unconditional branch.
                                let target = inst.successor(self.ctx, 0);
                                let target_block = self.blocks[&target];
                                let j = MInst::j(&mut self.mctx, target_block);
                                mblock.push_back(&mut self.mctx, j).unwrap();
                            } else {
                                // TODO: Handle conditional branch.
                                todo!()
                            }
                        }
                        // TODO: Add more instuctions.
                        _ => {
                            todo!()
                        }
                    }
                }
            }
        }
    }

    pub fn regalloc(&mut self) {
        // This is an extremely simple register allocator, which just assigns
        // the first available register to each virtual register. It's only for
        // demonstration.
        // TODO: You need to implement a real register allocator to replace this.
        let mut available_regs = vec![
            regs::t0(),
            regs::t1(),
            regs::t2(),
            regs::t3(),
            regs::t4(),
            regs::t5(),
            regs::t6(),
        ];

        for function in self.funcs.values() {
            let mut reg_map: HashMap<Reg, Reg> = HashMap::new();

            // Map virtual registers to physical registers.
            for block in function.iter(&self.mctx) {
                for inst in block.iter(&self.mctx) {
                    match inst.kind(self.mctx()) {
                        MInstKind::AluRRI { rd, rs, .. } => {
                            for reg in [rd, rs] {
                                if reg.is_vreg() && !reg_map.contains_key(reg) {
                                    let preg = available_regs.pop().unwrap();
                                    reg_map.insert(*reg, preg.into());
                                }
                            }
                        }
                        MInstKind::AluRRR { rd, rs1, rs2, .. } => {
                            for reg in [rd, rs1, rs2] {
                                if reg.is_vreg() && !reg_map.contains_key(reg) {
                                    let preg = available_regs.pop().unwrap();
                                    reg_map.insert(*reg, preg.into());
                                }
                            }
                        }
                        MInstKind::Load { rd, .. } => {
                            if rd.is_vreg() && !reg_map.contains_key(rd) {
                                let preg = available_regs.pop().unwrap();
                                reg_map.insert(*rd, preg.into());
                            }
                        }
                        MInstKind::Store { rs, .. } => {
                            if rs.is_vreg() && !reg_map.contains_key(rs) {
                                let preg = available_regs.pop().unwrap();
                                reg_map.insert(*rs, preg.into());
                            }
                        }
                        MInstKind::Li { rd, .. } => {
                            if rd.is_vreg() && !reg_map.contains_key(rd) {
                                let preg = available_regs.pop().unwrap();
                                reg_map.insert(*rd, preg.into());
                            }
                        }
                        MInstKind::J { .. } => {} /* XXX: We do not encourage using this naive
                                                   * register allocator in your work. But if you
                                                   * really want to use, you may need to handle
                                                   * other instructions. */
                    }
                }
            }

            // Replace virtual registers with physical registers.
            let mut curr_block = function.head(&self.mctx);
            while let Some(block) = curr_block {
                let mut curr_inst = block.head(&self.mctx);
                while let Some(inst) = curr_inst {
                    match &mut inst.kind_mut(&mut self.mctx) {
                        MInstKind::AluRRI { rd, rs, .. } => {
                            for reg in [rd, rs] {
                                if let Some(vreg) = reg_map.get(reg) {
                                    *reg = *vreg;
                                }
                            }
                        }
                        MInstKind::AluRRR { rd, rs1, rs2, .. } => {
                            for reg in [rd, rs1, rs2] {
                                if let Some(vreg) = reg_map.get(reg) {
                                    *reg = *vreg;
                                }
                            }
                        }
                        MInstKind::Load { rd, .. } => {
                            if let Some(vreg) = reg_map.get(rd) {
                                *rd = *vreg;
                            }
                        }
                        MInstKind::Store { rs, .. } => {
                            if let Some(vreg) = reg_map.get(rs) {
                                *rs = *vreg;
                            }
                        }
                        MInstKind::Li { rd, .. } => {
                            if let Some(vreg) = reg_map.get(rd) {
                                *rd = *vreg;
                            }
                        }
                        MInstKind::J { .. } => {} /* XXX: We do not encourage using this naive
                                                   * register allocator in your work. But if you
                                                   * really want to use, you may need to handle
                                                   * other instructions. */
                    }
                    curr_inst = inst.next(&self.mctx);
                }
                curr_block = block.next(&self.mctx);
            }
        }
    }

    /// Do the code generation after register allocation.
    pub fn after_regalloc(&mut self) {
        // TODO: The stack frame is determined after register allocation, so
        // we need to add instructions to adjust the stack frame.
        //
        // There should be two stages:
        //  1. Prologue: Save the callee-saved registers and adjust the stack frame.
        //  2. Epilogue: Restore the callee-saved registers and handle return.
        //
        // Depending on your implementation, you may need to adjust stack slots
        // offsets after these two stages.
        todo!("after register allocation");
    }

    /// Emit the assembly code.
    ///
    /// It's not necessary to implement this function, you can also handle the
    /// emission directly in `main.rs`.
    pub fn emit(&mut self) {
        // TODO: Emit the assembly code.
    }

    /// Generate a store instruction and append it to the current block.
    /// Here we just demonstrate the idea. Bugs may exist. You can
    /// refactor this entirely as your own way.
    pub fn gen_store(&mut self, val: Value, mem_loc: MemLoc) {
        let curr_block = self.curr_block.unwrap();

        // XXX: You might want to encapsulate this check into a function, since it'll be
        // used multiple times.
        let src = match &val.try_deref(self.ctx).unwrap().kind {
            ir::ValueKind::Constant { value } => {
                // XXX: You also might want to encapsulate this check into a function.
                match value {
                    ir::ConstantValue::Int32 { value, .. } => {
                        if value == &0 {
                            // Use zero register for zero value.
                            regs::zero().into()
                        } else {
                            // Sign-extend the 32-bit value to 64-bit
                            let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            r
                        }
                    }
                    ConstantValue::Undef { .. } => {
                        // We treat undef as zero temporarily.
                        regs::zero().into()
                    }
                    _ => todo!(),
                }
            }
            ir::ValueKind::InstResult { .. } => {
                // XXX: You also might want to encapsulate this check into a function.
                let mopd = self.lowered[&val];
                match mopd.kind {
                    MOperandKind::Reg(reg) => reg,
                    _ => todo!(),
                }
            }
            ir::ValueKind::Param { .. } => {
                // TODO: Handle parameters.
                todo!()
            }
        };

        let bitwidth = val.ty(self.ctx).bitwidth(self.ctx);
        // Different store instructions for different bitwidths.
        let op = match bitwidth {
            8 => StoreOp::Sb,
            16 => StoreOp::Sh,
            32 => StoreOp::Sw,
            64 => StoreOp::Sd,
            _ => unreachable!(),
        };

        // s[bhwd] src, loc
        let store = MInst::store(&mut self.mctx, op, src, mem_loc);
        // Append the store instruction to the current block.
        curr_block.push_back(&mut self.mctx, store).unwrap();
    }

    /// Generate a load instruction and append it to the current block.
    /// /// Here we just demonstrate the idea. Bugs may exist. You can
    /// refactor this entirely as your own way.
    pub fn gen_load(&mut self, ty: Ty, mem_loc: MemLoc) -> MOperand {
        let curr_block = self.curr_block.unwrap();

        let bitwidth = ty.bitwidth(self.ctx);
        let op = match bitwidth {
            8 => LoadOp::Lb,
            16 => LoadOp::Lh,
            32 => LoadOp::Lw,
            64 => LoadOp::Ld,
            _ => unreachable!(),
        };

        // l[bhwd] rd, loc
        let (load, rd) = MInst::load(&mut self.mctx, op, mem_loc);
        curr_block.push_back(&mut self.mctx, load).unwrap();

        MOperand {
            ty,
            kind: MOperandKind::Reg(rd),
        }
    }

    /// Generate an integer binary operation and append it to the current block.
    /// Here we just demonstrate the idea. Bugs may exist. You can
    /// refactor this entirely as your own way.
    pub fn gen_int_binary(&mut self, op: IntBinaryOp, lhs: Value, rhs: Value) -> MOperand {
        let curr_block = self.curr_block.unwrap();

        // TODO: We only handle reg + reg here, you need to handle other cases.
        let lhs_reg = match &lhs.try_deref(self.ctx).unwrap().kind {
            ir::ValueKind::Constant { .. } => {
                todo!()
            }
            ir::ValueKind::InstResult { .. } => {
                let mopd = self.lowered[&lhs];
                match mopd.kind {
                    MOperandKind::Reg(reg) => reg,
                    _ => todo!(),
                }
            }
            ir::ValueKind::Param { .. } => {
                todo!()
            }
        };

        let rhs_reg = match &rhs.try_deref(self.ctx).unwrap().kind {
            ir::ValueKind::Constant { .. } => {
                todo!()
            }
            ir::ValueKind::InstResult { .. } => {
                let mopd = self.lowered[&rhs];
                match mopd.kind {
                    MOperandKind::Reg(reg) => reg,
                    _ => todo!(),
                }
            }
            ir::ValueKind::Param { .. } => {
                todo!()
            }
        };

        let bitwidth = lhs.ty(self.ctx).bitwidth(self.ctx);

        match op {
            IntBinaryOp::Add => {
                // addw rd, rs1, rs2
                let alu_op = match bitwidth {
                    32 => AluOpRRR::Addw,
                    // TODO: other bitwidths?
                    _ => todo!(),
                };
                let (add, rd) = MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rhs_reg);
                curr_block.push_back(&mut self.mctx, add).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            // TODO: Add more ops.
            _ => todo!(),
        }
    }

    /// Generate a move instruction needed for return value and append it to the
    /// current block. Here we just demonstrate the idea. Bugs may exist.
    /// You can refactor this entirely as your own way.
    pub fn gen_ret_move(&mut self, val: Value) {
        let curr_block = self.curr_block.unwrap();

        // We only handle reg here, you need to handle other cases.
        let src = match &val.try_deref(self.ctx).unwrap().kind {
            ir::ValueKind::Constant { .. } => {
                todo!()
            }
            ir::ValueKind::InstResult { .. } => {
                let mopd = self.lowered[&val];
                match mopd.kind {
                    MOperandKind::Reg(reg) => reg,
                    _ => todo!(),
                }
            }
            ir::ValueKind::Param { .. } => {
                todo!()
            }
        };

        // addi a0, src, 0
        let mv = MInst::raw_alu_rri(
            &mut self.mctx,
            AluOpRRI::Addi,
            regs::a0().into(),
            src,
            Imm12::try_from_i64(0).unwrap(),
        );
        curr_block.push_back(&mut self.mctx, mv).unwrap();
    }

    // TODO: Add more helper functions.
}
