//! Target Code Generation.
//!
//! The assembly code is generated here.

use std::collections::{HashMap, VecDeque};

use super::block::MBlock;
use super::context::MContext;
use super::func::{MFunc, MLabel};
use super::imm::Imm12;
use super::inst::{
    AluOpRRI, AluOpRRR, FpuCompareOp, FpuMoveOp, FpuOpRRR, LoadOp, MInst, MInstKind, StoreOp,
};
use super::operand::{MOperand, MOperandKind, MemLoc};
use super::regs::{self, Reg, RegKind};
use crate::backend::context::RawData;
use crate::backend::inst::BranchOp;
use crate::infra::linked_list::{LinkedListContainer, LinkedListNode};
use crate::infra::storage::ArenaPtr;
use crate::ir::{
    self, Block, CastOp, ConstantValue, FloatBinaryOp, FloatCmpCond, IntBinaryOp, IntCmpCond, Ty,
    TyData, Value, ValueKind,
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

trait InstRegAccess {
    fn def_reg(&self, mctx: &MContext) -> Option<Reg>;
    fn use_regs(&self, mctx: &MContext) -> Vec<Reg>;
    fn replace_reg(&mut self, mctx: &mut MContext, old_reg: Reg, new_reg: Reg);
}

impl InstRegAccess for MInst {
    fn def_reg(&self, mctx: &MContext) -> Option<Reg> {
        match self.kind(mctx) {
            MInstKind::AluRRI { rd, .. } => Some(*rd),
            MInstKind::AluRRR { rd, .. } => Some(*rd),
            MInstKind::Load { rd, .. } => Some(*rd),
            MInstKind::Li { rd, .. } => Some(*rd),
            MInstKind::La { rd, .. } => Some(*rd),
            MInstKind::FpuRRR { rd, .. } => Some(*rd),
            MInstKind::FpuMove { rd, .. } => Some(*rd),
            MInstKind::Jal { rd, .. } => Some(*rd), // 添加这一行
            _ => None,
        }
    }

    fn use_regs(&self, mctx: &MContext) -> Vec<Reg> {
        match self.kind(mctx) {
            MInstKind::AluRRI { rs, .. } => vec![*rs],
            MInstKind::AluRRR { rs1, rs2, .. } => vec![*rs1, *rs2],
            MInstKind::Store { rs, loc, .. } => {
                let mut regs = vec![*rs];
                if let MemLoc::RegOffset { base, .. } = loc {
                    regs.push(*base);
                }
                regs
            }
            MInstKind::Load { rd: _, loc, .. } => {
                let mut regs = Vec::new();
                if let MemLoc::RegOffset { base, .. } = loc {
                    regs.push(*base);
                }
                regs
            }
            MInstKind::Branch { rs1, rs2, .. } => vec![*rs1, *rs2],
            MInstKind::FpuRRR { rs1, rs2, .. } => vec![*rs1, *rs2],
            MInstKind::FpuMove { rs, .. } => vec![*rs],
            MInstKind::Jal { rd: _, rs, .. } => {
                // rd 不应该在这里
                if let Some(rs) = rs {
                    vec![*rs]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    }

    fn replace_reg(&mut self, mctx: &mut MContext, old_reg: Reg, new_reg: Reg) {
        match self.kind_mut(mctx) {
            MInstKind::AluRRI { rd, rs, .. } => {
                if *rd == old_reg {
                    *rd = new_reg
                }
                if *rs == old_reg {
                    *rs = new_reg
                }
            }
            MInstKind::AluRRR { rd, rs1, rs2, .. } => {
                if *rd == old_reg {
                    *rd = new_reg
                }
                if *rs1 == old_reg {
                    *rs1 = new_reg
                }
                if *rs2 == old_reg {
                    *rs2 = new_reg
                }
            }
            MInstKind::Load { rd, loc, .. } => {
                if *rd == old_reg {
                    *rd = new_reg
                }
                if let MemLoc::RegOffset { base, .. } = loc {
                    if *base == old_reg {
                        *base = new_reg
                    }
                }
            }
            MInstKind::Store { rs, loc, .. } => {
                if *rs == old_reg {
                    *rs = new_reg
                }
                if let MemLoc::RegOffset { base, .. } = loc {
                    if *base == old_reg {
                        *base = new_reg
                    }
                }
            }
            MInstKind::Li { rd, .. } => {
                if *rd == old_reg {
                    *rd = new_reg
                }
            }
            MInstKind::La { rd, .. } => {
                if *rd == old_reg {
                    *rd = new_reg
                }
            }
            MInstKind::Branch { rs1, rs2, .. } => {
                if *rs1 == old_reg {
                    *rs1 = new_reg
                }
                if *rs2 == old_reg {
                    *rs2 = new_reg
                }
            }
            MInstKind::FpuRRR { rd, rs1, rs2, .. } => {
                if *rd == old_reg {
                    *rd = new_reg
                }
                if *rs1 == old_reg {
                    *rs1 = new_reg
                }
                if *rs2 == old_reg {
                    *rs2 = new_reg
                }
            }
            MInstKind::FpuMove { rd, rs, .. } => {
                if *rd == old_reg {
                    *rd = new_reg
                }
                if *rs == old_reg {
                    *rs = new_reg
                }
            }
            MInstKind::Jal { rd, rs, .. } => {
                if *rd == old_reg {
                    *rd = new_reg
                }
                if let Some(r) = rs {
                    if *r == old_reg {
                        *r = new_reg
                    }
                }
            }
            _ => {}
        }
    }
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
    pub fn finish(self) -> MContext {
        self.mctx
    }

    /// Get a reference to the machine code context.
    pub fn mctx(&self) -> &MContext {
        &self.mctx
    }

    /// Get a mutable reference to the machine code context.
    pub fn mctx_mut(&mut self) -> &mut MContext {
        &mut self.mctx
    }

    /// Do the code generation.
    pub fn codegen(&mut self) {
        // Generate plcaceholders for all the functions and blocks.
        for func in self.ctx.funcs() {
            let name = func.name(self.ctx);
            let label = MLabel::from(name);

            let mfunc = MFunc::new(&mut self.mctx, label);
            self.funcs.insert(name.to_string(), mfunc);

            if !func.is_define(&self.ctx) {
                mfunc.set_external(&mut self.mctx, true);
                continue;
            }

            for block in func.iter(self.ctx) {
                let mblock = MBlock::new(
                    &mut self.mctx,
                    format!(".L{}", block.name(self.ctx)[1..].to_string().to_uppercase()),
                );
                let _ = mfunc.push_back(&mut self.mctx, mblock);
                self.blocks.insert(block, mblock);
            }
        }

        // TODO: There are several things to be handled before translating instructions:
        //  1. External functions and corresponding signatures.
        //  2. Global variables/constants.
        for global in self.ctx.globals() {
            let name = global.name(self.ctx); // 获取全局变量的名称
            let label = MLabel::from(name); // 为全局变量生成符号地址
            self.globals.insert(name.to_string(), label.clone());

            let ty = global.ty(self.ctx); // 获取全局变量的类型
            let init_value = global.value(self.ctx); // 获取全局变量的初始值（如果有）
            self.emit_global_data(label, ty, init_value);
        }

        // XXX: This is just a demonstration, you may refactor this part entirely.
        for func in self.ctx.funcs() {
            self.curr_func = Some(self.funcs[func.name(self.ctx)]);
            let mfunc = self.curr_func.unwrap();

            if mfunc.is_external(&self.mctx) {
                continue;
            }

            // Incoming arguments.
            for (i, param) in func.params(self.ctx).iter().enumerate() {
                let mopd = if i < 8 {
                    let reg = match param.ty(self.ctx).kind(self.ctx) {
                        ir::TyData::Int1 | ir::TyData::Int8 | ir::TyData::Int32 => {
                            regs::get_arg(i).into()
                        }
                        ir::TyData::Float32 => regs::get_farg(i).into(),
                        ir::TyData::Ptr { .. } | ir::TyData::Array { .. } => {
                            regs::get_arg(i).into()
                        }
                        ir::TyData::Args => continue,
                        _ => {
                            eprintln!(
                                "Unsupported parameter type: {}",
                                param.ty(self.ctx).display(&self.ctx)
                            );
                            eprintln!("Error in func: {}", func.display(&self.ctx));
                            unreachable!()
                        }
                    };
                    MOperand {
                        ty: param.ty(self.ctx),
                        kind: MOperandKind::Reg(reg),
                    }
                } else {
                    let offset = (i - 8) * 8; // 假设每个参数占用 8 字节
                    let mem_loc = MemLoc::Slot {
                        offset: offset as i64,
                    };
                    MOperand {
                        ty: param.ty(self.ctx),
                        kind: MOperandKind::Mem(mem_loc),
                    }
                };
                self.lowered.insert(param.clone(), mopd);
            }

            // XXX: You can use dominance/cfg to generate better assembly.

            // Translate the instructions.
            for block in func.iter(self.ctx) {
                self.curr_block = Some(self.blocks[&block]);
                let mblock = self.curr_block.unwrap();

                for inst in block.iter(self.ctx) {
                    println!("{}", inst.display(&self.ctx));
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

                            // let default_value = self.get_default_value(ty);
                            // self.gen_store_from_constant(default_value, mem_loc);
                        }
                        ir::InstKind::Store => {
                            let val = inst.operand(self.ctx, 0);
                            let ptr: Value = inst.operand(self.ctx, 1);
                            let memloc = self.memloc_from_value(&ptr);
                            // Here we use a helper function to generate the store instruction.
                            // You can change the implementation of the helper functions as you
                            // like. Or you can also not use helper functions.
                            self.gen_store(val, memloc);
                        }
                        ir::InstKind::Load => {
                            let ptr = inst.operand(self.ctx, 0);
                            let memloc = self.memloc_from_value(&ptr);
                            let ty = inst.result(self.ctx).unwrap().ty(self.ctx);
                            let mopd = self.gen_load(ty, memloc);
                            self.lowered.insert(inst.result(self.ctx).unwrap(), mopd);
                        }
                        ir::InstKind::IntBinary { op } => {
                            let lhs = inst.operand(self.ctx, 0);
                            let rhs = inst.operand(self.ctx, 1);
                            let dst = inst.result(self.ctx).unwrap();
                            let mopd = self.gen_int_binary(*op, lhs, rhs);
                            self.lowered.insert(dst, mopd);
                        }
                        ir::InstKind::FloatBinary { op } => {
                            let lhs = inst.operand(self.ctx, 0);
                            let rhs = inst.operand(self.ctx, 1);
                            let dst = inst.result(self.ctx).unwrap();
                            let mopd = self.get_float_binary(*op, lhs, rhs);
                            self.lowered.insert(dst, mopd);
                        }
                        ir::InstKind::Ret => {
                            // TODO: You can handle multiple return values as you like.
                            if inst.operand_iter(self.ctx).count() == 1 {
                                let val = inst.operand(self.ctx, 0);
                                self.gen_ret_move(val);
                            }
                        }
                        ir::InstKind::Phi => {
                            let dst = inst.result(self.ctx).unwrap();
                            let incomings: Vec<_> = inst.incoming_iter(self.ctx).collect();
                            let mop = self.gen_phi(&dst, incomings);
                            self.lowered.insert(dst, mop);
                        }
                        ir::InstKind::GetElementPtr { bound_ty } => {
                            let base = inst.operand(&self.ctx, 0);
                            let offsets = inst.operand_iter(&self.ctx).skip(1).collect();
                            let dst = inst.result(&self.ctx).unwrap();
                            let mopd = self.gen_gep(&base, offsets, bound_ty);
                            self.lowered.insert(dst, mopd);
                        }
                        ir::InstKind::Call => {
                            let callee = inst.operand(&self.ctx, 0);
                            let args: Vec<_> = inst.operand_iter(&self.ctx).skip(1).collect();
                            let ret = match inst.result(&self.ctx) {
                                Some(ret) => {
                                    if ret.ty(&self.ctx).is_void(&self.ctx) {
                                        None
                                    } else {
                                        Some(ret)
                                    }
                                }
                                None => None,
                            };
                            let callee_name = match &callee.try_deref(&self.ctx).unwrap().kind {
                                ir::ValueKind::Constant { value } => match value {
                                    ir::ConstantValue::GlobalRef { name, .. } => name,
                                    _ => {
                                        eprintln!("Unsupported callee: {:?}", callee);
                                        unreachable!()
                                    }
                                },
                                _ => {
                                    eprintln!("Unsupported callee: {:?}", callee);
                                    unreachable!()
                                }
                            };
                            let reg = self.gen_call(callee_name, args, ret);
                            if let Some(reg) = reg {
                                self.lowered.insert(ret.unwrap(), reg);
                            }
                        }
                        ir::InstKind::Br => {
                            // You can also encapsulate this into a helper function for cleaner
                            // code.
                            if inst.operand_iter(self.ctx).count() == 0 {
                                // Unconditional branch.
                                let target = inst.successor(self.ctx, 0);
                                let target_block = self.blocks[&target];
                                let j = MInst::j(&mut self.mctx, target_block);
                                mblock.push_back(&mut self.mctx, j).unwrap();
                            }
                        }
                        ir::InstKind::CondBr => {
                            // 获取条件操作数和目标基本块
                            let cond = inst.operand(self.ctx, 0); // 条件操作数
                            let then_dst = inst.successor(self.ctx, 0); // 条件为真时跳转的目标块
                            let else_dst = inst.successor(self.ctx, 1); // 条件为假时跳转的目标块（如果存在）

                            // 获取对应的基本块
                            let then_block = self.blocks[&then_dst];

                            // 处理 else 块（如果映射失败，则为 None）
                            let else_block = if self.blocks.contains_key(&else_dst) {
                                Some(self.blocks[&else_dst])
                            } else {
                                None
                            };

                            // 获取当前块的下一个块，作为结束块
                            let end_block = self.blocks[&block.next(self.ctx).unwrap()];

                            // 调用 gen_cond_branch 方法生成条件分支指令
                            self.gen_cond_branch(cond, then_block, else_block, end_block);
                        }
                        ir::InstKind::Cast { op } => {
                            let src = inst.operand(self.ctx, 0);
                            let dst = inst.result(self.ctx).unwrap();
                            let mopd = self.gen_cast(op, &src, &dst);
                            self.lowered.insert(dst, mopd);
                        }
                    }
                }
            }
            // // sp adjustment
            // let offset = -(mfunc.storage_stack_size(&self.mctx) as i64);
            // if offset != 0 {
            //     let addi = MInst::raw_alu_rri(
            //         &mut self.mctx,
            //         AluOpRRI::Addi,
            //         regs::sp().into(),
            //         regs::sp().into(),
            //         Imm12::try_from_i64(offset).unwrap(),
            //     );
            //     mfunc
            //         .iter(&self.mctx)
            //         .nth(0)
            //         .unwrap()
            //         .push_front(&mut self.mctx, addi)
            //         .unwrap();
            // }

            // clear the lowered map
            self.lowered.clear();
        }

        // regalloc
        self.regalloc();
        self.after_regalloc();

        // set arch
        self.mctx
            .set_arch("rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0");
    }

    pub fn regalloc(&mut self) {
        for function in self.funcs.values() {
            let mut stack_offset: i64 = -8; // 从 -8 开始，为 s0 预留空间
            let mut reg_map: HashMap<Reg, MemLoc> = HashMap::new();
            let mut temp_reg_counter: usize = 0;

            let get_next_temp_reg = |counter: &mut usize| -> Reg {
                let reg = match *counter {
                    0 => regs::t0(),
                    1 => regs::t1(),
                    2 => regs::t2(),
                    3 => regs::t3(),
                    4 => regs::t4(),
                    5 => regs::t5(),
                    _ => regs::t6(),
                };
                *counter = (*counter + 1) % 7;
                reg.into()
            };

            // 第一遍:为所有虚拟寄存器分配栈空间
            for block in function.iter(&self.mctx) {
                for inst in block.iter(&self.mctx) {
                    if let Some(rd) = inst.def_reg(&self.mctx) {
                        if rd.is_vreg() && !reg_map.contains_key(&rd) {
                            stack_offset -= 8;
                            reg_map.insert(
                                rd,
                                MemLoc::RegOffset {
                                    base: regs::sp().into(),
                                    offset: stack_offset,
                                },
                            );
                        }
                    }
                }
            }

            // 更新函数的栈空间大小，确保是 16 字节对齐
            let stack_size = (-stack_offset + 15) & !15;
            function.add_storage_stack_size(&mut self.mctx, stack_size as u64);

            // 第二遍:插入加载/存储指令
            let mut curr_block = function.head(&self.mctx);
            while let Some(block) = curr_block {
                let mut curr_inst = block.head(&mut self.mctx);
                while let Some(mut inst) = curr_inst {
                    let next_inst = inst.next(&self.mctx);
                    temp_reg_counter = 0;

                    // 收集所有需要处理的寄存器
                    let mut regs_to_process = Vec::new();

                    // 1. 收集使用的寄存器
                    let use_regs = inst.use_regs(&self.mctx);
                    for use_reg in use_regs {
                        if use_reg.is_vreg() {
                            if let Some(&stack_loc) = reg_map.get(&use_reg) {
                                regs_to_process.push((use_reg, stack_loc, true));
                            }
                        }
                    }

                    // 2. 收集定义的寄存器
                    if let Some(def_reg) = inst.def_reg(&self.mctx) {
                        if def_reg.is_vreg() {
                            if let Some(&stack_loc) = reg_map.get(&def_reg) {
                                regs_to_process.push((def_reg, stack_loc, false));
                            }
                        }
                    }

                    // 3. 处理所有寄存器
                    let mut reg_mapping = HashMap::new();

                    // 先处理所有load
                    for (vreg, stack_loc, is_use) in regs_to_process.iter() {
                        if *is_use {
                            let temp_reg = get_next_temp_reg(&mut temp_reg_counter);
                            reg_mapping.insert(*vreg, temp_reg);

                            let (load, _) = MInst::load(&mut self.mctx, LoadOp::Lw, *stack_loc);
                            // 修改load指令使用正确的目标寄存器
                            match load.kind_mut(&mut self.mctx) {
                                MInstKind::Load { rd, .. } => *rd = temp_reg,
                                _ => unreachable!(),
                            }
                            inst.insert_before(&mut self.mctx, load).unwrap();
                        }
                    }

                    // 更新指令中的寄存器
                    for (vreg, temp_reg) in reg_mapping.iter() {
                        inst.replace_reg(&mut self.mctx, *vreg, *temp_reg);
                    }

                    // 再处理所有store
                    for (vreg, stack_loc, is_use) in regs_to_process {
                        if !is_use {
                            let temp_reg = get_next_temp_reg(&mut temp_reg_counter);
                            inst.replace_reg(&mut self.mctx, vreg, temp_reg);

                            let store =
                                MInst::store(&mut self.mctx, StoreOp::Sw, temp_reg, stack_loc);
                            inst.insert_after(&mut self.mctx, store).unwrap();
                        }
                    }

                    curr_inst = next_inst;
                }

                // 如果这是最后一个基本块，处理返回值
                if block.next(&self.mctx).is_none() {
                    // 遍历最后一个基本块的所有指令
                    let mut curr_inst = block.head(&mut self.mctx);
                    while let Some(inst) = curr_inst {
                        let next_inst = inst.next(&self.mctx);

                        // 替换所有虚拟寄存器为 s0
                        match inst.kind_mut(&mut self.mctx) {
                            MInstKind::Load { rd, loc, .. } => {
                                if rd.is_vreg() {
                                    *rd = regs::s0().into();
                                }
                                if let MemLoc::RegOffset { base, .. } = loc {
                                    if base.is_vreg() {
                                        *base = regs::sp().into();
                                    }
                                }
                            }
                            MInstKind::Store { rs, loc, .. } => {
                                if rs.is_vreg() {
                                    *rs = regs::s0().into();
                                }
                                if let MemLoc::RegOffset { base, .. } = loc {
                                    if base.is_vreg() {
                                        *base = regs::sp().into();
                                    }
                                }
                            }
                            MInstKind::AluRRI { rd, rs, .. } => {
                                if rd.is_vreg() {
                                    *rd = regs::s0().into();
                                }
                                if rs.is_vreg() {
                                    *rs = regs::s0().into();
                                }
                            }
                            MInstKind::AluRRR { rd, rs1, rs2, .. } => {
                                if rd.is_vreg() {
                                    *rd = regs::s0().into();
                                }
                                if rs1.is_vreg() {
                                    *rs1 = regs::s0().into();
                                }
                                if rs2.is_vreg() {
                                    *rs2 = regs::s0().into();
                                }
                            }
                            MInstKind::Jal { rd, rs, .. } => {
                                if rd.is_vreg() {
                                    *rd = regs::s0().into();
                                }
                                if let Some(r) = rs {
                                    if r.is_vreg() {
                                        *r = regs::s0().into();
                                    }
                                }
                            }
                            _ => {}
                        }

                        curr_inst = next_inst;
                    }
                }

                curr_block = block.next(&self.mctx);
            }
        }
    }

    /// Do the code generation after register allocation.
    pub fn after_regalloc(&mut self) {
        for function in self.funcs.values_mut() {
            // 计算栈帧大小，确保16字节对齐
            let stack_size = (function.storage_stack_size(&self.mctx) + 15) & !15;
            
            // 创建 Prologue 指令序列
            let entry_block = function.head(&self.mctx);
            if let Some(block) = entry_block {
                // 1. 分配栈空间
                let prologue_addi = MInst::raw_alu_rri(
                    &mut self.mctx,
                    AluOpRRI::Addi,
                    regs::sp().into(),
                    regs::sp().into(),
                    Imm12::try_from_i64(-(stack_size as i64)).unwrap(),
                );
                
                // 2. 保存调用者保存的寄存器
                // 保存 ra (返回地址)
                let save_ra = MInst::store(
                    &mut self.mctx,
                    StoreOp::Sw,
                    regs::ra().into(),
                    MemLoc::RegOffset {
                        base: regs::sp().into(),
                        offset: stack_size as i64 - 4,
                    },
                );
                
                // 保存 s0/fp (帧指针)
                let save_s0 = MInst::store(
                    &mut self.mctx,
                    StoreOp::Sw,
                    regs::s0().into(),
                    MemLoc::RegOffset {
                        base: regs::sp().into(),
                        offset: stack_size as i64 - 8,
                    },
                );
                
                // 设置帧指针
                let setup_fp = MInst::raw_alu_rri(
                    &mut self.mctx,
                    AluOpRRI::Addi,
                    regs::s0().into(),
                    regs::sp().into(),
                    Imm12::try_from_i64(stack_size as i64).unwrap(),
                );

                // 按照顺序插入 prologue 指令
                block.push_front(&mut self.mctx, setup_fp).unwrap();
                block.push_front(&mut self.mctx, save_s0).unwrap();
                block.push_front(&mut self.mctx, save_ra).unwrap();
                block.push_front(&mut self.mctx, prologue_addi).unwrap();
            }

            // 在每个 ret 指令之前插入 Epilogue 指令
            let mut curr_block = function.head(&self.mctx);
            while let Some(block) = curr_block {
                let mut curr_inst = block.head(&mut self.mctx);
                while let Some(inst) = curr_inst {
                    let next_inst = inst.next(&self.mctx);
                    
                    // 如果是 ret 指令，在其之前插入 epilogue 序列
                    if let MInstKind::Ret = inst.kind(&self.mctx) {
                        // 1. 恢复调用者保存的寄存器
                        // 恢复 ra
                        let (restore_ra, _) = MInst::load(
                            &mut self.mctx,
                            LoadOp::Lw,
                            MemLoc::RegOffset {
                                base: regs::sp().into(),
                                offset: stack_size as i64 - 4,
                            },
                        );
                        // 确保使用正确的物理寄存器
                        match restore_ra.kind_mut(&mut self.mctx) {
                            MInstKind::Load { rd, .. } => *rd = regs::ra().into(),
                            _ => unreachable!(),
                        }
                        
                        // 恢复 s0/fp
                        let (restore_s0, _) = MInst::load(
                            &mut self.mctx,
                            LoadOp::Lw,
                            MemLoc::RegOffset {
                                base: regs::sp().into(),
                                offset: stack_size as i64 - 8,
                            },
                        );
                        // 确保使用正确的物理寄存器
                        match restore_s0.kind_mut(&mut self.mctx) {
                            MInstKind::Load { rd, .. } => *rd = regs::s0().into(),
                            _ => unreachable!(),
                        }
                        
                        // 2. 恢复栈指针
                        let epilogue_addi = MInst::raw_alu_rri(
                            &mut self.mctx,
                            AluOpRRI::Addi,
                            regs::sp().into(),
                            regs::sp().into(),
                            Imm12::try_from_i64(stack_size as i64).unwrap(),
                        );

                        // 按照顺序插入 epilogue 指令
                        // 注意顺序：先恢复寄存器，再调整栈指针
                        inst.insert_before(&mut self.mctx, restore_ra).unwrap();
                        inst.insert_before(&mut self.mctx, restore_s0).unwrap();
                        inst.insert_before(&mut self.mctx, epilogue_addi).unwrap();
                    }
                    
                    curr_inst = next_inst;
                }
                curr_block = block.next(&self.mctx);
            }
        }
    }

    /// Emit the assembly code.
    ///
    /// It's not necessary to implement this function, you can also handle the
    /// emission directly in `main.rs`.
    pub fn emit(&mut self) {
        // TODO: Emit the assembly code.
    }

    pub fn emit_global_data(&mut self, label: MLabel, ty: Ty, init_value: &ir::ConstantValue) {
        let size = (ty.bitwidth(self.ctx) + 7) / 8;

        let raw_data = match init_value {
            ir::ConstantValue::Int1 { value, .. } => {
                let bytes = vec![*value as u8];
                RawData::Bytes(bytes)
            }
            ir::ConstantValue::Int8 { value, .. } => {
                let bytes = value.to_le_bytes().to_vec();
                RawData::Bytes(bytes)
            }
            ir::ConstantValue::Int32 { value, .. } => {
                // 将整数值按字节存储到 `.data` 段
                let bytes = value.to_le_bytes().to_vec();
                RawData::Bytes(bytes)
            }
            ir::ConstantValue::Float32 { value, .. } => {
                let float_value = f32::from_bits((*value as u32)); // 将解引用后的值转为 u32
                let float_bytes = float_value.to_bits().to_le_bytes().to_vec();
                RawData::Bytes(float_bytes)
            }
            ir::ConstantValue::AggregateZero { .. } => {
                // 如果是零初始化，直接用 BSS 段管理
                RawData::Bss(size)
            }
            ir::ConstantValue::Array { elems, .. } => {
                // 如果是数组，递归处理每个元素
                let mut queue = VecDeque::new();
                let mut bytes = Vec::new();
                if elems.len() == 0 {
                    // 空数组时，直接用 BSS 段管理
                    RawData::Bss(size)
                } else {
                    for elem in elems {
                        queue.push_back(elem.clone());

                        while let Some(elem) = queue.pop_front() {
                            let elem_bytes = match elem {
                                ir::ConstantValue::Int1 { value, .. } => vec![value as u8],
                                ir::ConstantValue::Int8 { value, .. } => {
                                    value.to_le_bytes().to_vec()
                                }
                                ir::ConstantValue::Int32 { value, .. } => {
                                    value.to_le_bytes().to_vec()
                                }
                                ir::ConstantValue::Float32 { value, .. } => {
                                    let float_value = f32::from_bits((value as u32));
                                    float_value.to_bits().to_le_bytes().to_vec()
                                }
                                ir::ConstantValue::Array { elems, .. } => {
                                    queue.extend(elems.iter().cloned());
                                    continue;
                                }
                                _ => panic!("Unsupported array element: {:?}", elem),
                            };
                            bytes.extend(elem_bytes);
                        }
                    }
                    RawData::Bytes(bytes)
                }
            }
            _ => panic!("Unsupported global variable type: {:?}", init_value),
        };

        // 添加到 `raw_data`
        self.mctx.add_raw_data(label, raw_data);
    }

    pub fn get_default_value(&self, ty: Ty) -> ir::ConstantValue {
        match ty.kind(self.ctx) {
            ir::TyData::Int1 => ir::ConstantValue::Int1 {
                ty,
                value: false, // 默认值为 false
            },
            ir::TyData::Int8 => ir::ConstantValue::Int8 {
                ty,
                value: 0, // 默认值为 0
            },
            ir::TyData::Int32 => ir::ConstantValue::Int32 {
                ty,
                value: 0, // 默认值为 0
            },
            ir::TyData::Float32 => ir::ConstantValue::Float32 {
                ty,
                value: 0, // 默认值为 0.0
            },
            ir::TyData::Ptr { .. } => ir::ConstantValue::Int32 {
                ty,
                value: 0, // 默认值为 0
            },
            ir::TyData::Array { .. } => ir::ConstantValue::AggregateZero { ty },
            _ => panic!(
                "Unsupported type for default initialization: {:?}",
                ty.kind(self.ctx)
            ),
        }
    }

    pub fn gen_store_from_constant(&mut self, value: ir::ConstantValue, mem_loc: MemLoc) {
        let curr_block = self.curr_block.unwrap();

        let src = match value {
            ir::ConstantValue::Int32 { value, .. } => {
                let (li, r) = MInst::li(&mut self.mctx, value as u64);
                curr_block.push_back(&mut self.mctx, li).unwrap();
                r
            }
            ir::ConstantValue::Float32 { value, .. } => {
                let (li, r) = MInst::li(&mut self.mctx, value as u64);
                curr_block.push_back(&mut self.mctx, li).unwrap();
                r
            }
            ir::ConstantValue::Int8 { value, .. } => {
                let (li, r) = MInst::li(&mut self.mctx, value as u64);
                curr_block.push_back(&mut self.mctx, li).unwrap();
                r
            }
            ir::ConstantValue::AggregateZero { .. } => regs::zero().into(),
            _ => panic!("Unsupported constant value: {:?}", value),
        };

        let bitwidth = match value {
            ir::ConstantValue::Int32 { .. } => 32,
            ir::ConstantValue::Float32 { .. } => 32,
            ir::ConstantValue::Int8 { .. } => 8,
            ir::ConstantValue::AggregateZero { .. } => 0, // 特殊情况
            _ => panic!("Unsupported constant value for bitwidth"),
        };

        let op = match bitwidth {
            8 => StoreOp::Sb,
            16 => StoreOp::Sh,
            32 => StoreOp::Sw,
            64 => StoreOp::Sd,
            _ => unreachable!(),
        };

        let store = MInst::store(&mut self.mctx, op, src, mem_loc);
        curr_block.push_back(&mut self.mctx, store).unwrap();
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
                    ConstantValue::Int1 { value, .. } => {
                        if value == &false {
                            // Use zero register for zero value.
                            regs::zero().into()
                        } else {
                            // Sign-extend the 32-bit value to 64-bit
                            let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            r
                        }
                    }
                    ConstantValue::Int8 { value, .. } => {
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
                    ConstantValue::Int32 { value, .. } => {
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
                    ConstantValue::Float32 { value, .. } => {
                        if value == &0 {
                            // Use zero register for zero value.
                            regs::f0().into()
                        } else {
                            // Sign-extend the 32-bit value to 64-bit
                            let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            r
                        }
                    }
                    _ => {
                        eprintln!("Unsupported constant: {:?}", value);
                        unreachable!()
                    }
                }
            }
            ir::ValueKind::InstResult { .. } => {
                // XXX: You also might want to encapsulate this check into a function.
                let mopd = self.lowered[&val];
                match mopd.kind {
                    MOperandKind::Reg(reg) => reg,
                    MOperandKind::Imm(.., imm) => {
                        let (li, r) = MInst::li(&mut self.mctx, imm as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        r
                    }
                    MOperandKind::Mem(mem) => {
                        let ty = val.ty(self.ctx);
                        let mop = self.gen_load(ty, mem);
                        match mop.kind {
                            MOperandKind::Reg(reg) => reg,
                            _ => todo!(),
                        }
                    }
                    MOperandKind::Undef => regs::zero().into(),
                }
            }
            ir::ValueKind::Param { ty, index, .. } => {
                let param_reg = match ty.kind(self.ctx) {
                    ir::TyData::Int1 | ir::TyData::Int8 | ir::TyData::Int32 => {
                        regs::get_arg(*index as usize).into()
                    }
                    ir::TyData::Float32 => regs::get_farg(*index as usize).into(),
                    // XXX：对于基本类型以外的参数均不支持
                    _ => {
                        eprintln!("Unsupported type: {:?}", ty.kind(self.ctx));
                        unreachable!()
                    }
                };
                param_reg
            }
            ir::ValueKind::Array { elems, .. } => {
                println!("-------------");
                let mut ty = val.ty(self.ctx);
                while let ir::TyData::Array { elem, .. } = ty.kind(self.ctx) {
                    ty = elem;
                }
                let bytewidth = ty.bitwidth(self.ctx) as i64 / 8;
                // Different store instructions for different bitwidths.
                let mut queue = VecDeque::new();
                let (reg, mut offset) = match mem_loc {
                    MemLoc::RegOffset { base, offset } => (base, offset),
                    MemLoc::Slot { offset } => (regs::sp().into(), offset),
                    MemLoc::Incoming { offset } => (regs::sp().into(), offset),
                };
                for elem in elems {
                    println!("elem: {}", elem.display(&self.ctx, true));
                    queue.push_back(elem.clone());
                    while let Some(elem) = queue.pop_front() {
                        match &elem.try_deref(&self.ctx).unwrap().kind {
                            ir::ValueKind::Array { elems, .. } => {
                                for elem in elems {
                                    queue.push_back(elem.clone());
                                }
                            }
                            _ => {
                                self.gen_store(elem, MemLoc::RegOffset { base: reg, offset });
                                offset += bytewidth;
                            }
                        }
                    }
                }

                return;
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
    ///
    /// ty: The type of the value to be loaded.
    /// mem_loc: The memory location to load from.
    ///
    /// Return the result operand.
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
    ///
    /// op: The binary operation.
    /// lhs: The left-hand side operand.
    /// rhs: The right-hand side operand.
    ///
    /// Return the result operand.
    pub fn gen_int_binary(&mut self, op: IntBinaryOp, lhs: Value, rhs: Value) -> MOperand {
        let curr_block = self.curr_block.unwrap();

        let lhs_reg = self.reg_from_value(&lhs);
        let (rhs_reg, rhs_imm) = self.reg_or_imm_from_value(&rhs);

        let bitwidth = lhs.ty(self.ctx).bitwidth(self.ctx);

        match op {
            IntBinaryOp::Add => {
                // addw rd, rs1, rs2 | imm
                let (add, rd) = if let Some(rhs_reg) = rhs_reg {
                    let alu_op = match bitwidth {
                        1 | 8 | 32 => AluOpRRR::Addw,
                        64 => AluOpRRR::Add,
                        _ => todo!(),
                    };
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    let alu_op = match bitwidth {
                        1 | 8 | 32 => AluOpRRI::Addiw,
                        64 => AluOpRRI::Addi,
                        _ => todo!(),
                    };
                    MInst::alu_rri(&mut self.mctx, alu_op, lhs_reg, rhs_imm)
                } else {
                    todo!()
                };

                curr_block.push_back(&mut self.mctx, add).unwrap();
                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::Sub => {
                // addw rd, rs1, rs2 | imm
                let (sub, rd) = if let Some(rhs_reg) = rhs_reg {
                    let alu_op = match bitwidth {
                        1 | 8 | 32 => AluOpRRR::Subw,
                        64 => AluOpRRR::Sub,
                        _ => todo!(),
                    };
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    let alu_op = match bitwidth {
                        1 | 8 | 32 => AluOpRRI::Addiw,
                        64 => AluOpRRI::Addi,
                        _ => todo!(),
                    };
                    MInst::alu_rri(
                        &mut self.mctx,
                        alu_op,
                        lhs_reg,
                        Imm12::try_from_i64(-rhs_imm.as_i16() as i64).unwrap(),
                    )
                } else {
                    todo!()
                };
                curr_block.push_back(&mut self.mctx, sub).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::Mul => {
                let alu_op = match bitwidth {
                    1 | 8 | 32 => AluOpRRR::Mulw,
                    64 => AluOpRRR::Mul,
                    _ => todo!(),
                };
                let (mul, rd) = if let Some(rhs_reg) = rhs_reg {
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rd)
                } else {
                    todo!()
                };

                curr_block.push_back(&mut self.mctx, mul).unwrap();
                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::SDiv => {
                let alu_op = match bitwidth {
                    1 | 8 | 32 => AluOpRRR::Divw,
                    64 => AluOpRRR::Div,
                    _ => todo!(),
                };

                let (sdiv, rd) = if let Some(rhs_reg) = rhs_reg {
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rd)
                } else {
                    todo!()
                };
                curr_block.push_back(&mut self.mctx, sdiv).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::UDiv => {
                let alu_op = match bitwidth {
                    1 | 8 | 32 => AluOpRRR::Divuw,
                    64 => AluOpRRR::Divu,
                    _ => todo!(),
                };
                let (udiv, rd) = if let Some(rhs_reg) = rhs_reg {
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rd)
                } else {
                    todo!()
                };
                curr_block.push_back(&mut self.mctx, udiv).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::SRem => {
                let alu_op = match bitwidth {
                    1 | 8 | 32 => AluOpRRR::Remw,
                    64 => AluOpRRR::Rem,
                    _ => todo!(),
                };
                let (srem, rd) = if let Some(rhs_reg) = rhs_reg {
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rd)
                } else {
                    todo!()
                };
                curr_block.push_back(&mut self.mctx, srem).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::URem => {
                let alu_op = match bitwidth {
                    1 | 8 | 32 => AluOpRRR::Remuw,
                    64 => AluOpRRR::Remu,
                    _ => todo!(),
                };
                let (urem, rd) = if let Some(rhs_reg) = rhs_reg {
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    MInst::alu_rrr(&mut self.mctx, alu_op, lhs_reg, rd)
                } else {
                    todo!()
                };
                curr_block.push_back(&mut self.mctx, urem).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::Shl => {
                let alu_op = match bitwidth {
                    1 | 8 | 32 => AluOpRRI::Slliw,
                    64 => AluOpRRI::Slli,
                    _ => todo!(),
                };
                let (shl, rd) = MInst::alu_rri(&mut self.mctx, alu_op, lhs_reg, rhs_imm.unwrap());
                curr_block.push_back(&mut self.mctx, shl).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::LShr => {
                let alu_op = match bitwidth {
                    1 | 8 | 32 => AluOpRRI::Srliw,
                    64 => AluOpRRI::Srli,
                    _ => todo!(),
                };
                let (lshr, rd) = MInst::alu_rri(&mut self.mctx, alu_op, lhs_reg, rhs_imm.unwrap());
                curr_block.push_back(&mut self.mctx, lshr).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::AShr => {
                let alu_op = match bitwidth {
                    1 | 8 | 32 => AluOpRRI::Sraiw,
                    64 => AluOpRRI::Srai,
                    _ => todo!(),
                };
                let (ashr, rd) = MInst::alu_rri(&mut self.mctx, alu_op, lhs_reg, rhs_imm.unwrap());
                curr_block.push_back(&mut self.mctx, ashr).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::And => {
                let (and, rd) = if let Some(rhs_reg) = rhs_reg {
                    MInst::alu_rrr(&mut self.mctx, AluOpRRR::And, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    MInst::alu_rri(&mut self.mctx, AluOpRRI::Andi, lhs_reg, rhs_imm)
                } else {
                    todo!()
                };
                curr_block.push_back(&mut self.mctx, and).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::Or => {
                let (or, rd) = if let Some(rhs_reg) = rhs_reg {
                    MInst::alu_rrr(&mut self.mctx, AluOpRRR::Or, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    MInst::alu_rri(&mut self.mctx, AluOpRRI::Ori, lhs_reg, rhs_imm)
                } else {
                    todo!()
                };
                curr_block.push_back(&mut self.mctx, or).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::Xor => {
                let (xor, rd) = if let Some(rhs_reg) = rhs_reg {
                    MInst::alu_rrr(&mut self.mctx, AluOpRRR::Xor, lhs_reg, rhs_reg)
                } else if let Some(rhs_imm) = rhs_imm {
                    MInst::alu_rri(&mut self.mctx, AluOpRRI::Xori, lhs_reg, rhs_imm)
                } else {
                    todo!()
                };
                curr_block.push_back(&mut self.mctx, xor).unwrap();

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            IntBinaryOp::ICmp { cond } => {
                let rd = match cond {
                    IntCmpCond::Eq => {
                        let (xor, rd) = if let Some(rhs_reg) = rhs_reg {
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Xor, lhs_reg, rhs_reg)
                        } else if let Some(rhs_imm) = rhs_imm {
                            let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Xor, lhs_reg, rd)
                        } else {
                            todo!()
                        };
                        curr_block.push_back(&mut self.mctx, xor).unwrap();
                        let (seqz, rd) = MInst::alu_rri(
                            &mut self.mctx,
                            AluOpRRI::Sltiu,
                            rd,
                            Imm12::try_from_u64(1).unwrap(),
                        );
                        curr_block.push_back(&mut self.mctx, seqz).unwrap();
                        rd
                    }
                    IntCmpCond::Ne => {
                        let (xor, rd) = if let Some(rhs_reg) = rhs_reg {
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Xor, lhs_reg, rhs_reg)
                        } else if let Some(rhs_imm) = rhs_imm {
                            let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Xor, lhs_reg, rd)
                        } else {
                            todo!()
                        };
                        curr_block.push_back(&mut self.mctx, xor).unwrap();
                        let (snez, rd) =
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Sltu, regs::zero().into(), rd);
                        curr_block.push_back(&mut self.mctx, snez).unwrap();
                        rd
                    }
                    IntCmpCond::Slt => {
                        let (slt, rd) = if let Some(rhs_reg) = rhs_reg {
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Slt, lhs_reg, rhs_reg)
                        } else if let Some(rhs_imm) = rhs_imm {
                            let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Slt, lhs_reg, rd)
                        } else {
                            todo!()
                        };
                        curr_block.push_back(&mut self.mctx, slt).unwrap();
                        rd
                    }
                    IntCmpCond::Sle => {
                        let (slt, rd) = if let Some(rhs_reg) = rhs_reg {
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Slt, rhs_reg, lhs_reg)
                        } else if let Some(rhs_imm) = rhs_imm {
                            let (li, rd) = MInst::li(&mut self.mctx, rhs_imm.bits() as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            MInst::alu_rrr(&mut self.mctx, AluOpRRR::Slt, rd, lhs_reg)
                        } else {
                            todo!()
                        };
                        curr_block.push_back(&mut self.mctx, slt).unwrap();
                        let (xor, rd) = MInst::alu_rri(
                            &mut self.mctx,
                            AluOpRRI::Xori,
                            rd,
                            Imm12::try_from_u64(1).unwrap(),
                        );
                        curr_block.push_back(&mut self.mctx, xor).unwrap();

                        rd
                    }
                };

                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
        }
    }

    /// Generate a float binary operation and append it to the current block.
    ///
    /// op: The float binary operation.
    /// lhs: The left-hand side value.
    /// rhs: The right-hand side value.
    ///
    /// Returns the result register.
    pub fn get_float_binary(&mut self, op: FloatBinaryOp, lhs: Value, rhs: Value) -> MOperand {
        let curr_block = self.curr_block.unwrap();

        let lhs_reg = self.reg_from_value(&lhs);
        let rhs_reg = self.reg_from_value(&rhs);

        let bitwidth = lhs.ty(self.ctx).bitwidth(self.ctx);

        match op {
            FloatBinaryOp::Fadd => {
                // fadd.s rd, rs1, rs2
                let fpu_op = match bitwidth {
                    1 | 8 | 32 => FpuOpRRR::FaddS,
                    64 => FpuOpRRR::FaddD,
                    _ => todo!(),
                };

                let (fadd, rd) = MInst::fpu_rrr(&mut self.mctx, fpu_op, lhs_reg, rhs_reg);

                curr_block.push_back(&mut self.mctx, fadd).unwrap();
                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            FloatBinaryOp::Fsub => {
                // fsub.s rd, rs1, rs2
                let fpu_op = match bitwidth {
                    1 | 8 | 32 => FpuOpRRR::FsubS,
                    64 => FpuOpRRR::FsubD,
                    _ => todo!(),
                };

                let (fsub, rd) = MInst::fpu_rrr(&mut self.mctx, fpu_op, lhs_reg, rhs_reg);

                curr_block.push_back(&mut self.mctx, fsub).unwrap();
                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            FloatBinaryOp::Fmul => {
                // fmul.s rd, rs1, rs2
                let fpu_op = match bitwidth {
                    1 | 8 | 32 => FpuOpRRR::FmulS,
                    64 => FpuOpRRR::FmulD,
                    _ => todo!(),
                };

                let (fmul, rd) = MInst::fpu_rrr(&mut self.mctx, fpu_op, lhs_reg, rhs_reg);

                curr_block.push_back(&mut self.mctx, fmul).unwrap();
                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            FloatBinaryOp::Fdiv => {
                // fdiv.s rd, rs1, rs2
                let fpu_op = match bitwidth {
                    1 | 8 | 32 => FpuOpRRR::FdivS,
                    64 => FpuOpRRR::FdivD,
                    _ => todo!(),
                };

                let (fdiv, rd) = MInst::fpu_rrr(&mut self.mctx, fpu_op, lhs_reg, rhs_reg);

                curr_block.push_back(&mut self.mctx, fdiv).unwrap();
                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
            FloatBinaryOp::FCmp { cond } => {
                // fcmp.s rd, rs1, rs2
                let (fcmp, rd) = match cond {
                    // XXX：暂时不考虑 NaN 的情况
                    FloatCmpCond::Oeq | FloatCmpCond::Ueq => {
                        let fpu_op = match bitwidth {
                            1 | 8 | 32 => FpuOpRRR::FCmp {
                                op: FpuCompareOp::FeqS,
                            },
                            64 => FpuOpRRR::FCmp {
                                op: FpuCompareOp::FeqD,
                            },
                            _ => todo!(),
                        };
                        MInst::fpu_rrr(&mut self.mctx, fpu_op, lhs_reg, rhs_reg)
                    }
                    FloatCmpCond::Olt | FloatCmpCond::Ult => {
                        let fpu_op = match bitwidth {
                            1 | 8 | 32 => FpuOpRRR::FCmp {
                                op: FpuCompareOp::FltS,
                            },
                            64 => FpuOpRRR::FCmp {
                                op: FpuCompareOp::FltD,
                            },
                            _ => todo!(),
                        };
                        MInst::fpu_rrr(&mut self.mctx, fpu_op, lhs_reg, rhs_reg)
                    }
                    FloatCmpCond::Ole | FloatCmpCond::Ule => {
                        let fpu_op = match bitwidth {
                            1 | 8 | 32 => FpuOpRRR::FCmp {
                                op: FpuCompareOp::FleS,
                            },
                            64 => FpuOpRRR::FCmp {
                                op: FpuCompareOp::FleD,
                            },
                            _ => todo!(),
                        };
                        MInst::fpu_rrr(&mut self.mctx, fpu_op, lhs_reg, rhs_reg)
                    }
                    FloatCmpCond::One | FloatCmpCond::Une => {
                        let fpu_op = match bitwidth {
                            1 | 8 | 32 => FpuOpRRR::FCmp {
                                op: FpuCompareOp::FeqS,
                            },
                            64 => FpuOpRRR::FCmp {
                                op: FpuCompareOp::FeqD,
                            },
                            _ => todo!(),
                        };
                        let (fcmp, rd) = MInst::fpu_rrr(&mut self.mctx, fpu_op, lhs_reg, rhs_reg);
                        curr_block.push_back(&mut self.mctx, fcmp).unwrap();
                        MInst::alu_rri(
                            &mut self.mctx,
                            AluOpRRI::Xori,
                            rd,
                            Imm12::try_from_u64(1).unwrap(),
                        )
                    }
                    _ => {
                        todo!("Unsupported condition: {:?}", cond);
                    }
                };
                curr_block.push_back(&mut self.mctx, fcmp).unwrap();
                MOperand {
                    ty: lhs.ty(self.ctx),
                    kind: MOperandKind::Reg(rd),
                }
            }
        }
    }

    /// Generate a move instruction needed for return value and append it to the
    /// current block.
    ///
    /// val: The value to be moved.
    ///
    /// return: The register that stores the return value.
    pub fn gen_ret_move(&mut self, val: Value) -> Reg {
        let curr_block = self.curr_block.unwrap();
        // let curr_func = self.curr_func.unwrap();

        // handle return value.
        let val_reg = self.reg_from_value(&val);
        let ret_reg = regs::a0().into();
        let mv = MInst::raw_alu_rri(
            &mut self.mctx,
            AluOpRRI::Addi,
            ret_reg,
            val_reg,
            Imm12::try_from_i64(0).unwrap(),
        );
        curr_block.push_back(&mut self.mctx, mv).unwrap();

        // // addi sp, sp, stack_size
        // let stack_size = curr_func.storage_stack_size(&self.mctx);
        // let mv = MInst::raw_alu_rri(
        //     &mut self.mctx,
        //     AluOpRRI::Addi,
        //     regs::sp().into(),
        //     regs::sp().into(),
        //     Imm12::try_from_u64(stack_size).unwrap(),
        // );
        // curr_block.push_back(&mut self.mctx, mv).unwrap();

        // add ret
        let ret = MInst::ret(&mut self.mctx);
        curr_block.push_back(&mut self.mctx, ret).unwrap();

        regs::a0().into()
    }

    pub fn gen_cond_branch(
        &mut self,
        cond: ir::Value,
        then_block: MBlock,
        else_block: Option<MBlock>,
        end_block: MBlock,
    ) {
        let curr_block = self.curr_block.unwrap();

        let cond_reg = self.reg_from_value(&cond);

        if let Some(else_block) = else_block {
            let bnez = MInst::new(
                &mut self.mctx,
                MInstKind::Branch {
                    op: BranchOp::Bne,
                    rs1: cond_reg,
                    rs2: regs::zero().into(),
                    target: then_block,
                },
            );
            curr_block.push_back(&mut self.mctx, bnez).unwrap();

            let j_else = MInst::j(&mut self.mctx, else_block);
            curr_block.push_back(&mut self.mctx, j_else).unwrap();

            let j_end = MInst::j(&mut self.mctx, end_block);
            then_block.push_back(&mut self.mctx, j_end).unwrap();
        } else {
            let bnez = MInst::new(
                &mut self.mctx,
                MInstKind::Branch {
                    op: BranchOp::Bne,
                    rs1: cond_reg,
                    rs2: regs::zero().into(),
                    target: then_block,
                },
            );
            curr_block.push_back(&mut self.mctx, bnez).unwrap();

            let j_end = MInst::j(&mut self.mctx, end_block);
            then_block.push_back(&mut self.mctx, j_end).unwrap();
        }
    }

    // TODO: Add more helper functions.

    /// Generate a function call instruction and append it to the current block.
    ///
    /// callee_name: The name of the callee function.
    /// args: The arguments of the function call.
    /// ret: The return value of the function call.
    ///
    /// return: The register that stores the return value.
    pub fn gen_call(
        &mut self,
        callee_name: &String,
        args: Vec<Value>,
        ret: Option<Value>,
    ) -> Option<MOperand> {
        let curr_block = self.curr_block.unwrap();

        // Prepare arguments
        for (index, arg) in args.iter().enumerate() {
            let (arg_reg, arg_imm) = self.reg_or_imm_from_value(&arg);
            let mv = if let Some(arg_reg) = arg_reg {
                MInst::raw_alu_rri(
                    &mut self.mctx,
                    AluOpRRI::Addi,
                    regs::get_arg(index).into(),
                    arg_reg,
                    Imm12::try_from_i64(0).unwrap(),
                )
            } else if let Some(arg_imm) = arg_imm {
                MInst::raw_alu_rri(
                    &mut self.mctx,
                    AluOpRRI::Addi,
                    regs::get_arg(index).into(),
                    regs::zero().into(),
                    arg_imm,
                )
            } else {
                continue;
            };
            curr_block.push_back(&mut self.mctx, mv).unwrap();
        }

        // Call function
        assert!(self.funcs.contains_key(callee_name));
        let call = MInst::call(&mut self.mctx, MLabel::from(callee_name.clone()));
        curr_block.push_back(&mut self.mctx, call).unwrap();

        // Handle return value if needed
        if let Some(ret) = ret {
            let ret_reg = regs::a0().into();
            Some(MOperand {
                ty: ret.ty(self.ctx),
                kind: MOperandKind::Reg(ret_reg),
            })
        } else {
            None
        }
    }

    /// Generate a getelementptr instruction and append it to the current block.
    ///
    /// base: The base address.
    /// indices: The indices.
    ///
    /// Return the result operand.
    pub fn gen_gep(&mut self, base: &Value, indices: Vec<Value>, bound_ty: &Ty) -> MOperand {
        let curr_block = self.curr_block.unwrap();
        let curr_func = self.curr_func.unwrap();

        // Get the base register.
        let base_loc = self.memloc_from_value(&base);
        let (base_reg, mut offset) = match base_loc {
            MemLoc::RegOffset { base, offset } => (base, offset),
            MemLoc::Slot { offset } => (
                regs::sp().into(),
                offset + curr_func.storage_stack_size(&self.mctx) as i64,
            ),
            MemLoc::Incoming { offset } => (regs::fp().into(), offset),
        };

        // Calculate the offset.
        let mut ty = bound_ty.clone();
        for op in indices {
            let size = (ty.bitwidth(&self.ctx) + 7) / 8;
            if let ValueKind::Constant { value } = &op.try_deref(self.ctx).unwrap().kind {
                let value = match value {
                    ConstantValue::Int32 { value, .. } => *value as i64,
                    ConstantValue::Int8 { value, .. } => *value as i64,
                    ConstantValue::Int1 { value, .. } => *value as i64,
                    _ => todo!(),
                };
                offset -= value * size as i64;
            }
            if let Some((inner_ty, ..)) = ty.as_array(&self.ctx) {
                ty = inner_ty;
            }
        }

        // addi dst, base, offset
        let dst_reg = self.mctx.new_vreg(RegKind::General).into();
        let mv = MInst::raw_alu_rri(
            &mut self.mctx,
            AluOpRRI::Addi,
            dst_reg,
            base_reg,
            Imm12::try_from_i64(offset).unwrap(),
        );
        curr_block.push_back(&mut self.mctx, mv).unwrap();

        MOperand {
            ty,
            kind: MOperandKind::Reg(dst_reg),
        }
    }

    /// Generate a phi instruction and append it to the current block.
    ///
    /// phi: The phi instruction.
    ///
    /// Return the result operand.
    pub fn gen_phi(&mut self, dst: &Value, incomings: Vec<(Block, Value)>) -> MOperand {
        let dst_reg = match dst.ty(&self.ctx).kind(&self.ctx) {
            TyData::Int1 | TyData::Int8 | TyData::Int32 => {
                self.mctx.new_vreg(RegKind::General).into()
            }
            TyData::Float32 => self.mctx.new_vreg(RegKind::General).into(),
            _ => unimplemented!("Unsupported phi type: {:?}", dst.ty(&self.ctx)),
        };
        for (block, val) in incomings {
            let mblock = self.blocks.get(&block).unwrap();
            if let Some(mop) = self.lowered.get(&val) {
                let reg = match mop.kind {
                    MOperandKind::Reg(reg) => reg,
                    MOperandKind::Imm(.., imm) => {
                        let raw_alu_rri = MInst::raw_alu_rri(
                            &mut self.mctx,
                            AluOpRRI::Addi,
                            dst_reg,
                            regs::zero().into(),
                            Imm12::try_from_i64(imm).unwrap(),
                        );
                        mblock.push_back(&mut self.mctx, raw_alu_rri).unwrap();
                        dst_reg
                    }
                    _ => todo!(),
                };
                // 更换寄存器
                let minsts = mblock.iter(&self.mctx).collect::<Vec<_>>();
                if reg != dst_reg {
                    for inst in minsts {
                        match inst.kind_mut(&mut self.mctx) {
                            MInstKind::AluRRI { rd, rs, .. }
                            | MInstKind::FpuMove { rd, rs, .. } => {
                                if rd == &reg {
                                    *rd = dst_reg;
                                }
                                if rs == &reg {
                                    *rs = dst_reg;
                                }
                            }
                            MInstKind::AluRRR { rd, rs1, rs2, .. }
                            | MInstKind::FpuRRR { rd, rs1, rs2, .. } => {
                                if rd == &reg {
                                    *rd = dst_reg;
                                }
                                if rs1 == &reg {
                                    *rs1 = dst_reg;
                                }
                                if rs2 == &reg {
                                    *rs2 = dst_reg;
                                }
                            }
                            MInstKind::Load { rd, .. }
                            | MInstKind::Li { rd, .. }
                            | MInstKind::La { rd, .. } => {
                                if rd == &reg {
                                    *rd = dst_reg;
                                }
                            }
                            MInstKind::Store { rs, .. } => {
                                if rs == &reg {
                                    *rs = dst_reg;
                                }
                            }
                            MInstKind::Jal { rd, rs, .. } => {
                                if rd == &reg {
                                    *rd = dst_reg;
                                }
                                if let Some(rs) = rs {
                                    if rs == &reg {
                                        *rs = dst_reg;
                                    }
                                }
                            }
                            MInstKind::Branch { rs1, rs2, .. } => {
                                if rs1 == &reg {
                                    *rs1 = dst_reg;
                                }
                                if rs2 == &reg {
                                    *rs2 = dst_reg;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            self.lowered.insert(
                val.clone(),
                MOperand {
                    ty: val.ty(self.ctx),
                    kind: MOperandKind::Reg(dst_reg),
                },
            );
        }

        MOperand {
            ty: dst.ty(&self.ctx),
            kind: MOperandKind::Reg(dst_reg),
        }
    }

    pub fn gen_cast(&mut self, op: &CastOp, src: &Value, dst: &Value) -> MOperand {
        let curr_block = self.curr_block.unwrap();

        let src_reg = self.reg_from_value(src);
        let src_ty = src.ty(self.ctx);
        let dst_ty = dst.ty(self.ctx);

        let dst_reg = match op {
            CastOp::Zext | CastOp::Sext => {
                if matches!(src_ty.kind(&self.ctx), TyData::Float32)
                    || matches!(dst_ty.kind(&self.ctx), TyData::Float32)
                {
                    unreachable!("zext and sext are not supported for float");
                }
                // 这样做的原因在于，load 指令会扩展载入的小数据类型
                let dst_reg = self.mctx.new_vreg(RegKind::General).into();
                let mv = MInst::raw_alu_rri(
                    &mut self.mctx,
                    AluOpRRI::Addi,
                    dst_reg,
                    src_reg,
                    Imm12::try_from_i64(0).unwrap(),
                );
                curr_block.push_back(&mut self.mctx, mv).unwrap();
                dst_reg
            }
            CastOp::Trunc => {
                let dst_reg = self.mctx.new_vreg(RegKind::General).into();
                let mv = MInst::raw_alu_rri(
                    &mut self.mctx,
                    AluOpRRI::Addi,
                    dst_reg,
                    src_reg,
                    Imm12::try_from_i64(0).unwrap(),
                );
                curr_block.push_back(&mut self.mctx, mv).unwrap();
                let (andi, dst_reg) = MInst::alu_rri(
                    &mut self.mctx,
                    AluOpRRI::Andi,
                    dst_reg,
                    Imm12::try_from_i64(-1 & ((1 << dst_ty.bitwidth(self.ctx)) - 1) as i64)
                        .unwrap(),
                );
                curr_block.push_back(&mut self.mctx, andi).unwrap();
                dst_reg
            }
            CastOp::Bitcast => match (src_ty.kind(&self.ctx), dst_ty.kind(&self.ctx)) {
                (TyData::Float32, TyData::Float32) => {
                    let dst_reg = self.mctx.new_vreg(RegKind::Float).into();
                    let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvS, dst_reg, src_reg);
                    curr_block.push_back(&mut self.mctx, fmv).unwrap();
                    dst_reg
                }
                (TyData::Float32, _) => {
                    let dst_reg = self.mctx.new_vreg(RegKind::General).into();
                    let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvXW, dst_reg, src_reg);
                    curr_block.push_back(&mut self.mctx, fmv).unwrap();
                    dst_reg
                }
                (_, TyData::Float32) => {
                    let dst_reg = self.mctx.new_vreg(RegKind::Float).into();
                    let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvWX, dst_reg, src_reg);
                    curr_block.push_back(&mut self.mctx, fmv).unwrap();
                    dst_reg
                }
                (TyData::Void, _) | (_, TyData::Void) => unreachable!(),
                (TyData::Array { .. }, _)
                | (_, TyData::Array { .. })
                | (TyData::Func { .. }, _)
                | (_, TyData::Func { .. })
                | (TyData::Ptr { .. }, _)
                | (_, TyData::Ptr { .. }) => {
                    todo!("Unsupported cast: {:?} -> {:?}", src_ty, dst_ty)
                }
                _ => {
                    let dst_reg = self.mctx.new_vreg(RegKind::General).into();
                    let mv = MInst::raw_alu_rri(
                        &mut self.mctx,
                        AluOpRRI::Addi,
                        dst_reg,
                        src_reg,
                        Imm12::try_from_i64(0).unwrap(),
                    );
                    curr_block.push_back(&mut self.mctx, mv).unwrap();
                    dst_reg
                }
            },
            CastOp::Fptoui => {
                let dst_reg = self.mctx.new_vreg(RegKind::General).into();
                let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvWX, dst_reg, src_reg);
                curr_block.push_back(&mut self.mctx, fmv).unwrap();
                dst_reg
            }
            CastOp::Fptosi => {
                let dst_reg = self.mctx.new_vreg(RegKind::General).into();
                let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvWX, dst_reg, src_reg);
                curr_block.push_back(&mut self.mctx, fmv).unwrap();
                dst_reg
            }
            CastOp::Uitofp => {
                let dst_reg = self.mctx.new_vreg(RegKind::Float).into();
                let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvSX, dst_reg, src_reg);
                curr_block.push_back(&mut self.mctx, fmv).unwrap();
                dst_reg
            }
            CastOp::Sitofp => {
                let dst_reg = self.mctx.new_vreg(RegKind::Float).into();
                let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvSX, dst_reg, src_reg);
                curr_block.push_back(&mut self.mctx, fmv).unwrap();
                dst_reg
            }
        };

        MOperand {
            ty: src.ty(self.ctx),
            kind: MOperandKind::Reg(dst_reg),
        }
    }

    pub fn reg_from_value(&mut self, val: &Value) -> Reg {
        let curr_block = self.curr_block.unwrap();
        let ty = val.ty(self.ctx);
        if let Some(mopd) = self.lowered.get(&val) {
            match mopd.kind {
                MOperandKind::Reg(reg) => reg,
                MOperandKind::Imm(.., imm) => {
                    let (li, r) = MInst::li(&mut self.mctx, imm as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    match ty.kind(&self.ctx) {
                        TyData::Int1 | TyData::Int8 | TyData::Int32 => r,
                        TyData::Float32 => {
                            let fs1 = self.mctx.new_vreg(RegKind::Float).into();
                            let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvSX, fs1, r);
                            curr_block.push_back(&mut self.mctx, fmv).unwrap();
                            fs1
                        }
                        _ => todo!(),
                    }
                }
                MOperandKind::Undef => regs::zero().into(),
                MOperandKind::Mem(loc) => {
                    let mop = self.gen_load(val.ty(self.ctx), loc);
                    match mop.kind {
                        MOperandKind::Reg(reg) => reg,
                        _ => todo!(),
                    }
                }
            }
        } else {
            match &val.try_deref(self.ctx).unwrap().kind {
                ir::ValueKind::Constant { value } => match value {
                    ConstantValue::Int32 { value, .. } => {
                        let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        r
                    }
                    ConstantValue::Int8 { value, .. } => {
                        let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        r
                    }
                    ConstantValue::Int1 { value, .. } => {
                        let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        r
                    }
                    ConstantValue::Float32 { value, .. } => {
                        let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        r
                    }
                    ConstantValue::Undef { .. } => {
                        let (li, r) = MInst::li(&mut self.mctx, 0);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        r
                    }
                    _ => {
                        eprintln!("Unsupported constant: {:?}", value);
                        unreachable!()
                    }
                },
                ir::ValueKind::InstResult { .. } | ir::ValueKind::Param { .. } => {
                    let vreg = match ty.kind(&self.ctx) {
                        TyData::Int1 | TyData::Int8 | TyData::Int32 => {
                            self.mctx.new_vreg(RegKind::General)
                        }
                        TyData::Float32 => self.mctx.new_vreg(RegKind::Float),
                        _ => todo!(),
                    };
                    let reg = vreg.into();
                    self.lowered.insert(
                        val.clone(),
                        MOperand {
                            ty,
                            kind: MOperandKind::Reg(reg),
                        },
                    );
                    reg
                }
                _ => todo!(),
            }
        }
    }

    pub fn reg_or_imm_from_value(&mut self, val: &Value) -> (Option<Reg>, Option<Imm12>) {
        let curr_block = self.curr_block.unwrap();
        let ty = val.ty(self.ctx);
        if let Some(mopd) = self.lowered.get(&val) {
            match mopd.kind {
                MOperandKind::Reg(reg) => (Some(reg), None),
                MOperandKind::Imm(.., imm) => match ty.kind(&self.ctx) {
                    TyData::Int1 | TyData::Int8 | TyData::Int32 => {
                        println!("try from i64: {}", imm);
                        if let Some(imm) = Imm12::try_from_i64(imm) {
                            (None, Some(imm))
                        } else {
                            let (li, r) = MInst::li(&mut self.mctx, imm as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            (Some(r), None)
                        }
                    }
                    TyData::Float32 => {
                        let (li, r) = MInst::li(&mut self.mctx, imm as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        let fs1 = self.mctx.new_vreg(RegKind::Float).into();
                        let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvSX, fs1, r);
                        curr_block.push_back(&mut self.mctx, fmv).unwrap();
                        (Some(fs1), None)
                    }
                    _ => todo!(),
                },
                MOperandKind::Undef => (Some(regs::zero().into()), None),
                MOperandKind::Mem(loc) => {
                    let mop = self.gen_load(val.ty(self.ctx), loc);
                    match mop.kind {
                        MOperandKind::Reg(reg) => (Some(reg), None),
                        _ => todo!(),
                    }
                }
            }
        } else {
            match &val.try_deref(self.ctx).unwrap().kind {
                ir::ValueKind::Constant { value } => match value {
                    ir::ConstantValue::Int32 { value, .. } => {
                        if let Some(imm) = Imm12::try_from_i64(*value as i64) {
                            (None, Some(imm))
                        } else {
                            let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            (Some(r), None)
                        }
                    }
                    ir::ConstantValue::Int8 { value, .. } => {
                        if let Some(imm) = Imm12::try_from_i64(*value as i64) {
                            (None, Some(imm))
                        } else {
                            let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            (Some(r), None)
                        }
                    }
                    ir::ConstantValue::Int1 { value, .. } => {
                        if let Some(imm) = Imm12::try_from_i64(*value as i64) {
                            (None, Some(imm))
                        } else {
                            let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                            curr_block.push_back(&mut self.mctx, li).unwrap();
                            (Some(r), None)
                        }
                    }
                    ir::ConstantValue::Float32 { value, .. } => {
                        let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        let fs1 = self.mctx.new_vreg(RegKind::Float).into();
                        let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvSX, fs1, r);
                        curr_block.push_back(&mut self.mctx, fmv).unwrap();
                        (Some(fs1), None)
                    }
                    ir::ConstantValue::Undef { .. } => (Some(regs::zero().into()), None),
                    _ => {
                        eprintln!("Unsupported constant: {:?}", value);
                        unreachable!()
                    }
                },
                ir::ValueKind::InstResult { .. } | ir::ValueKind::Param { .. } => {
                    let vreg = match ty.kind(&self.ctx) {
                        TyData::Int1 | TyData::Int8 | TyData::Int32 => {
                            self.mctx.new_vreg(RegKind::General)
                        }
                        TyData::Float32 => self.mctx.new_vreg(RegKind::Float),
                        _ => todo!(),
                    };
                    let reg = vreg.into();
                    self.lowered.insert(
                        val.clone(),
                        MOperand {
                            ty,
                            kind: MOperandKind::Reg(reg),
                        },
                    );
                    (Some(reg), None)
                }
                _ => todo!(),
            }
        }
    }

    pub fn memloc_from_value(&mut self, val: &Value) -> MemLoc {
        let curr_block = self.curr_block.unwrap();
        let curr_func = self.curr_func.unwrap();
        if let Some(mopd) = self.lowered.get(&val) {
            match mopd.kind {
                MOperandKind::Reg(reg) => MemLoc::RegOffset {
                    base: reg,
                    offset: 0,
                },
                MOperandKind::Imm(..) => todo!(),
                MOperandKind::Undef => MemLoc::RegOffset {
                    base: regs::zero().into(),
                    offset: 0,
                },
                MOperandKind::Mem(loc) => match loc {
                    MemLoc::RegOffset { .. } => loc,
                    MemLoc::Slot { offset } => MemLoc::RegOffset {
                        base: regs::sp().into(),
                        offset: offset + curr_func.storage_stack_size(&self.mctx) as i64,
                    },
                    MemLoc::Incoming { offset } => MemLoc::RegOffset {
                        base: regs::fp().into(),
                        offset,
                    },
                },
            }
        } else {
            match &val.try_deref(self.ctx).unwrap().kind {
                ir::ValueKind::Constant { value } => match value {
                    ir::ConstantValue::GlobalRef { name, .. } => {
                        let (la, rd) = MInst::la(&mut self.mctx, &name);
                        curr_block.push_back(&mut self.mctx, la).unwrap();
                        let loc = MemLoc::RegOffset {
                            base: rd,
                            offset: 0,
                        };
                        self.lowered.insert(
                            val.clone(),
                            MOperand {
                                ty: val.ty(self.ctx),
                                kind: MOperandKind::Mem(loc),
                            },
                        );
                        loc
                    }
                    _ => {
                        eprintln!("Unsupported constant: {:?}", value);
                        unreachable!()
                    }
                },
                _ => todo!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{irgen, preprocess, SysYParser};

    #[test]
    fn test_register_allocation() {
        let source = r#"
        int main() {
            int a = 1;
            int b = 2;
            int c = a + b;
            return c;
        }
        "#;

        let src = preprocess(source);
        let mut ast = SysYParser::new().parse(&src).unwrap();
        ast.type_check();

        let ir = irgen(&ast, 8);
        println!("=== IR ===");
        println!("{}", ir);

        let mut codegen_ctx = CodegenContext::new(&ir);
        codegen_ctx.mctx_mut().set_arch("rv64imafdc_zba_zbb");

        codegen_ctx.codegen();
        println!("\n=== Final Assembly ===");
        println!("{}", codegen_ctx.finish().display());
    }

    #[test]
    fn test_multi_function_calls() {
        let source = r#"
        int add(int a, int b) {
            return a + b;
        }

        int factorial(int n) {
            if (n <= 1) {
                return 1;
            }
            return n * factorial(n - 1);
        }

        int fibonacci(int n) {
            if (n <= 1) {
                return n;
            }
            return fibonacci(n - 1) + fibonacci(n - 2);
        }

        int main() {
            int result1 = add(3, 4);          // Should be 7
            int result2 = factorial(5);        // Should be 120
            int result3 = fibonacci(6);        // Should be 8
            return result1 + result2 + result3; // Should be 135
        }
        "#;

        let src = preprocess(source);
        let mut ast = SysYParser::new().parse(&src).unwrap();
        ast.type_check();

        let ir = irgen(&ast, 8);
        println!("=== IR ===");
        println!("{}", ir);

        let mut codegen_ctx = CodegenContext::new(&ir);
        codegen_ctx.mctx_mut().set_arch("rv64imafdc_zba_zbb");

        codegen_ctx.codegen();
        println!("\n=== Final Assembly ===");
        println!("{}", codegen_ctx.finish().display());
    }
}
