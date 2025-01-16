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
    self, ConstantValue, FloatBinaryOp, FloatCmpCond, IntBinaryOp, IntCmpCond, Ty, Value, ValueKind,
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
                let mblock = MBlock::new(&mut self.mctx, format!(".{}", block.name(self.ctx)));
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
                // self.curr_block.unwrap().push_back(&mut self.mctx, );
                continue;
            }

            // TODO: Incoming parameters can be handled here.
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
                            let ptr = inst.operand(self.ctx, 1);
                            let mem_loc = match self.lowered[&ptr].kind {
                                MOperandKind::Mem(mem_loc) => match mem_loc {
                                    MemLoc::RegOffset { .. } => mem_loc,
                                    MemLoc::Slot { offset } => MemLoc::RegOffset {
                                        base: regs::sp().into(),
                                        offset,
                                    },
                                    _ => todo!(),
                                },
                                MOperandKind::Reg(reg) => {
                                    let mem_loc = MemLoc::RegOffset {
                                        base: reg,
                                        offset: 0,
                                    };
                                    mem_loc
                                }
                                // There might be other cases, but i'll just panic for now.
                                _ => {
                                    eprintln!(
                                        "Unsupported instruction: {}",
                                        inst.display(&self.ctx)
                                    );
                                    panic!()
                                }
                            };
                            println!("store inst: {}", inst.display(&self.ctx));
                            // Here we use a helper function to generate the store instruction.
                            // You can change the implementation of the helper functions as you
                            // like. Or you can also not use helper functions.
                            self.gen_store(val, mem_loc);
                        }
                        ir::InstKind::Load => {
                            let ptr = inst.operand(self.ctx, 0);
                            let mem_loc = match self.lowered[&ptr].kind {
                                MOperandKind::Mem(mem_loc) => match mem_loc {
                                    MemLoc::RegOffset { .. } => mem_loc,
                                    MemLoc::Slot { offset } => MemLoc::RegOffset {
                                        base: regs::sp().into(),
                                        offset,
                                    },
                                    _ => todo!(),
                                },
                                MOperandKind::Reg(reg) => {
                                    let mem_loc = MemLoc::RegOffset {
                                        base: reg,
                                        offset: 0,
                                    };
                                    mem_loc
                                }
                                // There might be other cases, but i'll just panic for now.
                                _ => {
                                    eprintln!(
                                        "Unsupported instruction: {}",
                                        inst.display(&self.ctx)
                                    );
                                    panic!()
                                }
                            };
                            let ty = inst.result(self.ctx).unwrap().ty(self.ctx);
                            let mopd = self.gen_load(ty, mem_loc);
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
                        ir::InstKind::Phi => todo!(),//iakke
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
                            let ret = inst.result(&self.ctx);
                            let callee_name = match &callee.try_deref(&self.ctx).unwrap().kind {
                                ir::ValueKind::Constant { value } => match value {
                                    ir::ConstantValue::GlobalRef { ty, name, value_ty } => name,
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
                                let j = MInst::j(&mut self.mctx, None, target_block);
                                mblock.push_back(&mut self.mctx, j).unwrap();
                            } else {
                                // TODO: Handle conditional branch.
                                todo!()
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
                        ir::InstKind::Cast { op } => todo!(),//iakke
                    }
                }
            }
        }
    }

    pub fn regalloc(&mut self) {
        // 使用栈分配虚拟寄存器，初始化栈偏移和寄存器映射
        for function in self.funcs.values() {
            let mut stack_offset: i64 = 0; // 栈向下增长
            let mut reg_map: HashMap<Reg, MemLoc> = HashMap::new(); // 虚拟寄存器 -> 栈位置的映射
    
            // 遍历每个基本块中的每条指令，分配栈空间
            for block in function.iter(&self.mctx) {
                for inst in block.iter(&self.mctx) {
                    match inst.kind(self.mctx()) {
                        // 为所有操作数分配栈空间
                        MInstKind::AluRRI { rd, rs, .. } => {
                            for reg in [rd, rs] {
                                if reg.is_vreg() && !reg_map.contains_key(&reg) {
                                    stack_offset -= 8; // 每个虚拟寄存器占用8字节
                                    reg_map.insert(*reg, MemLoc::Slot { offset: stack_offset });
                                }
                            }
                        }
                        MInstKind::AluRRR { rd, rs1, rs2, .. } => {
                            for reg in [rd, rs1, rs2] {
                                if reg.is_vreg() && !reg_map.contains_key(&reg) {
                                    stack_offset -= 8;
                                    reg_map.insert(*reg, MemLoc::Slot { offset: stack_offset });
                                }
                            }
                        }
                        MInstKind::Load { rd, .. } => {
                            if rd.is_vreg() && !reg_map.contains_key(rd) {
                                stack_offset -= 8;
                                reg_map.insert(*rd, MemLoc::Slot { offset: stack_offset });
                            }
                        }
                        MInstKind::Store { rs, .. } => {
                            if rs.is_vreg() && !reg_map.contains_key(rs) {
                                stack_offset -= 8;
                                reg_map.insert(*rs, MemLoc::Slot { offset: stack_offset });
                            }
                        }
                        MInstKind::Li { rd, .. } => {
                            if rd.is_vreg() && !reg_map.contains_key(rd) {
                                stack_offset -= 8;
                                reg_map.insert(*rd, MemLoc::Slot { offset: stack_offset });
                            }
                        }
                        _ => {}
                    }
                }
            }
    
            // 更新函数所需的栈空间大小
            function.add_storage_stack_size(&mut self.mctx, (-stack_offset) as u64);
    
            // 遍历基本块并替换虚拟寄存器为栈操作
            let mut curr_block = function.head(&self.mctx);
            while let Some(block) = curr_block {
                let mut curr_inst = block.head(&mut self.mctx); // 获取可变引用
                while let Some(mut inst) = curr_inst {
                    match &mut inst.kind_mut(&mut self.mctx) {
                        MInstKind::AluRRI { rd, rs, .. } => {
                            let replacements: Vec<MInstKind> = [rd, rs]
                                .iter()
                                .filter_map(|reg| {
                                    reg_map.get(reg).map(|&mem_loc| {
                                        MInstKind::Store {
                                            op: StoreOp::Sw,
                                            rs: regs::sp().into(),
                                            loc: mem_loc,
                                        }
                                    })
                                })
                                .collect();
                        
                            for new_kind in replacements {
                                inst.replace(&mut self.mctx, new_kind); // 替换指令
                            }
                        }
                        
                        MInstKind::Li { rd, imm } => {
                            if let Some(mem_loc) = reg_map.get(rd) {
                                *rd = regs::sp().into(); // 替换寄存器为栈指针
                                let new_kind = MInstKind::Store {
                                    op: StoreOp::Sw,
                                    rs: *rd,
                                    loc: *mem_loc,
                                };
                                inst.replace(&mut self.mctx, new_kind); // 替换指令
                            }
                        }                        
                        MInstKind::Load { rd, loc,.. } => {
                            if let Some(mem_loc) = reg_map.get(rd) {
                                *rd = regs::sp().into();
                                *loc = *mem_loc; // 替换目标寄存器位置
                            }
                        }
                        MInstKind::Store { rs, loc,.. } => {
                            if let Some(mem_loc) = reg_map.get(rs) {
                                *rs = regs::sp().into();
                                *loc = *mem_loc; // 替换源寄存器位置
                            }
                        }
                        _ => {}
                    }
                    curr_inst = inst.next(&self.mctx); // 移动到下一个指令
                }
                curr_block = block.next(&self.mctx); // 移动到下一个基本块
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

    pub fn gen_block(&mut self, block: ir::Block) {
        // 设置当前块为给定块
        self.curr_block = Some(self.blocks[&block]);

        // 遍历块中的每条指令并生成机器码
        for inst in block.iter(self.ctx) {
            match inst.kind(self.ctx) {
                ir::InstKind::Alloca { ty } => {
                    let size = (ty.bitwidth(self.ctx) + 7) / 8;
                    let mfunc = self.curr_func.unwrap();
                    mfunc.add_storage_stack_size(&mut self.mctx, size as u64);

                    let offset = -(mfunc.storage_stack_size(&self.mctx) as i64);
                    let mem_loc = MemLoc::Slot { offset };
                    let ty = inst.result(self.ctx).unwrap().ty(self.ctx);
                    let mopd = MOperand {
                        ty,
                        kind: MOperandKind::Mem(mem_loc),
                    };
                    self.lowered.insert(inst.result(self.ctx).unwrap(), mopd);

                    let default_value = self.get_default_value(ty);
                    self.gen_store_from_constant(default_value, mem_loc);
                }
                ir::InstKind::Store => {
                    let val = inst.operand(self.ctx, 0);
                    let ptr = inst.operand(self.ctx, 1);
                    let mem_loc = match self.lowered[&ptr].kind {
                        MOperandKind::Mem(mem_loc) => mem_loc,
                        _ => unreachable!(),
                    };
                    self.gen_store(val, mem_loc);
                }
                ir::InstKind::Load => {
                    let ptr = inst.operand(self.ctx, 0);
                    let mem_loc = match self.lowered[&ptr].kind {
                        MOperandKind::Mem(mem_loc) => mem_loc,
                        _ => unreachable!(),
                    };
                    let ty = inst.result(self.ctx).unwrap().ty(self.ctx);
                    let mopd = self.gen_load(ty, mem_loc);
                    self.lowered.insert(inst.result(self.ctx).unwrap(), mopd);
                }
                ir::InstKind::Br => {
                    if inst.operand_iter(self.ctx).count() == 0 {
                        let target = inst.successor(self.ctx, 0);
                        let target_block = self.blocks[&target];
                        let j = MInst::j(&mut self.mctx, None, target_block);
                        self.curr_block.unwrap().push_back(&mut self.mctx, j).unwrap();
                    }
                }
                ir::InstKind::CondBr => {
                    let cond = inst.operand(self.ctx, 0);
                    let then_dst = inst.successor(self.ctx, 0);
                    let else_dst = inst.successor(self.ctx, 1);

                    let then_block = self.blocks[&then_dst];
                    let else_block = self.blocks.get(&else_dst).copied();
                    let end_block = self.blocks[&block.next(self.ctx).unwrap()];

                    self.gen_cond_branch(cond, then_block, else_block, end_block);
                }
                _ => {
                    println!("Unsupported instruction: {:?}", inst.kind(self.ctx));
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

        // We only handle reg here, you need to handle other cases.
        let src = self.reg_from_value(&val);

        // addi a0, src, 0
        let mv = MInst::raw_alu_rri(
            &mut self.mctx,
            AluOpRRI::Addi,
            regs::a0().into(),
            src,
            Imm12::try_from_i64(0).unwrap(),
        );
        curr_block.push_back(&mut self.mctx, mv).unwrap();

        regs::a0().into()
    }

    // pub fn gen_ret_move(&mut self, val: Value) {
    //     let curr_block = self.curr_block.unwrap();
    
    //     // 根据返回值类型确定目标寄存器
    //     let ty = val.ty(self.ctx);
    //     let ret_reg = match ty.kind(self.ctx) {
    //         ir::TyData::Int32 | ir::TyData::Int1 => regs::a0().into(),
    //         ir::TyData::Float32 => regs::fa0().into(),
    //         _ => panic!("Unsupported return type: {:?}", ty),
    //     };
    
    //     // 获取返回值来源
    //     let src = match &val.try_deref(self.ctx).unwrap().kind {
    //         ir::ValueKind::Constant { value } => {
    //             let imm = match value {
    //                 ir::ConstantValue::Int32 { value, .. } => *value as u64,
    //                 ir::ConstantValue::Float32 { value, .. } => *value as u64, // Bit cast for floats
    //                 _ => panic!("Unsupported constant value: {:?}", value),
    //             };
    //             let (li, reg) = MInst::li(&mut self.mctx, imm);
    //             curr_block.push_back(&mut self.mctx, li).unwrap();
    //             reg
    //         }
    //         ir::ValueKind::InstResult { .. } => {
    //             let mopd = self.lowered[&val];
    //             match mopd.kind {
    //                 MOperandKind::Reg(reg) => reg,
    //                 _ => panic!("Unsupported InstResult kind: {:?}", mopd.kind),
    //             }
    //         }
    //         ir::ValueKind::Param { .. } => {
    //             let param_reg = match ty.kind(self.ctx) {
    //                 ir::TyData::Int32 => regs::a0().into(),
    //                 ir::TyData::Float32 => regs::fa0().into(),
    //                 _ => panic!("Unsupported parameter type: {:?}", ty),
    //             };
    //             param_reg
    //         }
    //         _ => panic!("Unsupported return value kind: {:?}", val.kind(self.ctx)),
    //     };
    
    //     // 生成返回值存储指令
    //     let mv = MInst::raw_alu_rri(
    //         &mut self.mctx,
    //         AluOpRRI::Addi,
    //         ret_reg,
    //         src,
    //         Imm12::try_from_i64(0).unwrap(),
    //     );
    //     curr_block.push_back(&mut self.mctx, mv).unwrap();
    // }

    pub fn gen_cond_branch(
        &mut self,
        cond: ir::Value,
        then_block: MBlock,
        else_block: Option<MBlock>,
        end_block: MBlock,
    ) {
        let curr_block = self.curr_block.unwrap();

        let cond_reg = match self.lowered.get(&cond) {
            Some(mopd) => match mopd.kind {
                MOperandKind::Reg(reg) => reg,
                MOperandKind::Imm(_, imm) => {
                    let (li, reg) = MInst::li(&mut self.mctx, imm as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    reg
                }
                MOperandKind::Undef => regs::zero().into(),
                _ => panic!("Unsupported condition operand kind: {:?}", mopd.kind),
            },
            None => panic!("Condition value not lowered: {:?}", cond),
        };

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

            let j_else = MInst::j(&mut self.mctx, None, else_block);
            curr_block.push_back(&mut self.mctx, j_else).unwrap();

            let j_end = MInst::j(&mut self.mctx, None, end_block);
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

            let j_end = MInst::j(&mut self.mctx, None, end_block);
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

        // Prepare arguments.
        for arg in args {
            let arg_reg = self.reg_from_value(&arg);
            let mv = MInst::raw_alu_rri(
                &mut self.mctx,
                AluOpRRI::Addi,
                regs::a0().into(),
                arg_reg,
                Imm12::try_from_i64(0).unwrap(),
            );
            curr_block.push_back(&mut self.mctx, mv).unwrap();
        }

        // find the function block
        let func = self.funcs.get(callee_name).unwrap();
        let block = func.head(&self.mctx).unwrap();

        // jal func
        let call = MInst::call(&mut self.mctx, block);
        curr_block.push_back(&mut self.mctx, call).unwrap();

        // Handle return value.
        if let Some(ret) = ret {
            Some(MOperand {
                ty: ret.ty(self.ctx),
                kind: MOperandKind::Reg(self.gen_ret_move(ret)),
            })
        } else {
            None
        }
    }

    // pub fn gen_while(&mut self, cond: ir::Value, body: ir::Block) {
    //     // 创建块：条件检查块、循环体块、结束块
    //     let cond_block = MBlock::new(&mut self.mctx, "while_cond");
    //     let body_block = MBlock::new(&mut self.mctx, "while_body");
    //     let end_block = MBlock::new(&mut self.mctx, "while_end");
    
    //     // 将块插入当前函数
    //     let curr_func = self.curr_func.unwrap();
    //     curr_func.push_back(&mut self.mctx, cond_block).unwrap();
    //     curr_func.push_back(&mut self.mctx, body_block).unwrap();
    //     curr_func.push_back(&mut self.mctx, end_block).unwrap();
    
    //     // 当前块跳转到条件检查块
    //     let curr_block = self.curr_block.unwrap();
    //     let j_to_cond = MInst::j(&mut self.mctx, None, cond_block);
    //     curr_block.push_back(&mut self.mctx, j_to_cond).unwrap();
    
    //     // 生成条件检查块
    //     self.curr_block = Some(cond_block);
    //     self.gen_cond_branch(cond, body_block, None, end_block);
    
    //     // 生成循环体块
    //     self.curr_block = Some(body_block);
    //     self.gen_block(body); // 调用方法生成循环体代码
    //     let j_back_to_cond = MInst::j(&mut self.mctx, None, cond_block);
    //     body_block.push_back(&mut self.mctx, j_back_to_cond).unwrap();
    
    //     // 结束块
    //     self.curr_block = Some(end_block);
    // }

    /// Generate a getelementptr instruction and append it to the current block.
    ///
    /// base: The base address.
    /// indices: The indices.
    ///
    /// Return the result operand.
    pub fn gen_gep(&mut self, base: &Value, indices: Vec<Value>, bound_ty: &Ty) -> MOperand {
        let curr_block = self.curr_block.unwrap();

        // Get the base register.
        let base_reg = self.memloc_reg_from_value(&base);

        // Calculate the offset.
        let mut offset = 0;
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
                offset += value * size as i64;
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

    pub fn reg_from_value(&mut self, val: &Value) -> Reg {
        let curr_block = self.curr_block.unwrap();
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
            ir::ValueKind::InstResult { .. } => {
                let mopd = self.lowered[&val];
                match mopd.kind {
                    MOperandKind::Reg(reg) => reg,
                    MOperandKind::Imm(reg, imm) => {
                        let (li, r) = MInst::li(&mut self.mctx, imm as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        let fs1 = self.mctx.new_vreg(RegKind::Float).into();
                        let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvSX, fs1, r);
                        curr_block.push_back(&mut self.mctx, fmv).unwrap();
                        fs1
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
            }
            ir::ValueKind::Param { .. } => {
                let mopd = self.lowered[&val];
                match mopd.kind {
                    MOperandKind::Reg(reg) => reg,
                    MOperandKind::Imm(_, imm) => {
                        let (li, r) = MInst::li(&mut self.mctx, imm as u64);
                        curr_block.push_back(&mut self.mctx, li).unwrap();
                        let fs1 = self.mctx.new_vreg(RegKind::Float).into();
                        let fmv = MInst::fpu_move(&mut self.mctx, FpuMoveOp::FmvSX, fs1, r);
                        curr_block.push_back(&mut self.mctx, fmv).unwrap();
                        fs1
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
            }
            _ => todo!(),
        }
    }

    pub fn reg_or_imm_from_value(&mut self, val: &Value) -> (Option<Reg>, Option<Imm12>) {
        let curr_block = self.curr_block.unwrap();
        match &val.try_deref(self.ctx).unwrap().kind {
            ir::ValueKind::Constant { value } => match value {
                ir::ConstantValue::Int32 { value, .. } => {
                    let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    (Some(r), None)
                }
                ir::ConstantValue::Int8 { value, .. } => {
                    let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    (Some(r), None)
                }
                ir::ConstantValue::Int1 { value, .. } => {
                    let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    (Some(r), None)
                }
                ir::ConstantValue::Float32 { value, .. } => {
                    let (li, r) = MInst::li(&mut self.mctx, *value as u64);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    (Some(r), None)
                }
                ir::ConstantValue::Undef { .. } => {
                    let (li, r) = MInst::li(&mut self.mctx, 0);
                    curr_block.push_back(&mut self.mctx, li).unwrap();
                    (Some(r), None)
                }
                _ => {
                    eprintln!("Unsupported constant: {:?}", value);
                    unreachable!()
                }
            },
            ir::ValueKind::InstResult { .. } => {
                let mopd = self.lowered[&val];
                match mopd.kind {
                    MOperandKind::Reg(reg) => (Some(reg), None),
                    MOperandKind::Imm(_, imm) => (None, Imm12::try_from_i64(imm)),
                    MOperandKind::Undef => (Some(regs::zero().into()), None),
                    MOperandKind::Mem(loc) => {
                        let mop = self.gen_load(val.ty(self.ctx), loc);
                        match mop.kind {
                            MOperandKind::Reg(reg) => (Some(reg), None),
                            _ => todo!(),
                        }
                    }
                }
            }
            ir::ValueKind::Param { .. } => {
                let mopd = self.lowered[&val];
                match mopd.kind {
                    MOperandKind::Reg(reg) => (Some(reg), None),
                    MOperandKind::Imm(_, imm) => (None, Imm12::try_from_i64(imm)),
                    MOperandKind::Undef => (Some(regs::zero().into()), None),
                    MOperandKind::Mem(loc) => {
                        let mop = self.gen_load(val.ty(self.ctx), loc);
                        match mop.kind {
                            MOperandKind::Reg(reg) => (Some(reg), None),
                            _ => todo!(),
                        }
                    }
                }
            }
            _ => todo!(),
        }
    }

    pub fn memloc_reg_from_value(&mut self, val: &Value) -> Reg {
        let curr_block = self.curr_block.unwrap();
        match &val.try_deref(self.ctx).unwrap().kind {
            ir::ValueKind::Constant { value } => match value {
                ir::ConstantValue::GlobalRef { name, .. } => {
                    let (la, rd) = MInst::la(&mut self.mctx, &name);
                    curr_block.push_back(&mut self.mctx, la).unwrap();
                    rd
                }
                _ => {
                    eprintln!("Unsupported constant: {:?}", value);
                    unreachable!()
                }
            },
            ir::ValueKind::InstResult { .. }
            | ir::ValueKind::Param { .. }
            | ir::ValueKind::Array { .. } => {
                let mopd = self.lowered[&val];
                match mopd.kind {
                    MOperandKind::Reg(reg) => reg,
                    MOperandKind::Imm(..) => todo!("Imm"),
                    MOperandKind::Undef => regs::zero().into(),
                    MOperandKind::Mem(loc) => {
                        let loc = match loc {
                            MemLoc::RegOffset { .. } => loc,
                            MemLoc::Slot { offset, .. } => MemLoc::RegOffset {
                                base: regs::sp().into(),
                                offset: offset,
                            },
                            _ => todo!(),
                        };
                        let mop = self.gen_load(val.ty(self.ctx), loc);
                        match mop.kind {
                            MOperandKind::Reg(reg) => reg,
                            _ => todo!(),
                        }
                    }
                }
            }
        }
    }
}
