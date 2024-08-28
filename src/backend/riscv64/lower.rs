use super::imm::Imm12;
use super::inst::{AluOpRRI, AluOpRRR, BrOp, LoadOp, RvInst, StoreOp};
use super::regs::{self, CALLEE_SAVED_REGS, CALLER_SAVED_REGS};
use crate::backend::inst::MInst;
use crate::backend::lower::{LowerContext, LowerSpec, MValue, MValueKind, MemLoc};
use crate::backend::regs::Reg;
use crate::backend::{MBlock, MFunc, PReg, RegKind};
use crate::infra::linked_list::{LinkedListContainer, LinkedListNode};
use crate::ir;

pub struct RvLowerSpec;

impl LowerSpec for RvLowerSpec {
    type I = RvInst;

    fn stack_align() -> u32 { 16 }

    fn frame_pointer() -> PReg { regs::fp() }

    fn stack_pointer() -> PReg { regs::sp() }

    fn pointer_size() -> usize { 64 }

    fn allocatable_regs() -> Vec<PReg> {
        vec![
            regs::t1(),
            regs::t2(),
            regs::t3(),
            regs::t4(),
            regs::t5(),
            regs::t6(),
            regs::a0(),
            regs::a1(),
            regs::a2(),
            regs::a3(),
            regs::a4(),
            regs::a5(),
            regs::a6(),
            regs::a7(),
            regs::s0(),
            regs::s1(),
            regs::s2(),
            regs::s3(),
            regs::s4(),
            regs::s5(),
            regs::s6(),
            regs::s7(),
            regs::s8(),
            regs::s9(),
            regs::s10(),
            regs::s11(),
            regs::ra(),
        ]
    }

    fn non_allocatable_regs() -> Vec<PReg> {
        vec![regs::zero(), regs::t0(), regs::sp(), regs::gp(), regs::tp()]
    }

    fn callee_saved_regs() -> Vec<PReg> { CALLEE_SAVED_REGS.to_vec() }

    fn caller_saved_regs() -> Vec<PReg> { CALLER_SAVED_REGS.to_vec() }

    fn return_reg(ctx: &ir::Context, ty: ir::Ty) -> PReg {
        // TODO
        regs::a0()
    }

    fn total_stack_size(lower: &mut LowerContext<Self>, mfunc: MFunc<Self::I>) -> u64 {
        let saved_regs = mfunc.saved_regs(&lower.mctx);
        let raw = mfunc.storage_stack_size(&lower.mctx)
            + mfunc.outgoing_stack_size(&lower.mctx)
            + saved_regs.len() as u64 * 8;
        (raw + 15) & !15
    }

    fn gen_move(lower: &mut LowerContext<Self>, dst: Reg, src: MValue) {
        // TODO
        todo!()
    }

    fn gen_sp_adjust(lower: &mut LowerContext<Self>, offset: i64) {
        if let Some(imm) = Imm12::try_from_i64(offset) {
            let inst = RvInst::build_alu_rri(
                &mut lower.mctx,
                AluOpRRI::Addi,
                regs::sp().into(),
                regs::sp().into(),
                imm,
            );
            lower
                .curr_block
                .unwrap()
                .push_back(&mut lower.mctx, inst)
                .unwrap();
        } else {
            let (li, t) = RvInst::li(&mut lower.mctx, offset as u64);
            lower
                .curr_block
                .unwrap()
                .push_back(&mut lower.mctx, li)
                .unwrap();
            let inst = RvInst::build_alu_rrr(
                &mut lower.mctx,
                AluOpRRR::Add,
                regs::sp().into(),
                regs::sp().into(),
                t,
            );
            lower
                .curr_block
                .unwrap()
                .push_back(&mut lower.mctx, inst)
                .unwrap();
        }
    }

    fn gen_cond_br(lower: &mut LowerContext<Self>, cond: MValue, dst: MBlock<Self::I>) {
        let curr_block = lower.curr_block.unwrap();
        match cond.kind() {
            MValueKind::Reg(reg) => {
                let inst = RvInst::br(&mut lower.mctx, BrOp::Bne, reg, regs::zero().into(), dst);
                curr_block.push_back(&mut lower.mctx, inst).unwrap();
            }
            MValueKind::Imm(_, imm) => {
                if imm as u64 & 1 == 0 {
                    // do nothing, never jump
                } else {
                    let inst = RvInst::j(&mut lower.mctx, dst);
                    curr_block.push_back(&mut lower.mctx, inst).unwrap();
                }
            }
            MValueKind::Mem(_) => unreachable!(),
            MValueKind::Undef => {
                // do nothing, because it's undef
            }
        }
    }

    fn gen_int_binary(
        lower: &mut LowerContext<Self>,
        op: ir::IntBinaryOp,
        lhs: MValue,
        rhs: MValue,
        dst_ty: ir::Ty,
    ) -> MValue {
        todo!()
    }

    fn gen_load(lower: &mut LowerContext<Self>, ty: ir::Ty, mem_loc: MemLoc) -> MValue { todo!() }

    fn gen_store(lower: &mut LowerContext<Self>, val: MValue, mem_loc: MemLoc) { todo!() }

    fn gen_ret(lower: &mut LowerContext<Self>) {
        let curr_block = lower.curr_block.unwrap();
        let inst = RvInst::ret(&mut lower.mctx);
        curr_block.push_back(&mut lower.mctx, inst).unwrap();
    }

    fn gen_func_prologue(lower: &mut LowerContext<Self>, func: MFunc<Self::I>) {
        let mut clobber_ra = false;

        for block in func.iter(lower.mctx()) {
            for inst in block.iter(lower.mctx()) {
                if inst
                    .clobbers(lower.mctx(), &lower.config)
                    .contains(&Reg::from(regs::ra()))
                {
                    clobber_ra = true;
                }
            }
        }

        if clobber_ra {
            func.add_saved_reg(&mut lower.mctx, regs::ra());
        }

        // addi sp, sp, -frame_size or sub
        //
        // store reg#0, total_stack_size - 8(sp)
        // store reg#1, total_stack_size - 16(sp)
        // ...
        //
        // store reg#n, total_stack_size - 8n(sp)
        //
        // addi fp, sp, total_stack_size # if we omit fp, this is not needed

        // this is the first block, we need to insert in reverse order
        let curr_block = lower.curr_block.unwrap();

        if !lower.config.omit_frame_pointer {
            func.add_saved_reg(&mut lower.mctx, regs::fp());
        }

        let saved_regs = func.saved_regs(&lower.mctx);

        let total_stack_size = Self::total_stack_size(lower, func) as i64;

        let mut inst_buf = Vec::new();

        let sp = regs::sp();
        let fp = regs::fp();

        if let Some(imm) = Imm12::try_from_i64(-total_stack_size) {
            let addi =
                RvInst::build_alu_rri(&mut lower.mctx, AluOpRRI::Addi, sp.into(), sp.into(), imm);
            inst_buf.push(addi);
        } else {
            let t0 = regs::t0();
            let li = RvInst::build_li(&mut lower.mctx, t0.into(), total_stack_size as u64);
            let add = RvInst::build_alu_rrr(
                &mut lower.mctx,
                AluOpRRR::Sub,
                sp.into(),
                sp.into(),
                t0.into(),
            );
            inst_buf.push(li);
            inst_buf.push(add);
        }

        let mut curr_offset = total_stack_size - 8;
        for reg in saved_regs {
            // TODO: make this modular
            if let RegKind::General = reg.kind() {
                if let Some(imm) = Imm12::try_from_i64(curr_offset) {
                    let store = RvInst::store(
                        &mut lower.mctx,
                        StoreOp::Sd,
                        reg.into(),
                        MemLoc::RegOffset {
                            base: sp.into(),
                            offset: imm.as_i16() as i64,
                        },
                    );
                    inst_buf.push(store);
                } else {
                    let t0 = regs::t0();
                    let li = RvInst::build_li(&mut lower.mctx, t0.into(), curr_offset as u64);
                    let add = RvInst::build_alu_rrr(
                        &mut lower.mctx,
                        AluOpRRR::Add,
                        t0.into(),
                        sp.into(),
                        t0.into(),
                    );
                    let store = RvInst::store(
                        &mut lower.mctx,
                        StoreOp::Sd,
                        reg.into(),
                        MemLoc::RegOffset {
                            base: t0.into(),
                            offset: 0,
                        },
                    );
                    inst_buf.push(li);
                    inst_buf.push(add);
                    inst_buf.push(store);
                }
            } else {
                unimplemented!()
            }
            curr_offset -= 8;
        }

        if !lower.config.omit_frame_pointer {
            if let Some(imm) = Imm12::try_from_i64(total_stack_size) {
                let addi = RvInst::build_alu_rri(
                    &mut lower.mctx,
                    AluOpRRI::Addi,
                    fp.into(),
                    sp.into(),
                    imm,
                );
                inst_buf.push(addi);
            } else {
                let t0 = regs::t0();
                let li = RvInst::build_li(&mut lower.mctx, t0.into(), total_stack_size as u64);
                let add = RvInst::build_alu_rrr(
                    &mut lower.mctx,
                    AluOpRRR::Add,
                    fp.into(),
                    sp.into(),
                    t0.into(),
                );
                inst_buf.push(li);
                inst_buf.push(add);
            }
        }

        for inst in inst_buf.into_iter().rev() {
            curr_block.push_front(&mut lower.mctx, inst).unwrap();
        }
    }

    fn gen_func_epilogue(lower: &mut LowerContext<Self>, func: MFunc<Self::I>) {
        let curr_block = lower.curr_block.unwrap();
        let saved_regs = func.saved_regs(&lower.mctx);

        let total_stack_size = Self::total_stack_size(lower, func) as i64;

        // load reg#0, total_stack_size - 8(sp)
        // load reg#1, total_stack_size - 16(sp)
        // ...
        // load reg#n, total_stack_size - 8n(sp)
        //
        // addi sp, sp, total_stack_size
        // ret

        let sp = regs::sp();

        let mut curr_offset = total_stack_size - 8;

        for reg in saved_regs {
            if let RegKind::General = reg.kind() {
                if let Some(imm) = Imm12::try_from_i64(curr_offset) {
                    let load = RvInst::build_load(
                        &mut lower.mctx,
                        LoadOp::Ld,
                        reg.into(),
                        MemLoc::RegOffset {
                            base: sp.into(),
                            offset: imm.as_i16() as i64,
                        },
                    );
                    curr_block.push_back(&mut lower.mctx, load).unwrap();
                } else {
                    let t0 = regs::t0();
                    let li = RvInst::build_li(&mut lower.mctx, t0.into(), curr_offset as u64);
                    let add = RvInst::build_alu_rrr(
                        &mut lower.mctx,
                        AluOpRRR::Add,
                        t0.into(),
                        sp.into(),
                        t0.into(),
                    );
                    let load = RvInst::build_load(
                        &mut lower.mctx,
                        LoadOp::Ld,
                        reg.into(),
                        MemLoc::RegOffset {
                            base: t0.into(),
                            offset: 0,
                        },
                    );
                    curr_block.push_back(&mut lower.mctx, li).unwrap();
                    curr_block.push_back(&mut lower.mctx, add).unwrap();
                    curr_block.push_back(&mut lower.mctx, load).unwrap();
                }
            } else {
                unimplemented!()
            }
            curr_offset -= 8;
        }

        if let Some(imm) = Imm12::try_from_i64(total_stack_size) {
            let addi =
                RvInst::build_alu_rri(&mut lower.mctx, AluOpRRI::Addi, sp.into(), sp.into(), imm);
            curr_block.push_back(&mut lower.mctx, addi).unwrap();
        } else {
            let t0 = regs::t0();
            let li = RvInst::build_li(&mut lower.mctx, t0.into(), total_stack_size as u64);
            let add = RvInst::build_alu_rrr(
                &mut lower.mctx,
                AluOpRRR::Add,
                sp.into(),
                sp.into(),
                t0.into(),
            );
            curr_block.push_back(&mut lower.mctx, li).unwrap();
            curr_block.push_back(&mut lower.mctx, add).unwrap();
        }

        Self::gen_ret(lower);
    }

    fn gen_spill_load(lower: &mut LowerContext<Self>, reg: Reg, slot: MemLoc, inst: Self::I) {
        let load = match reg.kind() {
            RegKind::General => RvInst::build_load(&mut lower.mctx, LoadOp::Ld, reg, slot),
        };
        inst.insert_before(&mut lower.mctx, load).unwrap();
    }

    fn gen_spill_store(lower: &mut LowerContext<Self>, reg: Reg, slot: MemLoc, inst: Self::I) {
        let store = match reg.kind() {
            RegKind::General => RvInst::store(&mut lower.mctx, StoreOp::Sd, reg, slot),
        };
        inst.insert_after(&mut lower.mctx, store).unwrap();
    }

    fn display_reg(reg: Reg) -> String { regs::display(reg) }
}
