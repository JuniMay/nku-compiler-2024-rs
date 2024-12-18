use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    net::Incoming,
    result,
};

use crate::{
    backend::block,
    infra::{
        linked_list::{LinkedListContainer, LinkedListNode},
        storage::Arena,
    },
    ir::{Block, ConstantValue, Context, Func, Inst, InstKind, Usable, ValueKind},
};
pub struct Optimize {
    ir: Context,
    opt_level: u32,
}

impl Optimize {
    pub fn new(ir: Context, opt_level: u32) -> Self {
        Self { ir, opt_level }
    }

    pub fn optmize(&mut self) {
        match self.opt_level {
            0 => self.opt0(),
            1 => self.opt1(),
            2 => self.opt2(),
            3 => self.opt3(),
            _ => self.opt3(),
        }
    }

    pub fn opt0(&mut self) {
        // do nothing
    }

    pub fn opt1(&mut self) {
        while self.remove_unreachable_blocks() {} // 删除不可达块，不动点算法
        println!("==============================");
        println!("{}", self.ir.to_string());
        // self.remove_redundant_jumps(); // 删除冗余跳转
        self.mem2reg_special(); // mem2reg的特殊情况
        println!("mem2reg_special done");
        self.mem2reg(); // mem2reg
    }

    pub fn opt2(&mut self) {
        self.opt1();
        todo!();
    }

    pub fn opt3(&mut self) {
        self.opt2();
        todo!();
    }

    pub fn ir(&self) -> &Context {
        &self.ir
    }

    /// remove unreachable blocks
    /// Cause: Blocks not in CFG; Branchs that can ensure direction
    pub fn remove_unreachable_blocks(&mut self) -> bool {
        let funcs: Vec<_> = self.ir.funcs().collect();
        let mut changed = false;

        for func in funcs {
            // 不在head闭包中的块都不参与计算
            let head = func.head(&self.ir);
            if let None = head {
                continue;
            }
            let head = head.unwrap();
            let mut in_degree = self.get_in_degree(func);

            let mut block = Some(head);
            while block != None {
                let data = block.unwrap();
                block = data.next(&self.ir);
                // 最基本的不可达情况
                if let Some(cnt) = in_degree.get(&data) {
                    if data != head && *cnt == 0 {
                        for succ in data.successors(&self.ir) {
                            in_degree.entry(succ.to()).and_modify(|e| *e -= 1);
                        }
                        data.remove_self_from_neighbors(&mut self.ir);
                        data.clear_insts(&mut self.ir);
                        func.remove_block(&mut self.ir, data);
                        changed = true;
                        continue;
                    }
                } else {
                    data.remove_self_from_neighbors(&mut self.ir);
                    data.clear_insts(&mut self.ir);
                    func.remove_block(&mut self.ir, data);
                    changed = true;
                    continue;
                }

                // 由于条件判断导致的不可达
                let inst = data.tail(&self.ir);
                if let Some(inst) = inst {
                    match inst.kind(&self.ir) {
                        InstKind::CondBr => {
                            let cond = inst.operand(&self.ir, 0);
                            let true_br = inst.successor(&self.ir, 0);
                            let false_br = inst.successor(&self.ir, 1);
                            if let ValueKind::Constant {
                                value: ConstantValue::Int1 { ty: _, value },
                            } = cond.kind(&self.ir)
                            {
                                if *value {
                                    in_degree.entry(false_br).and_modify(|e| *e -= 1);
                                    data.remove_edge(&mut self.ir, false_br, inst, false);
                                } else {
                                    in_degree.entry(true_br).and_modify(|e| *e -= 1);
                                    data.remove_edge(&mut self.ir, true_br, inst, true);
                                }
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        changed
    }

    /// remove redundant jumps
    /// Cause: Only one edge is between those two blocks.
    pub fn remove_redundant_jumps(&mut self) -> bool {
        let funcs: Vec<_> = self.ir.funcs().collect();

        let block_cnt = funcs
            .iter()
            .map(|func| func.iter(&self.ir).into_iter().count())
            .sum::<usize>();
        // 主要是当前块算法复杂度太高，容易爆TLE
        if block_cnt > 500 {
            println!("TOO LARGE");
            return false;
        }
        let mut changed = false;

        for func in funcs {
            let blocks = func.iter(&self.ir).collect::<Vec<_>>();
            for block in blocks {
                if let Some(inst) = block.tail(&self.ir) {
                    if let InstKind::Br = inst.kind(&self.ir) {
                        let target_block = inst.successor(&self.ir, 0);
                        if block.successors(&self.ir).into_iter().count() == 1
                            && target_block.predecessors(&self.ir).into_iter().count() == 1
                        {
                            block.merge_with(&mut self.ir, target_block);
                            changed = true;
                        }
                    }
                }
            }
        }

        changed
    }

    pub fn dfs_preorder<F>(&mut self, from: Block, mut visit: F)
    where
        F: FnMut(&mut Context, Block),
    {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![from];

        while let Some(block) = stack.pop() {
            if visited.insert(block) {
                // Assuming `get_successors` returns an iterator of successor block indices
                visit(&mut self.ir, from);
                for succ in from.successors(&self.ir) {
                    let v = succ.to();
                    stack.push(v);
                }
            }
        }
    }

    pub fn get_in_degree(&self, func: Func) -> HashMap<Block, usize> {
        let mut in_degree = HashMap::new();

        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![];

        let head = func.head(&self.ir);
        stack.push(head);
        while let Some(block) = stack.pop() {
            if let None = block {
                continue;
            }
            let block = block.unwrap();
            if visited.insert(block) {
                in_degree.entry(block).or_insert(0);
                for succ in block.successors(&self.ir) {
                    let succ_block = succ.to();
                    *in_degree.entry(succ_block).or_insert(0) += 1;
                    stack.push(Some(succ_block));
                }
            }
        }

        in_degree
    }

    pub fn mem2reg_special(&mut self) {
        let funcs: Vec<_> = self.ir.funcs().collect();
        for func in funcs {
            let head = func.head(&self.ir);

            if let None = head {
                continue;
            }

            let head = head.unwrap();
            let alloca: Vec<_> = head
                .iter(&self.ir)
                .take_while(|inst| matches!(inst.kind(&self.ir), InstKind::Alloca { .. }))
                .collect();
            let mut inst_to_remove = HashSet::new();

            for alloca in alloca.clone() {
                let result = alloca.result(&self.ir).unwrap();
                println!("alloca: {}", alloca.display(&self.ir));
                for user in result.users(&self.ir) {
                    let inst = user.inst();
                    println!("inst: {}", inst.display(&self.ir));
                }
                println!("=====================");
            }

            let _alloca: Vec<_> = alloca
                .into_iter()
                .filter(|alloca| {
                    let mut users: Vec<_> = alloca
                        .result(&self.ir)
                        .unwrap()
                        .users(&self.ir)
                        .into_iter()
                        .collect();
                    users.sort_by_key(|user| user.inst().index());
                    let mut user_block = None;
                    // 清除没有load的alloca与相关指令
                    if users
                        .iter()
                        .all(|user| matches!(user.inst().kind(&self.ir), InstKind::Store))
                    {
                        for user in users {
                            user.inst().remove_without_dealloc(&mut self.ir);
                            inst_to_remove.insert(user.inst());
                        }
                        alloca.remove_without_dealloc(&mut self.ir);
                        inst_to_remove.insert(*alloca);
                        false
                    } else if users.iter().all(|user| {
                        //清除所有use与def在同一个块内的alloca与相关指令
                        if user_block.is_none() {
                            user_block = Some(user.inst().container(&self.ir).unwrap());
                            true
                        } else if user_block != Some(user.inst().container(&self.ir).unwrap()) {
                            false
                        } else {
                            true
                        }
                    }) {
                        let mut val = alloca.result(&self.ir).unwrap();
                        for user in users {
                            let inst = user.inst();
                            match inst.kind(&self.ir) {
                                InstKind::Load => {
                                    let dst = user.inst().result(&self.ir).unwrap();
                                    let dst_users =
                                        dst.users(&self.ir).into_iter().collect::<Vec<_>>();
                                    for user in dst_users {
                                        let inst = user.inst();
                                        inst.set_operand(&mut self.ir, val, user.idx());
                                    }
                                    inst.remove_without_dealloc(&mut self.ir);
                                    inst_to_remove.insert(inst);
                                }
                                InstKind::Store => {
                                    val = inst.operand(&self.ir, 0);
                                    inst.remove_without_dealloc(&mut self.ir);
                                    inst_to_remove.insert(inst);
                                }
                                _ => {}
                            }
                        }
                        alloca.remove_without_dealloc(&mut self.ir);
                        inst_to_remove.insert(*alloca);
                        false
                    } else {
                        true
                    }
                })
                .collect();

            for inst in inst_to_remove {
                self.ir.try_dealloc(inst);
            }
        }
    }

    pub fn mem2reg(&mut self) {
        let funcs: Vec<_> = self.ir.funcs().collect();

        let block_cnt = funcs
            .iter()
            .map(|func| func.iter(&self.ir).into_iter().count())
            .sum::<usize>();
        // 主要是当前块算法复杂度太高，容易爆TLE
        if block_cnt > 500 {
            println!("TOO LARGE");
            return;
        }

        let df = self.get_df();
        let dom_tree = self.get_dom_tree();

        println!("df:");
        for (block, succs) in &df {
            print!("{} -> ", block.name(&self.ir));
            for succ in succs {
                print!("    {} ", succ.name(&self.ir));
            }
            println!();
        }

        println!("dom_tree:");
        for (block, succs) in &dom_tree {
            print!("{} <- ", block.name(&self.ir));
            for succ in succs {
                print!("    {} ", succ.name(&self.ir));
            }
            println!();
        }

        for func in funcs {
            let head = func.head(&self.ir);

            if let None = head {
                continue;
            }

            let head = head.unwrap();
            let alloca: Vec<_> = head
                .iter(&self.ir)
                .filter(|inst| {
                    matches!(inst.kind(&self.ir), InstKind::Alloca { .. })
                        && self.is_alloca_promotable(inst, &dom_tree)
                })
                .collect();
            let alloca_store_bbs: HashMap<_, _> = alloca
                .iter()
                .map(|inst| {
                    let result = inst.result(&self.ir).unwrap();
                    let users = result.users(&self.ir).into_iter().collect::<Vec<_>>();
                    let mut stores = HashSet::new();
                    for user in users {
                        match user.inst().kind(&self.ir) {
                            InstKind::Store => {
                                stores.insert(user.inst().container(&self.ir).unwrap());
                            }
                            _ => {}
                        }
                    }
                    (inst, stores)
                })
                .collect();
            let alloca_res: HashSet<_> = alloca
                .iter()
                .map(|inst| inst.result(&self.ir).unwrap())
                .collect();
            let mut inst_to_remove = HashSet::new();

            let block_cnt = func.iter(&self.ir).into_iter().count();

            if alloca.is_empty() {
                println!("exit, block_cnt: {}", block_cnt);
                for inst in inst_to_remove {
                    self.ir.try_dealloc(inst);
                }
                continue;
            }

            // 插入phi节点
            let mut phi_alloca_map = HashMap::new();
            let mut block_phi_map = HashMap::new();

            for alloca in alloca.iter().rev() {
                let mut visited = HashSet::new();
                let mut work: Vec<_> = alloca_store_bbs
                    .get(&alloca)
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();
                let empty_hashset = HashSet::new();
                while let Some(block) = work.pop() {
                    for succ in df.get(&block).unwrap_or(&empty_hashset) {
                        if visited.insert(*succ) {
                            let ty = if let InstKind::Alloca { ty, .. } = alloca.kind(&self.ir) {
                                *ty
                            } else {
                                unreachable!();
                            };
                            let phi = Inst::phi(&mut self.ir, ty);
                            succ.push_front(&mut self.ir, phi).unwrap();
                            phi_alloca_map.insert(phi, alloca);
                            block_phi_map
                                .entry(*succ)
                                .or_insert_with(Vec::new)
                                .push(phi);
                            work.push(*succ);
                        }
                    }
                }
            }

            // 重命名
            let incoming_vals = HashMap::new();
            let mut visited = HashSet::new();
            let mut work = vec![(head, incoming_vals.clone())];
            while let Some((block, mut incoming_vals)) = work.pop() {
                if visited.insert(block) {
                    println!("block: {}", block.name(&self.ir));
                    let insts = block.iter(&self.ir).collect::<Vec<_>>();
                    for inst in insts {
                        let kind = inst.kind(&self.ir);
                        match kind {
                            InstKind::Alloca { .. } => {
                                if alloca.contains(&inst) {
                                    inst.remove_without_dealloc(&mut self.ir);
                                    inst_to_remove.insert(inst);
                                }
                            }
                            InstKind::Load => {
                                let dst = inst.operand(&self.ir, 0);
                                if alloca_res.contains(&dst) {
                                    let val = inst.result(&self.ir).unwrap();
                                    let _users =
                                        val.users(&self.ir).into_iter().collect::<Vec<_>>();

                                    let val = incoming_vals.get(&dst).unwrap();
                                    for _user in _users {
                                        let _inst = _user.inst();
                                        _inst.set_operand(&mut self.ir, *val, _user.idx());
                                    }
                                    inst.remove_without_dealloc(&mut self.ir);
                                    inst_to_remove.insert(inst);
                                }
                            }
                            InstKind::Store => {
                                let dst = inst.operand(&self.ir, 1);
                                if alloca_res.contains(&dst) {
                                    let val = inst.operand(&self.ir, 0);
                                    incoming_vals.insert(dst, val);
                                    inst.remove_without_dealloc(&mut self.ir);
                                    inst_to_remove.insert(inst);
                                }
                            }
                            InstKind::Phi => {
                                if phi_alloca_map.contains_key(&inst) {
                                    let alloca = phi_alloca_map.get(&inst).unwrap();
                                    let val = inst.result(&self.ir).unwrap();
                                    incoming_vals.insert(alloca.result(&self.ir).unwrap(), val);
                                }
                            }
                            _ => {}
                        }
                    }

                    let successors: Vec<_> = block.successors(&self.ir).into_iter().collect();
                    let mut phi_incomings = vec![];
                    let empty_vec = Vec::new();
                    for succ in &successors {
                        work.push((succ.to(), incoming_vals.clone()));
                        for phi in block_phi_map.get(&succ.to()).unwrap_or(&empty_vec) {
                            if phi_alloca_map.contains_key(&phi) {
                                let alloca = phi_alloca_map.get(&phi).unwrap();
                                let val = incoming_vals
                                    .get(&alloca.result(&self.ir).unwrap())
                                    .unwrap();
                                // let val = match incoming_vals.get(&alloca.result(&self.ir).unwrap())
                                // {
                                //     Some(val) => {
                                //         if *val == phi.result(&self.ir).unwrap() {
                                //             continue;
                                //         }
                                //         val
                                //     }
                                //     None => continue,
                                // };
                                phi_incomings.push((phi, block, val));
                            }
                        }
                    }
                    for (phi, block, val) in phi_incomings {
                        phi.insert_incoming(&mut self.ir, block, *val);
                    }
                }
            }

            // 清除冗余phi节点
            // for phi in block_phi_map.values().flatten() {
            //     let incoming: Vec<_> = phi.incoming_iter(&self.ir).into_iter().collect();
            //     if incoming.len() == 1 {
            //         let val = incoming[0].1;
            //         let result = phi.result(&self.ir).unwrap();
            //         let users = result.users(&self.ir).into_iter().collect::<Vec<_>>();
            //         for user in users {
            //             let inst = user.inst();
            //             inst.set_operand(&mut self.ir, val, user.idx());
            //         }
            //         phi.remove_without_dealloc(&mut self.ir);
            //         inst_to_remove.insert(*phi);
            //     } else if incoming.len() == 0 {
            //         phi.remove_without_dealloc(&mut self.ir);
            //         inst_to_remove.insert(*phi);
            //     }
            // }

            for inst in inst_to_remove {
                self.ir.try_dealloc(inst);
            }
        }
    }

    pub fn is_alloca_promotable(
        &self,
        alloca_inst: &Inst,
        dom_tree: &HashMap<Block, HashSet<Block>>,
    ) -> bool {
        let result = alloca_inst.result(&self.ir).unwrap();
        let users = result.users(&self.ir).into_iter().collect::<Vec<_>>();
        let head = alloca_inst.container(&self.ir).unwrap();

        // let mut users_by_block = HashMap::new();
        // for user in users.clone() {
        //     let user_inst = user.inst();
        //     let user_block = user_inst.container(&self.ir).unwrap();
        //     users_by_block
        //         .entry(user_block)
        //         .or_insert_with(Vec::new)
        //         .push(user_inst);
        // }
        // for (.., users) in users_by_block {
        //     let mut loads = Vec::new();
        //     let mut stores = Vec::new();
        //     for user in users {
        //         match user.kind(&self.ir) {
        //             InstKind::Load => loads.push(user),
        //             InstKind::Store => stores.push(user),
        //             InstKind::Ret => continue,
        //             _ => return false,
        //         }
        //     }
        //     if loads.len() > 1 || stores.len() > 1 {
        //         return false;
        //     }
        //     if loads.len() == 1 && stores.len() == 1 {
        //         let store = stores[0];
        //         let load = loads[0];
        //         if store.next(&self.ir) == Some(load) {
        //             continue;
        //         }
        //         return false;
        //     }
        // }
        // 确保操作在dom_tree中
        for user in users {
            let user_inst = user.inst();
            let user_block = user_inst.container(&self.ir).unwrap();
            match user_inst.kind(&self.ir) {
                InstKind::Load | InstKind::Store => match dom_tree.get(&user_block) {
                    Some(succs) => {
                        if succs.contains(&head) || head == user_block {
                            continue;
                        }
                        return false;
                    }
                    None => {
                        if head == user_inst.container(&self.ir).unwrap() {
                            continue;
                        }
                        return false;
                    }
                },
                _ => return false,
            }
        }

        true
    }

    pub fn get_idom_tree(&mut self) -> HashMap<Block, HashSet<Block>> {
        let mut idom_tree = HashMap::new();
        let dom_tree = self.get_dom_tree();

        for (idom, parents) in dom_tree {
            for parent in parents {
                if idom
                    .predecessors(&self.ir)
                    .iter()
                    .any(|succ| succ.from() == parent)
                {
                    idom_tree
                        .entry(idom)
                        .or_insert_with(HashSet::new)
                        .insert(parent);
                }
            }
        }

        idom_tree
    }

    pub fn get_idom_tree_tar(&mut self) -> HashMap<Block, HashSet<Block>> {
        // TODO: 如果需要进一步优化，考虑使用lengauertarjan算法，不过需要提前实现缩点图的结构，比较复杂
        let funcs: Vec<_> = self.ir.funcs().collect();
        let mut idom_tree = HashMap::new();

        for func in funcs {}

        idom_tree
    }

    pub fn get_dom_tree(&mut self) -> HashMap<Block, HashSet<Block>> {
        let mut dom_tree = HashMap::new();
        let funcs: Vec<_> = self.ir.funcs().collect();

        for func in funcs {
            let head = func.head(&self.ir);
            if head.is_none() {
                continue;
            }
            let head = head.unwrap();

            let blocks: Vec<_> = func.iter(&self.ir).collect();
            for block in &blocks {
                dom_tree.insert(*block, blocks.clone().into_iter().collect::<HashSet<_>>());
            }
            dom_tree.insert(head, vec![head].into_iter().collect());

            let mut changed = true;
            while changed {
                changed = false;
                for block in &blocks {
                    if *block == head {
                        continue;
                    }
                    let preds: Vec<_> = block.predecessors(&self.ir).iter().collect();
                    if preds.is_empty() {
                        continue;
                    }
                    let mut new_dom = dom_tree[&preds[0].from()].clone();
                    for pred in &preds[1..] {
                        new_dom = new_dom
                            .intersection(&dom_tree[&pred.from()])
                            .cloned()
                            .collect();
                    }
                    new_dom.insert(*block);
                    if dom_tree[block] != new_dom {
                        dom_tree.insert(*block, new_dom);
                        changed = true;
                    }
                }
            }
        }

        dom_tree
    }

    pub fn get_df(&mut self) -> HashMap<Block, HashSet<Block>> {
        let mut df = HashMap::new();
        let dom_tree: HashMap<Block, HashSet<Block>> = self.get_dom_tree();
        let cfg = self.get_cfg();

        let emptyset = HashSet::new();
        for (block, succs) in &cfg {
            for succ in succs {
                for v in dom_tree.get(block).unwrap_or_else(|| &emptyset) {
                    if !dom_tree.get(succ).unwrap_or_else(|| &emptyset).contains(v) {
                        df.entry(*v).or_insert_with(HashSet::new).insert(*succ);
                    }
                }
            }
        }

        df
    }

    pub fn get_cfg(&mut self) -> HashMap<Block, Vec<Block>> {
        let mut cfg = HashMap::new();

        let funcs: Vec<_> = self.ir.funcs().collect();
        for func in funcs {
            let head = func.head(&self.ir);
            if head.is_none() {
                continue;
            }
            let head = head.unwrap();

            let mut visited = HashSet::new();
            let mut stack = vec![head];
            while let Some(block) = stack.pop() {
                if visited.insert(block) {
                    let successors: Vec<_> = block.successors(&self.ir).into_iter().collect();
                    for succ in &successors {
                        stack.push(succ.to());
                    }
                    cfg.insert(block, successors.into_iter().map(|s| s.to()).collect());
                }
            }
        }

        cfg
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn df() {}
}
