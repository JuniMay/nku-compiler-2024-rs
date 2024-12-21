use crate::ir::Ty;
use std::collections::{HashMap, HashSet};

use crate::{
    infra::{
        linked_list::{LinkedListContainer, LinkedListNode},
        storage::{Arena, ArenaPtr},
    },
    ir::{Block, ConstantValue, Context, Func, Inst, InstKind, TyData, Usable, Value, ValueKind},
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
        println!("remove_unreachable_blocks done");
        // println!("{}", self.ir.to_string());
        // self.remove_redundant_jumps(); // 删除冗余跳转，简化CFG
        self.mem2reg_special(); // mem2reg的特殊情况
        println!("mem2reg_special done");
        self.mem2reg(); // mem2reg

        println!("{}", self.ir);

        // 添加死代码消除
        self.dead_code_elimination();
        println!("dead_code_elimination done");
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
                .filter(|inst| match inst.kind(&self.ir) {
                    InstKind::Alloca { ty } => match ty.try_deref(&self.ir).unwrap() {
                        TyData::Ptr { .. } => false,
                        TyData::Array { .. } => false,
                        _ => true,
                    },
                    _ => false,
                })
                .collect();
            let mut inst_to_remove = HashSet::new();

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

        // let block_cnt = funcs
        //     .iter()
        //     .map(|func| func.iter(&self.ir).into_iter().count())
        //     .sum::<usize>();
        // 主要是当前块算法复杂度太高，容易爆TLE
        // if block_cnt > 500 {
        //     println!("TOO LARGE: {}", block_cnt);
        //     return;
        // }

        let df = self.get_df();
        let dom_tree = self.get_dom_tree();

        for func in funcs {
            let head = func.head(&self.ir);

            if let None = head {
                continue;
            }

            // 首先确保所有的alloca都与数组无关（需要更复杂的mem2reg才能实现）
            let head = head.unwrap();
            if head.iter(&self.ir).any(|inst| match inst.kind(&self.ir) {
                InstKind::Alloca { ty } => match ty.try_deref(&self.ir).unwrap() {
                    TyData::Ptr { .. } => true,
                    TyData::Array { .. } => true,
                    _ => false,
                },
                _ => false,
            }) {
                println!("exist array");
                return;
            }
            let alloca: Vec<_> = head
                .iter(&self.ir)
                .filter(|inst| {
                    let result = match inst.kind(&self.ir) {
                        InstKind::Alloca { ty } => match ty.try_deref(&self.ir).unwrap() {
                            TyData::Ptr { .. } => false,
                            TyData::Array { .. } => false,
                            _ => true,
                        },
                        _ => false,
                    };
                    result && self.is_alloca_promotable(inst, &dom_tree)
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

            if alloca.is_empty() {
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
                                let val = incoming_vals.get(&alloca.result(&self.ir).unwrap());
                                if val.is_none() {
                                    continue;
                                }
                                let val = val.unwrap();
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
            let phis = phi_alloca_map.keys().cloned().collect::<Vec<_>>();
            for phi in phis.clone() {
                let incoming: Vec<_> = phi.incoming_iter(&self.ir).into_iter().collect();
                if incoming.len() == 1 {
                    let val = incoming[0].1;
                    let result = phi.result(&self.ir).unwrap();
                    let users = result.users(&self.ir).into_iter().collect::<Vec<_>>();
                    for user in users {
                        let inst = user.inst();
                        inst.set_operand(&mut self.ir, val, user.idx());
                    }
                    phi.remove_without_dealloc(&mut self.ir);
                    inst_to_remove.insert(phi);
                    phi_alloca_map.remove(&phi);
                } else if incoming.len() == 0 {
                    phi.remove_without_dealloc(&mut self.ir);
                    inst_to_remove.insert(phi);
                    phi_alloca_map.remove(&phi);
                }
            }

            let phis = phi_alloca_map.keys().cloned().collect::<Vec<_>>();
            for phi in phis {
                let incoming: Vec<_> = phi.incoming_iter(&self.ir).into_iter().collect();
                let ty = phi.result(&self.ir).unwrap().ty(&self.ir);
                let zero = Value::zero(&mut self.ir, ty);
                if incoming.len()
                    < phi
                        .container(&self.ir)
                        .unwrap()
                        .predecessors(&self.ir)
                        .len()
                {
                    let preds: Vec<_> = phi
                        .container(&self.ir)
                        .unwrap()
                        .predecessors(&self.ir)
                        .clone()
                        .into_iter()
                        .collect();
                    for pred in preds {
                        let mut flag = false;
                        for (block, _) in &incoming {
                            if *block == pred.from() {
                                flag = true;
                                break;
                            }
                        }
                        if !flag {
                            phi.insert_incoming(&mut self.ir, pred.from(), zero);
                        }
                    }
                }
            }

            // 清除alloca, load, store
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

        println!("cfg:");
        for (key, succs) in cfg.clone() {
            print!("{} -> ", key.name(&self.ir));
            for succ in succs {
                print!("{}, ", succ.name(&self.ir));
            }
            println!("");
        }

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
    /// 构建 def_map 和 use_map
    pub fn build_def_and_use_maps(
        &self,
        func: &Func,
    ) -> (HashMap<Value, Inst>, HashMap<Value, HashSet<Inst>>) {
        let mut def_map = HashMap::new();
        let mut use_map = HashMap::new();

        for block in func.iter(&self.ir) {
            for inst in block.iter(&self.ir) {
                // 如果指令有结果，记录定义点
                if let Some(result) = inst.result(&self.ir) {
                    if !result.ty(self.ir()).is_void(self.ir()) {
                        def_map.insert(result, inst);
                    }
                }
                // 遍历指令的所有操作数，记录使用点
                for operand in inst.operand_iter(&self.ir) {
                    use_map
                        .entry(operand)
                        .or_insert_with(HashSet::new)
                        .insert(inst);
                }
            }
        }

        (def_map, use_map)
    }

    pub fn init_work_list(
        &self,
        def_map: &HashMap<Value, Inst>,
        use_map: &HashMap<Value, HashSet<Inst>>,
        func: &Func, // 添加对函数的引用
    ) -> Vec<Value> {
        let return_value = func
            .head(&self.ir)
            .and_then(|head| head.tail(&self.ir)) // 获取最后一个块的最后一条指令
            .and_then(|inst| inst.result(&self.ir)); // 获取返回值

        def_map
            .keys()
            .filter(|val| {
                // 如果变量是函数返回值，则排除
                if Some(**val) == return_value {
                    return false;
                }
                // 检查变量是否没有使用点
                use_map.get(*val).map_or(true, |uses| uses.is_empty())
            })
            .cloned()
            .collect()
    }

    pub fn dead_code_elimination(&mut self) {
        let funcs: Vec<_> = self.ir.funcs().collect();
        for func in funcs {
            println!("Initial CFG:");
            for block in func.iter(&self.ir) {
                println!(
                    "Block {}: successors = {:?}, predecessors = {:?}",
                    block.name(&self.ir),
                    block
                        .successors(&self.ir)
                        .iter()
                        .map(|edge| edge.to().name(&self.ir))
                        .collect::<Vec<_>>(),
                    block
                        .predecessors(&self.ir)
                        .iter()
                        .map(|edge| edge.from().name(&self.ir))
                        .collect::<Vec<_>>()
                );
            }

            println!("Processing function: {}", func.name(&self.ir));

            let (mut def_map, mut use_map) = self.build_def_and_use_maps(&func);
            println!("Initial def_map: {:?}", def_map);
            println!("Initial use_map: {:?}", use_map);

            let mut work_list = self.init_work_list(&def_map, &use_map, &func);
            println!("Initial work_list: {:?}", work_list.iter().map(|val| val.display(self.ir(), true).to_string()).collect::<Vec<_>>());

            // 处理 work_list 中的死变量
            while let Some(dead_val) = work_list.pop() {
                println!("Processing dead_val: {:?}", dead_val);

                if let Some(&def_inst) = def_map.get(&dead_val) {
                    if !def_inst.has_side_effects(&self.ir) {
                        println!("Removing def_inst: {:?}", def_inst);

                        // 删除指令前清理其操作数
                        for operand in def_inst.operand_iter(&self.ir) {
                            if let Some(users) = use_map.get_mut(&operand) {
                                // 如果操作数变成死变量，加入 work_list
                                if users.is_empty() {
                                    println!(
                                        "Operand became dead, processing recursively: {:?}",
                                        operand
                                    );
                                    if let Some(operand_def_inst) = def_map.get(&operand) {//iakke
                                        // get
                                        work_list.push(operand_def_inst.result(&self.ir).unwrap());
                                    }
                                }

                                users.remove(&def_inst);

                                println!("operand removing successful!");
                            }
                        }

                        // 删除指令
                        let block = def_inst.container(&self.ir).unwrap();
                        block.remove_inst(&mut self.ir, def_inst);
                        def_map.remove(&dead_val);
                    }
                }
            }

            println!("Finished processing function: {}", func.name(&self.ir));

            println!(
                "CFG after dead_code_elimination for function {}:",
                func.name(&self.ir)
            );
            for block in func.iter(&self.ir) {
                println!(
                    "Block {}: successors = {:?}, predecessors = {:?}",
                    block.name(&self.ir),
                    block
                        .successors(&self.ir)
                        .iter()
                        .map(|edge| edge.to().name(&self.ir))
                        .collect::<Vec<_>>(),
                    block
                        .predecessors(&self.ir)
                        .iter()
                        .map(|edge| edge.from().name(&self.ir))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_maps() {
        // 创建一个模拟的 Context
        let ptr_size = std::mem::size_of::<usize>() as u32; // 动态获取指针大小
        let mut ctx = Context::new(ptr_size); // 创建 Context

        let ret_ty = Ty::i32(&mut ctx);
        let mut func = Func::new(&mut ctx, "complex_func".to_string(), ret_ty, Some(true));

        // 创建基本块
        let entry_block = {
            let b = Block::new(&mut ctx);
            func.set_head(&mut ctx, Some(b));
            b
        };

        let cond_block = Block::new(&mut ctx);
        let true_block = Block::new(&mut ctx);
        let false_block = Block::new(&mut ctx);
        let merge_block = Block::new(&mut ctx);

        func.set_tail(&mut ctx, Some(merge_block));

        // 在 entry_block 中创建变量和条件分支
        let cond_val = {
            let ty = Ty::i1(&mut ctx);
            let cond = Inst::alloca(&mut ctx, ty);
            entry_block.append_inst(&mut ctx, cond);
            cond.result(&ctx).unwrap()
        };

        // 设置块之间的控制流关系
        let br_inst = Inst::br(&mut ctx, cond_block);
        entry_block.append_inst(&mut ctx, br_inst);

        let cond_inst = Inst::cond_br(&mut ctx, cond_val, true_block, false_block);
        cond_block.append_inst(&mut ctx, cond_inst);

        entry_block.add_successor(&mut ctx, cond_block, br_inst, false);
        cond_block.add_successor(&mut ctx, true_block, cond_inst, true);
        cond_block.add_successor(&mut ctx, false_block, cond_inst, false);

        true_block.add_successor(&mut ctx, merge_block, br_inst, false);
        false_block.add_successor(&mut ctx, merge_block, br_inst, false);

        // 在 true_block 中添加未使用的变量和返回指令
        let unused_val = {
            let ty = Ty::i32(&mut ctx);
            let inst = Inst::alloca(&mut ctx, ty);
            true_block.append_inst(&mut ctx, inst);
            inst.result(&ctx).unwrap()
        };
        {
            // 在 true_block 中添加返回指令
            let ret_inst = Inst::ret(&mut ctx, None);
            true_block.append_inst(&mut ctx, ret_inst);
        }

        {
            // 在 false_block 中添加返回指令
            let ret_inst = Inst::ret(&mut ctx, None);
            false_block.append_inst(&mut ctx, ret_inst);
        }

        {
            // 在 merge_block 中添加返回指令
            let ret_inst = Inst::ret(&mut ctx, None);
            merge_block.append_inst(&mut ctx, ret_inst);
        }

        // 打印 CFG 调试信息
        println!("Initial CFG:");
        let cfg = func.display(&ctx);
        println!("{}", cfg);

        // 执行优化
        let mut optimizer = Optimize::new(ctx, 1);
        optimizer.opt1();

        // 验证优化效果
        let final_cfg = func.display(optimizer.ir());
        println!("Optimized CFG:");
        println!("{}", final_cfg);

        // 检查 true_block 和 unused_val 是否被删除
        let (def_map, _) = optimizer.build_def_and_use_maps(&func);
        assert!(!def_map.contains_key(&unused_val));
    }
}
