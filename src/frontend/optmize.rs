use std::collections::HashMap;

use crate::{
    infra::linked_list::{LinkedListContainer, LinkedListNode},
    ir::{Block, ConstantValue, Context, Func, ValueKind},
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

    pub fn remove_unreachable_blocks(&mut self) -> bool {
        let funcs: Vec<_> = self.ir.funcs().collect();
        let mut changed = false;
        println!("Remove unreachable blocks");

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
                        // println!("Block: {:?}", data.name(&self.ir));
                        for succ in data.successors(&self.ir) {
                            in_degree.entry(succ.to()).and_modify(|e| *e -= 1);
                        }
                        func.remove_block(&mut self.ir, data);
                        changed = true;
                        continue;
                    }
                } else {
                    // println!("Block not in in_degree: {:?}", data.name(&self.ir));
                    func.remove_block(&mut self.ir, data);
                    changed = true;
                    continue;
                }

                // 由于条件判断导致的不可达
                let inst = data.tail(&self.ir);
                if let Some(inst) = inst {
                    match inst.kind(&self.ir) {
                        crate::ir::InstKind::CondBr => {
                            let cond = inst.operand(&self.ir, 0);
                            let true_br = inst.successor(&self.ir, 0);
                            let false_br = inst.successor(&self.ir, 1);
                            if let ValueKind::Constant {
                                value: ConstantValue::Int1 { ty: _, value },
                            } = cond.kind(&self.ir)
                            {
                                if *value {
                                    in_degree.entry(false_br).and_modify(|e| *e -= 1);
                                    data.remove_successor(&mut self.ir, false_br, inst, false);
                                } else {
                                    in_degree.entry(true_br).and_modify(|e| *e -= 1);
                                    data.remove_successor(&mut self.ir, true_br, inst, true);
                                }
                            }
                            continue;
                        }
                        _ => {}
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
                println!("Block: {:?}", block.name(&self.ir));
                for succ in block.successors(&self.ir) {
                    let succ_block = succ.to();
                    *in_degree.entry(succ_block).or_insert(0) += 1;
                    stack.push(Some(succ_block));
                }
            }
        }

        in_degree
    }

    pub fn mem2reg(&mut self) {
        todo!("implement mem2reg");
    }

    // pub fn get_edges(&self, func: Func) -> HashMap<Block, Vec<BlockEdge>> {
    //     let mut edges = HashMap::new();

    //     let mut visited = HashSet::new();
    //     let mut stack = vec![];

    //     let head = func.head(&self.ir);
    //     match head {
    //         Some(head) => {
    //             while let Some(block) = stack.pop() {
    //                 visited.insert(block);
    //                 if visited.insert(block) {
    //                     for succ in head.successors(&self.ir) {
    //                         let to = succ.to();
    //                         edges.entry(head).or_insert(vec![]); // 确保head的in_degree存在
    //                         for succ in to.successors(&self.ir) {
    //                             let succ_block = succ.to();
    //                             edges.entry(succ_block).or_insert(vec![]).push(succ.clone());
    //                             stack.push(succ_block);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         None => {}
    //     }

    //     edges
    // }
}
