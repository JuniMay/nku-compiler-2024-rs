use std::collections::HashSet;
use std::fmt;

use super::context::Context;
use super::def_use::{Usable, User};
use super::func::Func;
use super::inst::Inst;
use super::InstKind;
use crate::infra::linked_list::{LinkedListContainer, LinkedListNode};
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr, Idx};
use std::hash::Hash;

pub struct BlockData {
    _self_ptr: Block,

    /// Users of this block.
    users: HashSet<User<Block>>,

    /// For linear traversal of the entire IR structure
    next: Option<Block>,
    prev: Option<Block>,

    /// Successors of this block.
    successors: HashSet<BlockEdge>,
    /// Predecessors of this block.
    predecessors: HashSet<BlockEdge>,

    /// The first instructions in the block.
    head: Option<Inst>,
    /// The last instruction in the block.
    tail: Option<Inst>,

    /// The function that contains this block.
    container: Option<Func>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Block(GenericPtr<BlockData>);

pub struct DisplayBlock<'ctx> {
    ctx: &'ctx Context,
    block: Block,
}

impl Block {
    pub fn new(ctx: &mut Context) -> Self {
        ctx.alloc_with(|self_ptr| BlockData {
            _self_ptr: self_ptr,
            users: HashSet::new(),
            next: None,
            prev: None,
            successors: HashSet::new(),
            predecessors: HashSet::new(),
            head: None,
            tail: None,
            container: None,
        })
    }

    /// Get the name of the block.
    pub fn name(self, _ctx: &Context) -> String {
        // We use the arena index directly as the block number. This is not a good way
        // to number blocks in a real compiler, but only for debugging purposes.
        format!("%bb_{}", self.0.index())
    }

    pub fn display(self, ctx: &Context) -> DisplayBlock {
        DisplayBlock { ctx, block: self }
    }

    pub fn remove(self, ctx: &mut Context) {
        let container = self.container(ctx).unwrap();
        container.remove_block(ctx, self);
    }

    /// Remove inst in the block.
    pub fn remove_inst(self, ctx: &mut Context, inst: Inst) {
        let mut head = self.head(ctx);
        let mut tail = self.tail(ctx);

        if head == Some(inst) {
            head = inst.next(ctx);
            self.set_head(ctx, head);
        }

        if tail == Some(inst) {
            tail = inst.prev(ctx);
            self.set_tail(ctx, tail);
        }

        let prev = inst.prev(ctx);
        let next = inst.next(ctx);

        if let Some(prev) = prev {
            prev.set_next(ctx, next);
        }
        if let Some(next) = next {
            next.set_prev(ctx, prev);
        }

        ctx.try_dealloc(inst);
    }

    /// Remove all insts in the block.
    pub fn clear_insts(self, ctx: &mut Context) {
        let mut head = self.head(ctx);
        while let Some(inst) = head {
            head = inst.next(ctx);
            inst.remove_all_use(ctx);
            ctx.try_dealloc(inst);
        }
        self.set_head(ctx, None);
        self.set_tail(ctx, None);
    }

    pub fn remove_inst_without_dealloc(self, ctx: &mut Context, inst: Inst) {
        let mut head = self.head(ctx);
        let mut tail = self.tail(ctx);

        if head == Some(inst) {
            head = inst.next(ctx);
            self.set_head(ctx, head);
        }

        if tail == Some(inst) {
            tail = inst.prev(ctx);
            self.set_tail(ctx, tail);
        }

        let prev = inst.prev(ctx);
        let next = inst.next(ctx);

        if let Some(prev) = prev {
            prev.set_next(ctx, next);
        }
        if let Some(next) = next {
            next.set_prev(ctx, prev);
        }
    }

    pub fn replace_inst(self, ctx: &mut Context, inst: Inst, new_inst: Inst) {
        inst.insert_after(ctx, new_inst).unwrap();
        self.remove_inst(ctx, inst);
    }

    pub fn append_inst(self, ctx: &mut Context, inst: Inst) -> Inst {
        let tail = self.tail(ctx);

        if let Some(tail_inst) = tail {
            // 如果有尾指令，将新指令插入到尾部之后
            tail_inst.set_next(ctx, Some(inst));
            inst.set_prev(ctx, Some(tail_inst));
        } else {
            // 如果块为空，将新指令设置为头和尾
            self.set_head(ctx, Some(inst));
        }

        self.set_tail(ctx, Some(inst));
        inst.set_container(ctx, Some(self)); // 将块设置为指令的容器
        inst
    }

    pub fn is_empty(&self, ctx: &Context) -> bool {
        self.iter(ctx).next().is_none()
    }
    
}

impl Block {
    /// 提供链表结构外的修改能力，构造CFG
    pub fn add_successor(self, ctx: &mut Context, successor: Block, inst: Inst, true_br: bool) {
        let edge = match inst.kind(ctx) {
            InstKind::Br => BlockEdge(successor, inst, false),
            InstKind::CondBr => BlockEdge(successor, inst, true_br),
            _ => {
                panic!("unavailble br");
            }
        };
        self.deref_mut(ctx).successors.insert(edge);
    }

    pub fn add_edge(self, ctx: &mut Context, successor: Block, inst: Inst, true_br: bool) {
        self.add_successor(ctx, successor, inst, true_br);
        successor.add_predecessor(ctx, self, inst, true_br);
    }

    pub fn remove_edge(self, ctx: &mut Context, successor: Block, inst: Inst, true_br: bool) {
        successor.remove_predessor(ctx, self, inst, true_br);
        self.remove_successor(ctx, successor, inst, true_br);
    }

    pub fn merge_with(self, ctx: &mut Context, successor: Block) {
        let successors: Vec<_> = successor
            .successors(ctx)
            .iter()
            .map(|edge| edge.clone())
            .collect();
        for edge in successors {
            let to = edge.to();
            let inst = edge.inst();
            let true_br = edge.is_true_branch();
            self.add_edge(ctx, to, inst, true_br);
        }
        successor.remove_self_from_neighbors(ctx);

        if let Some(tail) = self.tail(ctx) {
            tail.remove(ctx);
        }

        self.append(ctx, successor);

        successor.remove(ctx);
    }

    pub fn add_predecessor(self, ctx: &mut Context, predecessor: Block, inst: Inst, true_br: bool) {
        let edge = match inst.kind(ctx) {
            InstKind::Br => BlockEdge(predecessor, inst, false),
            InstKind::CondBr => BlockEdge(predecessor, inst, true_br),
            _ => {
                panic!("unavailble br");
            }
        };
        self.deref_mut(ctx).predecessors.insert(edge);
    }

    pub fn remove_successor(self, ctx: &mut Context, successor: Block, inst: Inst, true_br: bool) {
        let edge = match inst.kind(ctx) {
            InstKind::Br => BlockEdge(successor, inst, false),
            InstKind::CondBr => {
                let new_succ = if true_br {
                    inst.successor(ctx, 1)
                } else {
                    inst.successor(ctx, 0)
                };

                // 删除之前建立的两条边cond_br
                self.deref_mut(ctx)
                    .successors
                    .remove(&BlockEdge(successor, inst, true_br));
                self.deref_mut(ctx)
                    .successors
                    .remove(&BlockEdge(new_succ, inst, !true_br));
                successor.remove_predessor(ctx, self, inst, true_br);
                successor.remove_predessor(ctx, new_succ, inst, !true_br);

                let new_inst = Inst::br(ctx, new_succ);
                inst.insert_after(ctx, new_inst).unwrap(); // 先加结点后删结点，避免处理边界条件
                inst.remove(ctx);

                // 重新建立一条边br
                self.add_edge(ctx, new_succ, new_inst, false);
                return;
            }
            _ => {
                panic!("unavailble br");
            }
        };
        self.deref_mut(ctx).successors.remove(&edge);
    }

    pub fn remove_predessor(
        self,
        ctx: &mut Context,
        predecessor: Block,
        inst: Inst,
        true_br: bool,
    ) {
        let edge = match inst.kind(ctx) {
            InstKind::Br => BlockEdge(predecessor, inst, false),
            InstKind::CondBr => BlockEdge(predecessor, inst, true_br),
            _ => {
                println!("lty");
                panic!("unavailble br");
            }
        };
        self.deref_mut(ctx).predecessors.remove(&edge);
    }

    pub fn remove_self_from_neighbors(self, ctx: &mut Context) {
        // 从前驱和后继中删除所有关系
        let successors: Vec<_> = self
            .successors(ctx)
            .iter()
            .map(|edge| edge.clone())
            .collect();
        for succ in successors {
            succ.to()
                .remove_predessor(ctx, self, succ.inst(), succ.is_true_branch());
        }

        let predecessors: Vec<_> = self
            .predecessors(ctx)
            .iter()
            .map(|edge| edge.clone())
            .collect();
        for pred in predecessors {
            pred.from()
                .remove_successor(ctx, self, pred.inst(), pred.is_true_branch());
        }
    }

    pub fn clear_successors(self, ctx: &mut Context) {
        self.deref_mut(ctx).successors.clear();
    }

    pub fn clear_predecessors(self, ctx: &mut Context) {
        self.deref_mut(ctx).predecessors.clear();
    }

    pub fn copy_successors(self, ctx: &mut Context, successors: HashSet<BlockEdge>) {
        self.deref_mut(ctx).successors = successors;
    }

    pub fn copy_predecessors(self, ctx: &mut Context, predecessors: HashSet<BlockEdge>) {
        self.deref_mut(ctx).predecessors = predecessors;
    }

    pub fn successors(self, ctx: &Context) -> &HashSet<BlockEdge> {
        &self.deref(ctx).successors
    }

    pub fn predecessors(self, ctx: &Context) -> &HashSet<BlockEdge> {
        &self.deref(ctx).predecessors
    }
}

impl fmt::Display for DisplayBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "bb_{}:", self.block.0.index())?;

        for inst in self.block.iter(self.ctx) {
            write!(f, "\n\t{}", inst.display(self.ctx))?;
        }

        Ok(())
    }
}

impl ArenaPtr for Block {
    type Arena = Context;
    type Data = BlockData;
}

impl Arena<Block> for Context {
    fn alloc_with<F>(&mut self, f: F) -> Block
    where
        F: FnOnce(Block) -> BlockData,
    {
        Block(self.blocks.alloc_with(|ptr| f(Block(ptr))))
    }

    fn try_dealloc(&mut self, ptr: Block) -> Option<BlockData> {
        self.blocks.try_dealloc(ptr.0)
    }

    fn try_deref(&self, ptr: Block) -> Option<&BlockData> {
        self.blocks.try_deref(ptr.0)
    }

    fn try_deref_mut(&mut self, ptr: Block) -> Option<&mut BlockData> {
        self.blocks.try_deref_mut(ptr.0)
    }
}

impl LinkedListContainer<Inst> for Block {
    type Ctx = Context;

    fn head(self, ctx: &Self::Ctx) -> Option<Inst> {
        self.try_deref(ctx).expect("invalid pointer").head
    }

    fn tail(self, ctx: &Self::Ctx) -> Option<Inst> {
        self.try_deref(ctx).expect("invalid pointer").tail
    }

    fn set_head(self, ctx: &mut Self::Ctx, head: Option<Inst>) {
        self.try_deref_mut(ctx).expect("invalid pointer").head = head;
    }

    fn set_tail(self, ctx: &mut Self::Ctx, tail: Option<Inst>) {
        self.try_deref_mut(ctx).expect("invalid pointer").tail = tail;
    }
}

impl LinkedListNode for Block {
    type Container = Func;
    type Ctx = Context;

    fn next(self, ctx: &Self::Ctx) -> Option<Self> {
        self.try_deref(ctx).expect("invalid pointer").next
    }

    fn prev(self, ctx: &Self::Ctx) -> Option<Self> {
        self.try_deref(ctx).expect("invalid pointer").prev
    }

    fn container(self, ctx: &Self::Ctx) -> Option<Self::Container> {
        self.try_deref(ctx).expect("invalid pointer").container
    }

    fn set_next(self, ctx: &mut Self::Ctx, next: Option<Self>) {
        self.try_deref_mut(ctx).expect("invalid pointer").next = next;
    }

    fn set_prev(self, ctx: &mut Self::Ctx, prev: Option<Self>) {
        self.try_deref_mut(ctx).expect("invalid pointer").prev = prev;
    }

    fn set_container(self, ctx: &mut Self::Ctx, container: Option<Self::Container>) {
        self.try_deref_mut(ctx).expect("invalid pointer").container = container;
    }
}

impl Usable for Block {
    fn users(self, arena: &Self::Arena) -> impl IntoIterator<Item = User<Self>> {
        self.try_deref(arena).unwrap().users.iter().copied()
    }

    fn insert_user(self, arena: &mut Self::Arena, user: User<Self>) {
        self.try_deref_mut(arena).unwrap().users.insert(user);
    }

    fn remove_user(self, arena: &mut Self::Arena, user: User<Self>) {
        self.try_deref_mut(arena).unwrap().users.remove(&user);
    }
}

/// 表示(to, val)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockEdge(Block, Inst, bool);

impl BlockEdge {
    pub fn to(&self) -> Block {
        self.0.clone()
    }

    pub fn from(&self) -> Block {
        self.0.clone()
    }

    pub fn inst(&self) -> Inst {
        self.1.clone()
    }

    pub fn is_true_branch(&self) -> bool {
        self.2.clone()
    }
}
