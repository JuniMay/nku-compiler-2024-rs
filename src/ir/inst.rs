use std::collections::HashMap;
use std::fmt;

use super::block::Block;
use super::context::Context;
use super::def_use::{Operand, Usable};
use super::ty::Ty;
use super::value::Value;
use crate::infra::linked_list::LinkedListNode;
use crate::infra::storage::{Arena, ArenaPtr, GenericPtr};

pub enum IntCmpCond {
    Eq,
    Ne,
    Slt,
    Sle,
}

impl fmt::Display for IntCmpCond {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntCmpCond::Eq => write!(f, "eq"),
            IntCmpCond::Ne => write!(f, "ne"),
            IntCmpCond::Slt => write!(f, "slt"),
            IntCmpCond::Sle => write!(f, "sle"),
        }
    }
}

pub enum IntBinaryOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    ICmp { cond: IntCmpCond },
}

impl fmt::Display for IntBinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntBinaryOp::Add => write!(f, "add"),
            IntBinaryOp::Sub => write!(f, "sub"),
            IntBinaryOp::Mul => write!(f, "mul"),
            IntBinaryOp::SDiv => write!(f, "sdiv"),
            IntBinaryOp::UDiv => write!(f, "udiv"),
            IntBinaryOp::SRem => write!(f, "srem"),
            IntBinaryOp::URem => write!(f, "urem"),
            IntBinaryOp::Shl => write!(f, "shl"),
            IntBinaryOp::LShr => write!(f, "lshr"),
            IntBinaryOp::AShr => write!(f, "ashr"),
            IntBinaryOp::And => write!(f, "and"),
            IntBinaryOp::Or => write!(f, "or"),
            IntBinaryOp::Xor => write!(f, "xor"),
            IntBinaryOp::ICmp { cond } => write!(f, "icmp {}", cond),
        }
    }
}

pub enum CastOp {
    Zext,
    Sext,
    Trunc,
}

impl fmt::Display for CastOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CastOp::Zext => write!(f, "zext"),
            CastOp::Sext => write!(f, "sext"),
            CastOp::Trunc => write!(f, "trunc"),
        }
    }
}

pub enum InstKind {
    Alloca {
        /// The type of the allocated memory.
        ty: Ty,
    },
    Phi,
    Load,
    Store,
    GetElementPtr {
        bound_ty: Ty,
    },
    Call,
    Br,
    CondBr,
    Ret,
    IntBinary {
        op: IntBinaryOp,
    },
    Cast {
        op: CastOp,
    },
}

enum OperandEntry<T: Usable> {
    Occupied {
        operand: Operand<T>,
    },
    Vacant {
        /// The index of the next vacant slot. Together with `first_vacant`,
        /// this forms a free list.
        next_vacant: Option<usize>,
    },
}

impl<T: Usable> Default for OperandEntry<T> {
    fn default() -> Self { Self::Vacant { next_vacant: None } }
}

/// A list of operands.
///
/// When modifying IR, we need to insert and remove operands from instructions.
/// If we use [`Vec`] and remove the operands with [`Vec::remove`], the indices
/// of other operands might be changed. However, we use the index to distinguish
/// different uses of the same value in the instruction. To avoid this problem,
/// we use a linked list to store the operands. When we remove an operand, the
/// index of other operands will not be changed.
///
/// As a matter of fact, removing operands is a minor operation. Normally, we
/// can just replace the inner value of the operand with a new value.
struct OperandList<T: Usable> {
    operands: Vec<OperandEntry<T>>,
    /// The index of the first vacant slot.
    first_vacant: Option<usize>,
    /// The number of occupied slots.
    len: usize,
}

impl<T: Usable> Default for OperandList<T> {
    fn default() -> Self {
        Self {
            operands: Vec::default(),
            first_vacant: None,
            len: 0,
        }
    }
}

impl<T: Usable> OperandList<T> {
    /// Get the next index for operand creation.
    fn next_idx(&self) -> usize {
        match self.first_vacant {
            Some(vacant) => vacant,
            None => self.operands.len(),
        }
    }

    /// Create and insert an operand into the list and return its index in the
    /// operand list. The index grows monotonically, if there is no deletion
    /// between two insertions.
    fn insert(&mut self, operand: Operand<T>) -> usize {
        match self.first_vacant {
            Some(idx) => {
                // There is a vacant slot, use it and update the first vacant.
                let next_vacant = match self.operands[idx] {
                    // Get the next vacant index first, because we need to update `first_vacant`.
                    OperandEntry::Vacant { next_vacant } => next_vacant,
                    _ => unreachable!(),
                };
                self.operands[idx] = OperandEntry::Occupied { operand };
                self.first_vacant = next_vacant;
                self.len += 1; // Maintain the length.
                idx
            }
            None => {
                // There is no vacant slot, push the operand to the end.
                let idx = self.operands.len();
                self.operands.push(OperandEntry::Occupied { operand });
                self.len += 1; // Maintain the length.
                idx
            }
        }
    }

    /// Remove an operand from the list.
    ///
    /// # Panics
    ///
    /// - Panics if there is no operand at the given index.
    fn remove(&mut self, idx: usize) -> Operand<T> {
        let entry = std::mem::replace(
            &mut self.operands[idx],
            OperandEntry::Vacant {
                // The `first_vacant` will be updated to `idx` later.
                next_vacant: self.first_vacant,
            },
        );
        match entry {
            OperandEntry::Occupied { operand } => {
                self.first_vacant = Some(idx);
                self.len -= 1; // Maintain the length.
                operand
            }
            _ => panic!("invalid operand index"),
        }
    }

    /// Get the operand at the given index.
    ///
    /// # Panics
    ///
    /// - Panics if there is no operand at the given index.
    fn get(&self, idx: usize) -> &Operand<T> {
        match &self.operands[idx] {
            OperandEntry::Occupied { operand } => operand,
            _ => panic!("invalid operand index"),
        }
    }

    /// Get the operand at the given index mutably.
    ///
    /// # Panics
    ///
    /// - Panics if there is no operand at the given index.
    fn get_mut(&mut self, idx: usize) -> &mut Operand<T> {
        match &mut self.operands[idx] {
            OperandEntry::Occupied { operand } => operand,
            _ => panic!("invalid operand index"),
        }
    }

    /// Iterate over the operands.
    ///
    /// There might be vacant slots in the operand list, we need to filter them
    /// out. For instructions like add, sub, etc., the number of operands is
    /// fixed, there is no vacant slot in the operand list. However, for
    /// instructions like phi, there might be vacant slots in the operand
    /// list. But, if the instruction is a phi, one should use `incoming_iter`
    /// instead.
    fn iter(&self) -> impl Iterator<Item = &Operand<T>> + '_ {
        self.operands.iter().filter_map(|entry| match entry {
            OperandEntry::Occupied { operand } => Some(operand),
            _ => None,
        })
    }

    /// Get the length of the operand list.
    fn len(&self) -> usize { self.len }
}

pub struct InstData {
    /// Pointer to the instruction itself.
    self_ptr: Inst,
    /// The kind of the instruction.
    kind: InstKind,
    /// The operands of the instruction.
    ///
    /// This can either be a list of values or a phi node (with `predecessor ->
    /// value` mapping).
    operands: OperandList<Value>,
    /// The successors of the instruction.
    ///
    /// Only branch instructions have successors.
    ///
    /// We also use [`OperandList`] for successors, because blocks implement
    /// [`Usable`] trait. However, when we say `operand`, we usually refer to
    /// the values used by the instruction.
    successors: OperandList<Block>,
    /// The phi node information, with `predecessor block --> operand` idx
    /// mapping.
    phi_node: HashMap<Block, usize>,
    /// The result of the instruction.
    result: Option<Value>,
    // Linked list pointers.
    next: Option<Inst>,
    prev: Option<Inst>,
    container: Option<Block>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Inst(GenericPtr<InstData>);

pub struct DisplayInst<'ctx> {
    ctx: &'ctx Context,
    inst: Inst,
}

impl Inst {
    /// Create a new instruction. The new instruction is not linked to any
    /// block.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context to create the instruction.
    /// - `kind`: The kind of the instruction.
    /// - `ty`: The type of the instruction result.
    fn new(ctx: &mut Context, kind: InstKind, ty: Ty) -> Self {
        let inst = ctx.alloc_with(|self_ptr| InstData {
            self_ptr,
            kind,
            operands: OperandList::default(),
            phi_node: HashMap::default(),
            successors: OperandList::default(),
            result: None,
            next: None,
            prev: None,
            container: None,
        });
        let result = Value::new_inst_result(ctx, inst, ty);
        inst.try_deref_mut(ctx)
            .unwrap_or_else(|| unreachable!())
            .result = Some(result);
        inst
    }

    /// Create a new `alloca` instruction.
    pub fn alloca(ctx: &mut Context, alloca_ty: Ty) -> Self {
        let ptr = Ty::ptr(ctx);
        Self::new(ctx, InstKind::Alloca { ty: alloca_ty }, ptr)
    }

    /// Create a new `phi` instruction.
    pub fn phi(ctx: &mut Context, ty: Ty) -> Self { Self::new(ctx, InstKind::Phi, ty) }

    /// Create a new `load` instruction.
    pub fn load(ctx: &mut Context, ptr: Value, ty: Ty) -> Self {
        let inst = Self::new(ctx, InstKind::Load, ty);
        inst.add_operand(ctx, ptr);
        inst
    }

    /// Create a new `store` instruction.
    pub fn store(ctx: &mut Context, val: Value, ptr: Value) -> Self {
        let void = Ty::void(ctx);
        let inst = Self::new(ctx, InstKind::Store, void);
        inst.add_operand(ctx, val);
        inst.add_operand(ctx, ptr);
        inst
    }

    // TODO: Implement constructors for other instructions.

    fn add_operand(self, ctx: &mut Context, operand: Value) {
        let next_idx = self.try_deref_mut(ctx).unwrap().operands.next_idx();
        let operand = Operand::new(ctx, operand, self, next_idx);
        self.try_deref_mut(ctx)
            .unwrap_or_else(|| unreachable!())
            .operands
            .insert(operand);
    }

    /// Get the operand at the given index.
    ///
    /// # Panics
    ///
    /// - Panics if the index is out of bounds.
    pub fn operand(self, ctx: &Context, idx: usize) -> Value {
        self.try_deref(ctx)
            .expect("invalid pointer")
            .operands
            .get(idx)
            .used()
    }

    /// Iterate over operands
    ///
    /// # Panics
    ///
    /// - Panics if the instruction is a phi node.
    pub fn operand_iter(self, ctx: &Context) -> impl Iterator<Item = Value> + '_ {
        self.try_deref(ctx)
            .unwrap()
            .operands
            .iter()
            .map(|op| op.used())
    }

    /// Get the incoming value from the given block.
    ///
    /// # Panics
    ///
    /// - Panics if the block is not in the incoming list.
    /// - Panics if the block does not exist in the phi node.
    pub fn incoming(self, ctx: &Context, block: Block) -> Value {
        assert!(self.is_phi(ctx), "not a phi node");

        self.try_deref(ctx)
            .expect("invalid pointer")
            .phi_node
            .get(&block)
            .map(|&idx| self.operand(ctx, idx))
            .unwrap()
    }

    /// Iterate over incoming block and values
    ///
    /// # Panics
    ///
    /// - Panics if the instruction is not a phi node.
    pub fn incoming_iter(self, ctx: &Context) -> impl Iterator<Item = (Block, Value)> + '_ {
        assert!(self.is_phi(ctx), "not a phi node");

        self.try_deref(ctx)
            .unwrap()
            .phi_node
            .iter()
            .map(move |(&block, &idx)| (block, self.operand(ctx, idx)))
    }

    /// Add an incoming value to the phi node.
    ///
    /// # Panics
    ///
    /// - Panics if the instruction is not a phi node.
    pub fn insert_incoming(self, ctx: &mut Context, block: Block, value: Value) {
        assert!(self.is_phi(ctx), "not a phi node");

        let next_idx = self.try_deref_mut(ctx).unwrap().operands.next_idx();
        let operand = Operand::new(ctx, value, self, next_idx);

        // Add the operand into the operand list.
        let idx = self
            .try_deref_mut(ctx)
            .expect("invalid pointer")
            .operands
            .insert(operand);

        // Create the mapping from the predecessor block to the operand index.
        self.try_deref_mut(ctx)
            .expect("invalid pointer")
            .phi_node
            .insert(block, idx);
    }

    /// Remove an incoming value from the phi node.
    ///
    /// # Panics
    ///
    /// - Panics if the instruction is not a phi node.
    pub fn remove_incoming(self, ctx: &mut Context, block: Block) {
        assert!(self.is_phi(ctx), "not a phi node");

        // Get the index of the incoming value.
        let idx = self
            .try_deref_mut(ctx)
            .expect("invalid pointer")
            .phi_node
            .remove(&block)
            .expect("block not in the incoming list");

        // Remove it from the operand list.
        let operand = self
            .try_deref_mut(ctx)
            .expect("invalid pointer")
            .operands
            .remove(idx);

        operand.drop(ctx);
    }

    /// Get the successor at the given index.
    ///
    /// # Panics
    ///
    /// - Panics if the index is out of bounds (or the instruction is not a
    ///   branch instruction).
    pub fn successor(self, ctx: &Context, idx: usize) -> Block {
        self.try_deref(ctx)
            .expect("invalid pointer")
            .successors
            .get(idx)
            .used()
    }

    /// Iterate over successors
    pub fn successor_iter(self, ctx: &Context) -> impl Iterator<Item = Block> + '_ {
        self.try_deref(ctx)
            .unwrap()
            .successors
            .iter()
            .map(|op| op.used())
    }

    /// Get a displayable instance of the instruction.
    pub fn display(self, ctx: &Context) -> DisplayInst { DisplayInst { ctx, inst: self } }

    /// Get the result of the instruction.
    pub fn result(self, ctx: &Context) -> Option<Value> {
        self.try_deref(ctx).expect("invalid pointer").result
    }

    /// Get the kind of the instruction.
    pub fn kind(self, ctx: &Context) -> &InstKind {
        &self.try_deref(ctx).expect("invalid pointer").kind
    }

    /// Check if this is a phi node.
    pub fn is_phi(self, ctx: &Context) -> bool {
        matches!(
            self.try_deref(ctx).expect("invalid pointer").kind,
            InstKind::Phi
        )
    }
}

impl ArenaPtr for Inst {
    type Arena = Context;
    type Data = InstData;
}

impl Arena<Inst> for Context {
    fn alloc_with<F>(&mut self, f: F) -> Inst
    where
        F: FnOnce(Inst) -> InstData,
    {
        Inst(self.insts.alloc_with(|ptr| f(Inst(ptr))))
    }

    fn try_dealloc(&mut self, ptr: Inst) -> Option<InstData> { self.insts.try_dealloc(ptr.0) }

    fn try_deref(&self, ptr: Inst) -> Option<&InstData> { self.insts.try_deref(ptr.0) }

    fn try_deref_mut(&mut self, ptr: Inst) -> Option<&mut InstData> {
        self.insts.try_deref_mut(ptr.0)
    }
}

impl LinkedListNode for Inst {
    type Container = Block;
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
