use super::inst::Inst;
use crate::infra::storage::ArenaPtr;

/// Represents an entity that can be used in an instruction.
///
/// An instruction can use values and blocks.
pub trait Usable: ArenaPtr {
    /// Get all the uses of the entity.
    fn users(self, arena: &Self::Arena) -> impl IntoIterator<Item = User<Self>>;

    /// Insert a user into the entity.
    fn insert_user(self, arena: &mut Self::Arena, user: User<Self>);

    /// Remove a user from the entity.
    fn remove_user(self, arena: &mut Self::Arena, user: User<Self>);
}

/// Represents a user of an entity.
///
/// User is identified by the instruction and the index of the entity in the
/// instruction. For example, the instruction `add %1, %1` will generate two
/// users for the same entity `%1`, and the index will be `0` and `1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct User<T: Usable> {
    /// The user of the entity.
    inst: Inst,
    /// The index of the entity in the user.
    idx: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Usable> User<T> {
    /// Create a new user.
    pub fn new(inst: Inst, idx: usize) -> Self {
        Self {
            inst,
            idx,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the instruction that uses the entity.
    pub fn inst(&self) -> Inst { self.inst }

    /// Get the index of the entity in the user.
    pub fn idx(&self) -> usize { self.idx }
}

/// A user-side information of an operand.
///
/// Operand records the used operand, instruction and the index of the operand
/// in the instruction. It is used to track the uses of an operand and remove
/// them when the operand is dropped.
pub struct Operand<T>
where
    T: Usable,
{
    /// The used operand.
    used: T,
    /// The user of the operand.
    inst: Inst,
    /// The index of the operand in the user.
    ///
    /// There might be multiple operands in the user that refer to the same used
    /// operand, we need to distinguish them.
    idx: usize,
}

impl<T> Operand<T>
where
    T: Usable,
{
    /// Create a new operand and insert it into the user.
    pub fn new(arena: &mut T::Arena, used: T, inst: Inst, idx: usize) -> Self {
        let user = User::new(inst, idx);
        used.insert_user(arena, user);
        Self { used, inst, idx }
    }

    /// Get the used operand.
    pub fn used(&self) -> T { self.used }

    /// Get the instruction that uses the operand.
    pub fn inst(&self) -> Inst { self.inst }

    /// Get the index of the operand in the user.
    pub fn idx(&self) -> usize { self.idx }

    /// Drop this operand and remove it from the user.
    pub fn drop(self, arena: &mut T::Arena) {
        let user = User::new(self.inst, self.idx);
        self.used.remove_user(arena, user);
    }
}
