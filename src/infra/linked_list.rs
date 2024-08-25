//! Linked List Data Structure
//!
//! This implementation requires the linked list and its nodes to be associated
//! with a context type `Ctx`. The context type is used to store the actual data
//! and manage the memory allocation. For example, a `Ctx` can be an
//! [`Arena`](super::storage::Arena) that stores the nodes and the linked list
//! container.
//!
//! By using a context for the underlying storage, the lifetime and ownership
//! problems can be easily solved. Also, if the context is an arena-like
//! structure, there can be even better memory locality.
//!
//! Note that though the linked list can be used with an arena, it is not
//! compulsory. The [`Ctx`](LinkedListNode::Ctx) can be any type that is used to
//! access the data.

/// The error type for the linked list operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkedListError<N> {
    /// The node is already in a container and should be
    /// [`unlink`](LinkedListNode::unlink)-ed first.
    NodeAlreadyInContainer(N),
    /// The position node is not in this container, so the
    /// [`split`](LinkedListContainer::split) operation cannot be performed.
    PositionNodeNotInContainer(N),
    /// The current node is not linked and the container is [`None`], we cannot
    /// insert a node before or after it.
    CurrentNodeNotLinked(N),
}

/// A container of linked lists.
///
/// A container can store multiple linked lists, each linked list can be
/// represented by a pair of pointers to the head and tail nodes.
///
/// # Type parameters
///
/// - `Node`: The type of the linked nodes.
pub trait LinkedListContainer<Node>: Copy + Eq
where
    Node: LinkedListNode<Ctx = Self::Ctx, Container = Self>,
{
    /// The context type that is used to access the data.
    ///
    /// The context type here is assumed not have any complex lifetime generics.
    /// Because usually this will just be an arena.
    type Ctx;

    /// Get the head node of the linked list.
    ///
    /// The [`None`]-ness of the head and tail nodes should always be
    /// consistent. If the head node is [`None`], the tail node should also
    /// be [`None`], and vice versa. But if the head node is not [`None`],
    /// the tail node does not have to be the same node as the head node
    /// (apparently).
    fn head(self, ctx: &Self::Ctx) -> Option<Node>;

    /// Get the tail node of the linked list.
    ///
    /// # See also
    ///
    /// - [`LinkedListContainer::head`]
    fn tail(self, ctx: &Self::Ctx) -> Option<Node>;

    /// Set the head node of the linked list.
    ///
    /// This is a low-level method and is not intended to be used directly.
    fn set_head(self, ctx: &mut Self::Ctx, head: Option<Node>);

    /// Set the tail node of the linked list.
    ///
    /// This is a low-level method and is not intended to be used directly.
    fn set_tail(self, ctx: &mut Self::Ctx, tail: Option<Node>);

    /// Push a node to the front of the linked list.
    ///
    /// If the linked list is empty, the head and tail nodes will be set to the
    /// new node.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `node`: The node to be pushed.
    ///
    /// # Returns
    ///
    /// - [`Ok`] if the operation is successful.
    /// - [`LinkedListError::NodeAlreadyInContainer`] if the node is already in
    ///   a container.
    fn push_front(self, ctx: &mut Self::Ctx, node: Node) -> Result<(), LinkedListError<Node>> {
        if node.container(ctx).is_some() {
            return Err(LinkedListError::NodeAlreadyInContainer(node));
        }

        if let Some(head) = self.head(ctx) {
            // `insert_before` will handle setting the container
            // This should not return error, because the head is in the current container,
            // and node is checked.
            head.insert_before(ctx, node)
                .unwrap_or_else(|_| unreachable!());
        } else {
            // the linked list is empty, set head and tail to the new node
            self.set_head(ctx, Some(node));
            self.set_tail(ctx, Some(node));
            node.set_container(ctx, Some(self));
        }

        Ok(())
    }

    /// Push a node to the back of the linked list.
    ///
    /// If the linked list is empty, the head and tail nodes will be set to the
    /// new node.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `node`: The node to be pushed.
    ///
    /// # Returns
    ///
    /// - [`Ok`] if the operation is successful.
    /// - [`LinkedListError::NodeAlreadyInContainer`] if the node is already in
    ///   a container.
    fn push_back(self, ctx: &mut Self::Ctx, node: Node) -> Result<(), LinkedListError<Node>> {
        if node.container(ctx).is_some() {
            return Err(LinkedListError::NodeAlreadyInContainer(node));
        }

        if let Some(tail) = self.tail(ctx) {
            // `insert_after` will handle setting the container
            // This should not return error, because the head is in the current container,
            // and node is checked.
            tail.insert_after(ctx, node)
                .unwrap_or_else(|_| unreachable!());
        } else {
            // the linked list is empty, set head and tail to the new node
            self.set_head(ctx, Some(node));
            self.set_tail(ctx, Some(node));
            node.set_container(ctx, Some(self));
        }

        Ok(())
    }

    /// Move all the nodes in another linked list into this one.
    ///
    /// This operation will move all nodes from the other container to this one.
    /// The `other` container will be empty after this operation.
    ///
    /// The time complexity of this operation is O(N), because all the container
    /// of the nodes in `other` should be updated.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `other`: The other container to be drained.
    fn append(self, ctx: &mut Self::Ctx, other: Self) {
        let mut curr = other.head(ctx);
        while let Some(node) = curr {
            let next = node.next(ctx);
            node.unlink(ctx);
            // The node is unlinked, there should be no error.
            self.push_back(ctx, node).unwrap_or_else(|_| unreachable!());
            curr = next;
        }

        debug_assert!(other.head(ctx).is_none());
        debug_assert!(other.tail(ctx).is_none());
    }

    /// Split this linked list into two at the given position.
    ///
    /// The position node is included in the first linked list.
    ///
    /// The time complexity of this operation is O(N), because all the container
    /// fields of the nodes in the first linked list should be updated.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `other`: The other container to store the second linked list.
    /// - `pos`: The position node to split the linked list.
    ///
    /// # Returns
    ///
    /// - [`Ok`] if the operation is successful.
    /// - [`LinkedListError::PositionNodeNotInContainer`] if the position node
    ///   is not in this container.
    fn split(
        self,
        ctx: &mut Self::Ctx,
        other: Self,
        pos: Node,
    ) -> Result<(), LinkedListError<Node>> {
        if pos.container(ctx) != Some(self) {
            return Err(LinkedListError::PositionNodeNotInContainer(pos));
        }

        let mut curr = self.tail(ctx);
        while let Some(node) = curr {
            if node == pos {
                break;
            }
            let prev = node.prev(ctx);
            node.unlink(ctx);
            // The node is unlinked, there should be no error.
            other
                .push_front(ctx, node)
                .unwrap_or_else(|_| unreachable!());
            curr = prev;
        }

        debug_assert!(self.tail(ctx) == Some(pos));

        Ok(())
    }

    /// Create an iterator for the linked list.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    ///
    /// # Returns
    ///
    /// The created iterator.
    fn iter(self, ctx: &Self::Ctx) -> LinkedListIterator<Node> {
        LinkedListIterator {
            ctx,
            curr_forward: self.head(ctx),
            curr_backward: self.tail(ctx),
        }
    }

    /// Extend the linked list with nodes from an iterator.
    ///
    /// The order of the nodes in the iterator is preserved.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `iter`: The iterator of nodes to be extended.
    ///
    /// # Returns
    ///
    /// - [`Ok`] if the operation is successful.
    /// - [`LinkedListError`] if any error occurs when performing
    ///   [`push_back`](LinkedListContainer::push_back) operation.
    fn extend<I>(self, ctx: &mut Self::Ctx, iter: I) -> Result<(), LinkedListError<Node>>
    where
        I: IntoIterator<Item = Node>,
    {
        for node in iter {
            self.push_back(ctx, node)?;
        }
        Ok(())
    }

    /// Create a [forward](CursorDirection::Forward) cursor for the linked list.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `strategy`: The strategy for the cursor to fetch the next node.
    ///
    /// # Returns
    ///
    /// The created cursor.
    ///
    /// # See also
    ///
    /// - [`CursorStrategy`]
    /// - [`LinkedListCursor`]
    fn cursor(self, ctx: &Node::Ctx, strategy: CursorStrategy) -> LinkedListCursor<Node> {
        LinkedListCursor::new(self, ctx, strategy, CursorDirection::Forward)
    }
}

/// The linked node trait.
///
/// Any linked list node should only belong to one container at a time.
///
/// By using a context for the underlying storage, the linked list can be easily
/// constructed and managed. Also, if the context is an arena-like structure,
/// there can be even better memory locality.
pub trait LinkedListNode: Copy + Eq {
    /// The type of the container that stores the linked list.
    type Container: LinkedListContainer<Self, Ctx = Self::Ctx>;

    /// The context type that is used to access the data.
    type Ctx;

    /// Get the next node in the linked list.
    fn next(self, ctx: &Self::Ctx) -> Option<Self>;

    /// Get the previous node in the linked list.
    fn prev(self, ctx: &Self::Ctx) -> Option<Self>;

    /// Get the container that stores the linked list.
    fn container(self, ctx: &Self::Ctx) -> Option<Self::Container>;

    /// Set the next node in the linked list.
    ///
    /// This is a low-level method and is not intended to be used directly.
    fn set_next(self, ctx: &mut Self::Ctx, next: Option<Self>);

    /// Set the previous node in the linked list.
    ///
    /// This is a low-level method and is not intended to be used directly.
    fn set_prev(self, ctx: &mut Self::Ctx, prev: Option<Self>);

    /// Set the container that stores the linked list.
    ///
    /// This is a low-level method and is not intended to be used directly.
    fn set_container(self, ctx: &mut Self::Ctx, container: Option<Self::Container>);

    /// Insert a node after this node.
    ///
    /// If the current node is the tail of the linked list, the tail will be
    /// updated to the new node.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `node`: The node to be inserted.
    ///
    /// # Returns
    ///
    /// - [`Ok`] if the operation is successful.
    /// - [`LinkedListError::CurrentNodeNotLinked`] if the current node is not
    ///   linked and does not belong to any container.
    /// - [`LinkedListError::NodeAlreadyInContainer`] if the node is already in
    ///   a container.
    fn insert_after(self, ctx: &mut Self::Ctx, node: Self) -> Result<(), LinkedListError<Self>> {
        if self.container(ctx).is_none() {
            return Err(LinkedListError::CurrentNodeNotLinked(self));
        }

        if node.container(ctx).is_some() {
            return Err(LinkedListError::NodeAlreadyInContainer(node));
        }

        if let Some(next) = self.next(ctx) {
            next.set_prev(ctx, Some(node));
            node.set_next(ctx, Some(next));
        }

        node.set_prev(ctx, Some(self));
        self.set_next(ctx, Some(node));

        match self.container(ctx) {
            Some(container) => {
                if container.tail(ctx) == Some(self) {
                    container.set_tail(ctx, Some(node));
                }
            }
            None => unreachable!(),
        }

        node.set_container(ctx, self.container(ctx));

        Ok(())
    }

    /// Insert a node before this node.
    ///
    /// If the current node is the head of the linked list, the head will be
    /// updated to the new node.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `node`: The node to be inserted.
    ///
    /// # Returns
    ///
    /// - [`Ok`] if the operation is successful.
    /// - [`LinkedListError::CurrentNodeNotLinked`] if the current node is not
    ///   linked and does not belong to any container.
    /// - [`LinkedListError::NodeAlreadyInContainer`] if the node is already in
    ///   a container.
    fn insert_before(self, ctx: &mut Self::Ctx, node: Self) -> Result<(), LinkedListError<Self>> {
        if self.container(ctx).is_none() {
            return Err(LinkedListError::CurrentNodeNotLinked(self));
        }

        if node.container(ctx).is_some() {
            return Err(LinkedListError::NodeAlreadyInContainer(node));
        }

        if let Some(prev) = self.prev(ctx) {
            prev.set_next(ctx, Some(node));
            node.set_prev(ctx, Some(prev));
        }

        node.set_next(ctx, Some(self));
        self.set_prev(ctx, Some(node));

        match self.container(ctx) {
            Some(container) => {
                if container.head(ctx) == Some(self) {
                    container.set_head(ctx, Some(node));
                }
            }
            None => unreachable!(),
        }

        node.set_container(ctx, self.container(ctx));

        Ok(())
    }

    /// Unlink this node from the linked list without deallocating it from the
    /// arena.
    ///
    /// If the node is already unlinked, this method should do nothing.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    fn unlink(self, ctx: &mut Self::Ctx) {
        let prev = self.prev(ctx);
        let next = self.next(ctx);

        if let Some(prev) = prev {
            prev.set_next(ctx, next);
        }

        if let Some(next) = next {
            next.set_prev(ctx, prev);
        }

        if let Some(container) = self.container(ctx) {
            if container.head(ctx) == Some(self) {
                container.set_head(ctx, next);
            }

            if container.tail(ctx) == Some(self) {
                container.set_tail(ctx, prev);
            }
        }

        self.set_prev(ctx, None);
        self.set_next(ctx, None);

        self.set_container(ctx, None);
    }

    /// Extend the linked list after this node.
    ///
    /// The order of the nodes in the iterator is preserved.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `iter`: The iterator of nodes to be extended.
    ///
    /// # Returns
    ///
    /// - [`Ok`] if the operation is successful.
    /// - [`LinkedListError`] if any error occurs when performing
    ///   [`insert_after`](LinkedListNode::insert_after) operation.
    fn extend_after<I>(self, ctx: &mut Self::Ctx, iter: I) -> Result<(), LinkedListError<Self>>
    where
        I: IntoIterator<Item = Self>,
    {
        let mut last = self;
        for node in iter {
            last.insert_after(ctx, node)?;
            last = node;
        }
        Ok(())
    }

    /// Extend the linked list before this node.
    ///
    /// The order of the nodes in the iterator is preserved, the last node in
    /// the iterator will be the node before the insertion point.
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `iter`: The iterator of nodes to be extended.
    ///
    /// # Returns
    ///
    /// - [`Ok`] if the operation is successful.
    /// - [`LinkedListError`] if any error occurs when performing
    ///   [`insert_before`](LinkedListNode::insert_before) operation.
    fn extend_before<I>(self, ctx: &mut Self::Ctx, iter: I) -> Result<(), LinkedListError<Self>>
    where
        I: IntoIterator<Item = Self>,
    {
        for node in iter {
            self.insert_before(ctx, node)?;
        }
        Ok(())
    }
}

/// A double-ended iterator for the linked list.
///
/// The iterator is useful for immutable access to the linked list. If
/// modification is needed during the iteration, consider using
/// [`LinkedListCursor`] by calling [`LinkedListContainer::cursor`].
///
/// This is a double-ended iterator, and can be reversed by calling
/// [`rev`](Self::rev).
///
/// # Lifetimes
///
/// - `a`: The lifetime of the arena that stores the nodes.
///
/// # Type parameters
///
/// - `T`: The type of the linked nodes.
pub struct LinkedListIterator<'a, T: LinkedListNode> {
    ctx: &'a T::Ctx,
    curr_forward: Option<T>,
    curr_backward: Option<T>,
}

impl<'a, T: LinkedListNode> Iterator for LinkedListIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.curr_forward;
        self.curr_forward = curr.and_then(|node| node.next(self.ctx));
        curr
    }
}

impl<'a, T: LinkedListNode> DoubleEndedIterator for LinkedListIterator<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let curr = self.curr_backward;
        self.curr_backward = curr.and_then(|node| node.prev(self.ctx));
        curr
    }
}

/// The strategy for the cursor to fetch the next node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CursorStrategy {
    /// The next node is fetched before visiting the current node.
    ///
    /// If the current node is [`unlink`](LinkedListNode::unlink)-ed or
    /// modified, the next node will still be the original one.
    ///
    /// The first node will be fetched when constructing the cursor.
    Pre,
    /// The next node is fetched after visiting the current node.
    ///
    /// If the current node is modified, the next node will be updated
    /// accordingly. If the current node is
    /// [`unlink`](LinkedListNode::unlink)-ed, the next node will be
    /// [`None`] and the iteration will stop.
    ///
    /// The first node will be fetched when calling [`LinkedListCursor::next`].
    Post,
}

/// The direction of the [`LinkedListCursor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CursorDirection {
    /// The cursor is moving forward from [`head`](LinkedListContainer::head) to
    /// [`tail`](LinkedListContainer::tail).
    Forward,
    /// The cursor is moving backward from [`tail`](LinkedListContainer::tail)
    /// to [`head`](LinkedListContainer::head).
    Backward,
}

/// A cursor for the linked list.
///
/// Cursor is useful when modification is needed during the iteration. If only
/// immutable access is needed, consider using [`LinkedListIterator`] by calling
/// [`LinkedListContainer::iter`].
///
/// There is no lifetime parameter for the cursor, because the cursor does not
/// reference to the arena directly.
///
/// # Type parameters
///
/// - `T`: The type of the linked nodes.
pub struct LinkedListCursor<T: LinkedListNode> {
    /// The container of the iterating list.
    container: T::Container,
    /// The current node.
    ///
    /// For [`CursorStrategy::Pre`], this is the next-to-visit node. For
    /// [`CursorStrategy::Post`], this is the current node (or the previously
    /// visited node).
    curr: Option<T>,
    /// The strategy for iterating the linked list.
    strategy: CursorStrategy,
    /// The direction of the cursor.
    direction: CursorDirection,
    /// Indicator for the end of the forward iteration. Update when the visiting
    /// node is [`None`].
    done: bool,
}

impl<T: LinkedListNode> LinkedListCursor<T> {
    fn new(
        container: T::Container,
        ctx: &T::Ctx,
        strategy: CursorStrategy,
        direction: CursorDirection,
    ) -> Self {
        let mut cursor = Self {
            container,
            curr: None,
            strategy,
            direction,
            done: false,
        };

        match cursor.strategy {
            CursorStrategy::Pre => match cursor.direction {
                // fetch the next-to-visit node, the `done` flag will be updated when calling `next`
                CursorDirection::Forward => cursor.curr = cursor.container.head(ctx),
                CursorDirection::Backward => cursor.curr = cursor.container.tail(ctx),
            },
            CursorStrategy::Post => {}
        }

        cursor
    }

    /// Reverse the direction of the cursor.
    ///
    /// This method also clears the [`done`](Self::is_done) flag, and reset the
    /// current node.
    pub fn rev(mut self, ctx: &T::Ctx) -> Self {
        self.direction = match self.direction {
            CursorDirection::Forward => CursorDirection::Backward,
            CursorDirection::Backward => CursorDirection::Forward,
        };

        self.done = false; // reset the done flag

        match self.strategy {
            CursorStrategy::Pre => match self.direction {
                CursorDirection::Forward => self.curr = self.container.head(ctx),
                CursorDirection::Backward => self.curr = self.container.tail(ctx),
            },
            CursorStrategy::Post => self.curr = None,
        }

        self
    }

    /// If the iteration is done.
    pub fn is_done(&self) -> bool { self.done }

    /// Move the cursor and get the node to visit.
    ///
    /// - If the strategy is [`CursorStrategy::Pre`], the next node will be
    ///   fetched before visiting the current node.
    /// - If the strategy is [`CursorStrategy::Post`], the current node will be
    ///   fetched when calling this method.
    ///
    /// By default, [`LinkedListContainer::cursor`] will return a forward
    /// cursor, which can be reversed by calling [`rev`](Self::rev).
    pub fn next(&mut self, ctx: &T::Ctx) -> Option<T> {
        match self.strategy {
            CursorStrategy::Pre => {
                // the current node is already the next-to-visit node, so fetch the next node,
                // and then visit the current node
                let curr = self.curr;
                match curr {
                    // fetch the next node
                    Some(curr) => match self.direction {
                        CursorDirection::Forward => self.curr = curr.next(ctx),
                        CursorDirection::Backward => self.curr = curr.prev(ctx),
                    },
                    // the current node is none, so the iteration is done
                    None => self.done = true,
                }
                curr
            }
            CursorStrategy::Post => {
                // the current node is the previously visited node, so get the next node, and
                // return.
                match self.curr {
                    Some(curr) => match self.direction {
                        CursorDirection::Forward => self.curr = curr.next(ctx),
                        CursorDirection::Backward => self.curr = curr.prev(ctx),
                    },
                    // for none, check if the iteration is done
                    None if self.done => return None,
                    // this is the initial state, so fetch the first node
                    None => match self.direction {
                        CursorDirection::Forward => self.curr = self.container.head(ctx),
                        CursorDirection::Backward => self.curr = self.container.tail(ctx),
                    },
                }
                self.done = self.curr.is_none();
                self.curr
            }
        }
    }

    /// Iterate the linked list with a closure.
    ///
    /// This will consume the cursor and the closure can mutate the nodes. If
    /// mutation is not needed, consider using [`LinkedListIterator`] by
    /// calling [`LinkedListContainer::iter`].
    ///
    /// # Parameters
    ///
    /// - `ctx`: The context that is used to access the data.
    /// - `f`: The closure to be called for each node.
    pub fn for_each<F>(self, ctx: &mut T::Ctx, mut f: F)
    where
        F: FnMut(&mut T::Ctx, T),
    {
        let mut cursor = self;
        while let Some(node) = cursor.next(ctx) {
            f(ctx, node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infra::storage::{Arena, ArenaPtr, GenericArena, GenericPtr};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Node(GenericPtr<NodeData>);

    impl Node {
        fn new(arena: &mut TestArena, val: i32) -> Self {
            arena.alloc(NodeData {
                _val: val,
                prev: None,
                next: None,
                container: None,
            })
        }
    }

    struct NodeData {
        _val: i32,
        prev: Option<Node>,
        next: Option<Node>,
        container: Option<Container>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Container(GenericPtr<ContainerData>);

    struct ContainerData {
        head: Option<Node>,
        tail: Option<Node>,
    }

    impl Container {
        fn new(arena: &mut TestArena) -> Self {
            arena.alloc(ContainerData {
                head: None,
                tail: None,
            })
        }
    }

    #[derive(Default)]
    struct TestArena {
        nodes: GenericArena<NodeData>,
        containers: GenericArena<ContainerData>,
    }

    impl ArenaPtr for Node {
        type Arena = TestArena;
        type Data = NodeData;
    }

    impl ArenaPtr for Container {
        type Arena = TestArena;
        type Data = ContainerData;
    }

    impl Arena<Node> for TestArena {
        fn alloc_with<F>(&mut self, f: F) -> Node
        where
            F: FnOnce(Node) -> <Node as ArenaPtr>::Data,
        {
            Node(self.nodes.alloc_with(|ptr| f(Node(ptr))))
        }

        fn try_dealloc(&mut self, ptr: Node) -> Option<NodeData> { self.nodes.try_dealloc(ptr.0) }

        fn try_deref(&self, ptr: Node) -> Option<&<Node as ArenaPtr>::Data> {
            self.nodes.try_deref(ptr.0)
        }

        fn try_deref_mut(&mut self, ptr: Node) -> Option<&mut <Node as ArenaPtr>::Data> {
            self.nodes.try_deref_mut(ptr.0)
        }
    }

    impl Arena<Container> for TestArena {
        fn alloc_with<F>(&mut self, f: F) -> Container
        where
            F: FnOnce(Container) -> <Container as ArenaPtr>::Data,
        {
            Container(self.containers.alloc_with(|ptr| f(Container(ptr))))
        }

        fn try_dealloc(&mut self, ptr: Container) -> Option<ContainerData> {
            self.containers.try_dealloc(ptr.0)
        }

        fn try_deref(&self, ptr: Container) -> Option<&<Container as ArenaPtr>::Data> {
            self.containers.try_deref(ptr.0)
        }

        fn try_deref_mut(&mut self, ptr: Container) -> Option<&mut <Container as ArenaPtr>::Data> {
            self.containers.try_deref_mut(ptr.0)
        }
    }

    impl LinkedListNode for Node {
        type Container = Container;
        type Ctx = TestArena;

        fn next(self, ctx: &Self::Ctx) -> Option<Self> { ctx.try_deref(self).unwrap().next }

        fn prev(self, ctx: &Self::Ctx) -> Option<Self> { ctx.try_deref(self).unwrap().prev }

        fn container(self, ctx: &Self::Ctx) -> Option<Self::Container> {
            ctx.try_deref(self).unwrap().container
        }

        fn set_next(self, ctx: &mut Self::Ctx, next: Option<Self>) {
            ctx.try_deref_mut(self).unwrap().next = next;
        }

        fn set_prev(self, ctx: &mut Self::Ctx, prev: Option<Self>) {
            ctx.try_deref_mut(self).unwrap().prev = prev;
        }

        fn set_container(self, ctx: &mut Self::Ctx, container: Option<Self::Container>) {
            ctx.try_deref_mut(self).unwrap().container = container;
        }
    }

    impl LinkedListContainer<Node> for Container {
        type Ctx = TestArena;

        fn head(self, ctx: &Self::Ctx) -> Option<Node> { ctx.try_deref(self).unwrap().head }

        fn tail(self, ctx: &Self::Ctx) -> Option<Node> { ctx.try_deref(self).unwrap().tail }

        fn set_head(self, ctx: &mut Self::Ctx, head: Option<Node>) {
            ctx.try_deref_mut(self).unwrap().head = head;
        }

        fn set_tail(self, ctx: &mut Self::Ctx, tail: Option<Node>) {
            ctx.try_deref_mut(self).unwrap().tail = tail;
        }
    }

    #[test]
    fn test_linked_list() {
        let arena = &mut TestArena::default();

        let container = Container::new(arena);
        let node1 = Node::new(arena, 1);
        let node2 = Node::new(arena, 2);
        let node3 = Node::new(arena, 3);

        container.push_front(arena, node1).unwrap();

        assert_eq!(container.head(arena), Some(node1));
        assert_eq!(container.tail(arena), Some(node1));
        assert_eq!(node1.next(arena), None);
        assert_eq!(node1.prev(arena), None);

        container.push_back(arena, node2).unwrap();

        assert_eq!(container.head(arena), Some(node1));
        assert_eq!(container.tail(arena), Some(node2));
        assert_eq!(node1.next(arena), Some(node2));
        assert_eq!(node1.prev(arena), None);
        assert_eq!(node2.prev(arena), Some(node1));
        assert_eq!(node2.next(arena), None);

        container.push_back(arena, node3).unwrap();

        assert_eq!(container.head(arena), Some(node1));
        assert_eq!(container.tail(arena), Some(node3));
        assert_eq!(node1.next(arena), Some(node2));
        assert_eq!(node1.prev(arena), None);
        assert_eq!(node2.prev(arena), Some(node1));
        assert_eq!(node2.next(arena), Some(node3));
        assert_eq!(node3.prev(arena), Some(node2));
        assert_eq!(node3.next(arena), None);

        node1.unlink(arena);

        assert_eq!(container.head(arena), Some(node2));
        assert_eq!(container.tail(arena), Some(node3));
        assert_eq!(node1.next(arena), None);
        assert_eq!(node1.prev(arena), None);
        assert_eq!(node1.container(arena), None);

        node2.unlink(arena);

        assert_eq!(container.head(arena), Some(node3));
        assert_eq!(container.tail(arena), Some(node3));
        assert_eq!(node2.next(arena), None);
        assert_eq!(node2.prev(arena), None);
        assert_eq!(node2.container(arena), None);

        node3.unlink(arena);

        assert_eq!(container.head(arena), None);
        assert_eq!(container.tail(arena), None);
    }

    #[test]
    fn test_linked_list_insert_fail() {
        let arena = &mut TestArena::default();

        let container = Container::new(arena);
        let node1 = Node::new(arena, 1);
        let node2 = Node::new(arena, 2);

        container.push_back(arena, node1).unwrap();
        container.push_back(arena, node2).unwrap();

        assert_eq!(
            node1.insert_after(arena, node2),
            Err(LinkedListError::NodeAlreadyInContainer(node2))
        );
    }

    #[test]
    fn test_linked_list_iter() {
        let arena = &mut TestArena::default();

        let container = Container::new(arena);
        let node1 = Node::new(arena, 1);
        let node2 = Node::new(arena, 2);
        let node3 = Node::new(arena, 3);

        container.push_back(arena, node1).unwrap();
        container.push_back(arena, node2).unwrap();
        container.push_back(arena, node3).unwrap();

        let mut iter = container.iter(arena);
        assert_eq!(iter.next(), Some(node1));
        assert_eq!(iter.next(), Some(node2));
        assert_eq!(iter.next(), Some(node3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        let mut iter = container.iter(arena);
        assert_eq!(iter.next_back(), Some(node3));
        assert_eq!(iter.next_back(), Some(node2));
        assert_eq!(iter.next_back(), Some(node1));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_linked_list_cursor() {
        let arena = &mut TestArena::default();

        let container = Container::new(arena);
        let node1 = Node::new(arena, 1);
        let node2 = Node::new(arena, 2);
        let node3 = Node::new(arena, 3);

        let mut cursor = container.cursor(arena, CursorStrategy::Pre);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        let mut cursor = container.cursor(arena, CursorStrategy::Post);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        let mut cursor = cursor.rev(arena);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        container.push_back(arena, node1).unwrap();
        container.push_back(arena, node2).unwrap();
        container.push_back(arena, node3).unwrap();

        let mut cursor = container.cursor(arena, CursorStrategy::Pre);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node1));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node2));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node3));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        let mut cursor = cursor.rev(arena);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node3));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node2));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node1));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        let mut cursor = container.cursor(arena, CursorStrategy::Post);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node1));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node2));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node3));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        let mut cursor = cursor.rev(arena);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node3));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node2));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node1));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
    }

    #[test]
    fn test_linked_list_cursor_modify() {
        let arena = &mut TestArena::default();

        let container = Container::new(arena);
        let node1 = Node::new(arena, 1);
        let node2 = Node::new(arena, 2);
        let node3 = Node::new(arena, 3);
        let node4 = Node::new(arena, 4);

        container.push_back(arena, node1).unwrap();
        container.push_back(arena, node2).unwrap();
        container.push_back(arena, node3).unwrap();

        let mut cursor = container.cursor(arena, CursorStrategy::Pre);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node1));
        assert!(!cursor.is_done());
        node1.unlink(arena);
        assert_eq!(cursor.next(arena), Some(node2));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node3));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        container.push_front(arena, node1).unwrap(); // restore

        let mut cursor = container.cursor(arena, CursorStrategy::Pre);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node1));
        assert!(!cursor.is_done());
        node1.insert_after(arena, node4).unwrap();
        assert_eq!(cursor.next(arena), Some(node2));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node3));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        node4.unlink(arena); // restore

        let mut cursor = container.cursor(arena, CursorStrategy::Post);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node1));
        assert!(!cursor.is_done());
        node1.insert_after(arena, node4).unwrap();
        assert_eq!(cursor.next(arena), Some(node4));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node2));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node3));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());

        let mut cursor = cursor.rev(arena);
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node3));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node2));
        assert!(!cursor.is_done());
        assert_eq!(cursor.next(arena), Some(node4));
        assert!(!cursor.is_done());
        node4.unlink(arena);
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
        assert_eq!(cursor.next(arena), None);
        assert!(cursor.is_done());
    }

    #[test]
    fn test_linked_list_merge() {
        let arena = &mut TestArena::default();

        let container1 = Container::new(arena);
        let container2 = Container::new(arena);
        let node1 = Node::new(arena, 1);
        let node2 = Node::new(arena, 2);
        let node3 = Node::new(arena, 3);
        let node4 = Node::new(arena, 4);

        container1.push_back(arena, node1).unwrap();
        container1.push_back(arena, node2).unwrap();
        container2.push_back(arena, node3).unwrap();
        container2.push_back(arena, node4).unwrap();

        container1.append(arena, container2);

        assert_eq!(container1.head(arena), Some(node1));
        assert_eq!(container1.tail(arena), Some(node4));
        assert_eq!(container2.head(arena), None);
        assert_eq!(container2.tail(arena), None);

        let mut iter = container1.iter(arena);
        assert_eq!(iter.next(), Some(node1));
        assert_eq!(iter.next(), Some(node2));
        assert_eq!(iter.next(), Some(node3));
        assert_eq!(iter.next(), Some(node4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_linked_list_split() {
        let arena = &mut TestArena::default();

        let container1 = Container::new(arena);
        let container2 = Container::new(arena);
        let node1 = Node::new(arena, 1);
        let node2 = Node::new(arena, 2);
        let node3 = Node::new(arena, 3);
        let node4 = Node::new(arena, 4);

        container1.push_back(arena, node1).unwrap();
        container1.push_back(arena, node2).unwrap();
        container1.push_back(arena, node3).unwrap();
        container1.push_back(arena, node4).unwrap();

        container1.split(arena, container2, node2).unwrap();

        assert_eq!(container1.head(arena), Some(node1));
        assert_eq!(container1.tail(arena), Some(node2));
        assert_eq!(container2.head(arena), Some(node3));
        assert_eq!(container2.tail(arena), Some(node4));

        let mut iter = container1.iter(arena);
        assert_eq!(iter.next(), Some(node1));
        assert_eq!(iter.next(), Some(node2));
        assert_eq!(iter.next(), None);

        let mut iter = container2.iter(arena);
        assert_eq!(iter.next(), Some(node3));
        assert_eq!(iter.next(), Some(node4));
        assert_eq!(iter.next(), None);
    }
}
