Rust 编程基础
================================================

.. toctree::
    :hidden:
    :maxdepth: 4

系统学习 Rust
---------------------

这里有一些学习 Rust 的资源，如果你想系统学习 Rust，可以参考这些资源。如果你想要快速入门 Rust，推荐 Rust By Example。

- `Rust 程序设计语言 <https://kaisery.github.io/trpl-zh-cn/>`_
- `Rust 语言圣经 <https://course.rs/about-book.html>`_
- `Rust By Example <https://doc.rust-lang.org/rust-by-example/index.html>`_

开始实验之前你至少需要学习并掌握 Rust 程序设计语言的前七节和第十节，或者 Rust 语言圣经的第二节，或者 Rust By Example 的前十节和第十六节。

下面是一个用于自查的 Rust 基础知识点列表：

- ``mut`` 关键字的作用有哪些？
- 变量的 Shadowing 是什么？
- Rust 的基础数据类型有哪些？
- 注释有哪几种写法？
- ``if`` 语句语法是什么样的？
- Rust 有哪几种循环语句？
- 可变引用和不可变引用有什么区别？如何使用？
- Rust 中的结构体怎么写？如何控制结构体内部成员可见性？如何为结构体实现方法？
- Rust 中的枚举怎么写？如何为枚举实现方法？
- Rust 中模式匹配是什么？其语法是什么？
- Rust 中的 Trait 是什么？如何实现 Trait？
- Rust 中的范型是什么？如何使用范型？
- 如何组织 Rust 项目中模块的层级结构？

更进一步，如果能够熟悉 Rust 中 ``std::collections`` 中的数据结构，以及迭代器的使用，会对你完成这个实验有很大的帮助。

如果你还有时间，你可以进一步学习 `Effective Rust <https://www.lurklurk.org/effective-rust/title-page.html>`_ 中的内容。


使用 Cargo
--------------------------

你可以使用 Cargo 实现整个项目的构建、依赖的管理以及单元测试。以下是一个简单的例子：

.. code-block:: shell

    $ cargo new my_project
    $ cd my_project
    $ cargo run

之后你就会看到类似于下面的输出：

.. code-block:: shell

    $ cargo run
        Compiling my_project v0.1.0 (/path/to/my_project)
        Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.15s
        Running `target/debug/my_project`
    Hello, world!

你会发现整个项目的结构如下

.. code-block:: shell

    $ tree
        .
        ├── Cargo.lock
        ├── Cargo.toml
        └── src
            └── main.rs

其中 ``Cargo.toml`` 是项目的配置文件，``src/main.rs`` 是项目的入口文件。我们的编译器框架也会使用这样的结构。

.. tip::

    Cargo 是一个非常易用且可扩展的构建工具，你可以通过 ``cargo --help`` 查看所有的命令。
    关于更加详细的使用方法，可以参考 `The Cargo Book <https://doc.rust-lang.org/cargo/>`_ （强烈建议阅读，因为之后你会经常用到 Cargo）。

    此处列举几个比较基础的 Cargo 使用方式

    - 使用 Release 模式构建项目：``cargo build --release``
    - 运行单元测试：``cargo test``
    - 检查代码风格：``cargo fmt``
    - 添加依赖包：``cargo add <package-name>``

    Cargo 之于 Rust 就像 Pip 之于 Python，Npm 之于 Node.js，Maven 之于 Java 一样，是 Rust 生态中的一个重要组成部分。
    你可以在 `crates.io <https://crates.io>`_ 上找到很多 Rust 的第三方库，你可以通过 Cargo 添加这些库到你的项目中。
    如果速度太慢可以考虑使用 tuna 或者中科大的源，使用方法请自行 STFW。

.. tip::

    测试是保证你的代码质量的重要手段，Rust 语言的特性以及 Cargo 这个构建工具为测试提供了很大的便利，建议阅读： `How to Write Tests <https://doc.rust-lang.org/book/ch11-01-writing-tests.html>`_ 。
   
.. tip::

    你可以配置 Clippy 并运行 ``cargo clippy`` 来检查代码中更多的潜在问题，具体使用方式和配置方法可以参考 `Clippy Documentation <https://doc.rust-lang.org/clippy/index.html>`_ 。


Rust 编程技巧
--------------------------

.. note::

    在阅读下面的内容之前，请确保你已经至少掌握了 Rust 程序设计语言的前七节和第十节，或者 Rust 语言圣经的第二节，或者 Rust By Example 的前十节和第十六节的内容。

    以下内容只是对 Rust 语言一些惯用法或重要概念的介绍，这些内容只是你完成这个实验所需要的很少一部分，你仍然需要阅读前面给出的参考文档学习 Rust 的基础知识。

.. warning::

    如果你发现自己学习 Rust 所花的时间已经远远超过了学习编译原理理论所需要的时间，那么你可能需要重新确定一下课程实验所使用的编程语言了。
    使用 Rust 编写编译器只是实验的一个选项，其目的仍然是让你更好地去学习编译原理，而非 Rust 这个编程语言。
    如果你确实对 Rust 很感兴趣但是时间不够，可以考虑在课程实验结束后继续学习 Rust。


Arena 惯用法
^^^^^^^^^^^^^^^^^^^^^^^^

编译器的实现中我们会用到非常多的图、树、链表等数据结构。而图、树、链表这种存在自引用的数据结构在 Rust 中是非常难以实现的，
因为 Rust 的生命周期检查器会阻止这种自引用的数据结构的实现。在 C++ 中，我们可以使用指针来实现这种数据结构：

.. code-block:: cpp

    struct Node {
        Node* previous;
        Node* next;
        // ...
    };

但是相信大家已经遇到过了很多 C/C++ 中指针乱飞的问题。Rust 中一个可能的实现是使用 ``Rc`` 或者 ``Arc`` 来实现引用计数，
然而 ``Rc`` 不仅不好写，会带来额外开销，还有可能会导致循环引用，从而导致内存泄漏（**是的，Rust 不能防止内存泄漏**） [#rc_leak]_ 。
下面是一个使用 ``Rc`` 的例子 [#arena]_ ：

.. code-block:: rust

    use std::rc::Rc;
    use std::cell::RefCell;

    struct Node<T> {
        previous: Rc<RefCell<Box<Node<T>>>>,
        //        ^  ^       ^   ^
        //        |  |       |   |
        //        |  |       |   - 下一个节点 `T`
        //        |  |       |
        //        |  |       - 堆上动态分配的内存
        //        |  |
        //        |  - RefCell 实现了内部可变性，可以在不可变引用的情况下修改内部值
        //        |
        //        - 引用计数，因为一个节点可能被多个节点引用

        next: Vec<Rc<RefCell<Box<T>>>>,
        data: T,
        // ...
    }

.. note::

    如果你感兴趣，可以用这个方法写个图试试。

一个解决方法是使用 Arena，将所有的节点都放在一个 ``Vec`` 中，通过索引来表示其中的节点，这样就不会有生命周期检查器的问题了。

.. code-block:: rust

    pub struct Arena<T> {
        nodes: Vec<Node<T>>,
    }

    pub struct Node<T> {
        parent: Option<NodeId>,
        previous_sibling: Option<NodeId>,
        next_sibling: Option<NodeId>,
        first_child: Option<NodeId>,
        last_child: Option<NodeId>,

        /// 真实的数据存储在 Arena 中
        pub data: T,
    }

    pub struct NodeId {
        index: usize,
    }

    impl<T> Arena<T> {
        pub fn new_node(&mut self, data: T) -> NodeId {
            let node = Node {
                parent: None,
                previous_sibling: None,
                next_sibling: None,
                first_child: None,
                last_child: None,
                data,
            };

            // 将节点放入 Arena 中
            self.nodes.push(node);

            // 我们不返回数据，而是返回数据在 Arena 中的索引作为其标识符
            NodeId {
                index: self.nodes.len() - 1,
            }
        }
    }

在编译器的实现过程中，我们将大量用到这样的方式来实现各种带引用的数据结构。在代码框架的 ``src/infra/storage.rs``
中我们实现了一个较为通用的 Arena 基础设施。你也可以尝试使用别的方法实现这种自引用的数据结构。

.. note::

    如果你是一个 OI 选手，你可能会发现这种写法和你在写图论题目时的写法非常相似。

newtype 惯用法
^^^^^^^^^^^^^^^^^^^^^^^^

你可能想问，为什么我们要使用 ``NodeId`` 而不是直接使用 ``usize`` 作为索引呢？主要的原因仍然是类型安全。

在上面的实现中，如果我们要创建一个 ``NodeId``，我们必须要通过 ``Arena`` 的方法来创建，而不能直接构造一个 ``NodeId``
（当然如果处于同一个模块下，你仍然可以直接访问 ``NodeId`` 的 ``index`` 字段）。这样我们就能够保证 ``NodeId`` 永远不会被错误地使用。
如果我们使用 ``usize`` 直接索引 ``Arena``，我们就无法区分这个索引是不是来自于 ``Arena``，这样就可能导致程序在运行时 panic。

另外一个原因在于 ``usize`` 和 ``NodeId`` 的语义不同，``usize`` 往往用于表达整数、索引，而 ``NodeId`` 虽然内部是一个索引，
但是它实际上是用于标识一个节点的。另外，如果我们需要为 ``NodeId`` 添加一些方法或实现 Trait，我们可以直接在 ``NodeId`` 上实现，
而不会影响到 ``usize`` 的其他用途（一般对于 ``usize`` 而言，为他添加方法的实现会被编译器禁止，但是 ``usize`` 只是一个例子，
可能会有其他类型可以添加方法）。你可以参考 `Newtype Index Pattern <https://matklad.github.io/2018/06/04/newtype-index-pattern.html>`_ 
和 `Rust Design Patterns <https://rust-unofficial.github.io/patterns/patterns/behavioural/newtype.html>`_ 以及
`Embrace the newtype pattern -- Effective Rust <https://www.lurklurk.org/effective-rust/newtype.html>`_ 中对 newtype 模式的介绍。

在代码框架中，我们同样使用了很多 newtype 模式，但是他们内部并不都是 ``usize``，你可以在代码框架中找到这些例子。

标准库中的 Traits
^^^^^^^^^^^^^^^^^^^^^^^^

你可以阅读 Effective Rust 中的 `这一节 <https://www.lurklurk.org/effective-rust/std-traits.html>`_ 来熟悉 Rust 标准库提供的常用 Trait，
熟悉这些 Trait 可以帮助你更好的理解代码框架中的一些实现。此处主要对 ``Clone``， ``Copy`` 进行介绍。

``Clone`` 类似于 C++ 中的拷贝构造函数，实现这个 Trait 可以告诉编译器“我这个类型是可以拷贝的”，但需要注意的是，Rust 并不会自动插入 ``clone()`` 的调用，
你需要手动调用 ``clone()`` 方法来进行拷贝。

``Copy`` Trait 同样表示拷贝，但是 ``Copy`` 所表示的是 bitwise 拷贝，即直接将数据复制到新的变量中。另外，实现了 ``Copy`` 的类型会直接将移动语义变为拷贝语义，
这样就不会导致所有权的转移。下面是一个例子：

.. code-block:: rust

    #[derive(Clone)]
    struct Point {
        x: i32,
        y: i32,
    }

    fn main() {
        let p1 = Point { x: 1, y: 2 };
        let p3 = p1;

        println!("{:?}", p1); // 这里会报错，因为 p1 已经被移动到 p3 中了
    }

如果我们将 ``Point`` 实现为 ``Copy``，那么上面的代码就不会报错。

.. code-block:: rust

    #[derive(Copy, Clone)]
    struct Point {
        x: i32,
        y: i32,
    }

    fn main() {
        let p1 = Point { x: 1, y: 2 };
        let p3 = p1;

        println!("{:?}", p1); // 这里不会报错，因为 p1 会被拷贝到 p3 中，创建了一个副本
    }


.. [#rc_leak] 参考： `Reference Cycles Can Leak Memory <https://doc.rust-lang.org/book/ch15-06-reference-cycles.html>`_
.. [#arena] 代码来自于： `Idiomatic tree and graph like structures in Rust <https://rust-leipzig.github.io/architecture/2016/12/20/idiomatic-trees-in-rust/>`_
