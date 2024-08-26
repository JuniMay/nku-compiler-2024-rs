语法分析器与抽象语法树
================================================

.. toctree::
   :hidden:
   :maxdepth: 4


.. warning::

   本节施工中🏗️

实验描述
----------------

相信你已经在 C++ 文档中或者同学处听说了 Flex 和 Bison 这两个工具，它们是用于生成词法分析器和语法分析器的工具。在这一部分，
我们将使用 Rust 语言中的一系列库来实现 SysY 语言的 Parser，并且将 SysY 源代码转换到抽象语法树（AST）。


抽象语法树的构建
----------------

在 C++ 中我们可能需要使用基类 + 继承的方式构建 AST 的数据结构，但在 Rust 中，得益于 Rust 优秀的语言特性，我们可以使用枚举类型来构建 AST 的数据结构。

实现 Parser
----------------

有许多方法可以实现 Parser，例如手写递归下降 Parser、使用工具生成 Parser 等。
且由于 Rust 包管理的便利，有许多工具都可以帮助我们实现 Parser。此处我们并不限制 Parser 的实现方式，
你可以选择手写递归下降、Parser Combinator 库或者使用工具生成 Parser 等方式构建你的 Parser。

手写递归下降 Parser
^^^^^^^^^^^^^^^^^^^^

   人类的赞歌就是勇气的赞歌。

我们并不推荐你在 Rust 中手搓递归下降的 Parser。尽管 Rust 提供了各种安全性的保障，但是手写递归下降 Parser 仍然是一件非常繁琐的事情，
尤其是当你想完全手动实现 SysY 语言的语法时。如果你确定想要挑战自己，我们建议你先重温一下课件中所述的有关内容，然后参考以下这几个资料：

- GeeksforGeeks 上的 `Recursive Descent Parser <https://www.geeksforgeeks.org/recursive-descent-parser/>`_
- `Building Recursive Descent Parsers: The Definitive Guide <https://www.booleanworld.com/building-recursive-descent-parsers-definitive-guide/>`_
- `A Beginner's Guide to Parsing in Rust <https://depth-first.com/articles/2021/12/16/a-beginners-guide-to-parsing-in-rust/>`_

.. note::

   写一个能跑的递归下降 Parser 并不难，但是写一个运行效率高、保证正确性甚至支持错误恢复的递归下降 Parser 是有一定难度的。
   你可以在编写 Parser 的过程中进行单元测试，确保你的 Parser 能够正确地解析 SysY 语言的语法。

   值得注意的是，许多编译器都会手动实现递归下降 Parser 来构建前端，因为递归下降 Parser 通常有着很好的可读性和可维护性。


使用 Parser Combinator 库
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parser Combinator 实际上是递归下降 Parser 的一种实现方式，但是它使用了函数式编程的思想，将 Parser 看作是函数，通过组合这些函数来构建 Parser。

Rust 吸取了很多函数式编程语言的优点，其类型系统允许我们利用函数式编程的思想来构建 Parser。
如果你想要使用 Parser Combinator 来构建 Parser，建议阅读：

- 使用 ``nom`` 库： `Making a new parser from scratch <https://github.com/rust-bakery/nom/blob/main/doc/making_a_new_parser_from_scratch.md>`_
- 使用 ``chumsky`` 库： `Chumsky: A Tutorial <https://github.com/zesterer/chumsky/blob/main/tutorial.md>`_
- `A gentle introduction to parser-combinators <https://kat.bio/blog/parser-combinators>`_

使用工具生成 Parser
^^^^^^^^^^^^^^^^^^^^

Flex 和 Bison 都属于 Parser Generator，它们可以帮助我们生成词法分析器和语法分析器。
但是在 Rust 中，我们无法使用 Flex 和 Bison。作为替代，我们可以使用 LALRPOP 或者 Pest 来生成 Parser。

可以参考

- `LALRPOP 文档 <https://lalrpop.github.io/lalrpop/tutorial/002_paren_numbers.html>`_
- `Pest 文档 <https://docs.rs/pest/latest/pest/>`_
