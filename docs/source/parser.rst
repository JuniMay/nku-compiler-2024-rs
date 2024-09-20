语法分析器与抽象语法树
================================================

.. toctree::
    :hidden:
    :maxdepth: 4


.. note::

    Rust 版本实验中，词法分析和语法分析合为一个实验，DDL 与 C++ 版本实验的语法分析 DDL 一致。

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

.. note::

    提供的代码框架中使用了 LALRPOP 生成 Parser，你也可以选择使用其他方法实现 Parser。

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


代码框架
----------------

.. note::

    你也可以不使用我们给出的代码框架自己进行编写。如果你认为代码框架存在问题或者有不合理的地方也可以自行修改。

代码框架中包含了以下文件：

.. code-block:: bash

    .
    ├── Cargo.lock       
    ├── Cargo.toml
    ├── build.rs
    ├── rustfmt.toml
    ├── src
    │   ├── backend
    │   ├── backend.rs    # 后端实现
    │   ├── frontend
    │   ├── frontend.rs   # 前端实现
    │   ├── infra
    │   ├── infra.rs      # 基础设施代码，包括 Arena、链表等
    │   ├── ir
    │   ├── ir.rs         # IR 的定义
    │   ├── lib.rs
    │   └── main.rs
    └── tests
        ├── sysy
        └── testcase

代码框架使用 LALRPOP 作为 Parser 的生成器，你可以在 ``src/frontend`` 目录下找到 ``sysy.lalrpop`` 文件，
这是我们使用 LALRPOP 生成 Parser 的源文件。此外，LALRPOP 需要使用 ``build.rs`` 文件生成 Parser，
你可以在 ``build.rs`` 文件中找到相关的配置。

你可以从 ``main.rs`` 开始阅读代码。 ``main.rs`` 中的内容目前非常简单，它会加载
``tests/sysy`` 下的 ``basic.sy`` 这个源代码，使用 ``preprocess`` 函数进行预处理，
之后创建一个新的 Parser 解析源代码得到抽象语法树，在抽象语法树上
进行类型检查，用 Debug 方式输出 AST，并用 ``irgen`` 这个函数生成 IR 并输出。

.. code-block:: rust

    use nkucc::frontend::{irgen, preprocess, SysYParser};

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        println!("Hello, NKUCC!");

        let src = std::fs::read_to_string("tests/sysy/basic.sy")?;
        let src = preprocess(&src);

        let mut ast = SysYParser::new().parse(&src).unwrap();

        ast.type_check();

        println!("{:#?}", ast);

        let ir = irgen(&ast, 8);

        println!("{}", ir);

        Ok(())
    }

本次实验中你需要完成的工作包括以下几个部分：

#. 阅读 ``frontend`` 中与语法分析有关的代码，包括 ``ast.rs``， ``parse.rs``，
   ``preprocess.rs``， ``types.rs`` 以及 ``sysy.lalrpop`` 等文件，并且理解这些代码的作用。
#. 完善 ``sysy.lalrpop`` 中的语法规则，使其能够正确解析 SysY 语言的语法（主要是已经标注了 ``TODO`` 的部分），你可以用
   VSCode 中的 Todo Tree 来快速查看所有的 ``TODO``，RustRover 则自带了这个功能。
#. 在 ``main.rs`` 中添加命令行参数，通过命令行指定源文件并输出对应的 AST。

你可能需要注意以下几个问题：

- 本次实验并不要求实现类型检查或者符号表，只要能够打印解析得到的 AST 即可。
- ``main.rs`` 中的代码会有一些多余的部分，例如类型检查和 IR 生成，本次实验中你可以直接注释掉这部分代码来运行。
- 你可以使用 ``clap`` 这个库实现命令行参数的解析。
- 我们建议你在实现 Parser 之后完整阅读前端的代码，包括符号表和 IRGen 的结构，这样有助于你理解整个前端的结构。
- 完成本次实验之后，你的程序应当能够正确解析 ``tests/testcase`` 中的所有测试用例，并且输出 AST，你可能需要手动检查 AST 的正确性。
