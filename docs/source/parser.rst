词法、语法分析器与抽象语法树
================================================

.. toctree::
    :hidden:
    :maxdepth: 4

实验描述
----------------

相信你已经在 C++ 文档中或者同学处听说了 Flex 和 Bison 这两个工具，它们是用于生成词法分析器和语法分析器的工具。在这一部分，
我们将使用 Rust 语言中的一系列库来实现 SysY 语言的 Parser，并且将 SysY 源代码转换到抽象语法树（AST）。

根据实验要求，Rust 版本实验仍然拆分为词法分析和语法分析两个部分。我们所给出的框架中语法分析和词法分析是耦合的，所以你需要在阅读实验指导之后自己实现词法分析器这个实验中的内容。

实现词法分析器（Lexer）要求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. 能够识别简单的 SysY 源程序；
#. 输出解析得到的 Token 序列；

此外，从解析结果来看，你需要至少支持以下几种功能

#. 在解析过程中识别注释并且跳过；
#. 解析标识符，即变量名、函数名等；
#. 解析整数字面量，例如： ``123`` , ``0x123`` 等；

实现语法分析器（Parser）要求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. 理解 AST 结构的实现
#. 能够识别 SysY 语言的语法；
#. 输出解析得到的抽象语法树（AST）；

实现 Lexer 和 Parser
--------------------------------

有许多方法可以实现 Parser，例如手写递归下降 Parser、使用工具生成 Parser 等。
且由于 Rust 包管理的便利，有许多工具都可以帮助我们实现 Parser。此处我们并不限制 Parser 的实现方式，
你可以选择手写递归下降、Parser Combinator 库或者使用工具生成 Parser 等方式构建你的 Parser。

.. note::

    Rust 中并没有一个与 Flex + Bison 完全对应的工具组合，所以此处直接对前端实现可能需要对工具进行统一介绍。

    理论上，下面介绍的三种工具/方法都可以实现词法分析实验，但是我们的框架使用的是 LALRPOP，所以我们推荐你使用 LALRPOP 来实现编译器前端（包括词法分析和语法分析两个实验）

手写递归下降
^^^^^^^^^^^^^^^^^^^^

    人类的赞歌就是勇气的赞歌。

我们并不推荐你在 Rust 中手搓递归下降的 Parser。尽管 Rust 提供了各种安全性的保障，但是手写递归下降 Parser 仍然是一件非常繁琐的事情，
尤其是当你想完全手动实现 SysY 语言的语法时。如果你确定想要挑战自己，我们建议你先重温一下课件中所述的有关内容，然后参考以下这几个资料：

- GeeksforGeeks 上的 `Recursive Descent Parser <https://www.geeksforgeeks.org/recursive-descent-parser/>`_
- `Building Recursive Descent Parsers: The Definitive Guide <https://www.booleanworld.com/building-recursive-descent-parsers-definitive-guide/>`_
- `A Beginner's Guide to Parsing in Rust <https://depth-first.com/articles/2021/12/16/a-beginners-guide-to-parsing-in-rust/>`_

当然，如果你只是想手写一个词法分析器，那这个任务就相对简单一点，此处给出一个大致的思路：

1. 读取源代码，获得一个字符串；
2. 从头开始扫描字符串，根据当前字符的类型判断 Token 可能的类型；
3. 继续读取字符，直到 Token 的类型确定或者出现错误；
4. 将读入的字符串内容转换为 Token，存入 Token 列表中；
5. 重复 2-4 步，直到读取完整个源代码。

你可以利用 Rust 的枚举类型来定义 Token。

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

此外，如果你要使用 LALRPOP 来完成词法分析的实验，可以在其文档中找到 `这一章 <https://lalrpop.github.io/lalrpop/lexer_tutorial/index.html>`_ ，
里面的内容详细讲解了如何控制 LALRPOP 中的词法分析器。如果你时间有限，可以只阅读里面的 6.1。如果你发现 LALRPOP 对于词法分析来说并不好用，你可以参考上述文档中的 6.5 节，里面给出了一个更纯粹的用于词法分析的库。

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

语法分析和词法分析两次实验中你需要完成的工作包括以下几个部分：

1. 对于词法分析实验，你需要阅读 ``frontend`` 中的 ``parse.rs``， ``sysy.lalrpop`` 以及 ``main.rs``
   文件，理解代码的结构和作用。为了保持和后续试验的连贯性，同时验证对要求的代码的理解，你需要完善下面这个程序：

   .. code-block:: rust

       fn main() -> Result<(), Box<dyn std::error::Error>> {
           let int_str = String::from("/* xxxx */ 114514 // 123");
           let hex_str = String::from("/* xxxx */ 0xfff // 123");
           assert_eq!(parse_int(&int_str), 114514);
           assert_eq!(parse_int(&hex_str), 0xfff);
           println!("lexer test passed");
           Ok(())
       }
   
       fn parse_int(str: &str) -> i32 {
           // TODO: use NumberParser to parse int
           todo!()
       }

   你可以使用 ``frontend/parse.rs`` 中与 ``SysYParser`` 一样的方法找到
   ``NumberParser``，并且实现解析。你也可以替换这个程序里面的字符串、修改 ``sysy.lalrpop``
   中的规则来进一步理解里面的内容。另外你需要理解 LALRPOP 文件里面， ``Number`` 和 ``Ident`` 这两个规则中每一个 ``=>``
   左右的部分分别是什么含义，以及对 ``NumberParser`` 进行定义的 Rust 源代码在哪。

2. 对于语法分析实验：

   - 阅读 ``frontend`` 中与语法分析有关的代码，包括 ``ast.rs``， ``parse.rs``，``preprocess.rs``， ``types.rs`` 以及 ``sysy.lalrpop`` 等文件，并且理解这些代码的作用。
   - 完善 ``sysy.lalrpop`` 中的语法规则，使其能够正确解析 SysY 语言的语法（主要是已经标注了 ``TODO`` 的部分），你可以用VSCode 中的 Todo Tree 来快速查看所有的 ``TODO``，RustRover 则自带了这个功能。
   - 在 ``main.rs`` 中添加命令行参数，通过命令行指定源文件并输出对应的 AST。

你可能需要注意以下几个问题：

- 本次实验并不要求实现类型检查或者符号表，只要能够打印解析得到的 AST 即可。
- ``main.rs`` 中的代码会有一些多余的部分，例如类型检查和 IR 生成，本次实验中你可以直接注释掉这部分代码来运行。
- 你可以使用 ``clap`` 这个库实现命令行参数的解析。
- 我们建议你在实现 Parser 之后完整阅读前端的代码，包括符号表和 IRGen 的结构，这样有助于你理解整个前端的结构。
- 完成本次实验之后，你的程序应当能够正确解析 ``tests/testcase`` 中的所有测试用例，并且输出 AST，你可能需要手动检查 AST 的正确性。
