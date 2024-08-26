.. nku-compiler-2024 documentation master file, created by
   sphinx-quickstart on Sat Aug 24 19:53:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

编译系统原理课程项目文档
=============================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: 指导手册:

   chapter0/index
   rust-basics.rst
   llvm-basics.rst
   parser.rst
   irgen.rst
   codegen.rst
   optimization.rst


本文档为 Rust 实现版本文档。

课程项目简介
---------------------

.. attention::

   本实验要求对 Rust 语言有一定的了解，并且有很大一部分实验需要在没有完整参考代码的情况下完成项目，会有一定难度。
   
   请注意 Rust 版本实验仍然要求完成完整的编译器构建，所以在确定最终课程项目所使用的语言前请评估自身的能力，如果你对 Rust 语言不熟悉，并且没有足够的时间、精力学习 Rust，建议使用 C++ 完成课程项目。

.. note::
   
   Rust 版本为 2024 年课程新加入的实验，如果在实验过程中遇到问题，请及时与助教联系。

.. important::

   在开始实验前，请先仔细阅读本页面中的内容，**尤其是关于如何解决问题的部分**，这将有助于你更好地完成实验。

.. important::

   **本文档包括什么**

   - 对 Rust 语言以及实现一个编译器可能需要用到的设计模式和数据结构的介绍。
   - 对 LLVM IR 的基本介绍。
   - 对 Rust 中构建语法分析器、生成抽象语法树的介绍。
   - 对 AST 翻译到中间表示的一般方法的介绍。
   - 对中间表示翻译到目标代码的一般方法的介绍。
   - 对一些编译优化方法的概述。

   **本文档不包括什么**

   - 完整的 Rust 语言教程。本文档并非 Rust 教程，如果你对 Rust 不熟悉并且没有足够的时间学习 Rust，建议使用 C++ 完成课程项目。
   - 完整的编译器实现代码。本文档以及配套的代码框架并不包含完整的编译器实现，你可以参考下面给出的编译器框架，根据本文档的指导完成编译器的实现。
   - 详细的编译优化算法或实现。本文档只会对一些基本的编译优化方法进行概述，如果你对编译优化感兴趣，或者想实现课程实验的进阶要求，可以自行查阅相关资料。


由于 Rust 版本实验为新添加实验，许多内容可能并不成熟，此处给出几个可以参考的 Rust 编译器实现：

#. `orzcc <https://github.com/JuniMay/orzcc>`_ ：一个 SysY 到 RISC-V 后端编译器的实现，实验代码框架主要使用这个项目作为参考。

   .. note::

      orzcc 没有使用 LLVM IR，但是根据课程实验要求，你需要使用 LLVM IR 作为中间代码。

#. `vicis <https://github.com/maekawatoshiki/vicis>`_ ：在 Rust 中构建 LLVM IR 的实现，可以作为参考。
#. `cranelift <https://github.com/bytecodealliance/wasmtime/tree/main/cranelift/codegen>`_ ：一个较为成熟的 Rust 编译器框架，可以参考其中的数据结构和实现。


为什么使用 Rust ？
---------------------

   Rust 写图着色比 C++ 简单一个数量级。—— gdw

你可能会问，既然 C++ 已经是一门非常成熟的系统编程语言，为什么还要尝试使用 Rust 来实现编译器呢？答案是多方面的：

#. Rust 相比于 C++ 拥有更好的基础设施，通过 Cargo 可以方便地管理依赖、测试和构建项目，而不需要手动编写 Makefile 或者 CMakeLists.txt。
#. Rust 的类型检查更为严格，有许多问题可以在编译期就被检测出来。
#. 在正确使用的情况下，Rust 能够更方便地写出内存安全的代码，避免了许多 C++ 中常见的内存错误。

当然，还是如本文档开头所述，Rust 语言的学习曲线可能会比较陡峭，对于你来说，课程项目的得分可能仍然是非常重要的，所以请根据自己的实际情况选择合适的语言。

如果你确定要使用 Rust 完成课程项目，那么你需要注意以下几点：

#. 不要想当然地认为可以用 Rust 以同样的思路和方法实现 C++ 版本中的功能，Rust 语言的特性和 C++ 有很大的不同，你需要重新思考如何使用 Rust 的特性来实现编译器，往往这种思考过程会让你对编译器的原理有更深入的理解，并且写出更加优雅的代码。
#. Rust 编译器是你的朋友，而不是敌人，Rust 编译器会帮助你检测出许多潜在的错误，所以请不要忽略编译器的警告信息，尽量保持代码的可读性和可维护性。
#. 当你遇到 Rust 编译器报错时，不要惊慌，尝试仔细阅读错误信息，理解错误的原因，重新思考你的代码逻辑或者阅读有关的资料，很多时候错误信息会帮助你找到代码中的问题。

如何解决问题？
---------------------
在实现编译器的过程中，你可能会遇到各种各样的问题，对于这些问题，你可以依次尝试以下几种方法尝试解决：

#. 先尝试自己解决。仔细想想，问题出在哪里？是不是自己操作有误？是不是代码有问题？是不是对某个概念理解不正确？
#. **RTFM (Read The Friendly Manual) 先查文档！** 本文档以及本文档给出的参考资料中提及了许多常见问题以及解决方法，请仔细阅读。大部分问题都可以通过这些资料解决。
#. **STFW (Search The Fantastic Web) 善用搜索引擎！** 请至少尝试使用 Bing 或者 Google 搜索一下你的问题，再决定是否向助教求助。

   .. hint::

      在大部分情况下，Baidu/CSDN 远不如 Google/Bing/StackOverflow 等搜索引擎和平台给出的英文解决方案全面。

#. **Ask ChatGPT!** 现在是大模型时代！ChatGPT 比任何人都聪明，作为一个工科生，你应该至少有一个 ChatGPT 或者类似的工具，如果没有，请尽快找到一个。ChatGPT 将会在你的编程生涯中成为你一大利器，一定要善用 ChatGPT。

   .. warning::

      尽管大模型是个好工具，但是他们在专业知识上很有可能出错，请不要完全相信大模型的回答，特别是涉及到环境配置的指令，请自行验证后再执行。

#. 向他人求助。如果以上方法都无法解决你的问题，那么请向同学或者助教求助。

   .. important::

      请注意，助教也有自己的作息时间，不要期望他们能够立刻回复你的问题，尽量提前预留足够的时间来解决问题。

参考资料
---------------------

- `Rust 程序设计语言 <https://kaisery.github.io/trpl-zh-cn/>`_
- `Rust 语言圣经 <https://course.rs/about-book.html>`_
- `Rust By Example <https://doc.rust-lang.org/rust-by-example/index.html>`_
- 虎书（Modern Compiler Implementation in C）
- 龙书（Compilers: Principles, Techniques, and Tools）
- `The LALRPOP book <https://lalrpop.github.io/lalrpop/>`_
- `A thoughtful introduction to the pest parser <https://pest.rs/book/>`_
- `LLVM Programmer's Manual <https://llvm.org/docs/ProgrammersManual.html>`_
- `RISC-V ISA Specifications <https://riscv.org/technical/specifications/>`_
- `ARM Assembly Basics <https://azeria-labs.com/writing-arm-assembly-part-1/>`_