.. nku-compiler-2024 documentation master file, created by
   sphinx-quickstart on Sat Aug 24 19:53:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

编译系统原理课程项目文档
=============================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   rust-basics.rst
   llvm-basics.rst
   parser.rst
   irgen.rst
   codegen.rst
   optimization.rst


本文档为 Rust 实现版本文档。

课程项目简介
---------------------

.. warning::

   本实验要求对 Rust 语言有一定的了解，并且有很大一部分实验需要在没有参考代码的情况下完成项目，会有一定难度。
   
   请注意 Rust 版本实验仍然要求完成完整的编译器构建，所以在确定最终课程项目所使用的语言前请评估自身的能力，如果你对 Rust 语言不熟悉，并且没有足够的时间、精力学习 Rust，建议使用 C++ 完成课程项目。

.. note::
   
   Rust 版本为 2024 年课程新加入的实验，如果在实验过程中遇到问题，请及时与助教联系。

为什么使用 Rust ？
---------------------

   Rust 写图着色比 C++ 简单一个数量级。—— gdw

你可能会问，既然 C++ 已经是一门非常成熟的系统编程语言，为什么还要尝试使用 Rust 来实现编译器呢？答案是多方面的：

#. Rust 相比于 C++ 拥有更好的基础设施，通过 Cargo 可以方便地管理依赖、测试和构建项目，而不需要手动编写 Makefile 或者 CMakeLists.txt。
#. Rust 的类型系统相比 C++ 更加强大，有许多问题可以在编译期就被检测出来。
#. 在正确使用的情况下，Rust 能够更方便地写出内存安全的代码，避免了许多 C++ 中常见的内存错误。

当然，还是如本文档开头所述，Rust 语言的学习曲线可能会比较陡峭，对于你来说，课程项目的得分可能仍然是非常重要的，所以请根据自己的实际情况选择合适的语言。

如果你确定要使用 Rust 完成课程项目，那么你需要注意以下几点：

#. 不要想当然地认为可以用 Rust 以同样的思路和方法实现 C++ 版本中的功能，Rust 语言的特性和 C++ 有很大的不同，你需要重新思考如何使用 Rust 的特性来实现编译器，往往这种思考过程会让你对编译器的原理有更深入的理解，并且写出更加优雅的代码。
#. Rust 编译器是你的朋友，而不是敌人，Rust 编译器会帮助你检测出许多潜在的错误，所以请不要忽略编译器的警告信息，尽量保持代码的可读性和可维护性。
#. 当你遇到 Rust 编译器报错时，不要惊慌，尝试仔细阅读错误信息，理解错误的原因，重新思考你的代码逻辑或者阅读有关的资料，很多时候错误信息会帮助你找到代码中的问题。

参考资料
---------------------

- `Rust 程序设计语言 <https://kaisery.github.io/trpl-zh-cn/>`_
- `Rust 语言圣经 <https://course.rs/about-book.html>`_
- 虎书（Modern Compiler Implementation in C）
- 龙书（Compilers: Principles, Techniques, and Tools）
- `The LALRPOP book <https://lalrpop.github.io/lalrpop/>`_
- `LLVM Programmer's Manual <https://llvm.org/docs/ProgrammersManual.html>`_
- `RISC-V ISA Specifications <https://riscv.org/technical/specifications/>`_
- `ARM Assembly Basics <https://azeria-labs.com/writing-arm-assembly-part-1/>`_