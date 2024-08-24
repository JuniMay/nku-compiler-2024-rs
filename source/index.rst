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