汇编编程
================================================

.. toctree::
    :hidden:
    :maxdepth: 4


你可以参考以下内容学习 RISC-V 编程

- `An Introduction to Assembly Programming with RISC-V <https://riscv-programming.org/book/riscv-book.html>`_ ：较为完整的教程
- `RISC-V Assembly Tutorial <https://www.riscfive.com/2022/04/28/risc-v-assembly-tutorial/>`_ ：更简短的教程

在完成实验的过程中，你也可以通过以下两个文档快速查阅 RISC-V 指令集以及汇编语言的相关知识：

- `RISC-V Card <https://github.com/jameslzhu/riscv-card>`_
- `RISC-V Assembly Programmer's Manual <https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/riscv-asm.md>`_

如果你希望全方位了解 RISC-V 这个 ISA，我们建议你阅读 `RISC-V Specifications <https://riscv.org/technical/specifications/>`_ 中提供的文档。

.. hint::

    相比于 x86_64 和 ARM，RISC-V 的 Specification 可以说是非常简单了，如果你此前有阅读 ISA 文档（如 ARM 或 MIPS）的经验，那么你应该可以通过阅读
    RISC-V Assembly Programmer's Manual 以及 Spec 快速上手 RISC-V 汇编编程。

.. tip::

    就编译课程的要求而言，学习汇编编程最快的方法就是搞明白从 C 编译到 LLVM IR 再到汇编代码翻译到过程，然后自己尝试编写简单的汇编代码。

    在上一节我们提到了 `Compiler Explorer <https://godbolt.org>`_ 这个工具，你可以利用这个工具来查看 C 代码到汇编代码的转换过程。
