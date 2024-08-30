汇编编程
================================================

.. toctree::
    :hidden:
    :maxdepth: 4

系统学习 RISC-V 汇编
-----------------------

你可以参考以下教程学习 RISC-V 编程

- `An Introduction to Assembly Programming with RISC-V <https://riscv-programming.org/book/riscv-book.html>`_ ：较为完整的教程
- `RISC-V Assembly Tutorial <https://www.riscfive.com/2022/04/28/risc-v-assembly-tutorial/>`_ ：更简短的教程
- `RISC-V 手册中文书籍 <http://staff.ustc.edu.cn/~llxx/cod/reference_books/RISC-V-Reader-Chinese-v2p12017.pdf>`_ ：RISC-V 手册（中文版书籍）

在完成实验的过程中，你也可以通过以下两个文档快速查阅 RISC-V 指令集以及汇编语言的相关知识：

- `RISC-V Card <https://github.com/jameslzhu/riscv-card>`_
- `RISC-V Assembly Programmer's Manual <https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc>`_

如果你希望全方位了解 RISC-V 这个 ISA，我们建议你阅读 `RISC-V Specifications <https://riscv.org/technical/specifications/>`_ 中提供的文档。

其中 `The RISC-V Instruction Set Manual <https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf>`_ 里面可以找到 RISC-V 的所有指令集信息，在编译器的实现中，你可能需要 **反复查阅** 这个文档来了解某个指令的具体用法。

.. hint::

    相比于 x86_64 和 ARM，RISC-V 的 Specification 可以说是非常简单了，如果你此前有阅读 ISA 文档（如 ARM 或 MIPS）的经验，那么你应该可以通过阅读
    RISC-V Assembly Programmer's Manual 以及 Spec 快速上手 RISC-V 汇编编程。

.. tip::

    就编译课程的要求而言，学习汇编编程最快的方法就是搞明白从 C 编译到 LLVM IR 再到汇编代码翻译到过程，然后自己尝试编写简单的汇编代码。

    在上一节我们提到了 `Compiler Explorer <https://godbolt.org>`_ 这个工具，你可以利用这个工具来查看 C 代码到汇编代码的转换过程。

RISC-V 汇编简易入门
-----------------------

.. note::

    以下部分与 C++ 版本的实验文档基本一致，可能未针对 Rust 进行特别修改，如果发现问题请及时联系助教。

.. hint::

    此处介绍的 RISC-V 汇编的基础知识并不足以让你完成完整的编译器，我们建议你在需要时参考上面提到的教程进行深入学习。

    你可以通过查阅 `RISC-V 指令手册 <https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf>`_ 来获取完整的 RISC-V 指令集的知识。

    在本学期的所有实验中，我们只需要了解 RISC-V 的基础指令集，以及 M 扩展 (乘除法)，F 和 D 扩展 (浮点数指令集) 即可，所以你只需要阅读手册的一部分即可。

.. note::

    我们本学期实验中，RISC-V 的编译器框架使用的均为 64 位 RISC-V 指令集，而不是 32 位的 (不过 32 位和 64 位差别很小，你可以通过先学习 32 位再转 64 位的方式来学习)。


我们用下面 C 代码为例来介绍 RISC-V 汇编编程

.. code-block:: c

    #include<stdio.h>
    
    int a = 0;
    int b = 0;
    
    int max(int a, int b) {
        if(a >= b) {
            return a;
        } else {
            return b;
        }
    }
    
    int main() {
        scanf("%d %d", &a, &b);
        printf("max is: %d\n", max(a, b));
        return 0;
    }


使用 ``gcc -O3`` 编译后，我们可以得到以下汇编代码：

.. code-block:: asm

        .file   "test.c"
        .option nopic
        .attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"
        .attribute unaligned_access, 0
        .attribute stack_align, 16
        .text
        .align  1
        .globl  max
        .type   max, @function
    max:
        mv  a5,a0
        bge a0,a1,.L2
        mv  a5,a1
    .L2:
        sext.w  a0,a5
        ret
        .size   max, .-max
        .section    .rodata.str1.8,"aMS",@progbits,1
        .align  3
    .LC0:
        .string "%d %d"
        .align  3
    .LC1:
        .string "max is: %d\n"
        .section    .text.startup,"ax",@progbits
        .align  1
        .globl  main
        .type   main, @function
    main:
        addi    sp,sp,-32
        sd  s0,16(sp)
        sd  s1,8(sp)
        lui s0,%hi(a)
        lui s1,%hi(b)
        lui a0,%hi(.LC0)
        addi    a2,s1,%lo(b)
        addi    a1,s0,%lo(a)
        addi    a0,a0,%lo(.LC0)
        sd  ra,24(sp)
        call    __isoc99_scanf
        lw  a1,%lo(b)(s1)
        lw  a0,%lo(a)(s0)
        call    max
        mv  a1,a0
        lui a0,%hi(.LC1)
        addi    a0,a0,%lo(.LC1)
        call    printf
        ld  ra,24(sp)
        ld  s0,16(sp)
        ld  s1,8(sp)
        li  a0,0
        addi    sp,sp,32
        jr  ra
        .size   main, .-main
        .globl  b
        .globl  a
        .section    .sbss,"aw",@nobits
        .align  2
        .type   b, @object
        .size   b, 4
    b:
        .zero   4
        .type   a, @object
        .size   a, 4
    a:
        .zero   4
        .ident  "GCC: () 12.2.0"
        .section    .note.GNU-stack,"",@progbits

下面对一些关键代码进行解释：

1. ``.option nopic`` 表示不使用位置无关的代码。此时，汇编器生成的代码将假定它会被加载到固定的内存地址，对于生成静态链接的二进制文件比较重要，有时可以提高代码的执行效率。
2. ``.attribute arch, "rv64i2p1_m2p0"`` 是用于指定汇编代码所遵循的架构特性和扩展的指令。

   * ``rv64i2p1`` 表示代码遵循 RISC-V 64 位基础指令集，版本为 2.1。
   * ``m2p0`` 表示支持乘法和除法指令，版本为 2.0。其余的指令大家可以通过搜索查找相关意义，此处不再赘述。

3. ``.attribute stack_align, 16`` 表示栈空间需要 16 字节对齐，在我们 RISC-V 版本的编译实验作业中，同样需要遵循这一规定。
4. ``.text`` 表示接下来是代码区。
5. 接下来我们介绍 max 函数中出现的汇编指令：

   * 首先我们需要了解一点，RISC-V 函数调用约定中参数的传递首先使用 ``a0-a7`` 寄存器，如果寄存器不够用，则使用栈进行传递。所以在函数入口处，``a0`` 存放着变量 ``a`` 的值，``a1`` 存放着变量 ``b`` 的值。
   * ``mv a5 a0`` 的含义为 ``a5 = a0``，现在 ``a5`` 中也存放着变量 ``a`` 的值。
   * ``bge a0,a1,.L2`` 的含义为如果 ``a0 ≥ a1``，则跳转至 ``.L2``，对应源代码的 ``if(a>=b)``。
   * ``mv a5 a1`` 的含义为 ``a5 = a1``，这一条语句在 ``a >= b`` 为假的分支执行，表示将最大值 ``a1``（存放着变量 ``b`` 的值，此时有 ``b > a``）赋值给 ``a5``。
   * ``sext.w a0,a5`` 表示将 32 位的 ``a5`` 符号扩展到 64 位，值存放到 ``a0`` 中。
   * ``ret`` 表示函数返回，即源代码中的 ``return``。根据 RISC-V 函数调用约定，使用 ``a0`` 寄存器存放函数的返回值。
   * max 函数最后的 ``.size`` 和 ``.section`` 在我们的编译实验作业中并不需要生成，大家如果感兴趣可以自行查阅资料了解含义。``.align 3`` 表示之前有一段未对齐的代码或数据，汇编器会在这段内容前插入适当数量的填充字节，以确保接下来的数据或代码从一个 8 字节对齐的地址开始。这样做可以提高内存访问效率和程序的运行性能。

6. ``.LC0`` 和 ``.LC1`` 这些标签定义了字符串常量。
7. 接下来介绍 main 函数中出现的汇编指令：

   * ``addi sp,sp,-32`` 首先我们需要知道 ``sp`` 寄存器指向栈顶，该条指令的含义为开辟 32 字节的栈空间。
   * ``sd s0,16(sp)`` 和 ``sd s1,8(sp)`` 首先我们需要知道 RISC-V 的函数调用约定中，被调用者需要保存 ``sp``，``s0-s11`` 寄存器，该函数中使用到了 ``s0`` 和 ``s1`` 寄存器，所以我们在函数开头需要进行保存。``sd s0,16(sp)`` 表示将 ``s0`` 寄存器存储到地址为 ``16+sp`` 的内存中。
   * ``lui s0,%hi(a)`` 表示将 ``a`` 的标签（即全局变量 ``a`` 的地址）的高 20 位放到 ``s0`` 寄存器中，后两条指令同理。
   * ``addu a2,s1,%lo(b)`` 表示 ``a2 = s1 + b`` 的标签（即全局变量 ``b`` 的地址）的低 12 位，后两条指令同理，现在 ``a2`` 寄存器中存放着全局变量 ``b`` 的地址，同理，``a1`` 寄存器中存放着全局变量 ``a`` 的地址，``a0`` 寄存器中存放着字符串常量的地址。
   * ``sd ra,24(sp)`` 表示保存 ``ra`` 寄存器的值，原因是 RISC-V 函数调用约定需要调用者保存 ``ra`` 寄存器的值。
   * ``call __isoc99_scanf`` 表示 ``scanf`` 函数的调用，此时我们回忆一下刚才 ``a0``，``a1``，``a2`` 寄存器中变量的含义以及 RISC-V 函数调用约定，你大概就明白了此时 ``call`` 的含义。（``call`` 指令实际上是一条伪指令，具体做了什么就交给同学们自己探索了）。
   * ``lw a1,%lo(b)(s1)`` 含义为读取内存地址为 ``s1 + %lo(b)`` 的值（实际上，即为全局变量 ``b`` 的地址），并放到 ``a1`` 寄存器中，现在 ``a1`` 寄存器中为全局变量 ``b`` 的值。下一条指令同理，``a0`` 寄存器中为全局变量 ``a`` 的值。
   * ``call max`` 表示调用 ``max`` 函数，此时推荐再回忆一下 ``a0``，``a1`` 寄存器分别表示什么。
   * ``mv a1,a0`` 在函数调用完毕后，``a0`` 存放着函数返回值，此时我们将 ``a0`` 赋值给 ``a1``，即 ``a1`` 保存着 ``max(a,b)`` 的结果，也许你会问为什么这里要多此一举将 ``a0`` 赋值给 ``a1``，当你在下面看到 ``printf`` 函数的调用时你会明白一切（这也是 GCC ``O3`` 优化效果的体现）。
   * ``lui a0，%hi(.LC1)`` 和 ``addi a0,a0,%lo(.LC1)`` 的含义为将 ``.LC1`` 中字符串常量的地址赋值给 ``a0``。
   * ``call printf`` 表示调用 ``printf`` 函数，不妨再回忆一下此时 ``a0``，``a1`` 寄存器的含义。
   * ``ld ra 24(sp)`` 表示恢复之前保存的 ``ra`` 寄存器。
   * ``ld s0,16(sp)`` 和 ``ld s1,8(sp)`` 表示将 ``s0`` 和 ``s1`` 寄存器恢复至调用前的状态。
   * ``li a0，0`` 表示将 ``a0`` 赋值为 0，表示 ``main`` 函数的返回值。
   * ``addi sp,sp,32`` 表示恢复栈空间。
   * ``jr ra`` 表示函数返回，指令含义为跳转到 ``ra`` 寄存器中的地址（``ra`` 寄存器会保存函数的返回地址，此时即使你不知道 ``call`` 这一条伪指令的含义，应该也能猜到 ``call`` 做了什么了）。

8. 最后还有一些全局变量的定义，从文本上很容易理解其含义，这里就不赘述了。
9. 上述的 RISC-V 汇编由 GCC ``O3`` 优化选项编译而成。如果你想了解编译器不做任何优化生成的汇编是什么样的，可以自己使用在环境配置环节安装的 RISC-V 交叉编译 GCC，并使用编译选项 ``O0`` 进行编译。（添加编译选项 ``-S`` 表示生成汇编代码）。

对编写的 RISC-V 汇编进行测试
----------------------------

假设你编写了一个 RISC-V 汇编文件，命名为 ``test.s``，使用下面的命令即可进行测试。

.. code-block:: shell

    riscv64-unknown-linux-gnu-gcc test.s -o test -static
    # 如果你好奇为什么要加 -static，可以取消 -static 后使用 qemu 运行试试，
    # 看看会报什么错误
    # 你可以再根据错误信息去搜索引擎上搜索或者询问 ChatGPT 来了解原因。
    qemu-riscv64 test
