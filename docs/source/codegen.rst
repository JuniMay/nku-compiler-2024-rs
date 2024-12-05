完成编译器构造
================================================

.. toctree::
   :hidden:
   :maxdepth: 4

实验描述
----------------

经过一个学期的学习，编译器的构造来到了最终阶段。本试验中你需要完成中间代码到目标代码到构造，
并且运行你生成的编译器，检测其正确性。

实验流程
----------------

.. important::

   在开始这个实验之前，我们建议你先复习 RISC-V 的 ISA 和 ABI 等相关知识。

后端代码的生成主要分为三个部分：基于虚拟寄存器的代码生成、寄存器分配以及寄存器分配之后函数栈帧调整代码的插入。
虚拟寄存器的代码生成做的工作是将 IR 中的指令转换成目标 ISA 的指令，而不管寄存器的分配情况。
在这个阶段，我们和 IR 中一样，假定这个 CPU 有无限多的寄存器。
因此，在这个阶段我们可以直接将 IR 中的 Value 映射到后端的虚拟寄存器上。

.. important::

   需要注意的是，在代码框架的实现中，IR 中的 Value 并非全部映射到寄存器，而是映射到 ``MOperand`` 这个类型上，其中包括了寄存器、立即数、内存位置等。

   其中尤其重要的是内存位置等问题。在寄存器分配前，由于栈帧没有确定，所以我们只能够先存下中间信息，在寄存器分配之后再修改代码，确定内存中的偏移量。具体的代码参考后面的代码框架讲解。

在完成基于虚拟寄存器的代码生成之后，我们需要将虚拟寄存器映射到物理寄存器上。这个过程就是寄存器分配。
我们的实验并不要求你实现特别复杂的寄存器分配算法。这里给出几种你可以使用的寄存器分配方法：

- Top of Stack Allocation，即栈顶分配。这种方法是最简单的寄存器分配方法，将所有的运算结果保存到栈上，每次需要的时候再从栈上取出。
- Linear Scan，即线性扫描。这种方法是一种比较简单的寄存器分配方法，它会扫描整个函数的 IR，将活跃的变量映射到寄存器上。
- Graph Coloring，即图着色。这种方法是一种比较复杂的寄存器分配方法，它会将活跃变量的图着色，使得相邻的变量不会映射到同一个寄存器上。
- 各种优化（例如带上启发式的线性扫描等）。

这些寄存器分配的算法都能够比较轻松的找到参考资料（除了第一种你需要自己思考怎么实现）。如果你在实现的时候发现问题或者找不到相关资料，请咨询助教。

所有的寄存器分配有一个共同的特点，当寄存器不够用时（或者不想用时）使用栈来辅助存储。所以除了局部变量和传参所需要的栈帧空间外，
我们还需要为栈上存储的 Value 开辟一块空间，而这个空间的大小只能在寄存器分配之后才能确定。所以我们需要在寄存器分配之后才能够确定栈帧调整的代码。
另外，由于调用约定的关系，我们在函数的开头（prologue）和结尾（epilogue）需要插入一些代码来保存和恢复使用过的寄存器的值，而使用哪些寄存器也是在寄存器分配之后才能确定的。

.. important::

   不是所有分配的寄存器都需要保存。你需要阅读 RISC-V 的调用约定，了解哪些寄存器是需要保存的。

代码框架
----------------

.. code-block::

   src
   ├── backend
   │   ├── block.rs    ...... 基本块的定义（与 IR 结构类似）
   │   ├── codegen.rs  ...... 代码生成的规则
   │   ├── context.rs  ...... 代码生成的上下文（与 IR 结构类似）
   │   ├── func.rs     ...... 函数的定义（与 IR 结构类似）
   │   ├── imm.rs      ...... RISC-V 立即数的定义
   │   ├── inst.rs     ...... RISC-V 指令的定义
   │   ├── operand.rs  ...... 后端操作数的定义
   │   └── regs.rs     ...... RISC-V 寄存器的定义
   ├── backend.rs
   ├── frontend
   ├── frontend.rs
   ├── infra
   ├── infra.rs
   ├── ir
   ├── ir.rs
   ├── lib.rs
   └── main.rs

本次代码框架需要修改的内容集中在 ``backend`` 中。大部分结构，如函数、代码块、上下文的结构与
IR 中的组织方式类似，但是在内部存储的信息上会有细微差别。

在 ``inst.rs`` 中定义了 RISC-V 的指令，我们根据指令的格式（寄存器和立即数等）定义了一个枚举类型 ``MInstKind``，
并且在 ``MInst`` 中定义了一系列构造函数，用于生成不同类型的指令。

.. code-block:: rust

   /// Kinds of machine instructions.
   ///
   /// The instructions are classified with its format. This classification is
   /// derived from cranelift.
   pub enum MInstKind {
      /// ALU instructions with two registers (rd and rs) and an immediate.
      AluRRI {
         op: AluOpRRI,
         rd: Reg,
         rs: Reg,
         imm: Imm12,
      },
      /// ALU instructions with three registers (rd, and two rs-s).
      AluRRR {
         op: AluOpRRR,
         rd: Reg,
         rs1: Reg,
         rs2: Reg,
      },
      /// Load instructions.
      Load { op: LoadOp, rd: Reg, loc: MemLoc },
      /// Store instructions.
      Store { op: StoreOp, rs: Reg, loc: MemLoc },
      /// Load immediate pseudo instruction.
      Li { rd: Reg, imm: u64 },
      /// Jump instructions.
      J { target: MBlock },
      // TODO: add more instructions as you need.
   }

在 ``regs.rs`` 中我们定义了一个 ``Reg`` 的枚举。其中的 ``P`` 和 ``V`` 分别代表了物理寄存器和虚拟寄存器。

.. code-block:: rust
   
   /// The register.
   ///
   /// It can be either a physical register or a virtual register.
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
   pub enum Reg {
      P(PReg),
      V(VReg),
   }

为了将 IR 中的 Value 映射到后端代码中的实体，我们定义了 ``MOperand`` 这个类型，用来表示后端代码中的操作数。其代码位于 ``operand.rs`` 中。
其中最重要的是 ``MemLoc`` 这个类型，其中定义了一系列不同的内存位置。我们主要将内存位置分为三类：

.. code-block:: rust

   pub enum MemLoc {
      RegOffset { base: Reg, offset: i64 },
      Slot { offset: i64 },
      Incoming { offset: i64 },
   }

其中 ``RegOffset`` 是最一般的情况，同时也是我们寄存器分配完成之后需要修改得到的内存位置表示，只有这个表示才能够输出到最终的结果（其他的两种类型也能够输出，但是只是用于 Debug 目的）。
``Slot`` 表示函数内部局部变量的存储位置，我们只记录了一个 ``offset``，因为所有的局部变量都可以用同一个基址寄存器来表示（我们注释中使用了 fp），寄存器分配之后只需要将 ``offset`` 进行修改，加上对应的寄存器即可改为 ``RegOffset``。
最后一种 ``Incoming`` 对应的则是函数传入的参数，这些参数由调用者（Caller）负责传入，并且地址比 fp 的值更大（栈上位置更低）。

.. important::

   代码的注释中给出了一个函数栈帧的示意图：
   
   .. code-block::

                   param by stack #n
                          ...          -----> maybe depends on the calling convention
                   param by stack #0
            high +-------------------+ <-- frame pointer
                 |  Saved Registers  |
             |   +- - - - - - - - - -+
             |   | (maybe alignment) |
             |   +- - - - - - - - - -+ <-- start of local slots
             |   |                   |
       grow  |   |  Local Variables  |
             |   |                   |
             V   +- - - - - - - - - -+
                 |  for arg passing  |
            low  +-------------------+ <-- stack pointer
   
   我们建议你在实现的过程中充分理解这个示意图以及其注释中对应的说明。

   另外需要注意的是 RISC-V 的调用约定还要求栈对齐，在插入栈帧调整代码的时候需要注意这一点。

最后，你需要在 ``codegen.rs`` 中添加代码翻译的逻辑，包括寄存器分配和寄存器分配之后修改内存位置的逻辑。

另外，此处给出一个实现了后端之后 ``main.rs`` 中编译过程的逻辑

.. code-block:: rust

   let source = include_str!("../tests/sysy/basic.sy");

   // Frontend
   let src = preprocess(source);
   let mut ast = SysYParser::new().parse(&src).unwrap();
   ast.type_check();

   // IRGen
   let ir = irgen(&ast, 8);
   println!("{}", ir);

   // Initialize the codegen context.
   let mut codegen_ctx = CodegenContext::new(&ir);

   // Set architecture string.
   codegen_ctx.mctx_mut().set_arch("rv64imafdc_zba_zbb");

   // Do the codegen and emit virtual register assembly.
   codegen_ctx.codegen();
   println!("{}", codegen_ctx.mctx().display());

   // Do the register allocation.
   codegen_ctx.regalloc();
   println!("{}", codegen_ctx.mctx().display());

   // Additional work after register allocation.
   codegen_ctx.after_regalloc();

   // Emit the final assembly.
   let mctx = codegen_ctx.finish();
   println!("{}", mctx.display());

大部分需要修改或添加代码的部分我们都用 ``TODO`` 进行了注释，如果你发现有遗漏的地方，请及时联系助教。

线下提问示例
----------------

#. 请说明你是如何实现寄存器分配的。
#. 请说明你是如何实现栈帧调整的。
#. 请说明你对于 Phi 指令的处理（如果实现了并且需要 Phi 的话）。
#. 请说明你对于函数调用的处理。
