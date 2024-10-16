中间代码生成
================================================

.. toctree::
   :hidden:
   :maxdepth: 4

实验描述
----------------

在前面的实验中，你已经完成了语法分析器和 AST 的构建，接下来是整个编译器编写过程中最重要、工作量最大同时也是难度较大的一部分，
你需要在这一部分中完成对 SysY 语言的类型检查和中间代码生成。你需要从此前解析得到的 AST 生成 LLVM IR，这一工作也将会成为
之后目标代码生成的基础。

实验要求
----------------

#. 在语法分析实验的基础上，遍历 AST，进行类型检查；
#. 生成 LLVM IR；
#. 无需撰写完整实验报告，但需要在雨课堂提交本次实验的链接；

类型检查
----------------

类型检查是编译过程的重要一步，以确保操作对象与操作符相匹配。每一个表达式都有相关联的类型，
如关系运算表达式的类型为布尔型，而计算表达式的类型一般为整型或浮点型等。
类型检查的目的在于找出源代码中不符合类型表达式规定的代码，在最终的代码生成之前报错，
使得程序员根据错误信息对源代码进行修正。

SysY 的类型系统较为简单，你主要需要注意的内容包括以下几点：

#. 检查 ``main`` 函数是否存在；
#. 检查作用域内变量未定义或者重复定义问题；
#. 隐式类型转换问题（包括 ``int`` ``bool`` 和 ``float`` 之间的转换）；
#. 返回值类型问题；
#. 数组维度的问题；

类型检查最简单的实现方式是在建立语法树的过程中进行相应的识别和处理，
也可以在建树完成后，自底向上遍历语法树进行类型检查。类型检查过程中，
父结点需要检查孩子结点的类型，并根据孩子结点类型确定自身类型。
有一些表达式可以在语法制导翻译时就确定类型，比如整数就是整型，
这些表达式通常是语法树的叶结点。而有些表达式则依赖其子表达式确定类型，
这些表达式则是语法树中的内部结点。

下面是一个伪代码，在实现类型检查的时候可以参考：

.. code-block:: rust

   let lhs_ty = lhs.type_check();
   let rhs_ty = rhs.type_check();

   if lhs_ty != rhs_ty {
      // error, or coerce, depending on the situation
   }

   // set symbol type

首先获取两个子表达式的结点类型，判断两个类型是否相同，如果不同则报错或者进行隐式类型转换。

中间代码生成
----------------

中间代码生成是本次实验的重头戏，旨在前继词法分析、语法分析实验的基础上，
将 SysY 源代码翻译为中间代码。中间代码生成主要包含对数据流和控制流两种类型语句的翻译，
数据流包括表达式运算、变量声明与赋值等，控制流包括 if、while、break、continue 等语句。

表达式的翻译
~~~~~~~~~~~~~~~~

TBD

控制流的翻译
~~~~~~~~~~~~~~~~

TBD

代码框架
----------------

.. code-block::

   .
   ├── backend
   ├── backend.rs
   ├── frontend
   │   ├── ast.rs
   │   ├── irgen.rs
   │   ├── parse.rs
   │   ├── preprocess.rs
   │   ├── sysy.lalrpop
   │   └── types.rs
   ├── frontend.rs
   ├── infra
   │   ├── linked_list.rs
   │   └── storage.rs
   ├── infra.rs
   ├── ir
   │   ├── block.rs
   │   ├── context.rs
   │   ├── def_use.rs
   │   ├── func.rs
   │   ├── global.rs
   │   ├── inst.rs
   │   ├── ty.rs
   │   └── value.rs
   ├── ir.rs
   ├── lib.rs
   └── main.rs

本次实验需要阅读的代码更多，你需要阅读 ``ir`` 文件夹下的所有代码， ``frontend/irgen.rs`` 以及
``frontend/ast.rs`` 中与类型检查有关的代码。另外，我们建议你阅读 ``infra`` 模块下的代码，以更好
的理解一些框架实现。你可以从 ``context.rs`` 开始阅读，从 IR 的组织方式出发进行理解。

在预备工作中你应该已经发现，LLVM 是通过全局定义、函数定义、基本块、以及指令来组织 IR 的。此外，
你还需要知道的是，在进行中间代码优化的时候，我们往往需要对基本块、指令做插入、删除、遍历的操作，
对指令中的操作数（Operand）进行修改等。因此，我们使用链表来组织函数中的基本块和基本块中的指令，
链表的实现可以阅读 ``infra`` 中的有关代码，本实验中你并不需要自己实现。

``context.rs`` 中对 ``Context`` 类型的定义如下

.. code-block:: rust

   pub struct Context {
       /// Singleton storage for types.
       pub(super) tys: UniqueArena<TyData>,
       /// Storage for blocks.
       pub(super) blocks: GenericArena<BlockData>,
       /// Storage for instructions.
       pub(super) insts: GenericArena<InstData>,
       /// Storage for functions.
       pub(super) funcs: GenericArena<FuncData>,
       /// Storage for values.
       pub(super) values: GenericArena<ValueData>,
       /// Storage for global variables.
       pub(super) globals: GenericArena<GlobalData>,
 
       /// Target information.
       pub(super) target: TargetInfo,
   }

其中大部分的数据都是用 Arena 的方式存储，在使用的时候只存储一个轻量的 ``ArenaPtr`` ，而不是直接存储数据。
其中类型的存储使用了 ``UniqueArena`` ，以保证同样的两个类型，产生的 ``ArenaPtr`` 是相等的。

在 ``BlockData`` 和 ``InstData`` 中，存储了 ``next`` 和 ``prev`` 这两个字段，用于构成链表。
在 ``FuncData`` 和 ``BlockData`` 中，存储了 ``head`` 和 ``tail`` 这两个字段，分别表示基本块和指令的头尾。

另外，在 ``def_use.rs`` 中，我们设计了两个 Trait 用来描述 Def-Use 维护的接口，其中的内容已经实现，
但是我们建议你仔细阅读其中的内容以确保你能够理解其他的代码。另外，基于这一 Def-Use 链的设计， ``inst.rs``
中实现了一个 ``OperandList`` 类型用以存储指令中的操作数列表。在删除其中任何一个 Operand 的时候，其余操作数对应的
Index 都不会改变，从而保证一个 ``Operand`` 类型的有效性。这个设计可能比较难理解，你需要仔细阅读有关的
代码以理解这一设计。如果你认为对于这一问题有更好的解决方案，可以自行修改并且记录在最终的实验报告中。实际上，
对于 Def-Use 维护这个问题是有很多种写法的，你可以自行探索（如果有时间的话）。

最后，在阅读代码的过程中，要特别注意其中的一些注释内容，尤其是大段的文档注释，我们将一些注意的事项和设计思路
记录在了里面。
