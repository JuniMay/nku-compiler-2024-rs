中间代码生成
================================================

.. toctree::
   :hidden:
   :maxdepth: 4

实验描述
----------------

在前面的实验中，你已经完成了语法分析器和 AST 的构建，接下来是整个编译器编写过程中最重要、工作量最大同时也是难度较大的一部分，
你需要在这一部分中完成对 SysY 语言的类型检查和中间代码生成。你需要从此前解析得到的 AST 生成 LLVM IR，
这一工作也将会成为之后目标代码生成和中间代码优化的基础。

.. important::

   数组、浮点等内容属于进阶要求，但是本次实验的分数有一部分来自于进阶要求的实现。如果你希望获得更高的分数，我们建议你尽量实现这些内容。

   具体的分数要求可以参考学期初给出的《上机大作业总体要求》。

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
数据流包括表达式运算、变量声明与赋值等，控制流包括 ``if`` ``while`` ``break`` ``continue`` 等语句。

表达式的翻译
~~~~~~~~~~~~~~~~

表达式翻译的代码位于 ``irgen.rs`` 中 ``IrGenContext::gen_local_expr`` 的部分

.. code-block:: rust

    // Generate a new local expression in ir given an expression in AST.
    fn gen_local_expr(&mut self, expr: &Expr) -> Option<Value> {
        use BinaryOp as Bo;

        let curr_block = self.curr_block.unwrap();

        match &expr.kind {
            // Constants -> generate a local constant value
            ExprKind::Const(v) => Some(self.gen_local_comptime(v)),
            // Binary operations -> generate the operation
            ExprKind::Binary(op, lhs, rhs) => match op {
                Bo::Add | Bo::Sub | Bo::Mul | Bo::Div => {
                    let lhs = self.gen_local_expr(lhs).unwrap(); // Generate lhs
                    let rhs = self.gen_local_expr(rhs).unwrap(); // Generate rhs

                    let lhs_ty = lhs.ty(&self.ctx);

                    let inst = match op {
                        // Generate add instruction
                        Bo::Add => Inst::add(&mut self.ctx, lhs, rhs, lhs_ty),
                        // TODO: Implement other binary operations
                        Bo::Sub => {
                            todo!("implement sub");
                        }
                        Bo::Mul => {
                            todo!("implement mul");
                        }
                        Bo::Div => {
                            todo!("implement div");
                        }
                    };

                    // Push the instruction to the current block
                    curr_block.push_back(&mut self.ctx, inst).unwrap();
                    Some(inst.result(&self.ctx).unwrap())
                }
            },

            // ...
        }
    }

这里首先匹配表达式的种类，然后对其子节点生成中间代码，得到 IR 层级的 Value，然后再生成当前层级的指令，
并插入到当前基本块的尾部。

控制流的翻译
~~~~~~~~~~~~~~~~

控制流的翻译代码位于 ``irgen.rs`` 中 ``impl IrGen for Stmt`` 的部分。主要需要关注的是条件表达式中的短路求值的问题。
由于 SysY 的逻辑短路的语义只会出现在条件判断中（不会出现 ``int b = a || c;`` 这样的语句），所以可以根据表达式的与、或关系直接创建条件为真和假时进一步处理的 block，
以及两个 block 最终合并得到结果的 block。创建完 block 之后再设置当前块以正确插入指令。

具体的翻译方式可以参考 `这个代码 <https://godbolt.org/z/j8xYGP97G>`_ 。

.. tip::

   你可以使用 godbolt 来查看一些简单的 C 语言代码的翻译结果，以帮助你理解中间代码生成的过程。
   如果你选择 clang 编译器，你还可以查看中间代码的控制流图。

符号表
----------------

为了处理 SysY 中的作用域关系，我们需要一个带有层级关系的符号表。符号表的定义位于 ``ast.rs`` 中，其定义如下

.. code-block:: rust

    #[derive(Default)]
    pub struct SymbolTable {
       /// Stack of scopes.
       /// Each scope has its own hashmap of symbols.
       stack: Vec<HashMap<String, SymbolEntry>>,
 
       /// The current return type of the function.
       pub curr_ret_ty: Option<Type>,
    }

其中最主要的内容是用于处理作用域的 ``stack`` 字段，每一级作用域都有一个 ``HashMap`` 用于存储符号到其属性的映射。 ``SymbolEntry`` 包括了三个内容，
分别是符号在 SysY 中的类型，编译期的值（如果有的话），以及 IR 生成的结果（如果已经生成的话）。

.. code-block:: rust

    #[derive(Debug)]
    pub struct SymbolEntry {
        /// Type of the symbol.
        pub ty: Type,
        /// The possible compile time value of the symbol.
        pub comptime: Option<ComptimeVal>,
        /// The IR value of the symbol.
        /// Its generated during IR generation.
        pub ir_value: Option<IrGenResult>,
    }

代码框架的实现在类型检查和中间代码生成的时候实际上共创建了两个符号表。类型检查的时候我们只关心符号的类型和编译期常量，
而在中间代码生成的时候需要关心符号的 IR 生成结果。再进入一个新的作用域的时候，我们会在符号表的栈中压入一个新的 HashMap，
并在退出作用域的时候弹出这个 HashMap。

.. note::

   代码框架中实现的符号表实际上是一次性的，遍历完一遍语法树之后之前的内容都会被弹出。如果你有时间也可以尝试用更好的方式实现符号表。

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
``frontend/ast.rs`` 中与类型检查有关的代码。另外，我们建议你阅读 ``infra`` 模块下的代码，
以更好地理解一些框架实现。你可以从 ``context.rs`` 开始阅读，从 IR 的组织方式出发进行理解。

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
中实现了一个 ``OperandList`` 类型用以存储指令中的操作数列表。在删除其中任何一个 Operand 的时候，
其余操作数对应的 Index 都不会改变，从而保证一个 ``Operand`` 类型的有效性。这个设计可能比较难理解，
你需要仔细阅读有关的代码以理解这一设计。如果你认为对于这一问题有更好的解决方案，可以自行修改并且记录在最终的实验报告中。实际上，
对于 Def-Use 维护这个问题是有很多种写法的，你可以自行探索（如果有时间的话）。

.. note::

   中间代码生成阶段你并不会接触到操作数的替换、删除等操作，但是这些操作在优化的时候会非常频繁。
   如果你这次时间充裕，可以提前熟悉其中的内容。

最后，在阅读代码的过程中，要特别注意其中的一些注释内容，尤其是大段的文档注释，我们将一些注意的事项和设计思路记录在了里面。

线下提问示例
----------------

#. 请以 ``int a[5][4][3] =  {{{2,3},6,7},7,8,11};`` 说明对数组初始化语法的处理方式。
#. ``int a[5][4][3] = {{{2,3},6,7,5,4,3,2,11,2,4,5},7,8,11};`` 该初始化是否合法，请说明理由。
#. 请说明你是如何处理 ``int`` ``float`` ``bool`` 类型的隐式转换的，简单说明 ``int a = 5; int b = a + 3.5 + !a`` 的语法树结构，并说明各节点上的类型。
#. 请说明对于局部变量 ``int a[5][4][3] = {{{2,3},1}};`` 应当如何生成 LLVM IR。
#. 请说明符号表的设计方式，以及如何处理作用域的问题。
#. 请说明你是如何在语义分析步骤中检查 SysY 库函数调用是否合法的。
#. 请说明你是如何实现短路求值的控制流翻译的。
