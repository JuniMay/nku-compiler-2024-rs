//! IR generation from AST.

use super::ast::{
    self,
    BinaryOp,
    BlockItem,
    CompUnit,
    ComptimeVal as Cv, // alias, represents conptime values in the AST
    ConstDecl,
    ConstDef,
    Decl,
    Expr,
    ExprKind,
    ExprStmt,
    FuncDef,
    FuncFParam,
    Item,
    ReturnStmt,
    Stmt,
    SymbolEntry,
    SymbolTable,
    VarDecl,
    VarDef,
};
use super::types::{Type, TypeKind as Tk};
use super::{ArrayIdent, ArrayVal};
use crate::frontend::ast::{FuncCall, LVal, UnaryOp};
use crate::infra::linked_list::{LinkedListContainer, LinkedListCursor, LinkedListNode};
use crate::infra::storage::ArenaPtr;
use crate::ir::{
    Block, ConstantValue, Context, Func, Global, Inst, TargetInfo, Ty, TyData, Value, ValueKind,InstKind,
};


/// Generate IR from the AST.
pub fn irgen(ast: &CompUnit, pointer_width: u8) -> Context {
    let mut irgen = IrGenContext::default();

    // Set pointer width for target platform
    irgen.ctx.set_target_info(TargetInfo {
        ptr_size: pointer_width as u32,
    });

    // Generate IR
    ast.irgen(&mut irgen);

    // Transfer ownership of the generated IR.
    irgen.finish()
}

/// Generated IR result.
/// Its used to map AST nodes to IR values.
/// It can be either a Global or a Value.
#[derive(Debug, Clone, Copy)]
pub enum IrGenResult {
    Global(Global),
    Value(Value),
}

impl IrGenResult {
    /// Get the value if it is a value.
    ///
    /// # Panics
    /// - Panics if it is a global.
    pub fn unwrap_value(self) -> Value {
        match self {
            IrGenResult::Value(val) => val,
            IrGenResult::Global(_) => unreachable!("expected value"), //?iakke
        }
    }
}

/// IR generation context.
#[derive(Default)]
pub struct IrGenContext {
    pub ctx: Context,

    // Symbol table
    pub symtable: SymbolTable,

    // Current function and block
    pub curr_func: Option<Func>,
    pub curr_func_name: Option<String>,
    pub curr_block: Option<Block>,

    // Stacks for loop control flow.
    pub loop_entry_stack: Vec<Block>,
    pub loop_exit_stack: Vec<Block>,

    // Return block and slot
    pub curr_ret_slot: Option<Value>,
    pub curr_ret_block: Option<Block>,
}

impl IrGenContext {
    /// Consume the context and return the generated IR.
    pub fn finish(self) -> Context {
        self.ctx
    }

    // Generate a new global constant value in ir given a comptime value in AST.
    fn gen_global_comptime(&mut self, val: &Cv) -> ConstantValue {
        //从const里面提取value
        match val {
            Cv::Bool(a) => ConstantValue::i1(&mut self.ctx, *a),
            Cv::Char(a) => ConstantValue::i8(&mut self.ctx, *a as i8),
            Cv::Int(a) => ConstantValue::i32(&mut self.ctx, *a),
            Cv::Float(a) => ConstantValue::f32(&mut self.ctx, *a), //iakkefloattest,origin:f32
            Cv::Array(ty, vals) => {
                let elem_ty = self.gen_type(ty);
                let mut elems = Vec::new();
                for elem in vals {
                    elems.push(self.gen_global_comptime(elem));
                }
                ConstantValue::array(&mut self.ctx, elem_ty, elems)
            }
            Cv::Undef(ty) => {
                let ir_ty = self.gen_type(ty);
                ConstantValue::undef(&mut self.ctx, ir_ty)
            }
        }
    }

    // Gerate a new type in ir given a type in AST.
    fn gen_type(&mut self, ty: &Type) -> Ty {
        match ty.kind() {
            Tk::Void => Ty::void(&mut self.ctx),
            Tk::Bool => Ty::i1(&mut self.ctx),
            Tk::Char => Ty::i8(&mut self.ctx),
            Tk::Int => Ty::i32(&mut self.ctx),
            Tk::Float => Ty::f32(&mut self.ctx), //iakkefloattest,origin:f32
            Tk::Array(ty, size) => {
                let ir_ty = self.gen_type(ty);
                Ty::array(&mut self.ctx, ir_ty, size.clone())
            }
            Tk::Func(params, ret) => {
                let mut ir_params = Vec::new();
                for param in params {
                    ir_params.push(self.gen_type(param));
                }
                let ir_ret = self.gen_type(ret);
                Ty::func(&mut self.ctx, ir_ret, ir_params)
            }
            Tk::Ptr(ty) => {
                let ir_ty = self.gen_type(ty);
                Ty::ptr(&mut self.ctx, Some(ir_ty))
            }
            Tk::Args => Ty::args(&mut self.ctx),
        }
    }

    // Generate a new base_type of Array in AST.
    fn gen_array_base_type(&mut self, ty: &Type) -> (Ty, Ty) {
        match ty.kind() {
            Tk::Array(new_ty, _) => (self.gen_type(ty), self.gen_array_base_type(new_ty).1),
            _ => {
                let ty = self.gen_type(ty);
                (ty.clone(), ty)
            }
        }
    }

    // Generate a new local constant value in ir given a comptime value in AST.
    fn gen_local_comptime(&mut self, val: &Cv) -> Value {
        match val {
            Cv::Bool(a) => Value::i1(&mut self.ctx, *a),
            Cv::Int(a) => Value::i32(&mut self.ctx, *a),
            Cv::Char(a) => Value::i8(&mut self.ctx, *a as i8),
            Cv::Float(a) => Value::f32(&mut self.ctx, *a), //iakkefloattest,origin:f32
            Cv::Array(ty, vals) => {
                let elem_ty = self.gen_type(ty);
                let mut elems = Vec::new();
                for elem in vals {
                    elems.push(self.gen_local_comptime(elem));
                }
                Value::array(&mut self.ctx, elem_ty, elems)
            }
            Cv::Undef(ty) => {
                let ir_ty = self.gen_type(ty);
                Value::undef(&mut self.ctx, ir_ty)
            }
        }
    }

    // Generate a new local expression in ir given an expression in AST.
    fn gen_local_expr(&mut self, expr: &Expr) -> Option<Value> {
        use BinaryOp as Bo;

        let curr_block = self.curr_block.unwrap();

        match &expr.kind {
            // Constants -> generate a local constant value
            ExprKind::Const(v) => Some(self.gen_local_comptime(v)),
            // Binary operations -> generate the operation
            ExprKind::Binary(op, lhs, rhs) => match op {
                Bo::Add
                | Bo::Sub
                | Bo::Mul
                | Bo::Div
                | Bo::Mod
                | Bo::Lt
                | Bo::Le
                | Bo::Gt
                | Bo::Ge
                | Bo::Eq
                | Bo::Ne
                | Bo::And
                | Bo::Or => {
                    let lhs = self.gen_local_expr(lhs).unwrap(); // Generate lhs
                    let rhs = self.gen_local_expr(rhs).unwrap(); // Generate rhs

                    let lhs_ty = lhs.ty(&self.ctx);

                    let inst = match op {
                        // Generate add instruction
                        Bo::Add => Inst::add(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Sub => Inst::sub(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Mul => Inst::mul(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Div => Inst::sdiv(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Mod => Inst::srem(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Lt => Inst::lt(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Le => Inst::le(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Gt => Inst::gt(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Ge => Inst::ge(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Eq => Inst::eq(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Ne => Inst::ne(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::And => Inst::and(&mut self.ctx, lhs, rhs, lhs_ty),
                        Bo::Or => Inst::or(&mut self.ctx, lhs, rhs, lhs_ty),
                    };

                    // Push the instruction to the current block
                    curr_block.push_back(&mut self.ctx, inst).unwrap();
                    Some(inst.result(&self.ctx).unwrap())
                }
            },
            // Unary operations -> generate the operation
            ExprKind::Unary(op, expr) => {
                let ty = self.gen_type(expr.ty.as_ref().unwrap());
                let expr = self.gen_local_expr(expr).unwrap();
                let inst = match op {
                    // TOD?: Implement unary operations
                    UnaryOp::Neg => Inst::neg(&mut self.ctx, expr, ty),
                    UnaryOp::Not => Inst::not(&mut self.ctx, expr, ty),
                };
                curr_block.push_back(&mut self.ctx, inst).unwrap();
                Some(inst.result(&self.ctx).unwrap())
            }
            // LValues -> Get the value
            ExprKind::LVal(LVal { ident }) => {
                println!("ident: {}", ident);
                println!("{:#?}", self.symtable);
                // Look up the symbol in the symbol table to get the IR value
                let entry = self.symtable.lookup(ident).unwrap();
                let ir_value = entry.ir_value.unwrap();

                let ir_base_ty = self.gen_type(&entry.ty.clone());

                let slot = if let IrGenResult::Global(slot) = ir_value {
                    // If the value is a global, get the global reference
                    let name = slot.name(&self.ctx).to_string();
                    let value_ty = slot.ty(&self.ctx);
                    Value::global_ref(&mut self.ctx, name, value_ty)
                } else if let IrGenResult::Value(slot) = ir_value {
                    // If the value is a local, get the value
                    slot
                } else {
                    unreachable!()
                };

                if slot.is_param(&self.ctx) {
                    // If the value is a parameter, just return the value
                    Some(slot)
                } else {
                    let zero = Value::i32(&mut self.ctx, 0);
                    let indices = vec![zero, zero.clone()];
                    if let TyData::Array { .. } = ir_base_ty.kind(&self.ctx) {
                        println!("array access");
                        let getelmentptr =
                            Inst::getelementptr(&mut self.ctx, ir_base_ty, slot, indices);
                        curr_block.push_back(&mut self.ctx, getelmentptr).unwrap();
                        Some(getelmentptr.result(&self.ctx).unwrap())
                    } else {
                        // Otherwise, we need to load the value, generate a load instruction
                        let load = Inst::load(&mut self.ctx, slot, ir_base_ty);
                        curr_block.push_back(&mut self.ctx, load).unwrap();
                        Some(load.result(&self.ctx).unwrap())
                    }
                }
            }
            ExprKind::ArrayAccess(ident, pos) => {
                if let ExprKind::LVal(LVal { ident }) = &ident.as_ref().kind {
                    let entry = self.symtable.lookup(ident).unwrap();
                    let ir_value = entry.ir_value.unwrap();

                    let (ir_ty, ir_base_ty) = self.gen_array_base_type(&entry.ty.clone());

                    let slot = if let IrGenResult::Global(slot) = ir_value {
                        // If the value is a global, get the global reference
                        let name = slot.name(&self.ctx).to_string();
                        let value_ty = slot.ty(&self.ctx);
                        Value::global_ref(&mut self.ctx, name, value_ty)
                    } else if let IrGenResult::Value(slot) = ir_value {
                        // If the value is a local, get the value
                        slot
                    } else {
                        unreachable!()
                    };

                    let mut pos_values =
                        vec![self.gen_local_expr(&Expr::const_(Cv::Int(0))).unwrap()];
                    for p in pos.iter() {
                        pos_values.push(self.gen_local_expr(p).unwrap());
                    }
                    let ptr_dst = Inst::getelementptr(&mut self.ctx, ir_ty, slot, pos_values);
                    curr_block.push_back(&mut self.ctx, ptr_dst).unwrap();
                    let slot = ptr_dst.result(&self.ctx).unwrap();

                    if slot.is_param(&self.ctx) {
                        // If the value is a parameter, just return the value
                        Some(slot)
                    } else {
                        // Otherwise, we need to load the value, generate a load instruction
                        let load = Inst::load(&mut self.ctx, slot, ir_base_ty);
                        curr_block.push_back(&mut self.ctx, load).unwrap();
                        Some(load.result(&self.ctx).unwrap())
                    }
                } else {
                    unreachable!("array access should be a lval");
                }
            }
            ExprKind::Coercion(old_expr) => {
                // HACK: Implement coercion generation
                let expect_ty = expr.ty.as_ref().unwrap().kind();
                let origin_ty = old_expr.ty.as_ref().unwrap().kind();

                if expect_ty == origin_ty {
                    return Some(self.gen_local_expr(old_expr).unwrap());
                }

                match (expect_ty, origin_ty) {
                    (Tk::Bool, Tk::Int) => {
                        let val = self.gen_local_expr(old_expr).unwrap();
                        let ty = Ty::i1(&mut self.ctx);
                        let rhs = Value::i32(&mut self.ctx, 0);
                        let inst = Inst::eq(&mut self.ctx, val, rhs, ty);
                        curr_block.push_back(&mut self.ctx, inst).unwrap();
                        Some(inst.result(&self.ctx).unwrap())
                    }
                    (Tk::Bool, Tk::Float) => {
                        let val = self.gen_local_expr(old_expr).unwrap();
                        let ty = Ty::i1(&mut self.ctx);
                        let rhs = Value::f32(&mut self.ctx, 0.0); //iakkefloattest,origin:f32
                        let inst = Inst::eq(&mut self.ctx, val, rhs, ty);
                        curr_block.push_back(&mut self.ctx, inst).unwrap();
                        Some(inst.result(&self.ctx).unwrap())
                    }
                    (Tk::Char, Tk::Int) => {
                        let val = self.gen_local_expr(old_expr).unwrap();
                        let ty = Ty::i8(&mut self.ctx);
                        let inst = Inst::trunc(&mut self.ctx, val, ty);
                        curr_block.push_back(&mut self.ctx, inst).unwrap();
                        Some(inst.result(&self.ctx).unwrap())
                    }
                    (Tk::Int, Tk::Bool) => {
                        let val = self.gen_local_expr(old_expr).unwrap();
                        let ty = Ty::i32(&mut self.ctx);
                        let inst = Inst::zext(&mut self.ctx, val, ty);
                        curr_block.push_back(&mut self.ctx, inst).unwrap();
                        Some(inst.result(&self.ctx).unwrap())
                    }
                    (Tk::Int, Tk::Char) => {
                        let val = self.gen_local_expr(old_expr).unwrap();
                        let ty = Ty::i32(&mut self.ctx);
                        let inst = Inst::zext(&mut self.ctx, val, ty);
                        curr_block.push_back(&mut self.ctx, inst).unwrap();
                        Some(inst.result(&self.ctx).unwrap())
                    }
                    (Tk::Int, Tk::Float) => {
                        let val = self.gen_local_expr(old_expr).unwrap();
                        let ty = Ty::i32(&mut self.ctx);
                        let inst = Inst::fptosi(&mut self.ctx, val, ty);
                        curr_block.push_back(&mut self.ctx, inst).unwrap();
                        Some(inst.result(&self.ctx).unwrap())
                    }
                    (Tk::Float, Tk::Bool) => {
                        let val = self.gen_local_expr(old_expr).unwrap();
                        let ty = Ty::f32(&mut self.ctx); //iakkefloattest,origin:f32
                        let inst = Inst::uitofp(&mut self.ctx, val, ty);
                        curr_block.push_back(&mut self.ctx, inst).unwrap();
                        Some(inst.result(&self.ctx).unwrap())
                    }
                    (Tk::Float, Tk::Int) => {
                        let val = self.gen_local_expr(old_expr).unwrap();
                        let ty = Ty::f32(&mut self.ctx); //iakkefloattest,origin:f32
                        let inst = Inst::sitofp(&mut self.ctx, val, ty);
                        curr_block.push_back(&mut self.ctx, inst).unwrap();
                        Some(inst.result(&self.ctx).unwrap())
                    }
                    _ => {
                        println!("expr: {:#?}", expr);
                        println!("expect_ty: {:#?}", expect_ty);
                        println!("origin_ty: {:#?}", origin_ty);
                        unreachable!("invalid coercion");
                    }
                }
            }
            ExprKind::FuncCall(FuncCall { ident, args }) => {
                // HACK: Implement function call generation
                let entry = self.symtable.lookup(ident).unwrap();
                let ret_ty = expr.ty.as_ref().unwrap();

                let value_ty = {
                    let entry_ty = entry.ty.clone();
                    if let Tk::Func(params, ret) = entry_ty.kind() {
                        let params = params.iter().map(|ty| self.gen_type(ty)).collect();
                        let ret = self.gen_type(ret);
                        Ty::func(&mut self.ctx, ret, params)
                    } else {
                        unreachable!("invalid function type");
                    }
                };
                let func = Value::global_ref(&mut self.ctx, ident.clone(), value_ty);

                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.gen_local_expr(arg).unwrap());
                }

                let ret_ty = self.gen_type(ret_ty);
                let inst = Inst::call(&mut self.ctx, func, ir_args, ret_ty);
                curr_block.push_back(&mut self.ctx, inst).unwrap();

                inst.result(&self.ctx)
            }
        }
    }

    // HACK:使用dfs实现数组的生成
    fn gen_local_array(&mut self, ty: &Type, arr: &ArrayVal) -> Option<Value> {
        let ty = self.gen_type(ty);

        let (_, arr) = self.dfs(ty, arr);

        Some(arr)
    }

    fn dfs(&mut self, ty: Ty, arr: &ArrayVal) -> (Ty, Value) {
        match arr {
            ArrayVal::Val(val) => {
                let val = self.gen_local_expr(val).unwrap();
                // println!("{}", val.display(&self.ctx, true));
                (ty, val)
            }
            ArrayVal::Vals(arr) => {
                let mut elems = Vec::new();
                for elem in arr {
                    elems.push(self.dfs(ty, elem));
                }
                let arr = Value::array(
                    &mut self.ctx,
                    elems[0].0.clone(),
                    elems.iter().map(|(_, val)| val.clone()).collect(),
                );
                // println!("{}", arr.display(&self.ctx, true));
                let ty = Ty::array(&mut self.ctx, elems[0].0.clone(), elems.len());
                (ty, arr)
            }
        }
    }

    // Generate the system library function definitions.
    fn gen_sysylib(&mut self) {
        // TODO: Implement gen_sysylib
        // Since the system library is linked in the linking phase, we just need
        // to generate declarations here.

        let sysy_funcs = {
            vec![
                ("getint", Type::func(vec![], Type::int())),
                ("getch", Type::func(vec![], Type::char())),
                ("getfloat", Type::func(vec![], Type::float())),
                (
                    "getarray",
                    Type::func(vec![Type::ptr(Type::int())], Type::int()),
                ),
                (
                    "getfarray",
                    Type::func(vec![Type::ptr(Type::float())], Type::int()),
                ),
                ("putint", Type::func(vec![Type::int()], Type::void())),
                ("putch", Type::func(vec![Type::int()], Type::void())),
                ("putfloat", Type::func(vec![Type::float()], Type::void())),
                (
                    "putarray",
                    Type::func(vec![Type::int(), Type::ptr(Type::int())], Type::void()),
                ),
                (
                    "putfarray",
                    Type::func(vec![Type::int(), Type::ptr(Type::float())], Type::void()),
                ),
                (
                    "putf",
                    Type::func(vec![Type::ptr(Type::char()), Type::args()], Type::void()),
                ),
                ("starttime", Type::func(vec![], Type::void())),
                ("stoptime", Type::func(vec![], Type::void())),
            ]
        };

        for (name, ty) in sysy_funcs {
            if let Tk::Func(params, ret_ty) = ty.kind() {
                let ir_ret_ty = self.gen_type(&ret_ty);
                let func = Func::new(&mut self.ctx, name.to_string(), ir_ret_ty, Some(false));
                for param in params {
                    let ir_param_ty = self.gen_type(param);
                    func.add_param(&mut self.ctx, ir_param_ty);
                }
                self.symtable.insert(
                    name,
                    SymbolEntry {
                        ty,
                        comptime: None,
                        ir_value: None,
                    },
                );
            }
        }
    }
}

pub trait IrGen {
    fn irgen(&self, irgen: &mut IrGenContext);
}

impl IrGen for CompUnit {
    // Generate IR for the compilation unit.
    fn irgen(&self, irgen: &mut IrGenContext) {
        // Enter the global scope
        irgen.symtable.enter_scope();
        // Generate system library function definitions
        irgen.gen_sysylib();
        // Generate IR for each item in the compilation unit
        for item in &self.items {
            item.irgen(irgen);
        }
        // Leave the global scope
        irgen.symtable.leave_scope();
    }
}

impl IrGen for Item {
    // Generate IR for an item.
    fn irgen(&self, irgen: &mut IrGenContext) {
        match self {
            Item::Decl(decl) => match decl {
                Decl::ConstDecl(ConstDecl { ty, defs }) => {
                    for def in defs {
                        match def {
                            ConstDef::Val(ident, init) => {
                                let comptime = init
                                    .try_fold(&irgen.symtable)
                                    .expect("global def expected to have constant initializer");
                                let constant = irgen.gen_global_comptime(&comptime);
                                let slot = Global::new(
                                    &mut irgen.ctx,
                                    format!("__GLOBAL_CONST_{}", ident),
                                    constant,
                                    true,
                                );
                                irgen.symtable.insert(
                                    ident.clone(),
                                    SymbolEntry {
                                        ty: init.ty().clone(),
                                        comptime: Some(comptime),
                                        ir_value: Some(IrGenResult::Global(slot)),
                                    },
                                );
                            }
                            ConstDef::Array(arr_ident, init) => {
                                let comptime = init
                                    .try_fold(&irgen.symtable)
                                    .expect("global def expected to have constant initializer");

                                // println!("init:-----------\n{:#?}", init);
                                let array_ty = make_array(ty.clone(), 0, &arr_ident.size).unwrap();
                                let value = irgen.gen_global_comptime(&comptime.to_comptimeval(ty));
                                let slot = Global::new(
                                    &mut irgen.ctx,
                                    format!("__GLOBAL_CONST_{}", arr_ident.ident),
                                    value,
                                    true,
                                );

                                irgen.symtable.insert(
                                    arr_ident.ident.clone(),
                                    SymbolEntry {
                                        ty: array_ty,
                                        comptime: Some(comptime.to_comptimeval(ty)),
                                        ir_value: Some(IrGenResult::Global(slot)),
                                    },
                                );
                            }
                        }
                    }
                }
                Decl::VarDecl(VarDecl { ty, defs }) => {
                    for def in defs {
                        match def {
                            VarDef::Val(ident, init) => {
                                let comptime = init
                                    .as_ref()
                                    .unwrap() // None should have been assigned with undef
                                    .try_fold(&irgen.symtable)
                                    .expect("global def expected to have constant initializer");
                                let constant = irgen.gen_global_comptime(&comptime);
                                let slot = Global::new(
                                    &mut irgen.ctx,
                                    format!("__GLOBAL_VAR_{}", ident),
                                    constant,
                                    false,
                                );
                                irgen.symtable.insert(
                                    ident.clone(),
                                    SymbolEntry {
                                        ty: init.as_ref().unwrap().ty().clone(),
                                        comptime: Some(comptime),
                                        ir_value: Some(IrGenResult::Global(slot)),
                                    },
                                );
                            }
                            VarDef::Array(arr_ident, init) => {
                                let comptime = init
                                    .as_ref()
                                    .unwrap()
                                    .try_fold(&irgen.symtable)
                                    .expect("global def expected to have constant initializer");

                                // println!("init:-----------\n{:#?}", init);
                                let array_ty = make_array(ty.clone(), 0, &arr_ident.size).unwrap();
                                let value = irgen.gen_global_comptime(&comptime.to_comptimeval(ty));
                                let slot = Global::new(
                                    &mut irgen.ctx,
                                    format!("__GLOBAL_VAR_{}", arr_ident.ident),
                                    value,
                                    false,
                                );

                                irgen.symtable.insert(
                                    arr_ident.ident.clone(),
                                    SymbolEntry {
                                        ty: array_ty,
                                        comptime: Some(comptime.to_comptimeval(ty)),
                                        ir_value: Some(IrGenResult::Global(slot)),
                                    },
                                );
                            }
                        }
                    }
                }
            },
            Item::FuncDef(func_def) => func_def.irgen(irgen),
        }
        // println!("{:#?}", irgen.symtable);
    }
}

impl IrGen for FuncDef {
    fn irgen(&self, irgen: &mut IrGenContext) {
        irgen.symtable.enter_scope();

        let mut param_tys = Vec::new();
        for FuncFParam { ty, .. } in self.params.iter() {
            param_tys.push(ty.clone());
        }

        let func_ty = Type::func(param_tys.clone(), self.ret_ty.clone());

        let ir_ret_ty = irgen.gen_type(&self.ret_ty);
        let func = Func::new(&mut irgen.ctx, self.ident.clone(), ir_ret_ty, Some(true));

        irgen.symtable.insert_upper(
            self.ident.clone(),
            SymbolEntry {
                ty: func_ty,
                comptime: None,
                ir_value: None,
            },
            1,
        );

        let block = Block::new(&mut irgen.ctx);
        func.push_back(&mut irgen.ctx, block).unwrap();

        irgen.curr_func = Some(func);
        irgen.curr_func_name = Some(self.ident.clone());
        irgen.curr_block = Some(block);

        // block params
        for (FuncFParam { ident, .. }, ty) in self.params.iter().zip(param_tys.iter()) {
            let ir_ty = irgen.gen_type(ty);
            let param = func.add_param(&mut irgen.ctx, ir_ty);

            irgen.symtable.insert(
                ident.clone(),
                SymbolEntry {
                    ty: ty.clone(),
                    comptime: None,
                    ir_value: Some(IrGenResult::Value(param)),
                },
            );
        }

        // create slots for pass-by-value params
        for (FuncFParam { ident, .. }, ty) in self.params.iter().zip(param_tys.iter()) {
            if ty.is_int() {
                let ir_ty = irgen.gen_type(ty);
                let slot = Inst::alloca(&mut irgen.ctx, ir_ty);

                block.push_front(&mut irgen.ctx, slot).unwrap();
                let slot = slot.result(&irgen.ctx).unwrap();

                // get old entry
                let param = irgen
                    .symtable
                    .lookup(ident)
                    .unwrap()
                    .ir_value
                    .unwrap()
                    .unwrap_value();

                // store
                let store = Inst::store(&mut irgen.ctx, param, slot);

                block.push_back(&mut irgen.ctx, store).unwrap();

                // set new entry
                irgen.symtable.insert(
                    ident.clone(),
                    SymbolEntry {
                        ty: ty.clone(),
                        comptime: None,
                        ir_value: Some(IrGenResult::Value(slot)),
                    },
                );
            }
        }

        // create return block and slot
        let ret_block = Block::new(&mut irgen.ctx);
        irgen.curr_ret_block = Some(ret_block);

        if !self.ret_ty.is_void() {
            let ir_ret_ty = irgen.gen_type(&self.ret_ty);
            let ret_slot = Inst::alloca(&mut irgen.ctx, ir_ret_ty);

            block.push_front(&mut irgen.ctx, ret_slot).unwrap();
            irgen.curr_ret_slot = Some(ret_slot.result(&irgen.ctx).unwrap());
        }

        // generate body
        self.body.irgen(irgen);

        // generate br to return block
        if self.ret_ty.is_void() {
            let br_inst = Inst::br(&mut irgen.ctx, ret_block);
            block.push_back(&mut irgen.ctx, br_inst).unwrap();
        }

        // append return block
        func.push_back(&mut irgen.ctx, ret_block).unwrap();

        if !self.ret_ty.is_void() {
            // load, ret
            let ret_slot = irgen.curr_ret_slot.unwrap();
            let ty = irgen.gen_type(&self.ret_ty);

            let load = Inst::load(&mut irgen.ctx, ret_slot, ty);
            ret_block.push_back(&mut irgen.ctx, load).unwrap();
            let val = load.result(&irgen.ctx).unwrap();

            let ret = Inst::ret(&mut irgen.ctx, Some(val));
            ret_block.push_back(&mut irgen.ctx, ret).unwrap();
        } else {
            // just return
            let ret = Inst::ret(&mut irgen.ctx, None);
            ret_block.push_back(&mut irgen.ctx, ret).unwrap();
        }

        irgen.curr_func = None;
        irgen.curr_func_name = None;
        irgen.curr_block = None;
        irgen.curr_ret_slot = None;
        irgen.curr_ret_block = None;

        irgen.symtable.leave_scope();
    }
}

impl IrGen for Decl {
    fn irgen(&self, irgen: &mut IrGenContext) {
        let entry_block = irgen.curr_func.unwrap().head(&irgen.ctx).unwrap();
        let curr_block = irgen.curr_block.unwrap();
        match self {
            Decl::ConstDecl(ConstDecl { ty, defs }) => {
                for def in defs {
                    match def {
                        ConstDef::Val(ident, init) => {
                            let comptime = init
                                .try_fold(&irgen.symtable)
                                .expect("global def expected to have constant initializer");

                            let ir_ty = irgen.gen_type(init.ty());
                            // TODO: alloca 需要换成 constant，返回的变量应该是@var_name
                            let stack_slot = Inst::alloca(&mut irgen.ctx, ir_ty);

                            entry_block.push_front(&mut irgen.ctx, stack_slot).unwrap();
                            irgen.symtable.insert(
                                ident,
                                SymbolEntry {
                                    ty: init.ty().clone(),
                                    comptime: Some(comptime),
                                    ir_value: Some(IrGenResult::Value(
                                        stack_slot.result(&irgen.ctx).unwrap(),
                                    )),
                                },
                            );
                            let init = irgen.gen_local_expr(init).unwrap();
                            let slot = stack_slot.result(&irgen.ctx).unwrap();
                            let store = Inst::store(&mut irgen.ctx, init, slot);
                            curr_block.push_back(&mut irgen.ctx, store).unwrap();
                        }
                        ConstDef::Array(arr_ident, init) => {
                            let comptime = init
                                .try_fold(&irgen.symtable)
                                .expect("global def expected to have constant initializer");

                            // println!("init:-----------\n{:#?}", init);
                            let array_ty = make_array(ty.clone(), 0, &arr_ident.size).unwrap();
                            let ir_ty = irgen.gen_type(&array_ty);
                            // TODO: alloca 需要换成 constant，返回的变量应该是@var_name
                            let stack_slot = Inst::alloca(&mut irgen.ctx, ir_ty);

                            entry_block.push_front(&mut irgen.ctx, stack_slot).unwrap();
                            irgen.symtable.insert(
                                arr_ident.ident.clone(),
                                SymbolEntry {
                                    ty: array_ty,
                                    comptime: Some(comptime.to_comptimeval(ty)),
                                    ir_value: Some(IrGenResult::Value(
                                        stack_slot.result(&irgen.ctx).unwrap(),
                                    )),
                                },
                            );

                            let init = irgen.gen_local_array(ty, init).unwrap();
                            // println!("{}", init.display(&irgen.ctx, true));
                            let slot = stack_slot.result(&irgen.ctx).unwrap();
                            let store = Inst::store(&mut irgen.ctx, init, slot);
                            curr_block.push_back(&mut irgen.ctx, store).unwrap();
                        }
                    }
                }
            }
            Decl::VarDecl(VarDecl { ty, defs, .. }) => {
                for def in defs {
                    match def {
                        VarDef::Val(ident, init) => {
                            let init = init.as_ref().unwrap();
                            let ir_ty = irgen.gen_type(init.ty());
                            let stack_slot = Inst::alloca(&mut irgen.ctx, ir_ty);

                            entry_block.push_front(&mut irgen.ctx, stack_slot).unwrap();
                            irgen.symtable.insert(
                                ident,
                                SymbolEntry {
                                    ty: init.ty().clone(),
                                    comptime: None,
                                    ir_value: Some(IrGenResult::Value(
                                        stack_slot.result(&irgen.ctx).unwrap(),
                                    )),
                                },
                            );

                            let init = irgen.gen_local_expr(init).unwrap();
                            let slot = stack_slot.result(&irgen.ctx).unwrap();
                            let store = Inst::store(&mut irgen.ctx, init, slot);
                            curr_block.push_back(&mut irgen.ctx, store).unwrap();
                        }
                        VarDef::Array(arr_ident, init) => {
                            let init = init.as_ref().unwrap();
                            // println!("init:-----------\n{:#?}", init);
                            let array_ty = make_array(ty.clone(), 0, &arr_ident.size).unwrap();
                            let ir_ty = irgen.gen_type(&array_ty);
                            let stack_slot = Inst::alloca(&mut irgen.ctx, ir_ty);

                            entry_block.push_front(&mut irgen.ctx, stack_slot).unwrap();
                            irgen.symtable.insert(
                                arr_ident.ident.clone(),
                                SymbolEntry {
                                    ty: array_ty,
                                    comptime: None,
                                    ir_value: Some(IrGenResult::Value(
                                        stack_slot.result(&irgen.ctx).unwrap(),
                                    )),
                                },
                            );

                            let init = irgen.gen_local_array(ty, init).unwrap();
                            // println!("{}", init.display(&irgen.ctx, true));
                            let slot = stack_slot.result(&irgen.ctx).unwrap();
                            let store = Inst::store(&mut irgen.ctx, init, slot);
                            curr_block.push_back(&mut irgen.ctx, store).unwrap();
                        }
                    }
                }
            }
        }
    }
}

impl IrGen for Stmt {
    fn irgen(&self, irgen: &mut IrGenContext) {
        let curr_block = irgen.curr_block.unwrap();

        match self {
            Stmt::Assign(LVal { ident }, expr) => {
                let entry = irgen.symtable.lookup(ident).unwrap();
                let ir_value = entry.ir_value.unwrap();

                let slot = if let IrGenResult::Global(slot) = ir_value {
                    let name = slot.name(&irgen.ctx).to_string();
                    let value_ty = slot.ty(&irgen.ctx);
                    Value::global_ref(&mut irgen.ctx, name, value_ty)
                } else if let IrGenResult::Value(slot) = ir_value {
                    slot
                } else {
                    unreachable!()
                };

                let store_dst = slot;

                let val = irgen.gen_local_expr(expr).unwrap();
                let store = Inst::store(&mut irgen.ctx, val, store_dst);
                curr_block.push_back(&mut irgen.ctx, store).unwrap();
            }
            Stmt::ArrayAssign(ArrayIdent { ident, size: pos }, expr) => {
                let entry = irgen.symtable.lookup(ident).unwrap();
                let ir_value = entry.ir_value.unwrap();

                let slot = if let IrGenResult::Global(slot) = ir_value {
                    let name = slot.name(&irgen.ctx).to_string();
                    let value_ty = slot.ty(&irgen.ctx);
                    Value::global_ref(&mut irgen.ctx, name, value_ty)
                } else if let IrGenResult::Value(slot) = ir_value {
                    slot
                } else {
                    unreachable!()
                };

                let mut pos_values = vec![Value::i32(&mut irgen.ctx, 0)];
                let entry_ty = entry.ty.clone();
                let value_ty = irgen.gen_type(&entry_ty);
                for p in pos.iter() {
                    pos_values.push(irgen.gen_local_expr(p).unwrap());
                }
                let ptr_dst = Inst::getelementptr(&mut irgen.ctx, value_ty, slot, pos_values);
                curr_block.push_back(&mut irgen.ctx, ptr_dst).unwrap();
                let store_dst = ptr_dst.result(&irgen.ctx).unwrap();

                let val = irgen.gen_local_expr(expr).unwrap();
                let store = Inst::store(&mut irgen.ctx, val, store_dst);
                curr_block.push_back(&mut irgen.ctx, store).unwrap();
            }
            Stmt::Expr(ExprStmt { expr }) => {
                if let Some(ref expr) = expr {
                    irgen.gen_local_expr(expr);
                }
            }
            Stmt::Block(block) => block.irgen(irgen),
            //If实现
            Stmt::If(cond_expr, then_stmt, else_stmt) => {
                fn block_has_terminator(block: Block, ctx: &Context) -> bool {
                    let mut inst = block.head(ctx); // 从 head 开始
                    while let Some(current_inst) = inst {
                        if matches!(current_inst.kind(ctx), InstKind::Br | InstKind::Ret) {
                            return true;
                        }
                        inst = current_inst.next(ctx); // 遍历到下一条指令
                    }
                    false
                }
            
                if let Tk::Bool = cond_expr.ty().kind() {
                    // 创建基本块
                    let then_block = Block::new(&mut irgen.ctx);
                    let else_block = else_stmt.as_ref().map(|_| Block::new(&mut irgen.ctx));
                    let exit_block = Block::new(&mut irgen.ctx);
            
                    // 当前基本块和函数
                    let curr_block = irgen.curr_block.unwrap();
                    let curr_func = irgen.curr_func.unwrap();
            
                    // 生成条件表达式的值
                    let cond_value = irgen.gen_local_expr(cond_expr).unwrap();
                    let cond_br = Inst::cond_br(
                        &mut irgen.ctx,
                        cond_value,
                        then_block,
                        else_block.unwrap_or(exit_block),
                    );
                    curr_block.push_back(&mut irgen.ctx, cond_br).unwrap();
            
                    // Then 分支处理
                    curr_func.push_back(&mut irgen.ctx, then_block).unwrap();
                    irgen.curr_block = Some(then_block);
                    then_stmt.irgen(irgen);
            
                    // 如果 then_block 是空块，或者没有终止指令，添加跳转
                    if !block_has_terminator(then_block, &irgen.ctx) {
                        let then_exit_br = Inst::br(&mut irgen.ctx, exit_block);
                        then_block.push_back(&mut irgen.ctx, then_exit_br).unwrap();
                    }
            
                    // Else 分支处理
                    if let Some(else_stmt) = else_stmt {
                        let else_block = else_block.unwrap();
                        curr_func.push_back(&mut irgen.ctx, else_block).unwrap();
                        irgen.curr_block = Some(else_block);
            
                        else_stmt.irgen(irgen);
            
                        // 如果 else_block 是空块，或者没有终止指令，添加跳转
                        if !block_has_terminator(else_block, &irgen.ctx) {
                            let else_exit_br = Inst::br(&mut irgen.ctx, exit_block);
                            else_block.push_back(&mut irgen.ctx, else_exit_br).unwrap();
                        }
                    }
            
                    // 确保 exit_block 存在并链接
                    if exit_block.head(&irgen.ctx).is_none() {
                        // 如果 exit_block 是空的，添加占位指令
                        let unreachable_inst = Inst::br(&mut irgen.ctx, exit_block);
                        exit_block.push_back(&mut irgen.ctx, unreachable_inst).unwrap();
                    }
                    curr_func.push_back(&mut irgen.ctx, exit_block).unwrap();
                    irgen.curr_block = Some(exit_block);
                } else {
                    panic!("if condition must be a boolean expression");
                }
            }
            
            

            Stmt::While(expr, stmt) => {
                if let Tk::Bool = expr.ty().kind() {
                    let cond_block = Block::new(&mut irgen.ctx);
                    let body_block = Block::new(&mut irgen.ctx);
                    let exit_block = Block::new(&mut irgen.ctx);

                    let curr_block = irgen.curr_block.unwrap();
                    let curr_func = irgen.curr_func.unwrap();

                    irgen.loop_entry_stack.push(cond_block);
                    irgen.loop_exit_stack.push(exit_block);

                    let cond_branch = Inst::br(&mut irgen.ctx, cond_block);

                    curr_block.push_back(&mut irgen.ctx, cond_branch).unwrap();

                    curr_func.push_back(&mut irgen.ctx, cond_block).unwrap();
                    irgen.curr_block = Some(cond_block);

                    let cond = irgen.gen_local_expr(expr).unwrap();
                    let br = Inst::cond_br(&mut irgen.ctx, cond, body_block, exit_block);
                    cond_block.push_back(&mut irgen.ctx, br).unwrap();

                    irgen.curr_block = Some(body_block);
                    stmt.irgen(irgen);

                    let cond_branch = Inst::br(&mut irgen.ctx, cond_block);
                    body_block.push_back(&mut irgen.ctx, cond_branch).unwrap();
                    curr_func.push_back(&mut irgen.ctx, body_block).unwrap();

                    curr_func.push_back(&mut irgen.ctx, exit_block).unwrap();
                    irgen.curr_block = Some(exit_block);
                } else {
                    panic!("while condition must be a boolean expression");
                }
            }
            Stmt::Break => {
                let jump = Inst::br(
                    &mut irgen.ctx,
                    irgen.loop_exit_stack.last().unwrap().clone(),
                );
                curr_block.push_back(&mut irgen.ctx, jump).unwrap();
            }
            Stmt::Continue => {
                let jump = Inst::br(
                    &mut irgen.ctx,
                    irgen.loop_entry_stack.last().unwrap().clone(),
                );
                curr_block.push_back(&mut irgen.ctx, jump).unwrap();
            }
            Stmt::Return(ReturnStmt { expr }) => {
                if let Some(expr) = expr {
                    let val = irgen.gen_local_expr(expr).unwrap();
                    let store = Inst::store(&mut irgen.ctx, val, irgen.curr_ret_slot.unwrap());
                    irgen
                        .curr_block
                        .unwrap()
                        .push_back(&mut irgen.ctx, store)
                        .unwrap();
                }

                let jump = Inst::br(&mut irgen.ctx, irgen.curr_ret_block.unwrap());
                irgen
                    .curr_block
                    .unwrap()
                    .push_back(&mut irgen.ctx, jump)
                    .unwrap();
            }
        }
    }
}

impl IrGen for ast::Block {
    fn irgen(&self, irgen: &mut IrGenContext) {
        irgen.symtable.enter_scope();
        for item in self.items.iter() {
            match item {
                BlockItem::Decl(decl) => decl.irgen(irgen),
                BlockItem::Stmt(stmt) => stmt.irgen(irgen),
            }
        }
        irgen.symtable.leave_scope();
    }
}

pub fn make_array(ty: Type, index: usize, size: &Vec<Expr>) -> Option<Type> {
    let mut ty = ty.clone();
    if index == size.len() {
        if let Tk::Array(ty, _) = ty.kind() {
            return Some(ty.clone());
        }
        return Some(ty);
    }
    if let ExprKind::Const(Cv::Int(sz)) = &size[index].kind {
        ty = Type::make(Tk::Array(
            make_array(ty.clone(), index + 1, size).unwrap(),
            *sz as usize,
        ));
        return Some(ty);
    }

    None
}
