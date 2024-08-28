use std::collections::HashMap;

use super::irgen::IrGenResult;
use super::types::{Type, TypeKind as Tk};

/// Represents a constant value that can be evaluated at compile time.
#[derive(Debug, Clone)]
pub enum ComptimeVal {
    Bool(bool),
    Int(i32),
}

impl ComptimeVal {
    /// Unwrap the comptime value as a int
    pub fn unwrap_int(&self) -> i32 {
        match self {
            Self::Bool(b) => *b as i32,
            Self::Int(i) => *i,
        }
    }

    pub fn bool(b: bool) -> Self { Self::Bool(b) }

    pub fn int(i: i32) -> Self { Self::Int(i) }

    /// Get the type of the comptime value.
    pub fn get_type(&self) -> Type {
        match self {
            Self::Bool(_) => Type::bool(),
            Self::Int(_) => Type::int(),
        }
    }

    /// Check if the comptime value is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Self::Bool(b) => !*b,
            Self::Int(i) => *i == 0,
        }
    }

    /// Compute the logical OR of two comptime values.
    pub fn logical_or(&self, other: &Self) -> Self {
        let lhs = match self {
            Self::Bool(a) => *a,
            Self::Int(a) => *a != 0,
        };

        let rhs = match other {
            Self::Bool(b) => *b,
            Self::Int(b) => *b != 0,
        };

        Self::Bool(lhs || rhs)
    }

    /// Compute the logical AND of two comptime values.
    pub fn logical_and(&self, other: &Self) -> Self {
        let lhs = match self {
            Self::Bool(a) => *a,
            Self::Int(a) => *a != 0,
        };

        let rhs = match other {
            Self::Bool(b) => *b,
            Self::Int(b) => *b != 0,
        };

        Self::Bool(lhs && rhs)
    }

    // Comptime value operations are used in constant folding.
    // Your compiler can still work without these operations, but it will be less
    // efficient.
    //
    // TODO: Implement other operations for ComptimeVal
}

impl PartialEq for ComptimeVal {
    fn eq(&self, other: &Self) -> bool {
        use ComptimeVal as Cv;
        match (self, other) {
            (Cv::Bool(a), Cv::Bool(b)) => a == b,
            (Cv::Int(a), Cv::Int(b)) => a == b,

            // Coercion situations, bool -> int
            (Cv::Bool(a), Cv::Int(b)) => (*a as i32) == *b,
            (Cv::Int(a), Cv::Bool(b)) => *a == (*b as i32),
        }
    }
}

impl Eq for ComptimeVal {}

impl PartialOrd for ComptimeVal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use ComptimeVal as Cv;
        match (self, other) {
            (Cv::Bool(a), Cv::Bool(b)) => a.partial_cmp(b),
            (Cv::Int(a), Cv::Int(b)) => a.partial_cmp(b),

            // Coercion situations, bool -> int
            (Cv::Bool(a), Cv::Int(b)) => (*a as i32).partial_cmp(b),
            (Cv::Int(a), Cv::Bool(b)) => a.partial_cmp(&(*b as i32)),
        }
    }
}

impl std::ops::Neg for ComptimeVal {
    type Output = Self;

    fn neg(self) -> Self {
        use ComptimeVal as Cv;
        match self {
            Cv::Bool(a) => Cv::Int(-(a as i32)),
            Cv::Int(a) => Cv::Int(-a),
        }
    }
}

impl std::ops::Not for ComptimeVal {
    type Output = Self;

    fn not(self) -> Self {
        use ComptimeVal as Cv;
        match self {
            Cv::Bool(a) => Cv::Bool(!a),
            Cv::Int(a) => Cv::Bool(a != 0),
        }
    }
}

impl std::ops::Add for ComptimeVal {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        use ComptimeVal as Cv;
        match (self, other) {
            (Cv::Int(a), Cv::Int(b)) => Cv::Int(a + b),

            // coercion situations, bool -> int
            (Cv::Bool(a), Cv::Int(b)) => Cv::Int(a as i32 + b),
            (Cv::Int(a), Cv::Bool(b)) => Cv::Int(a + b as i32),
            (Cv::Bool(a), Cv::Bool(b)) => Cv::Int(a as i32 + b as i32),
        }
    }
}

impl std::ops::Sub for ComptimeVal {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        use ComptimeVal as Cv;

        match (self, other) {
            (Cv::Int(a), Cv::Int(b)) => Cv::Int(a - b),

            // coercion situations, bool -> int
            (Cv::Bool(a), Cv::Int(b)) => Cv::Int(a as i32 - b),
            (Cv::Int(a), Cv::Bool(b)) => Cv::Int(a - b as i32),
            (Cv::Bool(a), Cv::Bool(b)) => Cv::Int(a as i32 - b as i32),
        }
    }
}

impl std::ops::Mul for ComptimeVal {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        use ComptimeVal as Cv;
        match (self, other) {
            (Cv::Int(a), Cv::Int(b)) => Cv::Int(a * b),

            // coercion situations, bool -> int
            (Cv::Bool(a), Cv::Int(b)) => Cv::Int(a as i32 * b),
            (Cv::Int(a), Cv::Bool(b)) => Cv::Int(a * b as i32),
            (Cv::Bool(a), Cv::Bool(b)) => Cv::Int(a as i32 * b as i32),
        }
    }
}

impl std::ops::Div for ComptimeVal {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        use ComptimeVal as Cv;
        match (self, other) {
            (Cv::Int(a), Cv::Int(b)) => Cv::Int(a / b),

            // coercion situations, bool -> int
            (Cv::Bool(a), Cv::Int(b)) => Cv::Int(a as i32 / b),
            (Cv::Int(a), Cv::Bool(b)) => Cv::Int(a / b as i32),
            (Cv::Bool(a), Cv::Bool(b)) => Cv::Int(a as i32 / b as i32),
        }
    }
}

impl std::ops::Rem for ComptimeVal {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        use ComptimeVal as Cv;
        match (self, other) {
            (Cv::Int(a), Cv::Int(b)) => Cv::Int(a % b),

            // bool -> int
            (Cv::Bool(a), Cv::Bool(b)) => Cv::Int(a as i32 % b as i32),
            (Cv::Bool(a), Cv::Int(b)) => Cv::Int(a as i32 % b),
            (Cv::Int(a), Cv::Bool(b)) => Cv::Int(a % b as i32),
        }
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    /// Logical not.
    Not,
}

/// Function call.
#[derive(Debug, PartialEq, Eq)]
pub struct FuncCall {
    pub ident: String,
    pub args: Vec<Expr>,
}

/// Left value.
/// Left value refers to a specific memory location, typically allowing it to be
/// assigned a value.
/// Its usually on the left side of an assignment.
#[derive(Debug, PartialEq, Eq)]
pub struct LVal {
    pub ident: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExprKind {
    /// Constant value.
    Const(ComptimeVal),
    /// Binary operation.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// Unary operation.
    Unary(UnaryOp, Box<Expr>),
    /// Function call.
    FuncCall(FuncCall),
    /// Left value.
    LVal(LVal),
    /// Type coercion. This is used to convert one type to another.
    Coercion(Box<Expr>),
}

/// Expression.
#[derive(Debug)]
pub struct Expr {
    /// kind of the expression.
    pub kind: ExprKind,
    /// Type of the expression.
    /// Its generated during type checking.
    pub ty: Option<Type>,
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool { self.kind == other.kind }
}

impl Eq for Expr {}

impl Expr {
    pub fn const_(val: ComptimeVal) -> Self {
        let ty = val.get_type();
        Self {
            kind: ExprKind::Const(val),
            ty: Some(ty),
        }
    }

    pub fn binary(op: BinaryOp, lhs: Expr, rhs: Expr) -> Self {
        Self {
            kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
            ty: None,
        }
    }

    pub fn unary(op: UnaryOp, expr: Expr) -> Self {
        Self {
            kind: ExprKind::Unary(op, Box::new(expr)),
            ty: None,
        }
    }

    pub fn func_call(ident: String, args: Vec<Expr>) -> Self {
        Self {
            kind: ExprKind::FuncCall(FuncCall { ident, args }),
            ty: None,
        }
    }

    pub fn lval(lval: LVal) -> Self {
        Self {
            kind: ExprKind::LVal(lval),
            ty: None,
        }
    }

    pub fn coercion(expr: Expr, to: Type) -> Self {
        if let Some(ref from) = expr.ty {
            if from == &to {
                return expr;
            }
        }

        Self {
            kind: ExprKind::Coercion(Box::new(expr)),
            ty: Some(to),
        }
    }
}

/// Expression statement.
#[derive(Debug)]
pub struct ExprStmt {
    pub expr: Option<Expr>,
}

/// Return statement.
#[derive(Debug)]
pub struct ReturnStmt {
    pub expr: Option<Expr>,
}

/// Statement.
#[derive(Debug)]
pub enum Stmt {
    /// Assignment statement.
    /// e.g. `a = 1;`
    Assign(LVal, Expr),
    /// Expression statement.
    /// e.g. `1 + 2;`
    Expr(ExprStmt),
    /// Block statement.
    /// e.g. `{ ... }`
    Block(Block),
    /// If statement.
    /// e.g. `if (a) { ... } else { ... }`
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    /// While statement.
    /// e.g. `while (a) { ... }`
    While(Expr, Box<Stmt>),
    /// Break statement.
    /// e.g. `break;`
    Break,
    /// Continue statement.
    /// e.g. `continue;`
    Continue,
    /// Return statement.
    /// e.g. `return 1;`
    Return(ReturnStmt),
}

/// Block item.
/// This can be a declaration or a statement.
#[derive(Debug)]
pub enum BlockItem {
    /// Declaration.
    Decl(Decl),
    /// Statement.
    Stmt(Stmt),
}

/// Block.
/// A block is a sequence of statements and declarations enclosed in braces.
/// e.g. `{ ... }`
#[derive(Debug)]
pub struct Block {
    pub items: Vec<BlockItem>,
}

/// Declaration.
/// This can be a sieries of constant or variable definitions.
/// e.g. `const int a = 1, b = 2;`, `int a = 1, b = 2;`
#[derive(Debug)]
pub enum Decl {
    ConstDecl(ConstDecl),
    VarDecl(VarDecl),
}

/// Constant definition.
/// Definition a constant value.
/// e.g.
/// ```c
/// const int a = 1, b = 2;
///           ^^^^^
/// ```
#[derive(Debug)]
pub struct ConstDef {
    pub ident: String,
    pub init: Expr,
}

/// Variable definition.
/// Definition a variable.
/// e.g.
/// ```c
/// int a = 1, b = 2;
///     ^^^^^
/// ```
#[derive(Debug)]
pub struct VarDef {
    pub ident: String,
    pub init: Option<Expr>,
}

/// Constant declaration.
/// This can be a series of constant definitions.
/// e.g. `const int a = 1, b = 2;`
#[derive(Debug)]
pub struct ConstDecl {
    pub ty: Type,
    pub defs: Vec<ConstDef>,
}

/// Variable declaration.
/// This can be a series of variable definitions.
/// e.g. `int a = 1, b = 2;`
#[derive(Debug)]
pub struct VarDecl {
    pub ty: Type,
    pub defs: Vec<VarDef>,
}

/// Function Formal parameter.
/// A parameter of a function definition.
/// e.g.
/// ```c
/// int add(int a, int b) {}
///         ^^^^^
/// ```
#[derive(Debug)]
pub struct FuncFParam {
    pub ty: Type,
    pub ident: String,
}

/// Function definition.
/// e.g. `int add(int a, int b) {}`
#[derive(Debug)]
pub struct FuncDef {
    /// Type of the return value.
    pub ret_ty: Type,
    /// Identifier of the function. (Name of the function)
    pub ident: String,
    /// Parameters of the function.
    pub params: Vec<FuncFParam>,
    /// Body of the function. It contains a block of statements.
    pub body: Block,
}

/// A global item.
/// This can be a declaration or a function definition.
/// e.g. `const int a = 1;`, `int main() { ... }`
#[derive(Debug)]
pub enum Item {
    Decl(Decl),
    FuncDef(FuncDef),
}

/// Compilation unit.
/// This is the root node of the AST.
/// It contains a series of global items.
#[derive(Debug)]
pub struct CompUnit {
    pub items: Vec<Item>,
}

/// Symbol table entry.
/// This is used to store information about a symbol in the symbol table.
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

impl SymbolEntry {
    /// Create a new symbol entry from a type.
    pub fn from_ty(ty: Type) -> Self {
        Self {
            ty,
            comptime: None,
            ir_value: None,
        }
    }
}

/// Symbol table.
/// This is used to store information about symbols in the program.
#[derive(Default)]
pub struct SymbolTable {
    /// Stack of scopes.
    /// Each scope has its own hashmap of symbols.
    stack: Vec<HashMap<String, SymbolEntry>>,

    /// The current return type of the function.
    pub curr_ret_ty: Option<Type>,
}

impl SymbolTable {
    /// Enter a new scope.
    pub fn enter_scope(&mut self) { self.stack.push(HashMap::default()); }

    /// Leave the current scope.
    pub fn leave_scope(&mut self) { self.stack.pop(); }

    /// Insert a symbol into the current scope.
    pub fn insert(&mut self, name: impl Into<String>, entry: SymbolEntry) {
        self.stack.last_mut().unwrap().insert(name.into(), entry);
    }

    /// Insert a symbol into the `upper`-th scope from the current scope.
    pub fn insert_upper(&mut self, name: impl Into<String>, entry: SymbolEntry, upper: usize) {
        self.stack
            .iter_mut()
            .rev()
            .nth(upper)
            .unwrap()
            .insert(name.into(), entry);
    }

    /// Lookup a symbol in the symbol table.
    pub fn lookup(&self, name: &str) -> Option<&SymbolEntry> {
        for scope in self.stack.iter().rev() {
            if let Some(entry) = scope.get(name) {
                return Some(entry);
            }
        }
        None
    }

    /// Lookup a symbol in the symbol table.
    pub fn lookup_mut(&mut self, name: &str) -> Option<&mut SymbolEntry> {
        for scope in self.stack.iter_mut().rev() {
            if let Some(entry) = scope.get_mut(name) {
                return Some(entry);
            }
        }
        None
    }

    /// Register SysY library functions to the symbol table.
    pub fn register_sysylib(&mut self) {
        // TODO: Register SysY library functions to the symbol table
    }
}

impl CompUnit {
    /// Type check the compilation unit.
    pub fn type_check(&mut self) {
        let mut symtable = SymbolTable::default();
        symtable.enter_scope();

        // register SysY library functions in the top level scope
        symtable.register_sysylib();

        // type check each item
        for item in self.items.iter_mut() {
            item.type_check(&mut symtable);
        }

        symtable.leave_scope();
    }
}

impl Item {
    /// Type check the item.
    pub fn type_check(&mut self, symtable: &mut SymbolTable) {
        match self {
            Item::Decl(decl) => match decl {
                Decl::ConstDecl(decl) => decl.type_check(symtable),
                Decl::VarDecl(decl) => decl.type_check(symtable),
            },
            Item::FuncDef(FuncDef {
                ret_ty,
                ident,
                params,
                body,
            }) => {
                // Enter a new scope for function parameters
                symtable.enter_scope();

                // Insert the function parameters into the scope
                let mut param_tys = Vec::new();
                for param in params.iter() {
                    param_tys.push(param.ty.clone());
                    symtable.insert(param.ident.clone(), SymbolEntry::from_ty(param.ty.clone()));
                }

                let func_ty = Type::func(param_tys, ret_ty.clone());

                // Insert the function symbol into the scope above the current scope, since we
                // are in the parameters scope
                symtable.insert_upper(ident.clone(), SymbolEntry::from_ty(func_ty), 1);
                symtable.curr_ret_ty = Some(ret_ty.clone());

                // Type check the function body
                body.type_check(symtable);

                symtable.curr_ret_ty = None;
                symtable.leave_scope();
            }
        }
    }
}

impl ConstDecl {
    /// Type check the constant declaration.
    pub fn type_check(&mut self, symtable: &mut SymbolTable) {
        let mut new_defs = Vec::new();
        for mut def in self.defs.drain(..) {
            // TODO: array type checking

            let ty = self.ty.clone();

            // Type check the init expression
            def.init = def.init.type_check(Some(&ty), symtable);

            // Fold the init expression into a constant value
            let folded = def.init.try_fold(symtable).expect("non-constant init");
            def.init = Expr::const_(folded.clone());

            // Insert the constant into the symbol table
            symtable.insert(
                def.ident.clone(),
                SymbolEntry {
                    ty,
                    comptime: Some(folded),
                    ir_value: None,
                },
            );
            new_defs.push(def);
        }
        self.defs = new_defs;
    }
}

impl VarDecl {
    /// Type check the variable declaration.
    pub fn type_check(&mut self, symtable: &mut SymbolTable) {
        let mut new_defs = Vec::new();
        for mut def in self.defs.drain(..) {
            // TODO: array type checking

            let ty = self.ty.clone();

            // Type check the init expression, and fold it if possible
            let init = def
                .init
                .map(|init| {
                    // fold as much as possible
                    // XXX: what if we do not fold here?
                    let typed_init = init.type_check(Some(&ty), symtable);
                    match typed_init.try_fold(symtable) {
                        Some(val) => Expr::const_(val),
                        None => typed_init,
                    }
                })
                .unwrap_or_else(|| todo!("what if there is no init value?"));

            def.init = Some(init);

            // Insert the variable into the symbol table
            symtable.insert(def.ident.clone(), SymbolEntry::from_ty(ty));
            new_defs.push(def);
        }
        self.defs = new_defs;
    }
}

impl Block {
    /// Type check the block.
    pub fn type_check(&mut self, symtable: &mut SymbolTable) {
        // Enter a new scope
        symtable.enter_scope();
        let mut new_items = Vec::new();

        // Type check each block item in the block
        for item in self.items.drain(..) {
            let item = match item {
                BlockItem::Decl(decl) => match decl {
                    Decl::ConstDecl(mut decl) => {
                        decl.type_check(symtable);
                        BlockItem::Decl(Decl::ConstDecl(decl))
                    }
                    Decl::VarDecl(mut decl) => {
                        decl.type_check(symtable);
                        BlockItem::Decl(Decl::VarDecl(decl))
                    }
                },
                BlockItem::Stmt(stmt) => {
                    let stmt = stmt.type_check(symtable);
                    BlockItem::Stmt(stmt)
                }
            };
            new_items.push(item);
        }
        self.items = new_items;
        symtable.leave_scope();
    }
}

impl Stmt {
    /// Type check the statement.
    pub fn type_check(self, symtable: &mut SymbolTable) -> Self {
        match self {
            Stmt::Assign(LVal { ident }, expr) => {
                // lookup the variable in the symbol table
                let entry = symtable.lookup(&ident).expect("variable not found");

                // TODO: array type checking

                let ty = &entry.ty;

                // Type check the expression
                let expr = expr.type_check(Some(ty), symtable);
                Stmt::Assign(LVal { ident }, expr)
            }
            Stmt::Expr(ExprStmt { expr }) => {
                // Type check the expression
                let expr = expr.map(|expr| expr.type_check(None, symtable));
                Stmt::Expr(ExprStmt { expr })
            }
            Stmt::Block(mut block) => {
                // Type check the block
                block.type_check(symtable);
                Stmt::Block(block)
            }
            Stmt::Break => Stmt::Break,
            Stmt::Continue => Stmt::Continue,
            Stmt::Return(ReturnStmt { expr }) => {
                // Type check the return expression
                let expr =
                    expr.map(|expr| expr.type_check(symtable.curr_ret_ty.as_ref(), symtable));

                // Void return
                if expr.is_none() {
                    return Stmt::Return(ReturnStmt { expr });
                }

                let mut expr = expr.unwrap();
                let ret_ty = symtable.curr_ret_ty.as_ref().unwrap();

                if ret_ty.is_int() {
                    // Coerce the expression to int if needed
                    expr = Expr::coercion(expr, Type::int());
                } else {
                    panic!("unsupported return type");
                }

                Stmt::Return(ReturnStmt { expr: Some(expr) })
            }
            Stmt::If(cond, then_block, else_block) => {
                // Type check the condition expression and the blocks
                let cond = cond.type_check(Some(&Type::bool()), symtable);
                let then_block = then_block.type_check(symtable);
                let else_block = else_block.map(|block| block.type_check(symtable));
                Stmt::If(cond, Box::new(then_block), else_block.map(Box::new))
            }
            Stmt::While(cond, block) => {
                // Type check the condition expression and the block
                let cond = cond.type_check(Some(&Type::bool()), symtable);
                let block = block.type_check(symtable);
                Stmt::While(cond, Box::new(block))
            }
        }
    }
}

impl Expr {
    /// Get the type of the expression.
    pub fn ty(&self) -> &Type { self.ty.as_ref().unwrap() }

    /// Try to fold the expression into a constant value.
    pub fn try_fold(&self, symtable: &SymbolTable) -> Option<ComptimeVal> {
        match &self.kind {
            ExprKind::Const(val) => Some(val.clone()),
            ExprKind::Binary(op, lhs, rhs) => {
                let lhs = lhs.try_fold(symtable)?;
                let rhs = rhs.try_fold(symtable)?;

                use BinaryOp as Bo;

                match op {
                    Bo::Add => Some(lhs + rhs),
                    Bo::Sub => Some(lhs - rhs),
                    Bo::Mul => Some(lhs * rhs),
                    Bo::Div => Some(lhs / rhs),
                }
            }
            ExprKind::Unary(op, expr) => {
                let expr = expr.try_fold(symtable)?;

                match op {
                    UnaryOp::Neg => Some(-expr),
                    UnaryOp::Not => Some(!expr),
                }
            }
            ExprKind::FuncCall(_) => None,
            ExprKind::LVal(LVal { ident }) => {
                // TODO: what if there are indices?
                let entry = symtable.lookup(ident).unwrap();
                Some(entry.comptime.as_ref()?.clone())
            }
            ExprKind::Coercion(expr) => {
                // Coerce the expression to the target type
                let expr = expr.try_fold(symtable)?;
                match self.ty.as_ref().unwrap().kind() {
                    Tk::Bool => {
                        let expr = match expr {
                            ComptimeVal::Bool(val) => val,
                            ComptimeVal::Int(val) => val != 0,
                        };
                        Some(ComptimeVal::bool(expr))
                    }
                    Tk::Int => {
                        let expr = match expr {
                            ComptimeVal::Bool(val) => val as i32,
                            ComptimeVal::Int(val) => val,
                        };
                        Some(ComptimeVal::int(expr))
                    }
                    Tk::Void | Tk::Func(..) => {
                        panic!("unsupported type coercion")
                    }
                }
            }
        }
    }

    /// Type check the expression.
    /// If `expect` is `Some`, the expression is expected to be coerced to the
    /// given type.
    pub fn type_check(self, expect: Option<&Type>, symtable: &SymbolTable) -> Self {
        // If the expression is already known, and no expected type is
        // given, return the expression as is.
        if self.ty.is_some() && expect.is_none() {
            return self;
        }

        let mut expr = match self.kind {
            ExprKind::Const(_) => self,
            ExprKind::Binary(op, lhs, rhs) => {
                // Type check the left and right hand side expressions
                let mut lhs = lhs.type_check(None, symtable);
                let mut rhs = rhs.type_check(None, symtable);

                let lhs_ty = lhs.ty();
                let rhs_ty = rhs.ty();

                // Coerce the types if needed
                match (lhs_ty.kind(), rhs_ty.kind()) {
                    (Tk::Bool, Tk::Int) => {
                        lhs = Expr::coercion(lhs, Type::int());
                    }
                    (Tk::Int, Tk::Bool) => {
                        rhs = Expr::coercion(rhs, Type::int());
                    }
                    _ => {
                        if lhs_ty != rhs_ty {
                            panic!("unsupported type coercion: {:?} -> {:?}", lhs_ty, rhs_ty);
                        }
                    }
                }

                let lhs_ty = lhs.ty().clone();

                // Create the binary expression
                let mut expr = Expr::binary(op, lhs, rhs);
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                        expr.ty = Some(lhs_ty.clone());
                    } // TODO: support other binary operations
                }
                expr
            }
            ExprKind::Coercion(_) => unreachable!(),
            ExprKind::FuncCall(FuncCall { ident, args }) => {
                // Lookup the function in the symbol table
                let entry = symtable.lookup(&ident).unwrap();

                let (param_tys, ret_ty) = entry.ty.unwrap_func();

                // Type check the arguments
                let args = args
                    .into_iter()
                    .zip(param_tys)
                    .map(|(arg, ty)| arg.type_check(Some(ty), symtable))
                    .collect();

                // Create the function call expression
                let mut expr = Expr::func_call(ident, args);
                expr.ty = Some(ret_ty.clone());
                expr
            }
            ExprKind::LVal(LVal { ident }) => {
                // Lookup the variable in the symbol table
                let entry = symtable.lookup(&ident).unwrap();

                // Create the left value expression
                let mut expr = Expr::lval(LVal { ident });
                expr.ty = Some(entry.ty.clone());
                expr
            }
            ExprKind::Unary(op, expr) => {
                // Type check the expression
                let mut expr = expr.type_check(None, symtable);

                // Coerce the expression to int if needed
                let ty = match op {
                    UnaryOp::Neg => {
                        if expr.ty().is_bool() {
                            // If this is bool, convert to int first
                            expr = Expr::coercion(expr, Type::int());
                        }
                        let ty = expr.ty();
                        if ty.is_int() {
                            ty.clone()
                        } else {
                            panic!("unsupported type for negation: {:?}", ty);
                        }
                    }
                    UnaryOp::Not => {
                        let ty = expr.ty();
                        if ty.is_bool() {
                            // Do nothing
                        } else if ty.is_int() {
                            // TODO: How do we convert int to bool?
                        } else {
                            panic!("unsupported type for logical not: {:?}", ty);
                        }
                        Type::bool()
                    }
                };

                // Create the unary expression
                let mut expr = Expr::unary(op, expr);
                expr.ty = Some(ty);
                expr
            }
        };

        // Coerce the expression to the expected type if needed
        if let Some(ty) = expect {
            if ty.is_int() || ty.is_bool() {
                match ty.kind() {
                    Tk::Bool => expr = Expr::coercion(expr, Type::bool()),
                    Tk::Int => expr = Expr::coercion(expr, Type::int()),
                    Tk::Func(..) | Tk::Void => {
                        unreachable!()
                    }
                }
                expr.ty = Some(ty.clone());
            } else if ty != expr.ty() {
                panic!("unsupported type coercion: {:?}", ty);
            }
        }

        // try to fold the expression into a constant value
        if let Some(comptime) = expr.try_fold(symtable) {
            expr = Expr::const_(comptime);
        }

        expr
    }
}
