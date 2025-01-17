use nkucc::{
    frontend::{irgen, preprocess, SysYParser},
    backend::codegen::CodegenContext,
};

#[test]
fn test_register_allocation() {
    // 测试用的 SysY 代码
    let source = r#"
    int main() {
        int a = 1;
        int b = 2;
        int c = a + b;
        return c;
    }
    "#;

    // Frontend
    let src = preprocess(source);
    let mut ast = SysYParser::new().parse(&src).unwrap();
    ast.type_check();

    // IRGen
    let ir = irgen(&ast, 8);
    println!("=== IR ===");
    println!("{}", ir);

    // Initialize codegen
    let mut codegen_ctx = CodegenContext::new(&ir);
    codegen_ctx.mctx_mut().set_arch("rv64imafdc_zba_zbb");

    // 生成虚拟寄存器版本的汇编
    codegen_ctx.codegen();
    println!("\n=== Before Register Allocation ===");
    println!("{}", codegen_ctx.mctx().display());

    // 寄存器分配
    codegen_ctx.regalloc();
    println!("\n=== After Register Allocation ===");
    println!("{}", codegen_ctx.mctx().display());

    // 栈帧调整
    codegen_ctx.after_regalloc();
    println!("\n=== Final Assembly ===");
    println!("{}", codegen_ctx.finish().display());
}