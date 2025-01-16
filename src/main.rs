use clap::{Arg, ArgMatches, Command};
use nkucc::{
    backend::codegen::{self, CodegenContext},
    frontend::{irgen, preprocess, Optimize, SysYParser},
};

fn parse_arguments() -> ArgMatches {
    Command::new("nkucc")
        .arg(
            Arg::new("output")
                .short('o')
                .required(true)
                .help("The output assembly"),
        )
        .arg(Arg::new("source").required(true).help("The source code"))
        .arg(
            Arg::new("s_flag")
                .short('S')
                .action(clap::ArgAction::Count)
                .help("Output an assembly file"),
        )
        .arg(
            Arg::new("opt")
                .short('O')
                .help("Optimization level")
                .default_value("0"),
        )
        .arg(
            Arg::new("emit-ast")
                .long("emit-ast")
                .help("Emit the AST to the specified file"),
        )
        .arg(
            Arg::new("emit-llvm-ir")
                .long("emit-llvm-ir")
                .help("Emit the IR to the specified file"),
        )
        .arg(
            Arg::new("emit-ir")
                .long("emit-ir")
                .help("Emit the backend res to the specified file"),
        )
        .get_matches()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, NKUCC!");

    let matches = parse_arguments();

    // Extract arguments
    let output = matches.get_one::<String>("output");
    let emit_llvm_ir = matches.get_one::<String>("emit-llvm-ir");
    let emit_ir = matches.get_one::<String>("emit-ir");
    let opt_level = matches.get_one::<String>("opt").unwrap();
    let source = matches.get_one::<String>("source").unwrap();
    let emit_assembly = matches.get_count("s_flag") > 0;

    // Validate source file
    let src = std::fs::read_to_string(source)?;

    let src = preprocess(&src);

    let mut ast = SysYParser::new().parse(&src).unwrap();

    ast.type_check();

    // println!("{:#?}", ast);

    let ir = irgen(&ast, 8);

    let mut opt = Optimize::new(ir, opt_level.parse().unwrap());
    opt.optmize();

    if let Some(ir_file) = emit_llvm_ir {
        std::fs::write(ir_file, opt.ir().to_string()).unwrap();
    }

    if let Some(asm_file) = output {
        let mut ctx = CodegenContext::new(opt.ir());
        ctx.codegen();
        let asm = ctx.finish();
        std::fs::write(asm_file, asm.display().to_string()).unwrap();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use nkucc::{
        frontend::{irgen, preprocess, Optimize, SysYParser},
        ir::{Block, BlockEdge, Context, InstKind},
    };

    #[test]
    fn test_parse_arguments() {
        let src =
            std::fs::read_to_string("tests/testcase/functional_test/Advanced/000000dumbass.sy")
                .unwrap();
        let src = preprocess(&src);

        let mut ast = SysYParser::new().parse(&src).unwrap();

        ast.type_check();

        println!("{:#?}", ast);
    }

    #[test]
    fn test_opt_irgen() {
        // let src = std::fs::read_to_string("tests/testcase/optimize_test/eliUnreachablebb/blockafterret.sy").unwrap();
        let src = std::fs::read_to_string("tests/small-test/complex_phi.sy").unwrap();
        let src = preprocess(&src);

        let mut ast = SysYParser::new().parse(&src).unwrap();

        ast.type_check();

        // println!("{:#?}", ast);

        let ir = irgen(&ast, 8);

        println!("{}", &ir.to_string());

        let mut opt = Optimize::new(ir, 1);
        opt.optmize();

        println!("{}", opt.ir().to_string());
    }

    #[test]
    fn test_irgen_cfg() {
        let src = std::fs::read_to_string("tests/sysy/basic.sy").unwrap();
        let src = preprocess(&src);

        let mut ast = SysYParser::new().parse(&src).unwrap();

        ast.type_check();

        // println!("{:#?}", ast);

        let ir = irgen(&ast, 8);
        let funcs = ir.funcs();

        let mut st = HashSet::new();

        for func in funcs {
            println!("Func: {}", func.name(&ir));
            let head = func.head(&ir);
            if let None = head {
                println!("Head is None!");
                continue;
            }
            let head = head.unwrap();
            dfs(&head, &ir, &mut st);
            println!("========================");
        }

        println!("ir:");
        println!("{}", ir.to_string());
    }

    fn dfs(u: &Block, ctx: &Context, st: &mut HashSet<BlockEdge>) {
        for edge in u.successors(ctx) {
            if st.contains(edge) {
                continue;
            }
            st.insert(edge.clone());
            let v = edge.to();
            let inst = edge.inst();
            let true_br = edge.is_true_branch();
            dfs(&v, ctx, st);
            println!(
                "Edge: {:?} -> {:?}, {}",
                u,
                v,
                if let InstKind::Br = inst.kind(ctx) {
                    "Br"
                } else {
                    if true_br {
                        "Cond Br: true"
                    } else {
                        "Cond Br: false"
                    }
                }
            )
        }
    }
}
