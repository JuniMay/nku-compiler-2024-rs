// use std::path::Path;
// use std::process::Command as SysCommand;

// use clap::{Arg, ArgMatches, Command};
// use nkucc::frontend::{irgen, preprocess, SysYParser};

// fn parse_arguments() -> ArgMatches {
//     Command::new("nkucc")
//         .arg(
//             Arg::new("output")
//                 .short('o')
//                 .required(true)
//                 .help("The output assembly"),
//         )
//         .arg(Arg::new("source").required(true).help("The source code"))
//         .arg(
//             Arg::new("s_flag")
//                 .short('S')
//                 .action(clap::ArgAction::Count)
//                 .help("Output an assembly file"),
//         )
//         .arg(
//             Arg::new("opt")
//                 .short('O')
//                 .help("Optimization level")
//                 .default_value("0"),
//         )
//         .arg(
//             Arg::new("emit-ast")
//                 .long("emit-ast")
//                 .help("Emit the AST to the specified file"),
//         )
//         .arg(
//             Arg::new("emit-llvm-ir")
//                 .long("emit-llvm-ir")
//                 .help("Emit the IR to the specified file"),
//         )
//         .get_matches()
// }

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     println!("Hello, NKUCC!");

//     let matches = parse_arguments();

//     // Extract arguments
//     let output = matches.get_one::<String>("output");
//     let emit_llvm_ir = matches.get_one::<String>("emit-llvm-ir");
//     let opt_level = matches.get_one::<String>("opt").unwrap();
//     let source = matches.get_one::<String>("source").unwrap();
//     let emit_assembly = matches.get_count("s_flag") > 0;

//     // Validate source file
//     let src = std::fs::read_to_string(source)?;

//     let src = preprocess(&src);

//     let mut ast = SysYParser::new().parse(&src).unwrap();

//     ast.type_check();

//     println!("{:#?}", ast);

//     let ir = irgen(&ast, 8);

//     if let Some(ir_file) = emit_llvm_ir {
//         std::fs::write(ir_file, ir.to_string()).unwrap();
//     }

//     Ok(())
// }

// use nkucc::frontend::{parser::NumberParser, ComptimeVal};
use nkucc::frontend::{irgen, preprocess, SysYParser};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // println!("Hello, NKUCC!");
    // let int_str = String::from("/* xxxx */ 1145141919810 // 123");
    // let hex_str = String::from("/* xxxx */ 0xfff // 123");
    // let num = 1145141919810 as i64;
    // assert_eq!(parse_int(&int_str), num as i32);
    // assert_eq!(parse_int(&hex_str), 0xfff);
    // println!("lexer test passed");
    // Ok(())

    let src = std::fs::read_to_string("tests/testcase/functional_test/Advanced/005_float_defn.sy")?;
    let src = preprocess(&src);

    let mut ast = SysYParser::new().parse(&src).unwrap();

    ast.type_check();

    println!("{:#?}", ast);

    // Ok(());

    let ir = irgen(&ast, 8);

    println!("{}", ir);

    Ok(())
}

// fn parse_int(str: &str) -> i32 {
//     // TODO: use NumberParser to parse int
//     let num = NumberParser::new().parse(str).unwrap();
//     match num {
//         ComptimeVal::Int(n) => n,
//         ComptimeVal::Bool(n) => n as i32,
//         _ => panic!("parse_int failed"),
//     }
// }
