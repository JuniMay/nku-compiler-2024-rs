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

    let src = std::fs::read_to_string("tests/testcase/functional_test/Advanced/006_arr_defn3.sy")?;
    let src = preprocess(&src);

    let mut ast = SysYParser::new().parse(&src).unwrap();

    ast.type_check();

    // println!("{:#?}", ast);

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
