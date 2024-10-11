use nkucc::frontend::{parser::NumberParser, ComptimeVal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, NKUCC!");
    let int_str = String::from("/* xxxx */ 114514 // 123");
    let hex_str = String::from("/* xxxx */ 0xfff // 123");
    assert_eq!(parse_int(&int_str), 114514);
    assert_eq!(parse_int(&hex_str), 0xfff);
    println!("lexer test passed");
    Ok(())
}

fn parse_int(str: &str) -> i32 {
    // TODO: use NumberParser to parse int
    let num = NumberParser::new().parse(str).unwrap();
    match num {
        ComptimeVal::Int(n) => n,
        ComptimeVal::Bool(n) => n as i32,
    }
}
