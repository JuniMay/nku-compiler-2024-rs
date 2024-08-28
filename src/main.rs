use nkucc::frontend::{irgen, preprocess, SysYParser};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, NKUCC!");

    let src = std::fs::read_to_string("tests/sysy/basic.sy")?;
    let src = preprocess(&src);

    let mut ast = SysYParser::new().parse(&src).unwrap();

    ast.type_check();

    println!("{:#?}", ast);

    let ir = irgen(&ast, 8);

    println!("{}", ir);

    Ok(())
}
