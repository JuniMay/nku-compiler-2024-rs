use lalrpop_util::lalrpop_mod;

lalrpop_mod!(#[allow(clippy::all)] pub parser, "/frontend/sysy.rs");

pub use parser::SysYParser;
