//! Parser for SysY language.
//! This module guides LALRPOP to generate the parser for SysY language.

use lalrpop_util::lalrpop_mod;

// Define a module named `parser` and generate the parser according to the
// grammar in `sysy.lalrpop` to `src/frontend/sysy.rs`.
lalrpop_mod!(#[allow(clippy::all)] pub parser, "/frontend/sysy.rs");

// Make top-level parser public.
pub use parser::SysYParser;
