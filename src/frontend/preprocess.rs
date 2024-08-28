//! Preprocess of SysY source file.
//!
//! The only macro replacement in SysY is `starttime` and `stoptime`.
//! - `starttime()` is replaced by `_sysy_starttime(__LINE__)`
//! - `stoptime()` is replaced by `_sysy_stoptime(__LINE__)`
//!
//! Additionally, the `__LINE__` is replaced by the line number of the source
//! file.

/// Preprocess the source code of SysY.
///
/// TODO: Implement the macro replacement.
pub fn preprocess(src: &str) -> String { src.to_string() }
