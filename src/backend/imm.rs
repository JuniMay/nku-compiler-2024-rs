//! Immediates in RISC-V ISA.

use core::fmt;

/// 12-bit immediate used in some instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Imm12(u16);

impl Imm12 {
    /// Sign-extend imm12 to i16.
    pub fn as_i16(&self) -> i16 {
        (self.0 << 4) as i16 >> 4 // signext
    }

    /// Try to create an Imm12 from an i64.
    /// 
    /// Returns None if the value is out of range.
    pub fn try_from_i64(x: i64) -> Option<Self> {
        if (-2048..=2047).contains(&x) {
            Some(Imm12(x as u16 & 0xfff))
        } else {
            None
        }
    }

    /// Try to create an Imm12 from a u64.
    /// 
    /// Returns None if the value is out of range.
    pub fn try_from_u64(x: u64) -> Option<Self> { Self::try_from_i64(x as i64) }

    /// Get the raw bits.
    pub fn bits(&self) -> u16 { self.0 }
}

impl PartialOrd for Imm12 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.as_i16().partial_cmp(&other.as_i16())
    }
}

impl fmt::Display for Imm12 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.as_i16()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imm12_try_from_i64() {
        assert_eq!(Imm12::try_from_i64(0), Some(Imm12(0)));
        assert_eq!(Imm12::try_from_i64(1), Some(Imm12(1)));
        assert_eq!(Imm12::try_from_i64(2047), Some(Imm12(2047)));
        assert_eq!(
            Imm12::try_from_i64(-2048),
            Some(Imm12((-2048i16) as u16 & 0xfff))
        );
        assert_eq!(Imm12::try_from_i64(-2049), None);
        assert_eq!(Imm12::try_from_i64(2048), None);
    }

    #[test]
    fn imm12_try_from_u64() {
        assert_eq!(Imm12::try_from_u64(0), Some(Imm12(0)));
        assert_eq!(Imm12::try_from_u64(1), Some(Imm12(1)));
        assert_eq!(Imm12::try_from_u64(2047), Some(Imm12(2047)));
        assert_eq!(Imm12::try_from_u64(2048), None);
    }
}
