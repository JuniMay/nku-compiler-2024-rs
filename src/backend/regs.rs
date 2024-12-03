use std::fmt;

/// The kind of a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegKind {
    /// The general purpose register.
    General,
}

/// The register.
/// 
/// It can be either a physical register or a virtual register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    P(PReg),
    V(VReg),
}

impl Reg {
    /// Get the kind of the register.
    pub fn kind(&self) -> RegKind {
        match self {
            Reg::P(preg) => preg.kind(),
            Reg::V(vreg) => vreg.kind(),
        }
    }

    /// Check if the register is a physical register.
    pub fn is_preg(&self) -> bool { matches!(self, Reg::P(_)) }

    /// Check if the register is a virtual register.
    pub fn is_vreg(&self) -> bool { matches!(self, Reg::V(_)) }
}

/// The physical register.
///
/// Cranelift uses a bit-encoded representation, but here just separate the
/// register number and the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(u8, RegKind);

impl PReg {
    /// Create a new physical register.
    pub const fn new(num: u8, kind: RegKind) -> Self { Self(num, kind) }

    /// Get the number of the register.
    pub const fn num(&self) -> u8 { self.0 }

    /// Get the kind of the register.
    pub const fn kind(&self) -> RegKind { self.1 }
}

/// The virtual register.
///
/// Let's hope the number of virtual registers does not exceed [u32::MAX].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(u32, RegKind);

impl VReg {
    /// Create a new virtual register.
    pub fn new(num: u32, kind: RegKind) -> Self { Self(num, kind) }

    /// Get the number of the register.
    pub fn num(&self) -> u32 { self.0 }

    /// Get the kind of the register.
    pub fn kind(&self) -> RegKind { self.1 }
}

impl From<VReg> for Reg {
    fn from(vreg: VReg) -> Self { Self::V(vreg) }
}

impl From<PReg> for Reg {
    fn from(preg: PReg) -> Self { Self::P(preg) }
}

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            match self.1 {
                RegKind::General => "$r",
            },
            self.0
        )
    }
}
