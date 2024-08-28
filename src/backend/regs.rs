use std::fmt;

/// The kind of a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegKind {
    /// The general purpose register.
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    P(PReg),
    V(VReg),
}

impl Reg {
    pub fn kind(&self) -> RegKind {
        match self {
            Reg::P(preg) => preg.kind(),
            Reg::V(vreg) => vreg.kind(),
        }
    }

    pub fn is_preg(&self) -> bool { matches!(self, Reg::P(_)) }

    pub fn is_vreg(&self) -> bool { matches!(self, Reg::V(_)) }
}

/// The physical register.
///
/// Cranelift uses a bit-encoded representation, but here just separate the
/// register number and the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(u8, RegKind);

impl PReg {
    pub const fn new(num: u8, kind: RegKind) -> Self { Self(num, kind) }

    pub const fn num(&self) -> u8 { self.0 }

    pub const fn kind(&self) -> RegKind { self.1 }
}

/// The virtual register.
///
/// Let's hope the number of virtual registers does not exceed [u32::MAX].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(u32, RegKind);

impl VReg {
    pub fn new(num: u32, kind: RegKind) -> Self { Self(num, kind) }

    pub fn num(&self) -> u32 { self.0 }

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
