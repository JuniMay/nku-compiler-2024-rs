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

impl fmt::Display for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str = match self.kind() {
            RegKind::General => match self.num() {
                0 => "zero",
                1 => "ra",
                2 => "sp",
                3 => "gp",
                4 => "tp",
                5 => "t0",
                6 => "t1",
                7 => "t2",
                8 => "s0",
                9 => "s1",
                10 => "a0",
                11 => "a1",
                12 => "a2",
                13 => "a3",
                14 => "a4",
                15 => "a5",
                16 => "a6",
                17 => "a7",
                18 => "s2",
                19 => "s3",
                20 => "s4",
                21 => "s5",
                22 => "s6",
                23 => "s7",
                24 => "s8",
                25 => "s9",
                26 => "s10",
                27 => "s11",
                28 => "t3",
                29 => "t4",
                30 => "t5",
                31 => "t6",
                _ => "<invalid>",
            },
        };
        write!(f, "{}", str)
    }
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

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::P(preg) => write!(f, "{}", preg),
            Reg::V(vreg) => write!(f, "{}", vreg),
        }
    }
}

pub const fn zero() -> PReg { PReg::new(0, RegKind::General) }

pub const fn ra() -> PReg { PReg::new(1, RegKind::General) }

pub const fn sp() -> PReg { PReg::new(2, RegKind::General) }

pub const fn gp() -> PReg { PReg::new(3, RegKind::General) }

pub const fn tp() -> PReg { PReg::new(4, RegKind::General) }

pub const fn t0() -> PReg { PReg::new(5, RegKind::General) }

pub const fn t1() -> PReg { PReg::new(6, RegKind::General) }

pub const fn t2() -> PReg { PReg::new(7, RegKind::General) }

pub const fn s0() -> PReg { PReg::new(8, RegKind::General) }

pub const fn fp() -> PReg { s0() }

pub const fn s1() -> PReg { PReg::new(9, RegKind::General) }

pub const fn a0() -> PReg { PReg::new(10, RegKind::General) }

pub const fn a1() -> PReg { PReg::new(11, RegKind::General) }

pub const fn a2() -> PReg { PReg::new(12, RegKind::General) }

pub const fn a3() -> PReg { PReg::new(13, RegKind::General) }

pub const fn a4() -> PReg { PReg::new(14, RegKind::General) }

pub const fn a5() -> PReg { PReg::new(15, RegKind::General) }

pub const fn a6() -> PReg { PReg::new(16, RegKind::General) }

pub const fn a7() -> PReg { PReg::new(17, RegKind::General) }

pub const fn s2() -> PReg { PReg::new(18, RegKind::General) }

pub const fn s3() -> PReg { PReg::new(19, RegKind::General) }

pub const fn s4() -> PReg { PReg::new(20, RegKind::General) }

pub const fn s5() -> PReg { PReg::new(21, RegKind::General) }

pub const fn s6() -> PReg { PReg::new(22, RegKind::General) }

pub const fn s7() -> PReg { PReg::new(23, RegKind::General) }

pub const fn s8() -> PReg { PReg::new(24, RegKind::General) }

pub const fn s9() -> PReg { PReg::new(25, RegKind::General) }

pub const fn s10() -> PReg { PReg::new(26, RegKind::General) }

pub const fn s11() -> PReg { PReg::new(27, RegKind::General) }

pub const fn t3() -> PReg { PReg::new(28, RegKind::General) }

pub const fn t4() -> PReg { PReg::new(29, RegKind::General) }

pub const fn t5() -> PReg { PReg::new(30, RegKind::General) }

pub const fn t6() -> PReg { PReg::new(31, RegKind::General) }

// TODO: You may need to add more registers here.
