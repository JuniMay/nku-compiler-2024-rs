use std::fmt;

/// The kind of a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegKind {
    /// The general purpose register.
    General,
    Float,
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
    pub fn is_preg(&self) -> bool {
        matches!(self, Reg::P(_))
    }

    /// Check if the register is a virtual register.
    pub fn is_vreg(&self) -> bool {
        matches!(self, Reg::V(_))
    }
}

/// The physical register.
///
/// Cranelift uses a bit-encoded representation, but here just separate the
/// register number and the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(u8, RegKind);

impl PReg {
    /// Create a new physical register.
    pub const fn new(num: u8, kind: RegKind) -> Self {
        Self(num, kind)
    }

    /// Get the number of the register.
    pub const fn num(&self) -> u8 {
        self.0
    }

    /// Get the kind of the register.
    pub const fn kind(&self) -> RegKind {
        self.1
    }
}

/// The virtual register.
///
/// Let's hope the number of virtual registers does not exceed [u32::MAX].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(u32, RegKind);

impl VReg {
    /// Create a new virtual register.
    pub fn new(num: u32, kind: RegKind) -> Self {
        Self(num, kind)
    }

    /// Get the number of the register.
    pub fn num(&self) -> u32 {
        self.0
    }

    /// Get the kind of the register.
    pub fn kind(&self) -> RegKind {
        self.1
    }
}

impl From<VReg> for Reg {
    fn from(vreg: VReg) -> Self {
        Self::V(vreg)
    }
}

impl From<PReg> for Reg {
    fn from(preg: PReg) -> Self {
        Self::P(preg)
    }
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
            RegKind::Float => match self.num() {
                0 => "f0",
                1 => "f1",
                2 => "f2",
                3 => "f3",
                4 => "f4",
                5 => "f5",
                6 => "f6",
                7 => "f7",
                8 => "f8",
                9 => "f9",
                10 => "f10",
                11 => "f11",
                12 => "f12",
                13 => "f13",
                14 => "f14",
                15 => "f15",
                16 => "f16",
                17 => "f17",
                18 => "f18",
                19 => "f19",
                20 => "f20",
                21 => "f21",
                22 => "f22",
                23 => "f23",
                24 => "f24",
                25 => "f25",
                26 => "f26",
                27 => "f27",
                28 => "f28",
                29 => "f29",
                30 => "f30",
                31 => "f31",
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
                RegKind::Float => "$f",
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

pub const fn zero() -> PReg {
    PReg::new(0, RegKind::General)
}

pub const fn ra() -> PReg {
    PReg::new(1, RegKind::General)
}

pub const fn sp() -> PReg {
    PReg::new(2, RegKind::General)
}

pub const fn gp() -> PReg {
    PReg::new(3, RegKind::General)
}

pub const fn tp() -> PReg {
    PReg::new(4, RegKind::General)
}

pub const fn t0() -> PReg {
    PReg::new(5, RegKind::General)
}

pub const fn t1() -> PReg {
    PReg::new(6, RegKind::General)
}

pub const fn t2() -> PReg {
    PReg::new(7, RegKind::General)
}

pub const fn s0() -> PReg {
    PReg::new(8, RegKind::General)
}

pub const fn fp() -> PReg {
    s0()
}

pub const fn s1() -> PReg {
    PReg::new(9, RegKind::General)
}

pub const fn a0() -> PReg {
    PReg::new(10, RegKind::General)
}

pub const fn a1() -> PReg {
    PReg::new(11, RegKind::General)
}

pub const fn a2() -> PReg {
    PReg::new(12, RegKind::General)
}

pub const fn a3() -> PReg {
    PReg::new(13, RegKind::General)
}

pub const fn a4() -> PReg {
    PReg::new(14, RegKind::General)
}

pub const fn a5() -> PReg {
    PReg::new(15, RegKind::General)
}

pub const fn a6() -> PReg {
    PReg::new(16, RegKind::General)
}

pub const fn a7() -> PReg {
    PReg::new(17, RegKind::General)
}

pub const fn s2() -> PReg {
    PReg::new(18, RegKind::General)
}

pub const fn s3() -> PReg {
    PReg::new(19, RegKind::General)
}

pub const fn s4() -> PReg {
    PReg::new(20, RegKind::General)
}

pub const fn s5() -> PReg {
    PReg::new(21, RegKind::General)
}

pub const fn s6() -> PReg {
    PReg::new(22, RegKind::General)
}

pub const fn s7() -> PReg {
    PReg::new(23, RegKind::General)
}

pub const fn s8() -> PReg {
    PReg::new(24, RegKind::General)
}

pub const fn s9() -> PReg {
    PReg::new(25, RegKind::General)
}

pub const fn s10() -> PReg {
    PReg::new(26, RegKind::General)
}

pub const fn s11() -> PReg {
    PReg::new(27, RegKind::General)
}

pub const fn t3() -> PReg {
    PReg::new(28, RegKind::General)
}

pub const fn t4() -> PReg {
    PReg::new(29, RegKind::General)
}

pub const fn t5() -> PReg {
    PReg::new(30, RegKind::General)
}

pub const fn t6() -> PReg {
    PReg::new(31, RegKind::General)
}

// TODO: You may need to add more registers here.
pub const fn f0() -> PReg {
    PReg::new(0, RegKind::Float)
}

pub const fn f1() -> PReg {
    PReg::new(1, RegKind::Float)
}

pub const fn f2() -> PReg {
    PReg::new(2, RegKind::Float)
}

pub const fn f3() -> PReg {
    PReg::new(3, RegKind::Float)
}

pub const fn f4() -> PReg {
    PReg::new(4, RegKind::Float)
}

pub const fn f5() -> PReg {
    PReg::new(5, RegKind::Float)
}

pub const fn f6() -> PReg {
    PReg::new(6, RegKind::Float)
}

pub const fn f7() -> PReg {
    PReg::new(7, RegKind::Float)
}

pub const fn f8() -> PReg {
    PReg::new(8, RegKind::Float)
}

pub const fn f9() -> PReg {
    PReg::new(9, RegKind::Float)
}

pub const fn fa0() -> PReg {
    PReg::new(10, RegKind::Float)
}

pub const fn fa1() -> PReg {
    PReg::new(11, RegKind::Float)
}

pub const fn fa2() -> PReg {
    PReg::new(12, RegKind::Float)
}

pub const fn fa3() -> PReg {
    PReg::new(13, RegKind::Float)
}

pub const fn fa4() -> PReg {
    PReg::new(14, RegKind::Float)
}

pub const fn fa5() -> PReg {
    PReg::new(15, RegKind::Float)
}

pub const fn fa6() -> PReg {
    PReg::new(16, RegKind::Float)
}

pub const fn fa7() -> PReg {
    PReg::new(17, RegKind::Float)
}

pub const fn f18() -> PReg {
    PReg::new(18, RegKind::Float)
}

pub const fn f19() -> PReg {
    PReg::new(19, RegKind::Float)
}

pub const fn f20() -> PReg {
    PReg::new(20, RegKind::Float)
}

pub const fn f21() -> PReg {
    PReg::new(21, RegKind::Float)
}

pub const fn f22() -> PReg {
    PReg::new(22, RegKind::Float)
}

pub const fn f23() -> PReg {
    PReg::new(23, RegKind::Float)
}

pub const fn f24() -> PReg {
    PReg::new(24, RegKind::Float)
}

pub const fn f25() -> PReg {
    PReg::new(25, RegKind::Float)
}

pub const fn f26() -> PReg {
    PReg::new(26, RegKind::Float)
}

pub const fn f27() -> PReg {
    PReg::new(27, RegKind::Float)
}

pub const fn f28() -> PReg {
    PReg::new(28, RegKind::Float)
}

pub const fn f29() -> PReg {
    PReg::new(29, RegKind::Float)
}

pub const fn f30() -> PReg {
    PReg::new(30, RegKind::Float)
}

pub const fn f31() -> PReg {
    PReg::new(31, RegKind::Float)
}

pub const fn get_arg(index: usize) -> PReg {
    match index {
        0 => a0(),
        1 => a1(),
        2 => a2(),
        3 => a3(),
        4 => a4(),
        5 => a5(),
        6 => a6(),
        7 => a7(),
        _ => panic!("Invalid argument register index"),
    }
}

pub const fn get_farg(index: usize) -> PReg {
    match index {
        0 => fa0(),
        1 => fa1(),
        2 => fa2(),
        3 => fa3(),
        4 => fa4(),
        5 => fa5(),
        6 => fa6(),
        7 => fa7(),
        _ => panic!("Invalid floating-point argument register index"),
    }
}
