use std::fmt;

use crate::backend::regs::Reg;
use crate::backend::{PReg, RegKind};

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

pub const fn display_preg(reg: PReg) -> &'static str {
    match reg.kind() {
        RegKind::General => match reg.num() {
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
    }
}

impl fmt::Display for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.num() < 32 {
            write!(f, "{}", display_preg(*self))
        } else {
            unreachable!("invalid register number")
        }
    }
}

pub fn display(reg: Reg) -> String {
    match reg {
        Reg::P(reg) => display_preg(reg).into(),
        Reg::V(reg) => format!("{}", reg),
    }
}

pub const INT_ARG_REGS: [PReg; 8] = [a0(), a1(), a2(), a3(), a4(), a5(), a6(), a7()];

pub const CALLER_SAVED_REGS: [PReg; 16] = [
    ra(),
    t0(),
    t1(),
    t2(),
    t3(),
    t4(),
    t5(),
    t6(),
    a0(),
    a1(),
    a2(),
    a3(),
    a4(),
    a5(),
    a6(),
    a7(),
];

pub const CALLEE_SAVED_REGS: [PReg; 13] = [
    sp(),
    s0(),
    s1(),
    s2(),
    s3(),
    s4(),
    s5(),
    s6(),
    s7(),
    s8(),
    s9(),
    s10(),
    s11(),
];

pub const RETURN_REGS: [PReg; 1] = [
    a0(),
    // since we only support 64-bit return value, a1 is not used
    // a1(),
];
