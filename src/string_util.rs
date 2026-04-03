///  Verified string formatting utilities for WGSL emission.
///  No external_body — every function is fully verified.

use vstd::prelude::*;
use vstd::string::*;

verus! {

///  Get single-char string for digit 0-9.
pub fn digit_str(d: u8) -> (s: &'static str)
    requires d < 10
{
    if d == 0 { "0" } else if d == 1 { "1" } else if d == 2 { "2" }
    else if d == 3 { "3" } else if d == 4 { "4" } else if d == 5 { "5" }
    else if d == 6 { "6" } else if d == 7 { "7" } else if d == 8 { "8" }
    else { "9" }
}

///  Get single-char hex string for nibble 0-15.
pub fn hex_digit_str(d: u8) -> (s: &'static str)
    requires d < 16
{
    if d == 0 { "0" } else if d == 1 { "1" } else if d == 2 { "2" }
    else if d == 3 { "3" } else if d == 4 { "4" } else if d == 5 { "5" }
    else if d == 6 { "6" } else if d == 7 { "7" } else if d == 8 { "8" }
    else if d == 9 { "9" } else if d == 10 { "a" } else if d == 11 { "b" }
    else if d == 12 { "c" } else if d == 13 { "d" } else if d == 14 { "e" }
    else { "f" }
}

///  Convert a u32 to its decimal string representation.
pub fn u32_to_string(n: u32) -> (s: String)
{
    if n == 0 {
        return String::from_str("0");
    }
    //  Collect digits in reverse order
    let mut digits: Vec<u8> = Vec::new();
    let mut remaining: u32 = n;
    while remaining > 0
        invariant
            forall|i: int| 0 <= i < digits@.len() ==> digits@[i] < 10,
        decreases remaining,
    {
        let d: u8 = (remaining % 10) as u8;
        digits.push(d);
        remaining = remaining / 10;
    }
    //  Build string from digits in reverse
    let mut s = String::from_str("");
    let len = digits.len();
    let mut i: usize = 0;
    while i < len
        invariant
            i <= len,
            len == digits@.len(),
            forall|j: int| 0 <= j < digits@.len() ==> digits@[j] < 10,
        decreases len - i,
    {
        let idx = len - 1 - i;
        let d: u8 = digits[idx];
        s.append(digit_str(d));
        i = i + 1;
    }
    s
}

///  Convert a u64 to its decimal string representation.
pub fn u64_to_string(n: u64) -> (s: String)
{
    if n == 0 {
        return String::from_str("0");
    }
    let mut digits: Vec<u8> = Vec::new();
    let mut remaining: u64 = n;
    while remaining > 0
        invariant
            forall|i: int| 0 <= i < digits@.len() ==> digits@[i] < 10,
        decreases remaining,
    {
        let d: u8 = (remaining % 10) as u8;
        digits.push(d);
        remaining = remaining / 10;
    }
    let mut s = String::from_str("");
    let len = digits.len();
    let mut i: usize = 0;
    while i < len
        invariant
            i <= len,
            len == digits@.len(),
            forall|j: int| 0 <= j < digits@.len() ==> digits@[j] < 10,
        decreases len - i,
    {
        let idx = len - 1 - i;
        let d: u8 = digits[idx];
        s.append(digit_str(d));
        i = i + 1;
    }
    s
}

///  Convert an i64 to its decimal string representation.
pub fn i64_to_string(n: i64) -> (s: String)
{
    if n >= 0 {
        u64_to_string(n as u64)
    } else if n == i64::MIN {
        String::from_str("-9223372036854775808")
    } else {
        let mut s = String::from_str("-");
        let pos_str = u64_to_string((-n) as u64);
        s.append(pos_str.as_str());
        s
    }
}

///  Convert a u32 to 8-digit zero-padded hexadecimal with "0x" prefix.
///  Used for: bitcast<f32>(0xNNNNNNNNu) to emit float constants exactly.
pub fn u32_to_hex(n: u32) -> (s: String)
{
    let mut s = String::from_str("0x");
    let mut i: u32 = 0;
    while i < 8
        invariant i <= 8,
        decreases 8 - i,
    {
        let shift: u32 = (7 - i) * 4;
        let masked: u32 = (n >> shift) & 0xF;
        assert(masked < 16) by(bit_vector)
            requires masked == (n >> shift) & 0xF;
        let nibble: u8 = masked as u8;
        s.append(hex_digit_str(nibble));
        i = i + 1;
    }
    s
}

///  Helper: append a u32 as decimal suffix (e.g., "u" suffix for WGSL).
pub fn append_u32(s: &mut String, n: u32, suffix: &str)
{
    let num = u32_to_string(n);
    s.append(num.as_str());
    s.append(suffix);
}

///  Helper: append an i64 as decimal.
pub fn append_i64(s: &mut String, n: i64)
{
    let num = i64_to_string(n);
    s.append(num.as_str());
}

} // verus!
