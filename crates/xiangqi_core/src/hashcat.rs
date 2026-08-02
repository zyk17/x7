//! px0 `src/utils/hashcat.h:34-54`。

pub const fn hash(val: u64) -> u64 {
    0xfad0d7f2fbb059f1u64
        .wrapping_mul(val.wrapping_add(0xbaad41cdcb839961))
        .wrapping_add(0x7acec0050bf82f43u64.wrapping_mul((val >> 31).wrapping_add(0xd571b3a92b1b2755)))
}

pub const fn hash_cat(state: u64, x: u64) -> u64 {
    state
        ^ (0x299799adf0d95defu64
            .wrapping_add(hash(x))
            .wrapping_add(state << 6)
            .wrapping_add(state >> 2))
}

pub fn hash_cat_u128s(values: &[u128]) -> u64 {
    let mut hash = 0u64;
    for &value in values {
        hash = hash_cat(hash, (value >> 64) as u64);
        hash = hash_cat(hash, value as u64);
    }
    hash
}
