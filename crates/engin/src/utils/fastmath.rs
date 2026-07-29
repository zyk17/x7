//! px0 `src/utils/fastmath.h:38-103`。
//!
//! 这些近似是 px0 PUCT 语义的一部分：`FastLog` 参与 PUCT 和 WDL 分数换算，因此改用
//! 平台 `libm` 会改变决胜顺序和输出分数。

/// px0 `FastLog2`（`fastmath.h:42-57`）。不检查范围。
pub fn fast_log2(value: f32) -> f32 {
    let mut bits = value.to_bits();
    let exponent = bits >> 23;
    bits = (bits & 0x7f_ffff) | (0x7f << 23);
    let mantissa = f32::from_bits(bits) - 1.0;
    mantissa * (1.346_555_2 - 0.346_555_23 * mantissa) - 127.0 + exponent as f32
}

/// px0 `FastExp2` (`fastmath.h:64-79`).
pub fn fast_exp2(value: f32) -> f32 {
    if value < -126.0 {
        return 0.0;
    }
    let exponent = if value < 0.0 { value as i32 - 1 } else { value as i32 };
    let fraction = value - exponent as f32;
    let mantissa = 1.0 + fraction * (0.660_233_9 + 0.339_766_06 * fraction);
    let bits = mantissa.to_bits().wrapping_add((exponent as u32).wrapping_shl(23));
    f32::from_bits(bits)
}

/// px0 `FastLog` (`fastmath.h:81-83`).
#[allow(clippy::approx_constant)] // Keep px0's deliberately rounded f32 coefficient.
pub fn fast_log(value: f32) -> f32 {
    0.693_147_2 * fast_log2(value)
}

/// px0 `FastExp` (`fastmath.h:85-86`).
#[allow(clippy::approx_constant)] // Keep px0's deliberately rounded f32 coefficient.
pub fn fast_exp(value: f32) -> f32 {
    fast_exp2(1.442_695 * value)
}

/// px0 `FastLogistic` (`fastmath.h:88-92`).
pub fn fast_logistic(value: f32) -> f32 {
    if value > 20.0 {
        return 1.0;
    }
    if value < -20.0 {
        return 0.0;
    }
    1.0 / (1.0 + fast_exp(-value))
}

#[cfg(test)]
mod tests {
    use super::{fast_exp2, fast_log, fast_log2, fast_logistic};

    #[test]
    fn px0_fastmath_preserves_power_of_two_anchors() {
        assert!((fast_log2(1.0) - 0.0).abs() < f32::EPSILON);
        assert!((fast_log2(2.0) - 1.0).abs() < f32::EPSILON);
        assert!((fast_exp2(0.0) - 1.0).abs() < f32::EPSILON);
        assert!((fast_exp2(1.0) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn px0_fastmath_keeps_logistic_guards() {
        assert_eq!(fast_logistic(21.0), 1.0);
        assert_eq!(fast_logistic(-21.0), 0.0);
        assert!(fast_log(2.0) > 0.69);
    }
}
