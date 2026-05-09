//! ICCS 纵线记谱 → pyffish UCI（历史 Python 实现已移除；单源为本模块）。

use anyhow::{anyhow, bail, Result};

/// 单格 ICCS（如 `C3`、`a0`）→ pyffish 半串（纵坐标为条纹+1，如 `c4`、`a1`）。
pub fn iccs_half_to_pyffish(half: &str) -> Result<String> {
    let s = half.trim();
    let b = s.as_bytes();
    if b.len() < 2 {
        bail!("非法 ICCS 半格: {half:?}");
    }
    let f = b[0].to_ascii_lowercase();
    if !matches!(f, b'a'..=b'i') {
        bail!("非法 ICCS 半格: {half:?}");
    }
    let num_str = std::str::from_utf8(&b[1..]).map_err(|_| anyhow!("ICCS 非 UTF-8"))?;
    let r: i32 = num_str.parse().map_err(|_| anyhow!("非法 ICCS 条纹数字: {half:?}"))?;
    Ok(format!("{}{}", f as char, r + 1))
}

/// `C3-C4` / `c3c4` → pyffish UCI。
pub fn iccs_move_to_pyffish(mv: &str) -> Result<String> {
    let t = mv.trim().replace(' ', "");
    let (a, b) = if let Some((x, y)) = t.split_once('-') {
        (x, y)
    } else {
        let b = t.as_bytes();
        if b.len() < 4 {
            bail!("无法解析 ICCS 着法: {mv:?}");
        }
        let split = b
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, &ch)| matches!(ch, b'a'..=b'i' | b'A'..=b'I'))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if split == 0 {
            bail!("无法解析 ICCS 着法: {mv:?}");
        }
        (&t[..split], &t[split..])
    };
    Ok(iccs_half_to_pyffish(a)? + &iccs_half_to_pyffish(b)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iccs_pair_to_pyffish() {
        assert_eq!(iccs_move_to_pyffish("c3-c4").unwrap(), "c4c5");
    }

    #[test]
    fn iccs_compact() {
        assert_eq!(iccs_move_to_pyffish("c3c4").unwrap(), "c4c5");
    }
}
