//! ICCS 纵线记谱 → 引擎 UCI（Pikafish：`a0`～`i9`；单源为本模块）。

use anyhow::{anyhow, bail, Result};

/// 单格 ICCS（如 `C3`、`a0`）→ UCI 半串（`[a-i][0-9]`，纵坐标等于盘面条纹）。
pub fn iccs_half_to_uci(half: &str) -> Result<String> {
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
    if !(0..=9).contains(&r) {
        bail!("ICCS 纵坐标越界（须 0～9）: {half:?}");
    }
    Ok(format!("{}{}", f as char, r))
}

/// `C3-C4` / `c3c4` → 着法 UCI（四字符，纵坐标 0～9）。
pub fn iccs_move_to_uci(mv: &str) -> Result<String> {
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
    Ok(iccs_half_to_uci(a)? + &iccs_half_to_uci(b)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iccs_pair_to_uci() {
        assert_eq!(iccs_move_to_uci("c3-c4").unwrap(), "c3c4");
    }

    #[test]
    fn iccs_compact() {
        assert_eq!(iccs_move_to_uci("c3c4").unwrap(), "c3c4");
    }
}
