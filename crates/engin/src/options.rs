//! Engine 生命周期的 UCI option。
//!
//! 对照 px0 `src/engine.h` 的 `OptionsDict`：`Engine` 持有 option，搜索在
//! 启动 job 时读取需要的快照。

/// 正式 UCI 的最小 option 集。
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub weights_file: String,
    pub show_wdl: bool,
    pub show_eps: bool,
    /// UCI `MultiPV`：每次 info 最多输出的根变化数。
    pub multi_pv: usize,
    /// UCI `MiniBatchSize`：单次 NN 调用最多合并的局面数；0 使用 backend 建议值。
    pub mini_batch_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            weights_file: String::new(),
            show_wdl: false,
            show_eps: false,
            multi_pv: 1,
            mini_batch_size: 0,
        }
    }
}

impl Options {
    pub fn list_options_uci(&self) -> Vec<String> {
        vec![
            format!("option name UCI_ShowWDL type check default {}", self.show_wdl),
            format!("option name UCI_ShowEPS type check default {}", self.show_eps),
            format!("option name MultiPV type spin default {} min 1 max 500", self.multi_pv),
            format!(
                "option name MiniBatchSize type spin default {} min 0 max 1024",
                self.mini_batch_size
            ),
            format!("option name WeightsFile type string default {}", self.weights_file),
        ]
    }

    pub fn set_uci_option(&mut self, name: &str, value: &str) -> Result<(), crate::EnginError> {
        let flag = |name| match value {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(crate::EnginError::Uci(format!(
                "Flag '{name}' must be either true or false"
            ))),
        };
        match name {
            "UCI_ShowWDL" => self.show_wdl = flag(name)?,
            "UCI_ShowEPS" => self.show_eps = flag(name)?,
            "MultiPV" => {
                let value = value
                    .parse::<usize>()
                    .map_err(|_| crate::EnginError::Uci("MultiPV must be an integer within [1, 500]".into()))?;
                if !(1..=500).contains(&value) {
                    return Err(crate::EnginError::Uci("MultiPV must be within [1, 500]".into()));
                }
                self.multi_pv = value;
            }
            "WeightsFile" => self.weights_file = value.to_owned(),
            "MiniBatchSize" => {
                let value = value
                    .parse::<usize>()
                    .map_err(|_| crate::EnginError::Uci("MiniBatchSize must be an integer within [0, 1024]".into()))?;
                if value > 1024 {
                    return Err(crate::EnginError::Uci("MiniBatchSize must be within [0, 1024]".into()));
                }
                self.mini_batch_size = value;
            }
            _ => return Err(crate::EnginError::Uci(format!("Unknown option: {name}"))),
        }
        Ok(())
    }
}
