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
    /// UCI `VirtualLoss` 直接使用搜索参数 V，例如 `3.0`。
    pub virtual_loss: f32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            weights_file: String::new(),
            show_wdl: false,
            show_eps: false,
            virtual_loss: 3.0,
        }
    }
}

impl Options {
    pub fn list_options_uci(&self) -> Vec<String> {
        vec![
            format!("option name UCI_ShowWDL type check default {}", self.show_wdl),
            format!("option name UCI_ShowEPS type check default {}", self.show_eps),
            format!("option name WeightsFile type string default {}", self.weights_file),
            format!("option name VirtualLoss type string default {:.1}", self.virtual_loss),
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
            "WeightsFile" => self.weights_file = value.to_owned(),
            "VirtualLoss" => {
                let value = value
                    .parse::<f32>()
                    .map_err(|_| crate::EnginError::Uci("VirtualLoss must be a decimal within [0, 5]".into()))?;
                if !value.is_finite() || !(0.0..=5.0).contains(&value) {
                    return Err(crate::EnginError::Uci("VirtualLoss must be within [0, 5]".into()));
                }
                self.virtual_loss = value;
            }
            _ => return Err(crate::EnginError::Uci(format!("Unknown option: {name}"))),
        }
        Ok(())
    }
}
